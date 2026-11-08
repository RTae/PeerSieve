"""Fused Triton kernels for KVMend's tags (step 2 of decision 20: make the checks fast).

They compute exactly what kvmend.tags computes with separate PyTorch operations:
- page_tags: recompute the tag of each given page of one layer from its stored bytes, one program per page. A page
  is 16 slots of W 32-bit words for K and the same for V, contiguous in the pool, so each program reads two
  contiguous blocks; the work per word is a few integer operations, and the kernel runs at memory speed.
- store_update: at store time, reset the tag and mask of every page whose first slot is written, then add each
  stored token's share to its page's running tag (atomic add) and set its bit in the mask (atomic or). Two launches
  per layer replace about fifteen PyTorch operations, and neither waits for the host, so both run inside SGLang's
  decode CUDA graphs.
Tags are sums modulo 2^32: the kernels add in uint32, which wraps, and store the result zero-extended into int64.
"""
import torch
import triton
import triton.language as tl

PAGE = 16


@triton.jit
def _fmix32(h):
    """MurmurHash3's 32-bit finalizer on uint32 values (the multiplications wrap modulo 2^32)."""
    h ^= h >> 16
    h *= 0x85EBCA6B
    h ^= h >> 13
    h *= 0xC2B2AE35
    h ^= h >> 16
    return h


@triton.jit
def _page_tags_kernel(k_ptr, v_ptr, pages_ptr, mask_ptr, tk_ptr, pk_ptr, out_ptr,
                      W: tl.constexpr, BW: tl.constexpr, PG: tl.constexpr):
    i = tl.program_id(0)
    page = tl.load(pages_ptr + i).to(tl.int64)
    rows = tl.arange(0, PG)
    cols = tl.arange(0, BW)
    inside = (cols < W)[None, :] & (rows < PG)[:, None]
    words = (page * PG + rows)[:, None] * W + cols[None, :]
    kw = tl.load(k_ptr + words, mask=inside, other=0).to(tl.uint32, bitcast=True)
    vw = tl.load(v_ptr + words, mask=inside, other=0).to(tl.uint32, bitcast=True)
    keys = rows[:, None] * W + cols[None, :]
    pkk = tl.load(pk_ptr + keys, mask=inside, other=0).to(tl.uint32)
    pkv = tl.load(pk_ptr + PG * W + keys, mask=inside, other=0).to(tl.uint32)
    tk = tl.load(tk_ptr + i * PG + rows).to(tl.uint32)
    h = _fmix32(kw ^ pkk ^ tk[:, None]) + _fmix32(vw ^ pkv ^ tk[:, None])
    h = tl.where(inside, h, tl.zeros_like(h))
    per_token = tl.sum(h, axis=1)
    covered = ((tl.load(mask_ptr + i) >> rows) & 1) != 0
    tag = tl.sum(tl.where(covered, per_token, tl.zeros_like(per_token)), axis=0)
    tl.store(out_ptr + i, tag.to(tl.int64))


@triton.jit
def _store_reset_kernel(loc_ptr, tk_ptr, tags_ptr, mask_ptr, tkey_ptr, SET_KEYS: tl.constexpr, PG: tl.constexpr):
    t = tl.program_id(0)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    page = loc // PG
    if loc - page * PG == 0:  # a page's first slot is written: the page starts over (a reused page starts clean)
        tl.store(tags_ptr + page, tl.zeros((), dtype=tl.int64))
        tl.store(mask_ptr + page, tl.zeros((), dtype=tl.int32))
    if SET_KEYS:
        tl.store(tkey_ptr + loc, tl.load(tk_ptr + t))


@triton.jit
def _store_add_kernel(k_ptr, v_ptr, loc_ptr, tk_ptr, pk_ptr, tags_ptr, mask_ptr,
                      W: tl.constexpr, BW: tl.constexpr, PG: tl.constexpr):
    t = tl.program_id(0)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    page = loc // PG
    off = loc - page * PG
    cols = tl.arange(0, BW)
    inside = cols < W
    kw = tl.load(k_ptr + loc * W + cols, mask=inside, other=0).to(tl.uint32, bitcast=True)
    vw = tl.load(v_ptr + loc * W + cols, mask=inside, other=0).to(tl.uint32, bitcast=True)
    pkk = tl.load(pk_ptr + off * W + cols, mask=inside, other=0).to(tl.uint32)
    pkv = tl.load(pk_ptr + PG * W + off * W + cols, mask=inside, other=0).to(tl.uint32)
    tk = tl.load(tk_ptr + t).to(tl.uint32)
    h = _fmix32(kw ^ pkk ^ tk) + _fmix32(vw ^ pkv ^ tk)
    share = tl.sum(tl.where(inside, h, tl.zeros_like(h)), axis=0)
    tl.atomic_add(tags_ptr + page, share.to(tl.int64))
    tl.atomic_or(mask_ptr + page, (tl.full((), 1, tl.int64) << off).to(tl.int32))


@triton.jit
def _check_layers_kernel(kptrs, vptrs, pages_ptr, exp_ptr, tk_ptr, pk_ptr, tags_ptr, mask_ptr, out_ptr, n, n_pool,
                         W: tl.constexpr, BW: tl.constexpr, PG: tl.constexpr):
    """One program per (page of the list, layer). Code 0: the page's tag covers exactly the expected offsets and
    matches; 1: the coverage differs; 2: the tag differs. Each layer's K and V buffers come from a table of addresses,
    so one launch checks every layer."""
    i = tl.program_id(0)
    l = tl.program_id(1)
    page = tl.load(pages_ptr + i).to(tl.int64)
    k_ptr = tl.load(kptrs + l).to(tl.pointer_type(tl.int32))
    v_ptr = tl.load(vptrs + l).to(tl.pointer_type(tl.int32))
    rows = tl.arange(0, PG)
    cols = tl.arange(0, BW)
    inside = (cols < W)[None, :] & (rows < PG)[:, None]
    words = (page * PG + rows)[:, None] * W + cols[None, :]
    kw = tl.load(k_ptr + words, mask=inside, other=0).to(tl.uint32, bitcast=True)
    vw = tl.load(v_ptr + words, mask=inside, other=0).to(tl.uint32, bitcast=True)
    keys = rows[:, None] * W + cols[None, :]
    pkk = tl.load(pk_ptr + keys, mask=inside, other=0).to(tl.uint32)
    pkv = tl.load(pk_ptr + PG * W + keys, mask=inside, other=0).to(tl.uint32)
    tk = tl.load(tk_ptr + i * PG + rows).to(tl.uint32)
    h = _fmix32(kw ^ pkk ^ tk[:, None]) + _fmix32(vw ^ pkv ^ tk[:, None])
    h = tl.where(inside, h, tl.zeros_like(h))
    per_token = tl.sum(h, axis=1)
    want = tl.load(exp_ptr + i)
    covered = ((want >> rows) & 1) != 0
    tag = tl.sum(tl.where(covered, per_token, tl.zeros_like(per_token)), axis=0)
    stored = tl.load(tags_ptr + l * n_pool + page).to(tl.uint32)  # tags are kept modulo 2^32: compare the low bits
    have = tl.load(mask_ptr + l * n_pool + page)
    code = tl.where(have != want, 1, tl.where(tag != stored, 2, 0))
    tl.store(out_ptr + l * n + i, code.to(tl.int8))


def check_layers(kptrs: torch.Tensor, vptrs: torch.Tensor, layers: int, pages: torch.Tensor, expected: torch.Tensor,
                 tk: torch.Tensor, pkeys: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor, w: int) -> torch.Tensor:
    """Codes [layers, n] (int8) of the reader's check of `pages` in every layer, in one launch: 0 pass, 1 coverage,
    2 tag. kptrs, vptrs: int64 [layers] addresses of each layer's K and V buffers; expected [n]: offsets the reader
    reads; tk [n, PAGE]: its token keys; tags int64 and mask int32 [layers, pool pages]."""
    n = pages.numel()
    out = torch.empty((layers, n), dtype=torch.int8, device=pages.device)
    if n:
        _check_layers_kernel[(n, layers)](kptrs, vptrs, pages.contiguous(), expected.to(torch.int32).contiguous(),
                                          tk.to(torch.int64).contiguous(), pkeys, tags, mask, out, n, tags.shape[1],
                                          W=w, BW=triton.next_power_of_2(w), PG=PAGE)
    return out


def words(buf: torch.Tensor) -> torch.Tensor:
    """A layer's K or V buffer [slots, heads, dim] as [slots, W] 32-bit words."""
    return buf.view(torch.int32).view(buf.shape[0], -1)


def page_tags(k_words: torch.Tensor, v_words: torch.Tensor, pages: torch.Tensor, mask: torch.Tensor,
              tk: torch.Tensor, pkeys: torch.Tensor) -> torch.Tensor:
    """Tags of `pages` of one layer, from the stored bytes. mask [n]: offsets to cover; tk [n, PAGE]: token keys;
    pkeys [2, PAGE, W]: position keys. Returns int64 [n], to be compared modulo 2^32."""
    n = pages.numel()
    out = torch.empty(n, dtype=torch.int64, device=pages.device)
    if n:
        w = k_words.shape[1]
        _page_tags_kernel[(n,)](k_words, v_words, pages.contiguous(), mask.to(torch.int32).contiguous(),
                                tk.to(torch.int64).contiguous(), pkeys, out, W=w, BW=triton.next_power_of_2(w),
                                PG=PAGE)
    return out


def store_update(k_words: torch.Tensor, v_words: torch.Tensor, loc: torch.Tensor, tk: torch.Tensor,
                 pkeys: torch.Tensor, tags_row: torch.Tensor, mask_row: torch.Tensor, tkey: torch.Tensor = None):
    """Add the stored tokens' shares to one layer's running tags (tags_row int64 [pages], mask_row int32 [pages]).
    With tkey, also record each slot's token key."""
    n = loc.numel()
    if not n:
        return
    loc = loc.to(torch.int64).contiguous()
    tk = tk.to(torch.int64).contiguous()
    w = k_words.shape[1]
    _store_reset_kernel[(n,)](loc, tk, tags_row, mask_row, tkey if tkey is not None else tk,
                              SET_KEYS=tkey is not None, PG=PAGE)
    _store_add_kernel[(n,)](k_words, v_words, loc, tk, pkeys, tags_row, mask_row, W=w,
                            BW=triton.next_power_of_2(w), PG=PAGE)
