"""Checks on read inside the decode attention (kvmend/attn_check.py) against SGLang 0.5.1's own kernels. GPU only, and
SGLang must be importable; skipped otherwise.

Each test builds a fake KV pool of random K and V, writes the pieces' tags through KVMend's store path, and runs one
decode step of six requests (full pages, a prefix page shared by two requests, open pages of 1 to 12 tokens, up to
19 pages and 5 kv splits per request) through SGLang's decode_attention_fwd and through the copy with the check:
- the attention output and the stage-1 partial results are SGLang's bit for bit;
- each page's accumulated tag is readers x tag_kernels.page_tags over the offsets read (modulo 2^32), its coverage
  is those offsets and its reader count is right; no other page is touched;
- the verdict passes, then fails exactly the damaged piece after a flipped bit in K or V, and flags a slot that is
  read but was never written;
- in a CUDA graph with padding entries (sequence length 1 at a live page's first slot, as SGLang pads a replayed
  batch), every replay checks its own batch and the padding adds nothing;
- the patched TritonAttnBackend checks each decode step and the scheduler's verdict logs a fault with its reader.
Shapes: TinyLlama at TP=2 (2 KV heads, 8 query heads each), Llama-3.1-8B at TP=8 (1 KV head, 4 query heads), one
query head per KV head (SGLang's other kernel), 32 query heads per KV head (two programs read one KV head) and a
head dimension of 80 (masked lanes). Every kernel variant runs with the words formed from the attention's tiles
(reuse) and read again (reload).
"""
import inspect
import types

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
PAGE = 16
M32 = 0xFFFFFFFF
LAYERS = 2
POOL_PAGES = 48
MAX_SPLITS = 8
SHAPES = {  # one GPU's KV heads, query heads, head dim, dtype
    "tinyllama_tp2": (2, 16, 64, torch.bfloat16),
    "llama8b_tp8": (1, 4, 128, torch.bfloat16),
    "mha": (4, 4, 128, torch.float16),
    "group32": (1, 32, 64, torch.bfloat16),
    "dim80": (2, 8, 80, torch.bfloat16),
}
# One decode step: (tokens read, pages, kv splits). R0 and R1 share pages 3 and 4 (a cached prefix).
REQS = [(70, [3, 4, 5, 6, 7], 8), (40, [3, 4, 9], 2), (32, [11, 12], 1), (5, [14], 3),
        (300, list(range(20, 39)), 8), (17, [40, 41], 4)]


def _modules():
    sgl = pytest.importorskip("sglang.srt.layers.attention.triton_ops.decode_attention")
    from kvmend import attn_check
    return sgl, attn_check


def _same_bits(a, b):
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.view(view), b.view(view))


def _slot(pages, n):
    return pages[n // PAGE] * PAGE + n % PAGE


class Case:
    """A fake KV pool with tags written for REQS, and the batch tensors of one decode step."""

    def __init__(self, shape, seed=0, reuse=True):
        _modules()  # skip without SGLang
        from kvmend import hook
        from kvmend import tags as T

        self.heads, self.qh, self.dim, self.dtype = SHAPES[shape]
        self.name = shape
        g = torch.Generator(device="cuda").manual_seed(seed)

        def buf():
            return torch.randn(POOL_PAGES * PAGE, self.heads, self.dim, device="cuda", generator=g).to(self.dtype)

        self.pool = types.SimpleNamespace(page_size=PAGE, start_layer=0, k_buffer=[buf() for _ in range(LAYERS)],
                                          v_buffer=[buf() for _ in range(LAYERS)])
        self.st = hook.PoolState(self.pool)
        self.st.use_attn_check(reuse=reuse)
        self.chk = self.st.attn
        cpu = torch.Generator().manual_seed(seed)
        ids = [torch.randint(0, 32000, (n,), generator=cpu) for n, _, _ in REQS]
        ids[1][:32] = ids[0][:32]  # the shared prefix: same tokens at the same positions
        written = {}  # slot -> (token id, position); a shared page is written once
        for r, (n, pages, _) in enumerate(REQS):
            for i in range(n):
                written.setdefault(_slot(pages, i), (int(ids[r][i]), i))
        slots = sorted(written)
        loc = torch.tensor(slots, device="cuda")
        tk = T.token_keys(torch.tensor([written[s][0] for s in slots]), torch.tensor([written[s][1] for s in slots]))
        for layer in range(LAYERS):
            self.st.update(layer, loc, tk.cuda())  # KVMend's store path: running tags, masks and token keys
        lens = [n for n, _, _ in REQS]
        self.kv_indptr = torch.tensor([sum(lens[:i]) for i in range(len(lens) + 1)], dtype=torch.int32, device="cuda")
        self.kv_indices = torch.tensor([_slot(p, i) for n, p, _ in REQS for i in range(n)], dtype=torch.int64,
                                       device="cuda")
        self.splits = torch.tensor([s for _, _, s in REQS], dtype=torch.int32, device="cuda")
        self.q = torch.randn(len(REQS), self.qh, self.dim, device="cuda", generator=g).to(self.dtype)
        self.scale = self.dim ** -0.5
        exp = {}  # page -> (readers, offsets read)
        for n, pages, _ in REQS:
            for j, p in enumerate(pages):
                r, m = exp.get(p, (0, 0))
                exp[p] = (r + 1, m | ((1 << min(PAGE, n - j * PAGE)) - 1))
        self.exp = exp

    def buffers(self, bs=None):
        bs = len(REQS) if bs is None else bs
        return (torch.zeros(bs, self.qh, self.dim, dtype=self.dtype, device="cuda"),
                torch.zeros(bs, self.qh, MAX_SPLITS, self.dim, dtype=torch.float32, device="cuda"),
                torch.zeros(bs, self.qh, MAX_SPLITS, dtype=torch.float32, device="cuda"))

    def sglang(self, layer):
        sgl, _ = _modules()
        o, logits, lse = self.buffers()
        sgl.decode_attention_fwd(self.q, self.pool.k_buffer[layer], self.pool.v_buffer[layer], o, self.kv_indptr,
                                 self.kv_indices, logits, lse, self.splits, MAX_SPLITS, self.scale)
        return o, logits, lse

    def checked(self, layer, check=True, reuse=None):
        o, logits, lse = self.buffers()
        self.chk.decode(layer, self.q, self.pool.k_buffer[layer], self.pool.v_buffer[layer], o, self.kv_indptr,
                        self.kv_indices, logits, lse, self.splits, MAX_SPLITS, self.scale, check=check, reuse=reuse)
        return o, logits, lse

    def step(self, reuse=None):
        """One decode step through the check, every layer, then the verdict: (found, pieces read, anomalies)."""
        for layer in range(LAYERS):
            self.checked(layer, reuse=reuse)
        return self.chk.verdict()

    def expected(self, layer):
        """(pages, readers, offsets, readers x page_tags mod 2^32) of the pages this step reads."""
        from kvmend import tag_kernels as K

        pages = sorted(self.exp)
        p = torch.tensor(pages, device="cuda")
        readers = torch.tensor([self.exp[x][0] for x in pages], dtype=torch.int64, device="cuda")
        offs = torch.tensor([self.exp[x][1] for x in pages], dtype=torch.int32, device="cuda")
        tags = K.page_tags(K.words(self.pool.k_buffer[layer]), K.words(self.pool.v_buffer[layer]), p, offs,
                           self.st.tkey.view(-1, PAGE)[p], self.st.keys)
        return p, readers, offs, (tags * readers) & M32

    def assert_accumulated(self, layer, what):
        p, readers, offs, want = self.expected(layer)
        c = self.chk
        assert torch.equal(c.acc_mask[layer, p], offs), \
            f"{what}: coverage {c.acc_mask[layer, p].tolist()} != offsets read {offs.tolist()}"
        assert torch.equal(c.acc_cnt[layer, p].to(torch.int64), readers), \
            f"{what}: readers {c.acc_cnt[layer, p].tolist()} != {readers.tolist()}"
        got = c.acc_tag[layer, p] & M32
        bad = (got != want).nonzero().flatten().tolist()
        assert not bad, f"{what}: accumulated tags differ from readers x page_tags on pages {[int(p[i]) for i in bad]}"
        stored = (self.st.tags[layer, p] * readers) & M32  # the running tags cover exactly the offsets read
        assert torch.equal(stored, want), f"{what}: page_tags over the offsets read differs from the stored tags"
        rest = torch.ones(POOL_PAGES, dtype=torch.bool, device="cuda")
        rest[p] = False
        assert not (c.acc_mask[layer][rest].any() or c.acc_tag[layer][rest].any() or c.acc_cnt[layer][rest].any()), \
            f"{what}: pages that no request reads were touched: {rest.nonzero().flatten()[c.acc_mask[layer][rest] != 0].tolist()}"


def _flip(buf, slot, head, elem, bit):
    """Flip one bit of one 16-bit K or V value (bit < 15)."""
    buf.view(torch.int16)[slot, head, elem] ^= 1 << bit


@cuda
@pytest.mark.parametrize("reuse", [True, False], ids=["reuse", "reload"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_bit_exact_attention_and_accumulated_tags(shape, reuse):
    c = Case(shape, seed=1, reuse=reuse)
    what = f"{shape}, {'reuse' if reuse else 'reload'}"
    for layer in range(LAYERS):
        ref = c.sglang(layer)
        got = c.checked(layer)
        for name, a, b in zip(("output", "stage-1 partial output", "stage-1 log-sum-exp"), ref, got):
            diff = (a.float() - b.float()).abs().max().item()
            assert _same_bits(a, b), f"{what}, layer {layer}: {name} differs from SGLang's (max abs diff {diff})"
        torch.cuda.synchronize()
        c.assert_accumulated(layer, f"{what}, layer {layer}")
    found, read, layout = c.chk.verdict()
    assert found == [] and layout == 0, f"{what}: the verdict failed a clean step: {found}, layout anomalies {layout}"
    assert read == len(c.exp) * LAYERS, f"{what}: the verdict saw {read} pieces read, not {len(c.exp) * LAYERS}"
    for t in (c.chk.acc_tag, c.chk.acc_mask, c.chk.acc_cnt, c.chk.stat):
        assert not t.any(), f"{what}: the verdict left an accumulator nonzero"


@cuda
@pytest.mark.parametrize("shape", ["tinyllama_tp2", "mha"])
def test_the_copy_without_the_check_is_sglangs_kernel(shape):
    """With CHECK off, the copy compiles to SGLang's kernel: if this passes and the test above fails, the added check
    changed the attention's code generation."""
    c = Case(shape, seed=3)
    for layer in range(LAYERS):
        assert all(_same_bits(a, b) for a, b in zip(c.sglang(layer), c.checked(layer, check=False))), (shape, layer)
    torch.cuda.synchronize()
    assert not (c.chk.acc_mask.any() or c.chk.acc_tag.any()), "CHECK off must not accumulate"


@cuda
@pytest.mark.parametrize("reuse", [True, False], ids=["reuse", "reload"])
def test_a_flipped_bit_or_an_unwritten_slot_fails_exactly_its_piece(reuse):
    c = Case("tinyllama_tp2", seed=2, reuse=reuse)
    k1, v0 = c.pool.k_buffer[1], c.pool.v_buffer[0]
    assert c.step()[0] == [], "a clean step failed"
    _flip(k1, 5 * PAGE + 7, 1, 3, 9)  # layer 1, page 5 (read by R0 only), K of head 1
    assert c.step()[0] == [(1, 5, "tag")]
    _flip(k1, 5 * PAGE + 7, 1, 3, 9)
    _flip(v0, 3 * PAGE + 12, 0, 60, 2)  # layer 0, page 3 (read by R0 and R1), V of head 0
    assert c.step()[0] == [(0, 3, "tag")]
    _flip(v0, 3 * PAGE + 12, 0, 60, 2)
    _flip(c.pool.k_buffer[0], 14 * PAGE + 4, 0, 0, 0)  # the newest token of R3's open page, lowest bit
    assert c.step()[0] == [(0, 14, "tag")]
    _flip(c.pool.k_buffer[0], 14 * PAGE + 4, 0, 0, 0)
    c.st.mask[1, 12] ^= 1 << 2  # layer 1, page 12: slot 2 is read, but its write never happened
    assert c.step()[0] == [(1, 12, "coverage")]
    c.st.mask[1, 12] ^= 1 << 2
    assert c.step()[0] == [], "the restored pool still fails"


@cuda
def test_a_token_table_that_is_not_page_by_page_is_reported():
    """Two positions of one page whose slots are swapped: the kernel counts layout anomalies, and the page fails."""
    c = Case("tinyllama_tp2", seed=4)
    a, b = 40, 41  # R0's positions 40 and 41, page 5
    c.kv_indices[[a, b]] = c.kv_indices[[b, a]]
    found, _, layout = c.step()
    assert layout > 0, "the swapped slots were not counted"
    assert sorted(found) == [(layer, 5, "tag") for layer in range(LAYERS)], found


@cuda
def test_cuda_graph_replays_check_their_own_batch_and_padding_adds_nothing():
    """SGLang captures decode for a padded batch size and replays it with new contents in the same buffers. Padding
    entries have sequence length 1 and read the first slot of a stale request row, here a live page that a real entry
    also reads in the same step: if the padding added its share, that page would fail."""
    sgl, _ = _modules()
    c = Case("tinyllama_tp2", seed=6)
    bs, real = 4, 2
    batches = [(0, 3), (4, 1)]  # pairs of REQS rows
    most = max(REQS[i][0] + REQS[j][0] for i, j in batches) + bs
    q = torch.zeros(bs, c.qh, c.dim, dtype=c.dtype, device="cuda")
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    kv_indices = torch.zeros(most, dtype=torch.int64, device="cuda")
    splits = torch.full((bs,), MAX_SPLITS, dtype=torch.int32, device="cuda")
    outs = [c.buffers(bs) for _ in range(LAYERS)]

    def fill(rows):
        lens = [REQS[r][0] for r in rows] + [1] * (bs - real)
        idx = [_slot(REQS[r][1], i) for r in rows for i in range(REQS[r][0])]
        idx += [REQS[rows[0]][1][0] * PAGE] * (bs - real)  # padding: offset 0 of the first real entry's first page
        kv_indptr.copy_(torch.tensor([sum(lens[:i]) for i in range(bs + 1)], dtype=torch.int32))
        kv_indices[:len(idx)].copy_(torch.tensor(idx, dtype=torch.int64))
        splits.copy_(torch.tensor([REQS[r][2] for r in rows] + [MAX_SPLITS] * (bs - real), dtype=torch.int32))
        q.copy_(torch.randn(bs, c.qh, c.dim, device="cuda").to(c.dtype))

    def forward():
        for layer, (o, logits, lse) in enumerate(outs):
            c.chk.decode(layer, q, c.pool.k_buffer[layer], c.pool.v_buffer[layer], o, kv_indptr, kv_indices, logits,
                         lse, splits, MAX_SPLITS, c.scale)

    def expect(rows, layer):
        """Accumulated tags of the real entries only, and SGLang's output for them, run eagerly at bs=2."""
        exp = {}
        for r in rows:
            n, pages, _ = REQS[r]
            for j, p in enumerate(pages):
                k, m = exp.get(p, (0, 0))
                exp[p] = (k + 1, m | ((1 << min(PAGE, n - j * PAGE)) - 1))
        c.exp = exp
        o, logits, lse = c.buffers(real)
        sgl.decode_attention_fwd(q[:real], c.pool.k_buffer[layer], c.pool.v_buffer[layer], o, kv_indptr[:real + 1],
                                 kv_indices, logits, lse, splits[:real], MAX_SPLITS, c.scale)
        return o

    fill(batches[0])
    forward()  # eager: compiles every kernel before the capture, and checks that padding is skipped here too
    for layer in range(LAYERS):
        expect(batches[0], layer)
        c.assert_accumulated(layer, f"eager with padding, layer {layer}")
    assert c.chk.verdict()[0] == []
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    for rows in batches:
        fill(rows)
        graph.replay()
        torch.cuda.synchronize()
        for layer in range(LAYERS):
            ref = expect(rows, layer)
            assert _same_bits(outs[layer][0][:real], ref), f"replay of {rows}, layer {layer}: output differs"
            c.assert_accumulated(layer, f"replay of {rows}, layer {layer}")
        found, read, layout = c.chk.verdict()
        assert found == [] and layout == 0 and read == len(c.exp) * LAYERS, (rows, found, read, layout)
    _flip(c.pool.k_buffer[1], REQS[4][1][7] * PAGE + 9, 1, 30, 11)  # R4's 8th page, layer 1
    graph.replay()
    assert c.chk.verdict()[0] == [(1, REQS[4][1][7], "tag")]


class _Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


def _orig_decode(*args, **kwargs):  # stands in for SGLang's kernel where the patch must leave it alone
    raise AssertionError("not called")


@cuda
def test_the_patched_backend_checks_each_decode_step_and_the_verdict_logs_a_fault(monkeypatch):
    """The glue in kvmend/hook.py, without SGLang's scheduler: the patched pool stores K and V with running tags,
    the patched backend runs the check for the pool's own buffers and SGLang's kernel for any other buffer, and
    attn_verdict, as process_batch_result calls it, passes clean steps and logs a flipped bit with its reader."""
    sgl, _ = _modules()
    from kvmend import hook

    monkeypatch.setattr(hook, "CHECK_IN_ATTN", True)
    monkeypatch.setattr(hook, "_STATES", [])
    monkeypatch.setattr(hook, "_BUF", [])
    heads, qh, dim = 2, 16, 64
    g = torch.Generator(device="cuda").manual_seed(7)

    class FakePool:  # the fields of MHATokenToKVPool that the patch uses
        def __init__(self):
            self.page_size, self.start_layer = PAGE, 0
            self.k_buffer = [torch.randn(24 * PAGE, heads, dim, device="cuda", generator=g).to(torch.bfloat16)
                             for _ in range(LAYERS)]
            self.v_buffer = [torch.randn(24 * PAGE, heads, dim, device="cuda", generator=g).to(torch.bfloat16)
                             for _ in range(LAYERS)]

        def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
            self.k_buffer[layer.layer_id][loc] = cache_k
            self.v_buffer[layer.layer_id][loc] = cache_v

    class Backend:  # TritonAttnBackend: __init__ sets decode_attention_fwd, as SGLang 0.5.1's does
        sliding_window_size = None

        def __init__(self):
            self.decode_attention_fwd = torch.compiler.disable(sgl.decode_attention_fwd)

    class WindowBackend:
        sliding_window_size = 4096

        def __init__(self):
            self.decode_attention_fwd = _orig_decode

    hook.patch_memory_pool(types.SimpleNamespace(MHATokenToKVPool=FakePool))
    hook.patch_triton_backend(types.SimpleNamespace(TritonAttnBackend=Backend))
    hook.patch_triton_backend(types.SimpleNamespace(TritonAttnBackend=WindowBackend))
    assert WindowBackend().decode_attention_fwd is _orig_decode, "a backend with sliding windows must keep its kernel"
    pool, backend = FakePool(), Backend()
    st = pool._kvm
    assert st.attn is not None
    prompts, pages = (20, 35), ([2, 3], [5, 6, 7])
    table = torch.zeros(2, 64, dtype=torch.int32, device="cuda")
    for r in range(2):
        table[r, :len(pages[r]) * PAGE] = torch.tensor([_slot(pages[r], i) for i in range(len(pages[r]) * PAGE)])
    cpu = torch.Generator().manual_seed(8)
    ids = [torch.randint(0, 32000, (48,), generator=cpu) for _ in range(2)]
    reqs = [types.SimpleNamespace(rid=f"R{r}", req_pool_idx=r, origin_input_ids=ids[r][:prompts[r]].tolist(),
                                  output_ids=[int(ids[r][prompts[r]])]) for r in range(2)]
    sched = types.SimpleNamespace(tp_rank=0, req_to_token_pool=types.SimpleNamespace(req_to_token=table))
    batch = types.SimpleNamespace(reqs=reqs)

    def store(loc, tok, pos):
        fb = types.SimpleNamespace(out_cache_loc=loc, input_ids=tok, positions=pos)
        monkeypatch.setattr(hook, "_FB", [fb])
        n = loc.numel()
        for layer in range(LAYERS):
            pool.set_kv_buffer(_Layer(layer), loc,
                               torch.randn(n, heads, dim, device="cuda", generator=g).to(torch.bfloat16),
                               torch.randn(n, heads, dim, device="cuda", generator=g).to(torch.bfloat16))

    # the prefill of both prompts, one pass
    pos = [torch.arange(n) for n in prompts]
    store(torch.cat([table[r, :prompts[r]] for r in range(2)]).to(torch.int64),
          torch.cat([ids[r][:prompts[r]] for r in range(2)]).cuda(), torch.cat(pos).cuda())
    q = torch.randn(2, qh, dim, device="cuda", generator=g).to(torch.bfloat16)
    splits = torch.tensor([2, 8], dtype=torch.int32, device="cuda")
    damaged = None
    for t in range(4):
        n = [prompts[r] + t + 1 for r in range(2)]  # tokens read: the prompt, the outputs so far and the new token
        new = [prompts[r] + t for r in range(2)]
        store(torch.tensor([_slot(pages[r], new[r]) for r in range(2)], device="cuda"),
              torch.tensor([reqs[r].output_ids[-1] for r in range(2)], device="cuda"),
              torch.tensor(new, device="cuda"))
        if t == 2:  # a bit of R0's prompt (position 3, page 2) flips in layer 1 before this step reads it
            _flip(pool.k_buffer[1], _slot(pages[0], 3), 0, 10, 13)
            damaged = (0, 1, pages[0][0], "tag")
        kv_indptr = torch.tensor([0, n[0], n[0] + n[1]], dtype=torch.int32, device="cuda")
        kv_indices = torch.cat([table[r, :n[r]] for r in range(2)]).to(torch.int64)
        for layer in range(LAYERS):
            o = torch.empty(2, qh, dim, dtype=torch.bfloat16, device="cuda")
            logits = torch.zeros(2, qh, MAX_SPLITS, dim, device="cuda")
            lse = torch.zeros(2, qh, MAX_SPLITS, device="cuda")
            backend.decode_attention_fwd(q, pool.k_buffer[layer], pool.v_buffer[layer], o, kv_indptr, kv_indices,
                                         logits, lse, splits, MAX_SPLITS, dim ** -0.5)
            if t == 0 and layer == 1:  # the wrapper's output is SGLang's
                ref = torch.empty_like(o)
                sgl.decode_attention_fwd(q, pool.k_buffer[layer], pool.v_buffer[layer], ref, kv_indptr, kv_indices,
                                         torch.zeros_like(logits), torch.zeros_like(lse), splits, MAX_SPLITS,
                                         dim ** -0.5)
                assert _same_bits(o, ref), "the patched backend's output differs from SGLang's"
        found = hook.attn_verdict(st, sched, batch)
        assert found == ([damaged] if damaged else []), (t, found)
        for r in range(2):
            reqs[r].output_ids.append(int(ids[r][new[r] + 1]))
    logged = [x for x in hook._BUF if x["op"] == "attn_found"]
    assert len(logged) == 2 and [tuple(f) for f in logged[0]["found"]] == [damaged], logged
    assert logged[0]["rids"] == ["R0"], logged[0]
    assert st.attn.stats["steps"] == 4 and st.attn.stats["pieces"] > 0
    # buffers of no watched pool go to SGLang's kernel, untouched by the check
    other_k, other_v = pool.k_buffer[0].clone(), pool.v_buffer[0].clone()
    o = torch.empty(2, qh, dim, dtype=torch.bfloat16, device="cuda")
    backend.decode_attention_fwd(q, other_k, other_v, o, kv_indptr, kv_indices, torch.zeros_like(logits),
                                 torch.zeros_like(lse), splits, MAX_SPLITS, dim ** -0.5)
    torch.cuda.synchronize()
    assert not st.attn.acc_mask.any(), "the check ran on buffers it does not watch"


def _lines(kernel, drop_marked):
    lines = []
    for line in inspect.getsource(kernel.fn).splitlines():
        if drop_marked and "# KVMend" in line:
            continue
        s = line.strip()
        if s and not s.startswith(("#", "@")):
            lines.append(s)
    return lines


@pytest.mark.parametrize("name", ["_fwd_kernel_stage1", "_fwd_grouped_kernel_stage1"])
def test_the_copies_keep_every_line_of_the_installed_sglang_kernels(name):
    """The copies with the lines marked '# KVMend' removed are SGLang's kernels, line for line: a new SGLang that
    changes its kernels makes this fail instead of silently mixing two versions."""
    sgl, A = _modules()
    theirs = _lines(getattr(sgl, name), False)
    ours = [x.replace(f"def {name}_check(", f"def {name}(") for x in _lines(getattr(A, name + "_check"), True)]
    assert ours == theirs
