"""KVMend as a small patch on SGLang 0.5.1, loaded into every SGLang process by hook_site/sitecustomize.py.

It changes four places and nothing else:
- MHATokenToKVPool: keeps a running tag for every piece (one 16-token page, one layer, this GPU) as K and V are
  stored (set_kv_buffer). Each token's share is keyed by its token id and position (the token key), taken from the
  forward batch. With KVMEND_TAGS=0 (parity alone, as in GhostServe) it only tracks which pages are full;
- RadixAttention.forward: remembers the forward batch, so that set_kv_buffer knows the ids and positions it stores;
- PagedTokenToKVPoolAllocator.free: drops the tags of freed pages and moves them to a new generation;
- Scheduler: after every iteration (process_batch_result, self_check_during_idle), runs the actions of a JSON plan,
  and, when KVMEND_ENCODE_EVERY or KVMEND_SCRUB_PIECES is set, encodes parity and scrubs in the background.

Every tensor-parallel rank runs the actions at the same point of the same iteration, so their collectives line up.
The plan (KVMEND_PLAN) maps a request id to actions, each due when the request has produced `at` output tokens:
encode, scrub, inject, repair, retract, dump, and, for the comparison with GhostServe, repair_share and evict; for
the incident's faults, fault (a never-written page, a late write through or around the store path, a page holding
another sequence's KV), check (the reader's check of one request's pages) and mend (parity rebuild where one GPU is
damaged, recompute in place otherwise).
Each rank keeps one JSON line per action in memory and appends them to KVMEND_OUT/rank<r>.jsonl when the scheduler
is idle or flushes its cache, so slow file writes (the cluster's /home is NFS) never fall inside a timed step.
Run SGLang with page_size=16 and disable_overlap_schedule=True.
"""
import json
import os
import time
import traceback

import torch
import torch.distributed as dist

from kvmend import tags as T

PAGE = 16
FULL = (1 << PAGE) - 1
OUT = os.environ.get("KVMEND_OUT", "/tmp/kvmend")
TAGS_ON = os.environ.get("KVMEND_TAGS", "1") == "1"
CHUNK_PAGES = int(os.environ.get("KVMEND_CHUNK_PAGES", "256"))
SYNC_EVERY = int(os.environ.get("KVMEND_SYNC_EVERY", "64"))  # background scrub: batch steps between alarm sums
CTRL = os.environ.get("KVMEND_CTRL", "")  # JSON file read at every cache flush: encode_every, scrub_pieces, label
FENCE_ON = os.environ.get("KVMEND_FENCE", "1") == "1"  # stores that carry a stale generation are dropped
FUSED_ON = os.environ.get("KVMEND_FUSED", "0") == "1"  # tags by the fused Triton kernels (kvmend/tag_kernels.py)
STRIPE = os.environ.get("KVMEND_STRIPE", "gpus")  # parity stripes across the GPUs, or "local": within each GPU
# The release rule: request ids under strict release, as "rid" or "rid:every" ("*" for all). Their tokens leave only
# after a check covers them; a check runs every `every` output tokens (default KVMEND_STRICT_EVERY) and at finish.
STRICT = {rid: int(every) if every else None for rid, _, every in
          (x.partition(":") for x in os.environ.get("KVMEND_STRICT", "").split(",") if x)}
STRICT_EVERY = int(os.environ.get("KVMEND_STRICT_EVERY", "1"))
_FB = [None]  # the forward batch whose attention layers run now (RadixAttention.forward sets it)
_STATES = []  # the pool states of this process (one per KV pool), for patches that have no scheduler at hand


def new_stats():
    return dict(batch_steps=0, encodes=0, pages_encoded=0, encode_s=0.0, unclean=0, scrub_pieces=0,
                scrub_cpu_s=0.0, alarms=0, t_first=None, t_last=None)


# ---------------------------------------------------------------------------------------------------------------
# Per-rank state of the KV pool

class PoolState:
    """Tags, parity and repair state for one rank's KV pool."""

    def __init__(self, pool):
        _STATES.append(self)
        self.pool = pool
        k0 = pool.k_buffer[0]
        slots, self.heads, self.dim = k0.shape
        assert slots % PAGE == 0, f"KV pool of {slots} slots is not a whole number of {PAGE}-token pages"
        self.pages = slots // PAGE
        self.layers = len(pool.k_buffer)
        self.device = k0.device
        self.elem = k0.element_size()
        self.words = self.heads * self.dim * self.elem // 4
        self.keys = T.position_keys(PAGE, self.words, device=self.device)
        self.bit = (1 << torch.arange(PAGE, dtype=torch.int64, device=self.device)).to(torch.int32)
        self.tags = torch.zeros(self.layers, self.pages, dtype=torch.int64, device=self.device)
        self.mask = torch.zeros(self.layers, self.pages, dtype=torch.int32, device=self.device)
        self.encoded = torch.zeros(self.pages, dtype=torch.bool, device=self.device)
        self.holder = torch.full((self.pages,), -1, dtype=torch.int32)  # rank that keeps the page's parity
        self.parity = None  # host [pages, layers, 2, piece words] int32, pinned on GPUs, made at the first encode
        self.damaged = []   # [(rank, layer, page)] from the last scrub, the same list on every rank
        self.saved = {}     # (layer, page) -> clean piece copied at injection, to check a rebuild bit for bit
        self.tkey = torch.zeros(slots, dtype=torch.int64, device=self.device)  # token key of each slot's last write
        self.gen = torch.zeros(self.pages, dtype=torch.int64)  # generation of each page: +1 whenever it is freed
        self.req_gen = {}   # rid -> generations of the request's pages at its last good check
        self.found = []     # [(rank, layer, page, reason)] from the last reader check, the same list on every rank
        self.alarm_found = False  # the last check found damage (actions with only_if_found run only then)
        self.saved_span = None    # clean K and V of the span a fault replaced: (slots, [layers, 2, n, heads, dim])
        self.unkeyed = 0    # stores whose token ids were unknown (they keep the slots' previous token keys)
        self.stripe_of, self.stripe_pages, self.lparity = {}, {}, {}  # stripes within this GPU (holder -2)
        self.next_stripe = 0
        self.wstep = torch.zeros(self.pages, dtype=torch.int64, device=self.device)  # step of each page's last write
        self.step_t = torch.zeros(1, dtype=torch.int64, device=self.device)  # the step, readable inside CUDA graphs
        self.good_step = {}     # rid -> step of the request's last passing check
        self.pending_redo = {}  # rid -> tokens to keep: a reader of a mended page that must regenerate the rest
        self.checked_upto = {}  # rid -> output tokens covered by the request's last passing check (releasable)
        self.view_cache = {}    # rid -> token keys of the request's stored tokens, for the reader's check
        self.sched = None       # the scheduler, for checks that run outside on_step (the finish check)
        self.fused = FUSED_ON and self.device.type == "cuda"
        if self.fused:
            self.use_fused()
        self.step = 0
        self.last_good = None
        # background work (overhead benchmark): the same settings and step count on every rank
        self.ctrl = dict(encode_every=int(os.environ.get("KVMEND_ENCODE_EVERY", "0")),
                         scrub_pieces=int(os.environ.get("KVMEND_SCRUB_PIECES", "0")), label="")
        self.batch_steps = 0
        self.cursor = 0  # next piece of the background scrub, layer-major over pages 1 .. pages-1
        self.page_ids = torch.arange(1, self.pages, device=self.device)
        self.alarm = torch.zeros(1, dtype=torch.int64, device=self.device)  # mismatches since the last sum
        self.stats = new_stats()

    def use_fused(self):
        """Tags by the fused Triton kernels; each layer's K and V buffer address goes in a table, so the reader's
        check of every layer is one launch."""
        from kvmend import tag_kernels
        self.fused, self.kern = True, tag_kernels
        self.kptrs = torch.tensor([b.data_ptr() for b in self.pool.k_buffer], dtype=torch.int64, device=self.device)
        self.vptrs = torch.tensor([b.data_ptr() for b in self.pool.v_buffer], dtype=torch.int64, device=self.device)

    # Views of one layer: pages as [pages, PAGE, heads, dim]
    def kp(self, layer):
        return self.pool.k_buffer[layer].view(self.pages, PAGE, self.heads, self.dim)

    def vp(self, layer):
        return self.pool.v_buffer[layer].view(self.pages, PAGE, self.heads, self.dim)

    # Pieces as rows of 32-bit words: NCCL has no 16-bit integer type, and XOR on int32 words gives the same bits.
    def kflat(self, layer):
        return self.pool.k_buffer[layer].view(torch.int32).view(self.pages, -1)

    def vflat(self, layer):
        return self.pool.v_buffer[layer].view(torch.int32).view(self.pages, -1)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def piece_bytes(self):
        return 2 * PAGE * self.heads * self.dim * self.elem

    def reset(self):
        """A cache flush frees every page: all tags and parity go, and every page starts a new generation."""
        self.tags.zero_()
        self.mask.zero_()
        self.tkey.zero_()
        self.encoded.zero_()
        self.holder.fill_(-1)
        self.gen += 1
        self.damaged, self.saved, self.last_good = [], {}, None
        self.req_gen, self.found, self.alarm_found, self.saved_span = {}, [], False, None
        self.checked_upto, self.view_cache = {}, {}
        self.stripe_of, self.stripe_pages, self.lparity = {}, {}, {}
        self.wstep.zero_()
        self.good_step, self.pending_redo = {}, {}
        self.batch_steps = 0
        self.alarm.zero_()
        self.stats = new_stats()

    def drop(self, pages: torch.Tensor):
        """Freed pages lose their tags and parity, and start a new generation: a late write that still carries the
        old one is dropped, and a reader that recorded the old one is stopped."""
        self.tags[:, pages] = 0
        self.mask[:, pages] = 0
        self.encoded[pages] = False
        cpu = pages.cpu()
        self.holder[cpu] = -1
        self.gen[cpu] += 1
        for sid in {self.stripe_of[p] for p in cpu.tolist() if p in self.stripe_of}:  # a stripe within the GPU loses
            for q in self.stripe_pages.pop(sid):                                        # its parity with any page
                self.stripe_of.pop(q, None)
                self.encoded[q] = False
                self.holder[q] = -1
            self.lparity.pop(sid, None)

    # ---- running tags, called from set_kv_buffer --------------------------------------------------------------
    def _capturing(self):
        return self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()

    def update(self, layer: int, loc: torch.Tensor, tk: torch.Tensor = None):
        """Add the stored tokens' shares to their pieces' running tags. tk: the tokens' keys (token id and position);
        None keeps the keys the slots already have."""
        loc = loc.to(torch.int64)
        pg = torch.div(loc, PAGE, rounding_mode="floor")
        off = loc - pg * PAGE
        new_keys = tk is not None and layer == 0
        if tk is None:
            tk = self.tkey[loc]
        if layer == 0:  # stamp each written page with the step: cached KV newer than a last good check is suspect
            self.wstep.index_put_((pg,), self.step_t.expand(pg.shape[0]))
        if self.fused:
            K = self.kern
            K.store_update(K.words(self.pool.k_buffer[layer]), K.words(self.pool.v_buffer[layer]), loc, tk, self.keys,
                           self.tags[layer], self.mask[layer], self.tkey if new_keys else None)
            if layer == 0:  # a page whose first slot is written loses its parity; others write page 0 (padding).
                # index_fill_ takes the value as a kernel argument: a CUDA graph cannot copy a scalar from the host
                self.encoded.index_fill_(0, torch.where(off == 0, pg, torch.zeros_like(pg)), False)
            return
        if new_keys:
            self.tkey[loc] = tk
        share = T.token_shares(self.pool.k_buffer[layer][loc], self.pool.v_buffer[layer][loc], off, self.keys, tk)
        bits = self.bit[off]
        tags, mask = self.tags[layer], self.mask[layer]
        if self._capturing():
            # CUDA-graph decode: one token per request, so no two tokens share a page (padding uses page 0).
            first = off == 0
            tags[pg] = torch.where(first, 0, tags[pg]) + share
            mask[pg] = torch.where(first, 0, mask[pg]) | bits
            if layer == 0:
                self.encoded[pg] = torch.where(first, False, self.encoded[pg])
        else:
            fresh = pg[off == 0]  # a page's first token resets it, so a reused page starts clean
            if fresh.numel():
                tags[fresh] = 0
                mask[fresh] = 0
                if layer == 0:
                    self.encoded[fresh] = False
            tags.index_add_(0, pg, share)
            mask.index_add_(0, pg, bits)

    def seal(self, loc: torch.Tensor):
        """Parity alone (KVMEND_TAGS=0): track only which pages are full, from the last layer's writes."""
        loc = loc.to(torch.int64)
        pg = torch.div(loc, PAGE, rounding_mode="floor")
        off = loc - pg * PAGE
        bits = self.bit[off]
        mask = self.mask[-1]
        if self._capturing():
            first = off == 0
            mask[pg] = torch.where(first, 0, mask[pg]) | bits
            self.encoded[pg] = torch.where(first, False, self.encoded[pg])
        else:
            fresh = pg[off == 0]
            if fresh.numel():
                mask[fresh] = 0
                self.encoded[fresh] = False
            mask.index_add_(0, pg, bits)

    def check(self, layer: int, pages: torch.Tensor, tk: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """Recompute the pages' tags from their bytes and compare. tk [n, PAGE]: the token keys a reader expects
        (default: the keys stored with the slots, which checks the bytes only); mask: the offsets to cover (default:
        the written ones)."""
        if tk is None:
            tk = self.tkey.view(self.pages, PAGE)[pages]
        if mask is None:
            mask = self.mask[layer][pages]
        if self.fused:
            K = self.kern
            got = K.page_tags(K.words(self.pool.k_buffer[layer]), K.words(self.pool.v_buffer[layer]), pages, mask, tk,
                              self.keys)
        else:
            got = T.page_tags(self.kp(layer)[pages], self.vp(layer)[pages], mask, self.keys, tk)
        return T.same(got, self.tags[layer][pages])

    def check_codes(self, pages: torch.Tensor, expected: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        """The reader's test of `pages` in every layer: [layers, n] int8, 0 = the tag covers exactly the expected
        offsets and matches the reader's token keys, 1 = the coverage differs, 2 = the tag differs. The fused path is
        one kernel launch for every layer."""
        if self.fused:
            return self.kern.check_layers(self.kptrs, self.vptrs, self.layers, pages, expected, tk, self.keys,
                                          self.tags, self.mask, self.words)
        n = pages.numel()
        codes = torch.zeros(self.layers, n, dtype=torch.int8, device=self.device)
        for l in range(self.layers):
            for i0 in range(0, n, 4096):
                p, e, t = pages[i0:i0 + 4096], expected[i0:i0 + 4096], tk[i0:i0 + 4096]
                cover = self.mask[l][p] == e
                tag_ok = self.check(l, p, tk=t, mask=e)
                codes[l, i0:i0 + 4096] = torch.where(~cover, 1, torch.where(tag_ok, 0, 2)).to(torch.int8)
        return codes

    def pieces(self, pages: torch.Tensor) -> torch.Tensor:
        """[layers, n, 2, piece words] int32: a copy of this rank's pieces of the given pages."""
        return torch.stack([torch.stack([self.kflat(l)[pages], self.vflat(l)[pages]], 1) for l in range(self.layers)])


# ---------------------------------------------------------------------------------------------------------------
# Actions, run on every rank in the same scheduler iteration

def _group(sched):
    g = sched.tp_group
    return g, g.world_size, g.rank_in_group


def _start(st, g):
    """Start a timed collective together: without the barrier, a rank that arrives late adds its delay to the time."""
    st.sync()
    dist.barrier(group=g.cpu_group)
    return time.perf_counter()


def act_encode(st, sched, a, req):
    """Parity per page: each chunk of sealed pages goes to one encoder rank, which XORs it and keeps it in host
    memory. The encoder rotates by chunk. With check (KVMend), each rank checks its own pieces' tags before it sends
    them; GhostServe has no tags (check=False)."""
    g, n, me = _group(sched)
    check = a.get("check", True) and TAGS_ON
    cand = ((st.mask[-1] == FULL) & ~st.encoded).nonzero().flatten()
    box = [[p for p in cand.tolist() if p > 0]]
    dist.broadcast_object_list(box, src=g.ranks[0], group=g.cpu_group)
    pages = box[0]
    if a.get("stripe", STRIPE) == "local" or n == 1:
        return _encode_local(st, a, pages, check, me)
    if st.parity is None:
        st.parity = torch.empty((st.pages, st.layers, 2, st.kflat(0).shape[1]), dtype=torch.int32,
                                pin_memory=st.device.type == "cuda")
    st.sync()
    t0 = time.perf_counter()
    unclean = []
    for ci, start in enumerate(range(0, len(pages), CHUNK_PAGES)):
        chunk = pages[start:start + CHUNK_PAGES]
        enc = ci % n
        idx = torch.tensor(chunk, device=st.device)
        piece = st.pieces(idx).contiguous()
        if check:
            for l in range(st.layers):
                bad = (~st.check(l, idx)).nonzero().flatten().tolist()
                unclean += [(me, l, chunk[i]) for i in bad]
        bucket = [torch.empty_like(piece) for _ in range(n)] if me == enc else None
        dist.gather(piece, gather_list=bucket, dst=g.ranks[enc], group=g.device_group)
        if me == enc:
            par = bucket[0].clone()
            for t in bucket[1:]:
                par.bitwise_xor_(t)
            st.parity.index_copy_(0, torch.tensor(chunk), par.transpose(0, 1).contiguous().cpu())
        st.holder[torch.tensor(chunk)] = enc
        st.encoded[idx] = True
    st.sync()
    dt = time.perf_counter() - t0
    sent = len(pages) * st.layers * st.piece_bytes()
    return dict(pages=len(pages), bytes_sent_per_rank=sent, seconds=dt, checked=check, unclean_pieces=unclean)


def _encode_local(st, a, pages, check, me):
    """Stripes within this GPU (decision 20, step 4): every `k` consecutive sealed pages form a stripe, and the XOR of
    their pieces, layer by layer, goes to host memory. Each GPU encodes its own pieces and needs no other GPU, so it
    works for a model served on one GPU; it rebuilds one damaged page per stripe and layer, even when every GPU's copy
    of that page is damaged, but not a lost GPU. Pages short of a full stripe wait for the next encode."""
    k = a.get("k", 8)
    full = (len(pages) // k) * k
    use = pages[:full]
    st.sync()
    t0 = time.perf_counter()
    unclean = []
    if use:
        idx = torch.tensor(use, device=st.device)
        if check:
            for l in range(st.layers):
                unclean += [(me, l, use[i]) for i in (~st.check(l, idx)).nonzero().flatten().tolist()]
        piece = st.pieces(idx).view(st.layers, full // k, k, 2, -1)
        par = piece[:, :, 0].clone()
        for j in range(1, k):
            par.bitwise_xor_(piece[:, :, j])
        par = par.cpu()
        for s in range(full // k):
            sid = st.next_stripe
            st.next_stripe += 1
            st.lparity[sid] = par[:, s].clone()
            st.stripe_pages[sid] = use[s * k:(s + 1) * k]
            for p in st.stripe_pages[sid]:
                st.stripe_of[p] = sid
        st.holder[torch.tensor(use)] = -2
        st.encoded[idx] = True
    st.sync()
    return dict(pages=full, stripes=full // k, stripe="local", k=k, seconds=time.perf_counter() - t0,
                bytes_to_host_per_rank=(full // k) * st.layers * st.piece_bytes(), checked=check,
                unclean_pieces=unclean)


def act_scrub(st, sched, a, req):
    """Recompute every written piece's tag from its bytes and compare. A mismatch names (rank, layer, page)."""
    g, n, me = _group(sched)
    st.sync()
    t0 = time.perf_counter()
    found, checked = [], 0
    for l in range(st.layers):
        idx = (st.mask[l] != 0).nonzero().flatten()
        idx = idx[idx > 0]
        for chunk in idx.split(4096):
            bad = (~st.check(l, chunk)).nonzero().flatten().tolist()
            found += [(me, l, int(chunk[i])) for i in bad]
            checked += chunk.numel()
    st.sync()
    dt = time.perf_counter() - t0
    every = [None] * n
    dist.all_gather_object(every, found, group=g.cpu_group)
    st.damaged = sorted({tuple(x) for lst in every for x in lst})
    if not st.damaged:
        st.last_good = st.step
    nbytes = checked * st.piece_bytes()
    return dict(pieces_checked=checked, bytes_checked=nbytes, seconds=dt, gib_per_s=nbytes / dt / 2**30 if dt else None,
                found_here=found, damaged_all=st.damaged, last_good_step=st.last_good)


def act_inject(st, sched, a, req):
    """Flip one bit of one stored K or V value of request `rid` at token position `pos`, on rank `rank` only."""
    g, n, me = _group(sched)
    loc = int(sched.req_to_token_pool.req_to_token[req.req_pool_idx, a["pos"]])
    layer, page = a["layer"], loc // PAGE
    rec = dict(loc=loc, page=page, layer=layer, target_rank=a["rank"], kv=a["kv"], elem=a["elem"], bit=a["bit"])
    if me == a["rank"]:
        st.saved[(layer, page)] = torch.stack([st.kflat(layer)[page], st.vflat(layer)[page]]).clone()
        buf = (st.pool.k_buffer if a["kv"] == "k" else st.pool.v_buffer)[layer]
        words = buf.view(torch.int16)[loc].view(-1)
        before = int(words[a["elem"]]) & 0xFFFF
        after = before ^ (1 << a["bit"])
        words[a["elem"]] = after - 0x10000 if after >= 0x8000 else after

        def value(u):
            return torch.tensor([u - 0x10000 if u >= 0x8000 else u], dtype=torch.int16).view(buf.dtype).item()

        rec.update(dtype=str(buf.dtype), before=f"0x{before:04X}", after=f"0x{after:04X}",
                   value_before=value(before), value_after=value(after))
    return rec


CHUNK_PIECES = 1024  # pieces rebuilt per gather


def act_repair(st, sched, a, req):
    """Rebuild each damaged piece in place: the other ranks' pieces XOR the parity. The pieces of one damaged rank
    are rebuilt together, up to CHUNK_PIECES per gather. Each rebuilt piece must match its tag, and the clean copy
    saved at injection, bit for bit."""
    g, n, me = _group(sched)
    out, by_rank, local = [], {}, []
    for bad_rank, layer, page in st.damaged:
        h = int(st.holder[page])
        if h == -1 or not bool(st.encoded[page]):
            out.append(dict(rank=bad_rank, layer=layer, page=page, rebuilt=False, reason="no parity yet"))
        elif h == -2:
            local.append((bad_rank, layer, page))
        else:
            by_rank.setdefault(bad_rank, []).append((layer, page))
    if local:
        out += _rebuild_local(st, me, local)
    for bad_rank, items in by_rank.items():
        for i0 in range(0, len(items), CHUNK_PIECES):
            chunk = items[i0:i0 + CHUNK_PIECES]
            t0 = _start(st, g)
            mine = torch.stack([torch.stack([st.kflat(l)[p], st.vflat(l)[p]]) for l, p in chunk])
            if me == bad_rank:
                mine.zero_()
            for i, (l, p) in enumerate(chunk):
                if int(st.holder[p]) == me:
                    mine[i].bitwise_xor_(st.parity[p, l].to(st.device))
            bucket = [torch.empty_like(mine) for _ in range(n)] if me == bad_rank else None
            dist.gather(mine, gather_list=bucket, dst=g.ranks[bad_rank], group=g.device_group)
            recs = [dict(rank=bad_rank, layer=l, page=p, holder=int(st.holder[p]), bytes_read=n * st.piece_bytes())
                    for l, p in chunk]
            if me == bad_rank:
                rebuilt = bucket[0].clone()
                for t in bucket[1:]:
                    rebuilt.bitwise_xor_(t)
                for i, (l, p) in enumerate(chunk):
                    st.kflat(l)[p] = rebuilt[i, 0]
                    st.vflat(l)[p] = rebuilt[i, 1]
                st.sync()
                dt = time.perf_counter() - t0
                ok = {}
                for l in sorted({l for l, _ in chunk}):
                    pages = [p for l2, p in chunk if l2 == l]
                    good = st.check(l, torch.tensor(pages, device=st.device)).tolist()
                    ok.update({(l, p): bool(x) for p, x in zip(pages, good)})
                for i, ((l, p), rec) in enumerate(zip(chunk, recs)):
                    rec.update(rebuilt=True, seconds=dt / len(chunk), tag_matches=ok[(l, p)],
                               bit_exact=bool(torch.equal(st.saved[(l, p)], rebuilt[i])) if (l, p) in st.saved
                               else None)
            out += recs
    st.damaged = []
    return dict(repairs=out)


def _rebuild_local(st, me, items):
    """Rebuild pieces from stripes within their own GPU: each rank rebuilds its own damaged pieces from the stripe's
    other pages on the same GPU and the stripe's parity in host memory. No other GPU takes part."""
    st.sync()
    t0 = time.perf_counter()
    done = {}
    for r, l, p in items:
        if r != me:
            continue
        sid = st.stripe_of[p]
        acc = st.lparity[sid][l].to(st.device).clone()
        for q in st.stripe_pages[sid]:
            if q != p:
                acc.bitwise_xor_(torch.stack([st.kflat(l)[q], st.vflat(l)[q]]))
        st.kflat(l)[p] = acc[0]
        st.vflat(l)[p] = acc[1]
        done[(l, p)] = acc
    st.sync()
    dt = time.perf_counter() - t0
    out = []
    for r, l, p in items:
        sid = st.stripe_of.get(p)
        rec = dict(rank=r, layer=l, page=p, holder="local", stripe=sid,
                   bytes_read=len(st.stripe_pages.get(sid, ())) * st.piece_bytes())
        if r == me:
            rec.update(rebuilt=True, seconds=dt / max(1, len(done)),
                       tag_matches=bool(st.check(l, torch.tensor([p], device=st.device)).all()),
                       bit_exact=bool(torch.equal(st.saved[(l, p)], done[(l, p)])) if (l, p) in st.saved else None)
        out.append(rec)
    return out


def act_repair_share(st, sched, a, req):
    """GhostServe's repair, re-implemented for the comparison. GhostServe cannot find silent damage, so an oracle
    tells it which GPU (`rank`) is bad, at the step where KVMend's scrub finds the damage. It rebuilds that GPU's
    share from the other GPUs' pieces and the parity, in every layer:
      scope "gpu":   every encoded page, as GhostServe recovers a failed GPU;
      scope "chunk": only the request's pages in the `chunk_tokens`-token chunk that holds position `pos`
                     (GhostServe's encoding unit), as if the oracle also named the chunk.
    Nothing is rolled back: GhostServe keeps the tokens computed from the damaged value."""
    g, n, me = _group(sched)
    bad, scope = a["rank"], a.get("scope", "gpu")
    if scope == "gpu":
        pages = st.encoded.nonzero().flatten()
    else:
        m = a.get("chunk_tokens", 2048)
        c0 = (a["pos"] // m) * m
        hi = min(c0 + m, len(req.origin_input_ids))
        locs = sched.req_to_token_pool.req_to_token[req.req_pool_idx, c0:hi].to(torch.int64)
        pages = torch.unique(torch.div(locs, PAGE, rounding_mode="floor"))
        pages = pages[st.encoded[pages]]
    pages = pages[pages > 0]
    t0 = _start(st, g)
    for chunk in pages.split(CHUNK_PAGES):
        mine = st.pieces(chunk)
        if me == bad:
            mine.zero_()
        ccpu = chunk.cpu()
        held = (st.holder[ccpu] == me).nonzero().flatten()
        if held.numel():
            hd = held.to(st.device)
            mine[:, hd] = mine[:, hd] ^ st.parity[ccpu[held]].to(st.device).transpose(0, 1)
        bucket = [torch.empty_like(mine) for _ in range(n)] if me == bad else None
        dist.gather(mine, gather_list=bucket, dst=g.ranks[bad], group=g.device_group)
        if me == bad:
            rebuilt = bucket[0]
            for t in bucket[1:]:
                rebuilt.bitwise_xor_(t)
            for l in range(st.layers):
                st.kflat(l)[chunk] = rebuilt[l, :, 0]
                st.vflat(l)[chunk] = rebuilt[l, :, 1]
    st.sync()
    dt = time.perf_counter() - t0
    k = int(pages.numel())
    rec = dict(scope=scope, target_rank=bad, pages=k, pieces=k * st.layers,
               bytes_read=n * k * st.layers * st.piece_bytes(), seconds=dt, t_end=time.time())
    if me == bad:
        rec["tag_mismatches_after"] = (sum(int((~st.check(l, pages)).sum()) for l in range(st.layers))
                                       if TAGS_ON and k else None)
        on = set(pages.tolist())
        rec["damaged_pieces"] = [dict(layer=l, page=p, rebuilt=p in on,
                                      bit_exact=bool(torch.equal(st.saved[(l, p)],
                                                                 torch.stack([st.kflat(l)[p], st.vflat(l)[p]]))))
                                 for (l, p) in sorted(st.saved)]
    st.damaged = []
    return rec


def act_retract(st, sched, a, req):
    """Roll the request back with SGLang's own retract steps (ScheduleBatch.retract_decode), for this one request:
    free its KV after the cached prefix and send it back to the waiting queue. SGLang then re-prefills only the
    tokens that the radix cache does not hold. With only_if_found, only when the last check found damage."""
    if a.get("only_if_found") and not st.alarm_found:
        return dict(retracted=False, reason="the last check found nothing")
    batch = sched.running_batch
    i = next((j for j, r in enumerate(batch.reqs) if r.rid == req.rid), None)
    if i is None:
        return dict(retracted=False, reason="request is not in the running batch")
    seq_len = int(batch.seq_lens[i].item())
    start = (len(req.prefix_indices) // sched.page_size) * sched.page_size
    sched.token_to_kv_pool_allocator.free(sched.req_to_token_pool.req_to_token[req.req_pool_idx, start:seq_len])
    sched.req_to_token_pool.free(req.req_pool_idx)
    sched.tree_cache.dec_lock_ref(req.last_node)
    req.reset_for_retract()
    rec = dict(retracted=True, kept_prefix=start, freed_tokens=seq_len - start, seq_len=seq_len)
    if "keep" in a:
        # Strict release: output tokens after the last good check were never released, so drop them and let the
        # engine generate them again. Only allowed while no token past `keep` has been sent to the client.
        sent = getattr(req, "send_token_offset", 0)
        if sent <= a["keep"]:
            rec.update(regenerated=len(req.output_ids) - a["keep"], kept_outputs=a["keep"])
            del req.output_ids[a["keep"]:]
        else:
            rec.update(regenerated=0, kept_outputs=len(req.output_ids), refused="tokens already sent")
    batch.filter_batch(keep_indices=[j for j in range(len(batch.reqs)) if j != i])
    sched._extend_requests_to_queue([req], is_retracted=True)
    return rec


def act_abort(st, sched, a, req):
    """End the request with an abort, as a system that can only detect damage does; the client may resubmit. With
    only_if_found, only when the last check found damage."""
    if a.get("only_if_found") and not st.alarm_found:
        return dict(aborted=False, reason="the last check found nothing")
    req.to_abort = True
    req.to_abort_message = "KV damage found; resubmit the request"
    return dict(aborted=True)


def act_evict(st, sched, a, req):
    """Drop every unlocked page of the radix cache (after a retract, that includes the request's prompt), so the
    request re-prefills from its first token: a system that detects damage but cannot repair it, such as SGLang's KV
    checksum, aborts the request, and the retry recomputes everything."""
    tree = sched.tree_cache
    before = int(tree.evictable_size())
    t0 = time.perf_counter()
    tree.evict(before)
    return dict(evictable_before=before, evictable_after=int(tree.evictable_size()), seconds=time.perf_counter() - t0)


def act_dump(st, sched, a, req):
    """Save this rank's K and V of positions [from, to) of the request, every layer, for the spread analysis."""
    g, n, me = _group(sched)
    hi = min(a["to"], len(req.origin_input_ids) + len(req.output_ids) - 1)
    locs = sched.req_to_token_pool.req_to_token[req.req_pool_idx, a["from"]:hi].to(torch.int64)
    k = torch.stack([st.pool.k_buffer[l][locs] for l in range(st.layers)]).cpu()
    v = torch.stack([st.pool.v_buffer[l][locs] for l in range(st.layers)]).cpu()
    path = os.path.join(OUT, f"dump_{req.rid}_{a.get('name', 'kv')}_rank{me}.pt")
    os.makedirs(OUT, exist_ok=True)
    torch.save(dict(k=k, v=v, start=a["from"], output_ids=list(req.output_ids)), path)
    return dict(path=path, positions=[a["from"], hi])


# ---------------------------------------------------------------------------------------------------------------
# The incident's faults: KV that is well formed but wrong, on every GPU, and the reader's check that finds it

def _reader_view(st, sched, req):
    """What request R reads: the pool page of each 16-token block of its stored sequence (the prompt and every output
    token but the newest), the offsets it reads in each, and the token keys it expects there. The token keys of a
    request are cached and extended as it grows, so a check every step does not rehash the whole prompt. The prompt
    never changes, so only the output tokens behind the cache are compared."""
    prompt, outs = req.origin_input_ids, req.output_ids
    n = len(prompt) + max(len(outs) - 1, 0)
    slots = sched.req_to_token_pool.req_to_token[req.req_pool_idx, :n].to(torch.int64).to(st.device)
    blocks = (n + PAGE - 1) // PAGE
    pad = blocks * PAGE - n
    pages = torch.div(slots[::PAGE], PAGE, rounding_mode="floor")
    cache = st.view_cache.get(req.rid)
    if (cache is None or cache["n"] > n or cache["prompt"] != len(prompt)
            or cache["outs"] != list(outs[:max(cache["n"] - len(prompt), 0)])):
        cache = dict(n=0, prompt=len(prompt), outs=[], tk=torch.zeros(0, dtype=torch.int64, device=st.device),
                     blocks_ok=0)
    if cache["blocks_ok"] < blocks:  # with page_size 16, block j fills offsets 0-15 of one page
        grid = torch.cat([slots, slots.new_full((pad,), -1)]).view(blocks, PAGE)
        want = pages.unsqueeze(1) * PAGE + torch.arange(PAGE, device=st.device)
        if not bool(((grid == want) | (grid < 0)).all()):
            raise RuntimeError(f"request {req.rid}: its positions are not laid out page by page")
        cache["blocks_ok"] = blocks - (1 if pad else 0)  # the open block is checked again as it fills
    if cache["n"] < n:
        lo, np_ = cache["n"], len(prompt)
        fresh = (list(prompt[lo:np_]) if lo < np_ else []) + list(outs[max(lo - np_, 0):n - np_])
        new = T.token_keys(torch.tensor(fresh, dtype=torch.int64, device=st.device),
                           torch.arange(lo, n, device=st.device))
        cache.update(tk=torch.cat([cache["tk"], new]), n=n, outs=list(outs[:max(n - np_, 0)]))
    st.view_cache[req.rid] = cache
    tk = torch.cat([cache["tk"][:n], cache["tk"].new_zeros(pad)]).view(blocks, PAGE)
    expected = torch.full((blocks,), FULL, dtype=torch.int32, device=st.device)
    if pad:
        expected[-1] = (1 << (PAGE - pad)) - 1
    return pages, expected, tk


def _reader_failures(st, me, pages, expected, tk):
    """This rank's failures of the reader's test on the given pages: (rank, layer, page, reason). The GPU is waited
    for once, not once per layer."""
    codes = st.check_codes(pages, expected, tk)
    if not bool(codes.any()):
        return []
    on = pages.tolist()
    return ([(me, l, on[i], "coverage") for l, i in (codes == 1).nonzero().tolist()]
            + [(me, l, on[i], "tag") for l, i in (codes == 2).nonzero().tolist()])


def _gather_found(st, g, found):
    """Every rank's findings, the same sorted list on every rank. One small collective first; the details travel
    only when some rank found something."""
    count = torch.tensor([len(found)], dtype=torch.int64, device=st.device)
    dist.all_reduce(count, group=g.device_group)
    if not int(count.item()):
        return []
    every = [None] * g.world_size
    dist.all_gather_object(every, found, group=g.cpu_group)
    return sorted({tuple(x) for lst in every for x in lst})


def _summary(found):
    """{page: {reason: {ranks, layers}}}: the findings, short enough for one log line."""
    out = {}
    for r, l, p, why in found:
        d = out.setdefault(str(p), {}).setdefault(why, (set(), set()))
        d[0].add(r)
        d[1].add(l)
    return {p: {w: dict(ranks=sorted(rs), layers=len(ls)) for w, (rs, ls) in v.items()} for p, v in out.items()}


def _generation_failures(st, me, req, pages):
    """A page the request recorded at its last good check that has since moved to a new generation was freed and
    handed out again while the request still read it (a read after free). The record is keyed by page, so a request
    that legitimately moved to new pages (after a rollback, which clears the record) is not flagged."""
    old = st.req_gen.get(req.rid)
    if old is None:
        return []
    rec_pages, rec_gens = old
    on = pages.cpu()
    seen = torch.full((st.pages,), -1, dtype=torch.int64)
    seen[rec_pages] = rec_gens
    was = seen[on]
    bad = ((was >= 0) & (was != st.gen[on])).nonzero().flatten().tolist()
    return [(me, -1, int(on[i]), "generation") for i in bad]


def _mark_good(st, req, pages):
    """A passing check: record the step and the generation of every page the request read."""
    st.last_good = st.step
    st.good_step[req.rid] = st.step
    on = pages.cpu()
    st.req_gen[req.rid] = (on, st.gen[on].clone())


def forget(st, rid):
    """A request rolled back (by KVMend or by SGLang under memory pressure) will read new pages: drop its records."""
    st.req_gen.pop(rid, None)
    st.view_cache.pop(rid, None)


def act_check(st, sched, a, req):
    """The reader's check of request R before its KV is used again or its tokens are released. Every page R reads
    must carry a tag that covers exactly the tokens R reads there, and that tag, recomputed from the bytes with R's
    own token keys, must match. A skipped write fails the first test (coverage); changed bytes, or another
    sequence's KV with tags of its own, fail the second (tag). A page that moved to a new generation since R's last
    good check fails as well (generation: a read after free). The findings, the same on every rank, go to st.found."""
    g, n, me = _group(sched)
    pages, expected, tk = _reader_view(st, sched, req)
    st.sync()
    t0 = time.perf_counter()
    found = _reader_failures(st, me, pages, expected, tk) + _generation_failures(st, me, req, pages)
    st.found = _gather_found(st, g, found)
    dt = time.perf_counter() - t0
    st.alarm_found = bool(st.found)
    st.damaged = [(r, l, p) for r, l, p, why in st.found if why == "tag"]
    if not st.found:
        _mark_good(st, req, pages)
    return dict(pages=int(pages.numel()), pieces_checked=int(pages.numel()) * st.layers, seconds=dt,
                bytes_checked=int(pages.numel()) * st.layers * st.piece_bytes(), found=_summary(st.found),
                found_pieces=len(st.found), last_good_step=st.last_good)


def _span(req, a):
    """The token span a fault replaces: `pages` pages from position `pos`, or the last `pages` pages of the prompt."""
    n = a.get("pages", 4) * PAGE
    start = a["pos"] if "pos" in a else (len(req.origin_input_ids) // PAGE) * PAGE - n
    if start < 0 or start % PAGE:
        raise ValueError(f"a fault span must start at a page boundary, not at {start}")
    return start, n


def act_fault(st, sched, a, req):
    """Make R's KV well formed but wrong in `pages` pages, in every layer and on every GPU, as a software bug does.
    The wrong KV is that of R's tokens at position `donor`: real KV of other tokens at other positions.
      unwritten    the pages were never written for R: they hold an earlier owner's bytes and have no tags, as after
                   a free (a skipped write);
      late_bypass  a dead request's late write lands around the store path, as a transfer would: bytes change, tags
                   do not;
      late_store   a dead request's late write arrives through the store path, carrying the generation the pages
                   had before they were freed and given to R; with fences on (fence, KVMEND_FENCE) it is dropped;
      foreign      the pages hold another sequence's KV with tags that match it: what R reads when its token table
                   points at another live page.
    The clean K and V of the span are kept, to measure how close a recompute comes."""
    start, n = _span(req, a)
    donor = a.get("donor", 512)
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx]
    tgt = row[start:start + n].to(torch.int64).to(st.device)
    src = row[donor:donor + n].to(torch.int64).to(st.device)
    pages = torch.unique(torch.div(tgt, PAGE, rounding_mode="floor"))
    kind = a["kind"]
    st.saved_span = (tgt.clone(), torch.stack([torch.stack([st.pool.k_buffer[l][tgt], st.pool.v_buffer[l][tgt]])
                                               for l in range(st.layers)]).clone())
    rec = dict(kind=kind, start=start, tokens=n, donor=donor, pages=pages.tolist())
    if kind == "burst":
        return _burst(st, sched, a, tgt, pages, rec)
    ids = list(req.origin_input_ids) + list(req.output_ids)
    tk = T.token_keys(torch.tensor(ids[donor:donor + n], dtype=torch.int64, device=st.device),
                      torch.arange(donor, donor + n, device=st.device))
    if kind == "late_store" and a.get("fence", True) and FENCE_ON:
        tok_pages = torch.div(tgt, PAGE, rounding_mode="floor").cpu()
        writer_gen = st.gen[tok_pages] - 1  # the dead writer still carries the generation from before the free
        keep = (st.gen[tok_pages] == writer_gen).to(st.device)
        rec.update(fenced=True, dropped_tokens=int((~keep).sum()), written_tokens=int(keep.sum()))
        tgt, src, tk = tgt[keep], src[keep], tk[keep]
        if not tgt.numel():
            return rec
    for l in range(st.layers):
        st.pool.k_buffer[l][tgt] = st.pool.k_buffer[l][src]
        st.pool.v_buffer[l][tgt] = st.pool.v_buffer[l][src]
    if kind in ("late_store", "foreign"):
        if TAGS_ON:
            for l in range(st.layers):
                st.update(l, tgt, tk)  # the store path keys the tags by the writer's tokens
    elif kind == "unwritten":
        st.tags[:, pages] = 0
        st.mask[:, pages] = 0
        st.encoded[pages] = False
        st.holder[pages.cpu()] = -1
        st.tkey[tgt] = 0
    elif kind != "late_bypass":
        raise ValueError(f"unknown fault kind {kind}")
    st.sync()
    rec.setdefault("written_tokens", int(tgt.numel()))
    return rec


def _burst(st, sched, a, tgt, pages, rec):
    """A failing memory region on one GPU: one low mantissa bit changes in every token's K of the span's pages, in
    `layers` layers from `layer`, on rank `rank` only. Each damaged piece's clean copy is kept, so that a rebuild
    can be checked bit for bit."""
    g, n, me = _group(sched)
    rank = a.get("rank", 2)
    first = a.get("layer", 12)
    layers = list(range(first, min(first + a.get("layers", 1), st.layers)))
    rec.update(rank=rank, layers=layers, pieces=len(layers) * int(pages.numel()))
    if me == rank:
        for l in layers:
            for p in pages.tolist():
                st.saved[(l, p)] = torch.stack([st.kflat(l)[p], st.vflat(l)[p]]).clone()
            words = st.pool.k_buffer[l].view(torch.int16).view(st.pool.k_buffer[l].shape[0], -1)
            words[tgt, 3] = words[tgt, 3] ^ (1 << 3)
    st.sync()
    return rec


def _recompute_range(st, sched, req, s, e):
    """Recompute R's KV of positions [s, e) in place, in every layer: one extend forward of a side request that holds
    R's tokens up to e, whose token table maps [0, s) to R's cached slots (the prefix) and [s, e) to R's slots of the
    damaged span, so the forward writes new K and V over the damaged ones. R itself is not touched."""
    import copy

    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx]
    slots = row[:e].to(torch.int64).clone()
    ids = (list(req.origin_input_ids) + list(req.output_ids))[:e]
    params = copy.copy(req.sampling_params)
    params.max_new_tokens = 1
    side = Req(f"kvmend-recompute-{req.rid}-{s}", "", ids, params, vocab_size=sched.model_config.vocab_size)
    side.fill_ids = ids
    side.prefix_indices = slots[:s]
    side.extend_input_len = e - s
    batch = ScheduleBatch.init_new([side], sched.req_to_token_pool, sched.token_to_kv_pool_allocator,
                                   sched.tree_cache, sched.model_config, sched.enable_overlap, sched.spec_algorithm)
    target = slots[s:e]
    batch.alloc_paged_token_slots_extend = lambda *args, **kwargs: target  # write over R's slots, allocate nothing
    batch.alloc_token_slots = lambda *args, **kwargs: target
    batch.prepare_for_extend()
    try:
        sched.tp_worker.forward_batch_generation(batch.get_model_worker_batch(), skip_sample=True)
    finally:
        sched.req_to_token_pool.free(side.req_pool_idx)


def act_mend(st, sched, a, req):
    """Repair what the last check found. A page damaged on one GPU only, still covered by its tags and with parity,
    is rebuilt from parity, bit for bit (parity=False skips this path). Every other page (damage on several GPUs, a
    skipped write, another sequence's KV, a new generation) is recomputed in place from R's tokens, with R's cached KV
    before it as the prefix. With force, the span of `pages`/`pos` is recomputed whatever the check found: the
    fault-free control of this path."""
    g, n, me = _group(sched)
    pages, expected, tk = _reader_view(st, sched, req)
    block_of = {int(p): j for j, p in enumerate(pages.tolist())}
    rebuild, redo = [], set()
    if a.get("force"):
        start, length = _span(req, a)
        redo = {int(pages[j]) for j in range(start // PAGE, (start + length) // PAGE)}
        tgt = sched.req_to_token_pool.req_to_token[req.req_pool_idx, start:start + length].to(torch.int64).to(st.device)
        st.saved_span = (tgt.clone(), torch.stack([torch.stack([st.pool.k_buffer[l][tgt], st.pool.v_buffer[l][tgt]])
                                                   for l in range(st.layers)]).clone())
    elif not st.found:
        return dict(mended=False, reason="the last check found nothing")
    else:
        by_page = {}
        for r, l, p, why in st.found:
            by_page.setdefault(p, []).append((r, l, why))
        for p, fs in by_page.items():
            h, one_gpu = int(st.holder[p]), len({r for r, _, _ in fs}) == 1
            # parity across GPUs rebuilds damage on one GPU; a stripe within each GPU rebuilds it on any number
            if (a.get("parity", True) and all(w == "tag" for _, _, w in fs) and bool(st.encoded[p])
                    and (h == -2 or (h >= 0 and one_gpu))):
                rebuild += [(r, l, p) for r, l, _ in fs]
            else:
                redo.add(p)
        # a stripe within a GPU rebuilds one damaged page per layer: pages sharing a stripe and a layer on one GPU
        # with another damaged page are recomputed instead
        crowd = {}
        for r, l, p in rebuild:
            if int(st.holder[p]) == -2:
                crowd.setdefault((r, l, st.stripe_of[p]), set()).add(p)
        crowded = {p for ps in crowd.values() if len(ps) > 1 for p in ps}
        rebuild = [x for x in rebuild if x[2] not in crowded]
        redo |= crowded
    rec = dict(mended=True, rebuilt=[], recomputed=[])
    t_mend = _start(st, g)
    if rebuild:
        st.damaged = rebuild
        rec["rebuilt"] = act_repair(st, sched, {}, req)["repairs"]
        # a rebuilt piece that still fails R's check (damaged parity or tag, or a second damaged page) is recomputed
        rp = sorted({p for _, _, p in rebuild if p in block_of})
        js = torch.tensor([block_of[p] for p in rp], dtype=torch.int64, device=st.device)
        failed = _gather_found(st, g, _reader_failures(st, me, pages[js], expected[js], tk[js]) if rp else [])
        if failed:
            rec["rebuild_failed_pages"] = sorted({p for _, _, p, _ in failed})
            redo |= set(rec["rebuild_failed_pages"])
    blocks = sorted(block_of[p] for p in redo if p in block_of)
    spans, stored = [], len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
    for j in blocks:
        if spans and spans[-1][1] == j:
            spans[-1][1] = j + 1
        else:
            spans.append([j, j + 1])
    for j0, j1 in spans:
        s, e = j0 * PAGE, min(j1 * PAGE, stored)
        t0 = _start(st, g)
        _recompute_range(st, sched, req, s, e)
        st.sync()
        rec["recomputed"].append(dict(start=s, end=e, tokens=e - s, seconds=time.perf_counter() - t0))
    st.sync()
    rec["mend_seconds"] = time.perf_counter() - t_mend  # the whole repair, before the test's own checks below
    rec["mended_pages"] = sorted({p for _, _, p in rebuild} | redo)
    if blocks:
        idx = torch.tensor(blocks, device=st.device)
        rec["still_failing_here"] = len(_reader_failures(st, me, pages[idx], expected[idx], tk[idx]))
    if st.saved_span is not None and (rec["recomputed"] or rec["rebuilt"]):
        tgt, clean = st.saved_span
        now = torch.stack([torch.stack([st.pool.k_buffer[l][tgt], st.pool.v_buffer[l][tgt]]) for l in range(st.layers)])
        rec.update(max_abs_diff_vs_clean=float((now.float() - clean.float()).abs().max()),
                   bit_equal_share_vs_clean=float((now.view(torch.int16) == clean.view(torch.int16)).float().mean()))
    st.found, st.damaged = [], []
    return rec


ACTIONS = dict(encode=act_encode, scrub=act_scrub, inject=act_inject, repair=act_repair, retract=act_retract,
               dump=act_dump, repair_share=act_repair_share, evict=act_evict, abort=act_abort,
               check=act_check, fault=act_fault, mend=act_mend)


# ---------------------------------------------------------------------------------------------------------------
# The release rule: a strict request's tokens leave only after a check covers what they read

def is_strict(req):
    return bool(STRICT) and ("*" in STRICT or req.rid in STRICT) and not str(req.rid).startswith("kvmend-")


def strict_every(req):
    every = STRICT.get(req.rid)
    return every or STRICT.get("*") or STRICT_EVERY


def release_check(st, sched, req, finishing=False):
    """Check a strict request's pages before its next tokens leave. On a pass, every output token so far may be
    released. On a failure, the pages are mended, and the tokens chosen since the last pass, which were never
    released, are dropped and generated again. A request that is finishing is ended with an abort instead, which
    releases only its checked tokens (the proof of concept does not regenerate at finish)."""
    rec = dict(op="release_check", rid=req.rid, n_out=len(req.output_ids), finishing=finishing)
    rec.update(act_check(st, sched, {}, req))
    if not st.found:
        st.checked_upto[req.rid] = len(req.output_ids)
        rec["releasable"] = len(req.output_ids)
        return rec
    keep = st.checked_upto.get(req.rid, 0)
    since = st.good_step.get(req.rid, 0)  # the damage struck after this request's last passing check
    rec["mend"] = act_mend(st, sched, {}, req)
    if finishing:
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        req.finished_reason = FINISH_ABORT(message="KV damage found at the end; only checked tokens were released")
        rec["aborted_at_finish"] = True
    else:
        rec["regenerate"] = _redo(st, sched, req, keep)
    # every other strict reader of a mended page drops its unreleased tokens too: they may have read the damage
    mended = set(rec["mend"].get("mended_pages", []))
    rec["other_readers"] = []
    for other in _batch_reqs(sched):
        if (other is req or not is_strict(other) or other.req_pool_idx is None or other.finished() or not mended
                or not mended & set(_reader_view(st, sched, other)[0].tolist())):
            continue
        rec["other_readers"].append(dict(rid=other.rid, **_redo(st, sched, other, st.checked_upto.get(other.rid, 0))))
    # cached KV that finished readers computed after the damage may have read it: drop it from the prefix cache
    rec["evicted_tokens"] = evict_suspect(st, sched.tree_cache, mended, since)
    return rec


def _redo(st, sched, req, keep):
    """Drop a strict request's tokens after `keep` and generate them again, now, or at its next step if it is not in
    the running batch yet (it has just finished its prefill); until then, nothing past `keep` is released."""
    st.checked_upto[req.rid] = keep
    rec = act_retract(st, sched, dict(keep=keep), req)
    if not rec.get("retracted"):
        st.pending_redo[req.rid] = keep
        rec["pending"] = True
    else:
        forget(st, req.rid)
    return rec


def evict_suspect(st, tree, damaged_pages, since):
    """Drop from the prefix cache the KV that may have read damaged pages: unlocked nodes, leaves first, in the
    subtrees of nodes that hold a damaged page, any of whose pages was written after step `since`. The damaged pages
    themselves are repaired in place and stay; a later request recomputes what was dropped instead of reusing it."""
    if tree is None or not damaged_pages or not hasattr(tree, "root_node"):
        return 0
    damaged = torch.tensor(sorted(damaged_pages), dtype=torch.int64, device=st.device)
    evicted = [0]

    def pages_of(node):
        return torch.unique(torch.div(node.value.to(torch.int64).to(st.device), PAGE, rounding_mode="floor"))

    def drop(node):  # post-order: children first, so a parent that becomes a leaf can go in the same pass
        for child in list(node.children.values()):
            drop(child)
        if node is tree.root_node or node.lock_ref > 0 or node.children or node.value is None or not len(node.value):
            return
        pg = pages_of(node)
        pg = pg[~torch.isin(pg, damaged)]  # the damaged pages were just repaired in place: written now, but not suspect
        if pg.numel() and bool((st.wstep[pg] > since).any()):
            tree.token_to_kv_pool_allocator.free(node.value)
            tree._delete_leaf(node)
            evicted[0] += len(node.value)

    def visit(node):
        if node is not tree.root_node and node.value is not None and len(node.value) and bool(
                torch.isin(pages_of(node), damaged).any()):
            drop(node)
            return
        for child in list(node.children.values()):
            visit(child)

    visit(tree.root_node)
    return evicted[0]


def _batch_reqs(sched):
    """The requests of the running batch and of the current batch (one that has just finished its prefill is still
    only in the current batch), each once."""
    reqs, seen = [], set()
    for batch in (getattr(sched, "running_batch", None), getattr(sched, "cur_batch", None)):
        for req in (batch.reqs if batch is not None else []):
            if id(req) not in seen:
                seen.add(id(req))
                reqs.append(req)
    return reqs


def release_checks(st, sched):
    """After a step: check every strict request that has unreleased tokens, right after its prefill (its first
    output token) and every strict_every(req) output tokens. The check after the prefill gives every strict request a
    last good check before any decode step reads its KV: without it, damage found at the first scheduled check leaves
    no bound on what read it, so all of the request's cached KV is suspect and its whole prompt is computed again.
    First, a reader of a mended page that could not regenerate yet does so now."""
    if not STRICT:
        return
    reqs = _batch_reqs(sched)
    running = set(map(id, getattr(sched.running_batch, "reqs", []) if getattr(sched, "running_batch", None) else []))
    for req in reqs:
        if req.rid in st.pending_redo and id(req) in running and req.req_pool_idx is not None:
            keep = st.pending_redo.pop(req.rid)
            rec = act_retract(st, sched, dict(keep=keep), req)
            if rec.get("retracted"):
                forget(st, req.rid)
            _log(sched, dict(op="pending_redo", step=st.step, rid=req.rid, t0=time.time(), **rec))
    due = [req for req in _batch_reqs(sched)
           if is_strict(req) and not req.finished() and req.req_pool_idx is not None
           and (len(req.output_ids) % strict_every(req) == 0 or len(req.output_ids) == 1)
           and st.checked_upto.get(req.rid, 0) < len(req.output_ids)
           and req.rid not in st.pending_redo]
    if not due:
        return
    # One check for every request that is due: one pass over all their pages, one collective. Only when something
    # fails does each affected request go through the full path (mend, regenerate, other readers, cache).
    g, n, me = _group(sched)
    views = [(req,) + _reader_view(st, sched, req) for req in due]
    st.sync()
    t0 = time.perf_counter()
    found = _reader_failures(st, me, torch.cat([v[1] for v in views]), torch.cat([v[2] for v in views]),
                             torch.cat([v[3] for v in views]))
    for req, pages, _, _ in views:
        found += _generation_failures(st, me, req, pages)
    found = _gather_found(st, g, found)
    bad = {p for _, _, p, _ in found}
    passed, redone = [], set()
    for req, pages, _, _ in views:
        if req.rid in redone or req.req_pool_idx is None or req.rid in st.pending_redo:
            continue  # already regenerated this step, as a reader of a page another request's check mended
        if bad and bad & set(pages.tolist()):
            rec = release_check(st, sched, req)
            redone |= {req.rid} | {o["rid"] for o in rec.get("other_readers", [])}
            _log(sched, dict(step=st.step, t0=time.time(), **rec))
        else:
            st.checked_upto[req.rid] = len(req.output_ids)
            _mark_good(st, req, pages)
            passed.append(req.rid)
    st.sync()
    _log(sched, dict(op="release_checks", step=st.step, t0=time.time(), requests=len(due), passed=passed,
                     pages=sum(int(v[1].numel()) for v in views), seconds=time.perf_counter() - t0,
                     found_pieces=len(found)))


def held_back(st, reqs):
    """Before SGLang streams output: cut each strict request's output tokens to the ones its last check covered.
    Returns what was cut, to be put back after the stream call."""
    held = []
    if STRICT and st is not None:
        for req in reqs:
            if is_strict(req):
                upto = st.checked_upto.get(req.rid, 0)
                if len(req.output_ids) > upto:
                    held.append((req, req.output_ids[upto:]))
                    del req.output_ids[upto:]
    return held


# ---------------------------------------------------------------------------------------------------------------
# Background work for the overhead benchmark

def scrub_slice(st, sched, n):
    """Check the next n pieces of one layer (layer-major over pages 1 .. pages-1) without waiting for the GPU. Every
    SYNC_EVERY batch steps the ranks add up their mismatch counts; a nonzero sum is an alarm."""
    span = st.pages - 1
    layer, first = divmod(st.cursor, span)
    n = min(n, span - first)
    st.alarm += (~st.check(layer, st.page_ids[first:first + n])).sum()
    st.cursor = (st.cursor + n) % (st.layers * span)
    st.stats["scrub_pieces"] += n
    if st.batch_steps % SYNC_EVERY == 0:
        g, _, _ = _group(sched)
        dist.all_reduce(st.alarm, group=g.device_group)
        st.stats["alarms"] += int(st.alarm.item())
        st.alarm.zero_()


def periodic(st, sched):
    """Background work, the same on every rank: encode sealed pages every `encode_every` batch steps (parity, as in
    GhostServe), and check `scrub_pieces` pieces per batch step (KVMend's background scrub)."""
    c = st.ctrl
    if not (c["encode_every"] or c["scrub_pieces"]):
        return
    st.batch_steps += 1
    s = st.stats
    s["batch_steps"] += 1
    now = time.perf_counter()
    s["t_first"] = now if s["t_first"] is None else s["t_first"]
    s["t_last"] = now
    if c["encode_every"] and st.batch_steps % c["encode_every"] == 0:
        r = act_encode(st, sched, {}, None)
        s["encodes"] += 1
        s["pages_encoded"] += r["pages"]
        s["unclean"] += len(r["unclean_pieces"])
        s["encode_s"] += time.perf_counter() - now
    if c["scrub_pieces"] and TAGS_ON:
        t0 = time.perf_counter()
        scrub_slice(st, sched, c["scrub_pieces"])
        s["scrub_cpu_s"] += time.perf_counter() - t0


def _read_ctrl(st):
    if CTRL and os.path.exists(CTRL):
        with open(CTRL) as fh:
            st.ctrl.update(json.load(fh))


# ---------------------------------------------------------------------------------------------------------------
# Plan and step

class Plan:
    def __init__(self, path):
        self.items = {}
        if path and os.path.exists(path):
            with open(path) as fh:
                self.items = {rid: sorted(acts, key=lambda x: x["at"]) for rid, acts in json.load(fh).items()}
        self.done = set()

    def due(self, sched):
        for rid, acts in self.items.items():
            req = _find(sched, rid)
            if req is None or req.req_pool_idx is None:
                continue
            for i, a in enumerate(acts):
                if (rid, i) in self.done:
                    continue
                if len(req.output_ids) < a["at"]:
                    break
                yield rid, i, a, req


PLAN = Plan(os.environ.get("KVMEND_PLAN", ""))
_BUF = []  # log lines not yet written


def _find(sched, rid):
    for batch in (sched.running_batch, sched.cur_batch):
        if batch is not None:
            for r in batch.reqs:
                if r.rid == rid:
                    return r
    return None


def _log(sched, rec, now=False):
    _BUF.append({**rec, "time": time.time(), "rank": sched.tp_rank})  # "rank" is always the rank that wrote the line
    if now:
        _flush_log(sched)


def _flush_log(sched):
    if not _BUF:
        return
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"rank{sched.tp_rank}.jsonl"), "a") as fh:
        for rec in _BUF:
            fh.write(json.dumps(rec, default=str) + "\n")
    _BUF.clear()


def _state(sched):
    return getattr(sched.token_to_kv_pool_allocator.get_kvcache(), "_kvm", None)


def on_step(sched):
    st = _state(sched)
    if st is None:
        return
    st.step += 1
    st.step_t.fill_(st.step)
    st.sched = sched
    for rid, i, a, req in list(PLAN.due(sched)):
        PLAN.done.add((rid, i))
        rec = dict(step=st.step, rid=rid, at=a["at"], op=a["op"], n_out=len(req.output_ids), t0=time.time())
        rec.update(ACTIONS[a["op"]](st, sched, a, req))
        _log(sched, rec)
    release_checks(st, sched)


# ---------------------------------------------------------------------------------------------------------------
# Patches, applied by sitecustomize when each module is imported

def patch_memory_pool(mod):
    cls = mod.MHATokenToKVPool
    orig_init, orig_set = cls.__init__, cls.set_kv_buffer

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._kvm = PoolState(self) if self.page_size == PAGE else None

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
        orig_set(self, layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override)
        st = getattr(self, "_kvm", None)
        if st is not None:
            lid = (layer_id_override if layer_id_override is not None else layer.layer_id) - self.start_layer
            if TAGS_ON:
                tk = store_keys(loc, first_layer=lid == 0)
                if tk is None:
                    st.unkeyed += 1
                st.update(lid, loc, tk)
            elif lid == st.layers - 1:
                st.seal(loc)

    cls.__init__, cls.set_kv_buffer = __init__, set_kv_buffer


def store_keys(loc, first_layer=True):
    """The token keys of the tokens a store writes, from the forward batch that is running: the attention backend
    stores forward_batch.out_cache_loc, row for row with forward_batch.input_ids and positions. None when the store
    comes from elsewhere. Computed at the first layer of every forward pass and reused by the later layers of the same
    pass. Never reused across passes: SGLang runs a CUDA graph's batch eagerly before it captures it, and keys kept
    from that run would be baked into the graph, so every replay would key its tokens with the warm-up's ids. Under
    capture the ids and positions are the graph's input buffers, so a replay keys each step's own tokens."""
    fb = _FB[0]
    if fb is None or getattr(fb, "out_cache_loc", None) is not loc:
        return None
    tk = None if first_layer else getattr(fb, "_kvm_tk", None)
    if tk is None:
        if fb.input_ids is None or fb.input_ids.shape[0] != loc.shape[0]:
            return None
        tk = T.token_keys(fb.input_ids, fb.positions)
        fb._kvm_tk = tk
    return tk


def patch_radix_attention(mod):
    cls = mod.RadixAttention
    orig_forward = cls.forward

    def forward(self, q, k, v, forward_batch, *args, **kwargs):
        _FB[0] = forward_batch
        return orig_forward(self, q, k, v, forward_batch, *args, **kwargs)

    cls.forward = forward


def patch_allocator(mod):
    cls = mod.PagedTokenToKVPoolAllocator
    orig_free = cls.free

    def free(self, free_index):
        orig_free(self, free_index)
        st = getattr(getattr(self, "_kvcache", None), "_kvm", None)
        if st is not None and free_index.numel() and self.is_not_in_free_group:
            st.drop(torch.unique(free_index.to(torch.int64) // PAGE))

    cls.free = free


def patch_scheduler(mod):
    cls = mod.Scheduler
    orig_result, orig_idle, orig_flush = cls.process_batch_result, cls.self_check_during_idle, cls.flush_cache

    def process_batch_result(self, batch, result, *args, **kwargs):
        out = orig_result(self, batch, result, *args, **kwargs)
        try:
            if PLAN.items and batch.forward_mode.is_extend():
                for r in batch.reqs:
                    if r.rid in PLAN.items:
                        _log(self, dict(op="extend", rid=r.rid, extend_input_len=int(r.extend_input_len),
                                        prefix_len=len(r.prefix_indices), n_out=len(r.output_ids),
                                        retracted=bool(getattr(r, "is_retracted", False))))
            on_step(self)
            st = _state(self)
            if st is not None:
                periodic(st, self)
        except Exception:
            _log(self, dict(op="error", trace=traceback.format_exc()), now=True)
            raise
        return out

    def self_check_during_idle(self):
        orig_idle(self)
        on_step(self)
        _flush_log(self)

    def flush_cache(self):
        ok = orig_flush(self)
        st = _state(self)
        if ok and st is not None:
            if st.stats["batch_steps"]:
                _log(self, dict(op="periodic", label=st.ctrl.get("label", ""), tags=TAGS_ON,
                                encode_every=st.ctrl["encode_every"], scrub_pieces_per_step=st.ctrl["scrub_pieces"],
                                piece_bytes=st.piece_bytes(), pool_pages=st.pages - 1, layers=st.layers, **st.stats))
            st.reset()
            _read_ctrl(st)
        _flush_log(self)
        return ok

    orig_stream = cls.stream_output_generation

    def stream_output_generation(self, reqs, return_logprob, skip_req=None, *args, **kwargs):
        held = held_back(_state(self), reqs)  # the release rule: unchecked tokens of strict requests stay
        try:
            return orig_stream(self, reqs, return_logprob, skip_req, *args, **kwargs)
        finally:
            for req, tail in held:
                req.output_ids.extend(tail)

    cls.process_batch_result = process_batch_result
    cls.self_check_during_idle = self_check_during_idle
    cls.flush_cache = flush_cache
    cls.stream_output_generation = stream_output_generation


def patch_radix_cache(mod):
    """The release rule's last check: before a finished strict request's KV goes to the prefix cache, check what its
    final tokens read, so that they too leave only after a check."""
    cls = mod.RadixCache
    orig_finished = cls.cache_finished_req

    def cache_finished_req(self, req, *args, **kwargs):
        if STRICT and is_strict(req) and req.req_pool_idx is not None:
            st = getattr(self.token_to_kv_pool_allocator.get_kvcache(), "_kvm", None)
            if st is not None and st.sched is not None and st.checked_upto.get(req.rid, 0) < len(req.output_ids):
                _log(st.sched, dict(step=st.step, t0=time.time(), **release_check(st, st.sched, req, finishing=True)))
        return orig_finished(self, req, *args, **kwargs)

    cls.cache_finished_req = cache_finished_req


def patch_schedule_batch(mod):
    """A request rolled back by SGLang itself (retract_decode under memory pressure) will read new pages, as after a
    KVMend rollback: drop its generation record, or its next check would see a read after free."""
    cls = mod.Req
    orig_reset = cls.reset_for_retract

    def reset_for_retract(self, *args, **kwargs):
        out = orig_reset(self, *args, **kwargs)
        for st in _STATES:
            forget(st, self.rid)
        return out

    cls.reset_for_retract = reset_for_retract


PATCHES = {
    "sglang.srt.mem_cache.memory_pool": patch_memory_pool,
    "sglang.srt.mem_cache.allocator": patch_allocator,
    "sglang.srt.managers.scheduler": patch_scheduler,
    "sglang.srt.layers.radix_attention": patch_radix_attention,
    "sglang.srt.mem_cache.radix_cache": patch_radix_cache,
    "sglang.srt.managers.schedule_batch": patch_schedule_batch,
}
