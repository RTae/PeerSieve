"""KVMend as a small patch on SGLang 0.5.1, loaded into every SGLang process by hook_site/sitecustomize.py.

It changes these places and nothing else:
- MHATokenToKVPool: keeps a running tag for every piece (one 16-token page, one layer, this GPU) as K and V are
  stored (set_kv_buffer). Each token's share is keyed by its token id and position (the token key), taken from the
  forward batch. With KVMEND_TAGS=0 (parity alone, as in GhostServe) it only tracks which pages are full;
- RadixAttention.forward: remembers the forward batch, so that set_kv_buffer knows the ids and positions it stores;
- PagedTokenToKVPoolAllocator.free: drops the tags of freed pages and moves them to a new generation;
- Scheduler: after every iteration (process_batch_result, self_check_during_idle), runs the actions of a JSON plan,
  and, when KVMEND_ENCODE_EVERY or KVMEND_SCRUB_PIECES is set, encodes parity and scrubs in the background; after
  every prefill forward, records the positions it computed for each request (one entry per chunk, note_fills), so a
  failed check can tell what read the damage; right before a prefill forward (run_batch), arms the plan's test fault
  skipped_store;
- HiCache (--enable-hierarchical-cache), whose copies to and from host memory bypass set_kv_buffer:
  HiCacheController.write records the tags of the pages it backs up under their host pages, HiCacheController.load
  gives the new device pages those tags before the load thread copies a byte, MHATokenToKVPoolHost
  .load_to_device_per_layer checks each loaded layer on arrival (before the forward reads it) and copies a piece that
  fails again from the host copy, and HostKVCache.free drops the records of freed host pages.
- with KVMEND_CHECK_IN_ATTN=1, TritonAttnBackend's decode attention: copies of its stage-1 kernels also add up the tag
  of every piece they read (kvmend/attn_check.py), and after every decode step, before SGLang frees what finished
  requests held, the scheduler compares those sums with the stored tags (checks on read).

Every tensor-parallel rank runs the actions at the same point of the same iteration, so their collectives line up.
The plan (KVMEND_PLAN) maps a request id to actions, each due when the request has produced `at` output tokens:
encode, scrub, inject, repair, retract, dump, and, for the comparison with GhostServe, repair_share and evict; for
the incident's faults, fault (a never-written page, a late write through or around the store path, a page holding
another sequence's KV, a store skipped during the prefill), check (the reader's check of one request's pages), mend
(parity rebuild where one GPU is damaged, recompute in place otherwise) and digest (fingerprints of a request's KV, to
compare runs); for HiCache, offload (evict the device copy of what is backed up to host memory) and fault kind
partial_load (the next load from host memory does not land on one GPU).
The release rule (KVMEND_STRICT) checks strict requests as they run. A failed check mends the pages and regenerates
the unreleased tokens, or, with on_fail "abort" (KVMEND_ON_FAIL, or an entry rid:every:abort), aborts the request and
every other reader of the damaged pages and drops those pages from the prefix cache, for a client that resubmits.
With KVMEND_CHECK_AFTER_PREFILL=all, every other request is checked once, right after its prefill, in the same batched
check: a pass is its last good check; a failure mends the pages and reports that its first token, already released,
may be wrong (op released_suspect).
A check right after a prefill bounds nothing, and a chunked prefill's later chunks read the earlier chunks' KV from the
pool. So when it fails, both mends also recompute every position the prefill computed in a later forward than the one
that wrote the first damaged page (all it computed, if that page came from the prefix cache or a host load), before
any decode step reads them (act_mend with suffix, suspect_suffix).
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
# The release rule: request ids under strict release, as "rid", "rid:every" or "rid:every:on_fail" ("*" for all; an
# id ending in "*" names every id that starts with what comes before the "*"). Their tokens leave only after a check
# covers them; a check runs every `every` output tokens (default KVMEND_STRICT_EVERY) and at finish. on_fail is what a
# failed check does (default KVMEND_ON_FAIL): "mend" repairs the pages and generates the unreleased tokens again;
# "abort" ends the request with an abort, which the client answers by resubmitting it (abort_on_fail).
ON_FAIL_ACTIONS = ("mend", "abort")


def _rules(spec):
    """KVMEND_STRICT as ({rid: every or None}, {rid: on_fail}), the second only for entries that name an action."""
    every, on_fail = {}, {}
    for x in spec.split(","):
        if x:
            rid, ev, act = (x.split(":") + ["", ""])[:3]
            if act and act not in ON_FAIL_ACTIONS:
                raise ValueError(f"KVMEND_STRICT entry {x!r}: the action after the second ':' must be one of "
                                 f"{ON_FAIL_ACTIONS}")
            every[rid] = int(ev) if ev else None
            if act:
                on_fail[rid] = act
    return every, on_fail


STRICT, ON_FAIL = _rules(os.environ.get("KVMEND_STRICT", ""))
STRICT_EVERY = int(os.environ.get("KVMEND_STRICT_EVERY", "1"))
ON_FAIL_DEFAULT = os.environ.get("KVMEND_ON_FAIL", "mend")
if ON_FAIL_DEFAULT not in ON_FAIL_ACTIONS:
    raise ValueError(f"KVMEND_ON_FAIL must be one of {ON_FAIL_ACTIONS}, not {ON_FAIL_DEFAULT!r}")
# Which requests get a check right after their prefill (their first output token): "strict" (default) only strict
# requests, as the release rule needs; "all" every request. A pass is the request's last good check, which bounds what
# damage found later can have reached. A request outside the release rule has released its first token before the
# check, so a failure mends its pages and reports that token as suspect; it is never held back, generated again or
# aborted (on_fail is a rule for strict requests).
AFTER_PREFILL_RULES = ("strict", "all")
CHECK_AFTER_PREFILL = os.environ.get("KVMEND_CHECK_AFTER_PREFILL", "strict")
if CHECK_AFTER_PREFILL not in AFTER_PREFILL_RULES:
    raise ValueError(f"KVMEND_CHECK_AFTER_PREFILL must be one of {AFTER_PREFILL_RULES}, not {CHECK_AFTER_PREFILL!r}")
LOAD_CHECK_ON = os.environ.get("KVMEND_LOAD_CHECK", "1") == "1"  # HiCache: check loaded pages on arrival
# The pages each request read (those its checks covered, and those it holds at the end), logged once as it finishes
# (op read_pages), so that a driver can tell whether a request read a damaged page (run_arrivals.py)
LOG_READS = os.environ.get("KVMEND_LOG_READS", "0") == "1"
# Checks on read: the decode attention kernel adds up the tags of the pieces it reads (kvmend/attn_check.py). With
# KVMEND_ATTN_RELOAD=1 the kernel reads the words again from memory instead of reusing the attention's tiles.
CHECK_IN_ATTN = os.environ.get("KVMEND_CHECK_IN_ATTN", "0") == "1"
ATTN_RELOAD = os.environ.get("KVMEND_ATTN_RELOAD", "0") == "1"
ATTN_LOG_MAX = int(os.environ.get("KVMEND_ATTN_LOG_MAX", "200"))  # attn_found lines per flush; later ones are counted
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
        self.pending_suspect = {}  # rid -> (mended pages, since) for evict_suspect when that rollback happens
        self.poisoned = {}      # page -> its generation: damaged pages of aborted requests, to leave the prefix cache
        self.read_pages = {}    # rid -> pages its checks covered (KVMEND_LOG_READS)
        self.read_since = {}    # rid -> wall time of its first check, right after its prefill (KVMEND_LOG_READS)
        self.checked_upto = {}  # rid -> output tokens covered by the request's last passing check (releasable)
        self.prefill_checked = set()  # rids outside the release rule whose one check, after the prefill, has run
        # rid -> the forwards of the request's current prefill, [(step, first, end)]: the positions [first, end) each
        # computed (one per chunk), so that a failed check can tell what read the damage (note_fills, suspect_suffix)
        self.fills = {}
        self.skip = None        # a skipped store armed for the forward that runs now (test fault, _arm_skip)
        self.view_cache = {}    # rid -> token keys of the request's stored tokens, for the reader's check
        self.sched = None       # the scheduler, for checks that run outside on_step (the finish check)
        # HiCache: what each host page left this GPU with (made at the first backup), the host slot each device slot
        # was loaded from (-1: not loaded, or freed since), an armed partial-load fault and the device pages it
        # strikes, and the records of the HiCache hooks, which on_step moves to the log (the load thread never
        # writes the log itself)
        self.hrec = None
        self.src_slot = torch.full((slots,), -1, dtype=torch.int64)
        self.partial = None
        self.partial_target = None
        self.load_events = []
        self.load_batch = None  # the load thread's view of the load it is copying, made at its layer 0
        self.fused = FUSED_ON and self.device.type == "cuda"
        if self.fused:
            self.use_fused()
        self.attn = None        # the check on read inside the decode attention (attn_check.AttnCheck)
        self.attn_found = []    # [(step, rank, layer, page, reason)] that check found, this rank, since the last flush
        if CHECK_IN_ATTN and TAGS_ON and self.device.type == "cuda":
            self.use_attn_check()
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

    def use_attn_check(self, reuse=None):
        """Checks on read: accumulators for the decode attention's tag sums, allocated once, before SGLang captures
        its CUDA graphs. reuse: form the words from the attention's own tiles (default: unless KVMEND_ATTN_RELOAD)."""
        from kvmend import attn_check
        self.attn = attn_check.AttnCheck(self, reuse=not ATTN_RELOAD if reuse is None else reuse)

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
        self.checked_upto, self.view_cache, self.prefill_checked = {}, {}, set()
        self.fills, self.skip = {}, None
        self.stripe_of, self.stripe_pages, self.lparity = {}, {}, {}
        self.wstep.zero_()
        self.good_step, self.pending_redo, self.pending_suspect, self.poisoned = {}, {}, {}, {}
        self.read_pages, self.read_since = {}, {}
        self.batch_steps = 0
        self.alarm.zero_()
        self.stats = new_stats()
        if self.hrec is not None:  # the flush clears HiCache's host pool too
            self.hrec.valid.zero_()
        self.src_slot.fill_(-1)
        self.partial = self.partial_target = self.load_batch = None
        if self.attn is not None:
            self.attn.reset()
        self.attn_found = []

    def drop(self, pages: torch.Tensor):
        """Freed pages lose their tags and parity, and start a new generation: a late write that still carries the
        old one is dropped, and a reader that recorded the old one is stopped."""
        self.tags[:, pages] = 0
        self.mask[:, pages] = 0
        self.encoded[pages] = False
        cpu = pages.cpu()
        self.holder[cpu] = -1
        self.gen[cpu] += 1
        self.src_slot.view(self.pages, PAGE)[cpu] = -1  # a freed page no longer holds what it was loaded from
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

    def check(self, layer: int, pages: torch.Tensor, tk: torch.Tensor = None, mask: torch.Tensor = None,
              fused: bool = None) -> torch.Tensor:
        """Recompute the pages' tags from their bytes and compare. tk [n, PAGE]: the token keys a reader expects
        (default: the keys stored with the slots, which checks the bytes only); mask: the offsets to cover (default:
        the written ones); fused=False: plain PyTorch even when the fused kernels are on (HiCache's load thread)."""
        if tk is None:
            tk = self.tkey.view(self.pages, PAGE)[pages]
        if mask is None:
            mask = self.mask[layer][pages]
        if self.fused if fused is None else fused:
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

    # ---- HiCache: tags travel with the pages' host copies ------------------------------------------------------
    def carry_out(self, dev_slots: torch.Tensor, host_slots: torch.Tensor, host_size: int):
        """A backup to host memory (HiCache's write): record each copied page's tags, coverage, token keys and write
        step under its host page. The pages are locked in the radix cache until the copy lands, and nothing writes a
        cached page, so the record describes the bytes the write thread copies."""
        if dev_slots.numel() == 0 or dev_slots.numel() % PAGE:
            return
        hp = _whole_pages(host_slots.cpu())
        if hp is None:
            self.load_events.append(dict(op="backup", tokens=int(dev_slots.numel()), carried=False,
                                         reason="host slots are not whole pages"))
            return
        if self.hrec is None:
            self.hrec = HostRecord(self, host_size)
        r, hp = self.hrec, hp.to(self.device)
        dp = torch.div(dev_slots.to(self.device, torch.int64)[::PAGE], PAGE, rounding_mode="floor")
        r.tags[:, hp] = self.tags[:, dp]
        r.mask[:, hp] = self.mask[:, dp]
        r.tkey[hp] = self.tkey.view(self.pages, PAGE)[dp]
        r.wstep[hp] = self.wstep[dp]
        r.valid[hp] = True
        self.load_events.append(dict(op="backup", tokens=int(dev_slots.numel()), carried=True))

    def carry_in(self, host_slots: torch.Tensor, dev_slots: torch.Tensor):
        """A load from host memory (HiCache's load) into new device pages: each page gets the tags, coverage, token
        keys and write step its host page left with, before the load thread copies a byte. Nothing is recomputed from
        the bytes that land, so a copy that does not land fails the check. No generation moves: the new pages moved
        to a new one when they were freed. They have no parity yet. A host page with no record (never backed up from
        this GPU, or freed since) loads uncovered, so its check fails rather than passing on no evidence."""
        dcpu, hcpu = dev_slots.cpu().to(torch.int64), host_slots.cpu().to(torch.int64)
        dp_cpu, hp_cpu = _whole_pages(dcpu), _whole_pages(hcpu)
        if dp_cpu is None:  # the paged allocator always hands out whole pages
            self.load_events.append(dict(op="load_issue", tokens=int(dcpu.numel()), carried_pages=0,
                                         reason="device slots are not whole pages"))
            return
        dp = dp_cpu.to(self.device)
        tk = self.tkey.view(self.pages, PAGE)
        if self.hrec is None or hp_cpu is None:
            valid = torch.zeros(dp.numel(), dtype=torch.bool, device=self.device)
            self.tags[:, dp], self.mask[:, dp], tk[dp], self.wstep[dp] = 0, 0, 0, 0
        else:
            r, hp = self.hrec, hp_cpu.to(self.device)
            valid = r.valid[hp]
            self.tags[:, dp] = torch.where(valid, r.tags[:, hp], torch.zeros_like(r.tags[:, hp]))
            self.mask[:, dp] = torch.where(valid, r.mask[:, hp], torch.zeros_like(r.mask[:, hp]))
            tk[dp] = torch.where(valid.unsqueeze(1), r.tkey[hp], torch.zeros_like(r.tkey[hp]))
            self.wstep[dp] = torch.where(valid, r.wstep[hp], torch.zeros_like(r.wstep[hp]))
        self.encoded[dp] = False
        self.holder[dp_cpu] = -1
        self.src_slot[dcpu] = hcpu if hp_cpu is not None else -1
        rec = dict(op="load_issue", tokens=int(dcpu.numel()), pages=int(dp_cpu.numel()),
                   carried_pages=int(valid.sum()))
        f, self.partial = self.partial, None
        if f is not None:  # an armed partial load strikes this load: choose its pages, the same on every GPU
            n = int(dp_cpu.numel())
            k = max(1, min(int(f["pages"]), n))
            j0 = int(f["pos"]) // PAGE if int(f.get("pos", -1)) >= 0 else n - k
            j0 = max(0, min(j0, n - k))
            pages = dp_cpu[j0:j0 + k]
            self.partial_target = dict(rank=f["rank"], here=f["rank"] == f["me"], mode=f["mode"],
                                       check=f["check_on_arrival"], pages=pages.tolist(),
                                       slots=(pages.unsqueeze(1) * PAGE + torch.arange(PAGE)).flatten().to(self.device),
                                       load_positions=[j0 * PAGE, (j0 + k) * PAGE])
            rec["fault"] = dict(kind="partial_load", rank=f["rank"], mode=f["mode"], pages=pages.tolist(),
                                load_positions=[j0 * PAGE, (j0 + k) * PAGE], load_tokens=n * PAGE,
                                check_on_arrival=f["check_on_arrival"])
        self.load_events.append(rec)

    def forget_host(self, host_slots: torch.Tensor):
        """Host slots freed (HostKVCache.free): their pages' records go, so a later load of them is uncovered."""
        if self.hrec is None or host_slots.numel() == 0:
            return
        hp = torch.unique(torch.div(host_slots.cpu().to(torch.int64), PAGE, rounding_mode="floor"))
        self.hrec.valid[hp.to(self.device)] = False

    def host_copy(self, page: int):
        """The 16 host slots a device page was loaded from, while their record is valid (the host copy is still
        there and is what the page's tags describe); None otherwise."""
        hs = self.src_slot[page * PAGE:(page + 1) * PAGE]
        if self.hrec is None or bool((hs < 0).any()):
            return None
        hp = _whole_pages(hs)
        if hp is None or not bool(self.hrec.valid[hp.to(self.device)].all()):
            return None
        return hs

    def arrival_codes(self, layer: int, pages: torch.Tensor) -> torch.Tensor:
        """The check on arrival of loaded pages in one layer: 0 = the bytes match the tag their page left with,
        1 = no record covers the whole page, 2 = the tag differs. Plain PyTorch, in chunks: it runs in HiCache's
        load thread, where no Triton kernel should be compiled."""
        out = []
        for chunk in pages.split(4096):
            cover = self.mask[layer][chunk] == FULL
            ok = self.check(layer, chunk, fused=False)
            out.append(torch.where(~cover, 1, torch.where(ok, 0, 2)).to(torch.int8))
        return torch.cat(out) if out else torch.zeros(0, dtype=torch.int8, device=self.device)


class HostRecord:
    """What each host page left this GPU with (HiCache keeps one host pool per GPU): per host page of 16 slots, every
    layer's tag and coverage, the page's 16 token keys and its write step. It lives on the GPU, keyed by host page, so
    recording and restoring are index copies with no wait. A page loaded back gets these, never tags recomputed from
    the bytes that landed."""

    def __init__(self, st, host_slots):
        n = host_slots // PAGE
        self.tags = torch.zeros(st.layers, n, dtype=torch.int64, device=st.device)
        self.mask = torch.zeros(st.layers, n, dtype=torch.int32, device=st.device)
        self.tkey = torch.zeros(n, PAGE, dtype=torch.int64, device=st.device)
        self.wstep = torch.zeros(n, dtype=torch.int64, device=st.device)
        self.valid = torch.zeros(n, dtype=torch.bool, device=st.device)


def _whole_pages(slots: torch.Tensor):
    """The page of each group of 16 slots, or None unless every group is one whole page in order (CPU tensors only:
    no wait on the GPU). HiCache allocates and frees host memory in whole pages, and the paged allocator hands out
    whole device pages, so this holds; anything else is not carried, and its check fails."""
    if slots.numel() == 0 or slots.numel() % PAGE:
        return None
    s = slots.view(-1, PAGE)
    first = s[:, 0]
    if bool((first % PAGE != 0).any()) or not bool((s == first.unsqueeze(1) + torch.arange(PAGE)).all()):
        return None
    return torch.div(first, PAGE, rounding_mode="floor")


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
    tokens that the radix cache does not hold. With only_if_found, only when the last check found damage. With
    from_prefill, a request that has just finished its prefill, and so is still only in the current batch (the
    scheduler merges that batch into the running batch at its next step), is taken out of the current batch instead,
    which keeps it out of that merge, and that prefill is undone whole: the damage struck before it, so everything it
    computed is suspect. The request goes back to the prefix its prefill started from (its first chunk's, for a
    chunked prefill: _prefill_start), and what SGLang cached of the prefill (cache_unfinished_req, after every chunk)
    leaves the prefix cache (_drop_cached), so the next prefill computes it again. Without from_prefill such a request
    is left alone (the release rule retries it a step later)."""
    if a.get("only_if_found") and not st.alarm_found:
        return dict(retracted=False, reason="the last check found nothing")
    batch = sched.running_batch
    i = next((j for j, r in enumerate(batch.reqs) if r.rid == req.rid), None)
    cur, prefill = getattr(sched, "cur_batch", None), False
    if i is None and a.get("from_prefill") and cur is not None and cur is not batch and cur.forward_mode.is_extend():
        batch, prefill = cur, True
        i = next((j for j, r in enumerate(batch.reqs) if r.rid == req.rid), None)
    if i is None:
        return dict(retracted=False, reason="request is not in the running batch")
    seq_len = int(batch.seq_lens[i].item())
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx]
    held = (len(req.prefix_indices) // sched.page_size) * sched.page_size  # held by the prefix cache, the rest by R
    start = held
    if prefill:  # the prefix the prefill started from (SGLang has set prefix_indices to all it cached since)
        start = min(held, _prefill_start(st, req) // sched.page_size * sched.page_size)
    cached = (torch.unique(torch.div(row[start:held].to(torch.int64), PAGE, rounding_mode="floor")).tolist()
              if start < held else [])
    last = req.last_node
    sched.token_to_kv_pool_allocator.free(row[held:seq_len])
    sched.req_to_token_pool.free(req.req_pool_idx)
    sched.tree_cache.dec_lock_ref(last)
    rec = dict(retracted=True, kept_prefix=start, freed_tokens=seq_len - start, seq_len=seq_len)
    if prefill:
        rec["dropped_from_cache"] = _drop_cached(st, sched.tree_cache, cached, last)
    req.reset_for_retract()
    if "keep" in a:
        # Strict release: output tokens after the last good check were never released, so drop them and let the
        # engine generate them again. Only allowed while no token past `keep` has been sent to the client. keep
        # "last_good": the output tokens the request's last passing check covered (checked_upto, which the release
        # rule and the check right after the prefill keep), 0 if it has none.
        keep = st.checked_upto.get(req.rid, 0) if a["keep"] == "last_good" else a["keep"]
        sent = getattr(req, "send_token_offset", 0)
        if sent <= keep:
            rec.update(regenerated=len(req.output_ids) - keep, kept_outputs=keep)
            del req.output_ids[keep:]
        else:
            rec.update(regenerated=0, kept_outputs=len(req.output_ids), refused="tokens already sent")
    batch.filter_batch(keep_indices=[j for j in range(len(batch.reqs)) if j != i])
    sched._extend_requests_to_queue([req], is_retracted=True)
    return rec


def _prefill_start(st, req):
    """The first position of the request's current prefill: its first forward's, from the record of its forwards
    (note_fills) when that record reaches what it fills now, else its last forward's, which is all SGLang keeps
    (fill_ids and extend_input_len of the last chunk)."""
    end = len(req.fill_ids)
    fills = getattr(st, "fills", {}).get(req.rid) if st is not None else None
    if fills and fills[-1][2] == end:
        return fills[0][1]
    return max(end - int(req.extend_input_len), 0)


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


def _kv_digest(st, slots):
    """A fingerprint of this rank's K and V at `slots`, every layer: the sum of every 32-bit word mixed with its place
    (layer, K or V, token, word), so that a change in any word changes it, but for a chance of about 2^-32."""
    total, idx = 0, None
    for layer in range(st.layers if slots.numel() else 0):
        for kv, buf in enumerate((st.pool.k_buffer[layer], st.pool.v_buffer[layer])):
            w = T.as_words(buf[slots])
            if idx is None:
                idx = torch.arange(w.numel(), dtype=torch.int64, device=w.device).view_as(w)
            key = T.fmix32((idx * 0x9E3779B1 + (2 * layer + kv) * 0x85EBCA77) & T.M32)
            total += int(T.fmix32(w ^ key).sum())
    return total & ((1 << 63) - 1)


def act_digest(st, sched, a, req):
    """Fingerprints of this rank's K and V of R's prompt (_kv_digest), to compare runs bit for bit without saving any
    KV: the whole prompt; before the span `pages`/`pos`; the span; the rest of the prefill forward that computed the
    span's first position (its own chunk), up to the page that holds the next forward's first position; and from that
    page on (the later chunks; R's record of its prefill forwards, note_fills)."""
    start, n = _span(req, a)
    P = len(req.origin_input_ids)
    fills = st.fills.get(req.rid) or []
    row = sched.req_to_token_pool.req_to_token[req.req_pool_idx, :P].to(torch.int64).to(st.device)
    parts = dict(whole=(0, P), before=(0, start), span=(start, min(start + n, P)))
    c = next((f for f in fills if f[1] <= start < f[2]), None)
    if c is not None:
        b = min(max(c[2] // PAGE * PAGE, start + n), P)
        parts.update(same_chunk=(min(start + n, P), b), later=(b, P))
    return dict(positions={k: [x, y] for k, (x, y) in parts.items()}, forwards=[[f[1], f[2]] for f in fills],
                digest={k: _kv_digest(st, row[x:y]) for k, (x, y) in parts.items()})


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
    _note_reads(st, req.rid, on)


def _note_reads(st, rid, pages_cpu):
    if LOG_READS:
        st.read_pages.setdefault(rid, set()).update(pages_cpu.tolist())
        st.read_since.setdefault(rid, time.time())


def _ranges(pages):
    """Sorted page numbers as [first, last] runs: a request reads its prefix pages and its own pages in a few runs."""
    out = []
    for p in sorted(pages):
        if out and p == out[-1][1] + 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return out


def read_record(st, tree, req):
    """The pages request `req` read: those its checks covered and those it holds now (a request never checked,
    under fast release, holds all it read, since nothing rolled it back), and since when: its first check, right after
    its prefill (None if it had none). The record's own time is the end of its reading."""
    n = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
    slots = tree.req_to_token_pool.req_to_token[req.req_pool_idx, :n].to(torch.int64)
    now = set(torch.unique(torch.div(slots, PAGE, rounding_mode="floor")).tolist())
    return dict(op="read_pages", rid=req.rid, n_out=len(req.output_ids),
                pages=_ranges(now | st.read_pages.pop(req.rid, set())), since=st.read_since.pop(req.rid, None))


def forget(st, rid):
    """A request rolled back (by KVMend or by SGLang under memory pressure) will read new pages: drop its records. Its
    next prefill starts a new record of forwards (note_fills)."""
    st.req_gen.pop(rid, None)
    st.view_cache.pop(rid, None)
    st.fills.pop(rid, None)


def act_check(st, sched, a, req, mark_good=True):
    """The reader's check of request R before its KV is used again or its tokens are released. Every page R reads
    must carry a tag that covers exactly the tokens R reads there, and that tag, recomputed from the bytes with R's
    own token keys, must match. A skipped write fails the first test (coverage); changed bytes, or another
    sequence's KV with tags of its own, fail the second (tag). A page that moved to a new generation since R's last
    good check fails as well (generation: a read after free). The findings, the same on every rank, go to st.found.
    A pass is R's last good check, unless mark_good is False."""
    g, n, me = _group(sched)
    pages, expected, tk = _reader_view(st, sched, req)
    st.sync()
    t0 = time.perf_counter()
    found = _reader_failures(st, me, pages, expected, tk) + _generation_failures(st, me, req, pages)
    st.found = _gather_found(st, g, found)
    dt = time.perf_counter() - t0
    st.alarm_found = bool(st.found)
    st.damaged = [(r, l, p) for r, l, p, why in st.found if why == "tag"]
    if not st.found and mark_good:
        _mark_good(st, req, pages)
    else:
        _note_reads(st, req.rid, pages.cpu())
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
      partial_load (HiCache) arms the next load from host memory to land on every GPU but one (see _arm_partial).
      skipped_store  during R's prefill: the forward that computes the span stores none of it (_arm_skip), so its
                   pages keep an earlier owner's bytes, and every later chunk of the prefill reads them. It runs right
                   before that forward (before_forward), not after a step, and its `at` is 0.
    The clean K and V of the span are kept, to measure how close a recompute comes."""
    if a["kind"] == "partial_load":
        return _arm_partial(st, sched, a)
    if a["kind"] == "skipped_store":
        return _arm_skip(st, sched, a, req)
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
    if kind in ("bad_tag", "bad_parity"):
        return _meta_damage(st, sched, a, tgt, pages, rec)
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


def _meta_damage(st, sched, a, tgt, pages, rec):
    """Damage to what KVMend keeps rather than to the KV alone, in one layer (`layer`) on one GPU (rank `rank`):
      bad_tag     one bit of each page's stored tag flips; the KV is intact, but its check fails, and a rebuild from
                  parity restores the same bytes, which still fail: KVMend must recompute the pages, which writes new
                  tags;
      bad_parity  one low mantissa bit of every token's K flips on that GPU, and one bit of the parity that would
                  rebuild each page flips too, on the rank that holds it: the rebuilt pieces fail their tags, and KVMend
                  must recompute the pages.
    The clean pieces are kept, to compare a repair with them."""
    g, n, me = _group(sched)
    rank = a.get("rank", 1)
    layer = a.get("layer", 1)
    rec.update(rank=rank, layer=layer, holders=[int(st.holder[int(p)]) for p in pages.tolist()])
    if me == rank:
        for p in pages.tolist():
            st.saved[(layer, p)] = torch.stack([st.kflat(layer)[p], st.vflat(layer)[p]]).clone()
    if rec["kind"] == "bad_tag":
        if me == rank:
            st.tags[layer, pages] = st.tags[layer, pages] ^ (1 << 7)
    else:
        if me == rank:
            words = st.pool.k_buffer[layer].view(torch.int16).view(st.pool.k_buffer[layer].shape[0], -1)
            words[tgt, 3] = words[tgt, 3] ^ (1 << 3)
        if st.parity is not None:
            for p in pages.tolist():
                if int(st.holder[p]) == me:
                    st.parity[p, layer, 0, 5] = st.parity[p, layer, 0, 5] ^ (1 << 9)
    st.sync()
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


def _arm_skip(st, sched, a, req):
    """A store skipped during R's prefill, armed right before the forward that is about to run (before_forward): of
    the span `pages`/`pos`, the positions this forward computes are not stored, on every GPU, neither their K and V
    nor their tags. Their pages keep the bytes an earlier owner left there, with no tags. The forward itself reads the
    span's K and V from its own tensors, not from the pool, so only the later chunks of the prefill (and every decode
    step) read the stale bytes. With keep_bytes (the fault-free control of this path), the K and V land and only their
    tags are skipped: the check after the prefill fails the same way, and the same repair runs, on undamaged KV."""
    start, n = _span(req, a)
    end = len(req.fill_ids)
    first = end - int(req.extend_input_len)
    lo, hi = max(first, start), min(end, start + n)
    keep_bytes = bool(a.get("keep_bytes", False))
    rec = dict(kind=a["kind"], start=start, tokens=n, keep_bytes=keep_bytes, forward=[first, end],
               prompt=len(req.origin_input_ids), layers=st.layers)
    if lo >= hi:
        return dict(rec, armed=False, reason="this forward computes none of the span")
    slots = sched.req_to_token_pool.req_to_token[req.req_pool_idx, lo:hi].to(torch.int64).to(st.device)
    if st.skip is None:
        empty = torch.zeros(0, dtype=torch.int64, device=st.device)
        st.skip = dict(drop=empty, untag=empty, stores=0, stale_same=0)
    key = "untag" if keep_bytes else "drop"
    st.skip[key] = torch.cat([st.skip[key], slots])
    return dict(rec, armed=True, positions=[lo, hi], skipped_tokens=hi - lo,
                pages=torch.unique(torch.div(slots, PAGE, rounding_mode="floor")).tolist(),
                later_forwards=end < len(req.origin_input_ids))


def _store_with_skip(st, pool, orig_set, layer, loc, cache_k, cache_v, k_scale, v_scale, override):
    """set_kv_buffer while a skipped store is armed (_arm_skip): the armed slots are not stored (K, V and tags), or,
    for keep_bytes, only their tags are not. Each store that drops K and V compares the stale bytes it leaves with
    those it would have written (stale_same counts the stores where they agree, so the fault changed nothing there)."""
    sk = st.skip
    lid = (override if override is not None else layer.layer_id) - pool.start_layer
    tk = store_keys(loc, first_layer=lid == 0) if TAGS_ON else None  # first: it matches the store by `loc` itself
    loc64 = loc.to(torch.int64)
    drop = torch.isin(loc64, sk["drop"])
    untag = drop | torch.isin(loc64, sk["untag"])
    if bool(drop.any()):
        keep = ~drop
        stale_k, stale_v = pool.k_buffer[lid][loc64[drop]], pool.v_buffer[lid][loc64[drop]]
        new_k, new_v = cache_k[drop], cache_v[drop]
        sk["stores"] += 1
        sk["stale_same"] += int(new_k.dtype == stale_k.dtype and new_v.dtype == stale_v.dtype
                                and torch.equal(new_k.reshape(stale_k.shape), stale_k)
                                and torch.equal(new_v.reshape(stale_v.shape), stale_v))
        if bool(keep.any()):
            orig_set(pool, layer, loc[keep], cache_k[keep], cache_v[keep], k_scale, v_scale, override)
    else:
        orig_set(pool, layer, loc, cache_k, cache_v, k_scale, v_scale, override)
    tagged = ~untag
    if not bool(tagged.any()):
        return
    if TAGS_ON:
        if tk is None:
            st.unkeyed += 1
        st.update(lid, loc[tagged], tk[tagged] if tk is not None else None)
    elif lid == st.layers - 1:
        st.seal(loc[tagged])


def before_forward(st, sched, batch):
    """Right before a prefill forward (Scheduler.run_batch): arm the plan's skipped stores due in it, for this forward
    only. An action waits while its span lies beyond this forward (a later chunk computes it), and is done once a
    forward has reached the end of its span. Returns their records, which after_forward completes and logs once the
    forward has run."""
    out = []
    for rid, i, a, req in list(PLAN.before_forward(batch)):
        start, n = _span(req, a)
        if start >= len(req.fill_ids):
            continue
        rec = dict(step=st.step, rid=rid, at=a["at"], op=a["op"], n_out=len(req.output_ids), t0=time.time())
        rec.update(act_fault(st, sched, a, req))
        if len(req.fill_ids) >= start + n:
            PLAN.done.add((rid, i))
        out.append(rec)
    return out


def after_forward(st, sched, recs):
    """After the forward that a skipped store struck: disarm it and log what it did."""
    sk, st.skip = st.skip, None
    for rec in recs:
        if rec.get("armed") and sk is not None:
            rec.update(stores_dropped=sk["stores"], stale_same_stores=sk["stale_same"])
        _log(sched, rec)


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


def note_fills(st, batch):
    """After a prefill forward (an extend batch), before the step's checks: record, for each request it extended, the
    positions it computed, [first, end) (SGLang's fill_ids and extend_input_len of this chunk), with the step its
    stores were stamped with (st.step, which on_step advances after). A forward that starts where the request's last
    one ended continues its prefill (the next chunk); any other starts a new record. Decode tokens of a mixed batch
    are not prefill and are left out."""
    decoding = {id(r) for r in (getattr(batch, "decoding_reqs", None) or [])}
    for req in batch.reqs:
        if id(req) in decoding or getattr(req, "is_retracted", False) or req.finished():
            continue
        end = len(req.fill_ids)
        first = end - int(req.extend_input_len)
        if first >= end:
            continue
        rec = st.fills.get(req.rid)
        if rec and rec[-1][2] == first:
            rec.append((st.step, first, end))
        else:
            st.fills[req.rid] = [(st.step, first, end)]


def suspect_suffix(st, req, blocks, n, use_last_good=True):
    """What else read damage found in R's 16-token `blocks` (pages that failed a check, or a forced span), beyond
    those pages: every position of R's prefill computed in a forward after the one that computed the first damaged
    block, up to the n positions R stores. A forward reads the KV of earlier forwards from the pool but its own from
    its own tensors, so the other positions of that same forward did not read the damage; a page that two forwards
    share counts as the earlier one's, the safe side. A first damaged block before the prefill's first position
    came from the prefix cache or a host load, and the prefill's every forward read it. One after the prefill (a
    decode page) was read by no forward of it. A passing check bounds it too (use_last_good): damage in a page it
    covered struck after it, so only forwards after R's last good check can have read it, and a forward that ran
    after the check carries its step (good_step) or a later one. None when nothing else is suspect, or R has no
    record of its prefill; else the record of the suffix, from the page holding its first position (start) to n."""
    fills = st.fills.get(req.rid) if blocks else None
    if not fills:
        return None
    lo = min(blocks) * PAGE
    if lo >= fills[-1][2]:
        return None
    if lo < fills[0][1]:
        after, where = None, "prefix"
    else:
        c0 = next(f for f in fills if f[1] < lo + PAGE and f[2] > lo)
        after, where = c0[0], [c0[1], c0[2]]
    good = st.good_step.get(req.rid) if use_last_good else None
    later = [f for f in fills if (after is None or f[0] > after) and (good is None or f[0] >= good)]
    if not later or later[0][1] >= n:
        return None
    first = later[0][1]
    start = first // PAGE * PAGE
    return dict(first=first, start=start, end=n, tokens=n - start, forwards=len(later), damage_in=where,
                after_last_good=good is not None and (after is None or good > after))


def _recompute_piece(sched):
    """The most tokens one in-place recompute computes: the engine's prefill chunk, in whole pages (None: no limit)."""
    c = getattr(sched, "chunked_prefill_size", None)
    return c // PAGE * PAGE if isinstance(c, int) and c >= PAGE else None


def _suspect_backups(st, tree, req, pages):
    """HiCache: the nodes of R's path in the prefix cache that hold one of `pages` on the GPU and a host copy that is
    not older than them (_clean_host_copy), so made from the bytes about to be recomputed. Taken before the recompute,
    which moves the pages to a newer write step."""
    if getattr(tree, "cache_controller", None) is None or st.hrec is None or not pages:
        return []
    want = torch.tensor(sorted(pages), dtype=torch.int64)
    out, node = [], getattr(req, "last_node", None)
    while node is not None and node is not getattr(tree, "root_node", None) and hasattr(node, "parent"):
        if (getattr(node, "host_value", None) is not None and node.value is not None and len(node.value)
                and bool(torch.isin(torch.div(node.value.cpu().to(torch.int64), PAGE, rounding_mode="floor"),
                                    want).any()) and not _clean_host_copy(st, node)):
            out.append(node)
        node = node.parent
    return out


def _drop_backups(tree, nodes):
    """Free the host copies of `nodes` (HiCache), once every backup in flight has landed: they hold KV computed from
    the damage, with tags that match it, so a later load would bring it back unseen. Each node keeps its device
    copy, now recomputed; HiCache backs it up again at its next hit (write_through) or evicts it as a node with no
    host copy. Returns the host tokens freed."""
    if not nodes:
        return 0
    _settle_backups(tree)
    freed = 0
    for node in nodes:
        hv = node.host_value
        if hv is not None:
            tree.cache_controller.mem_pool_host.free(hv)
            node.host_value = None
            freed += len(hv)
    return freed


def act_mend(st, sched, a, req):
    """Repair what the last check found. A page loaded from host memory (HiCache) whose host copy is still there is
    copied again from it, on each GPU whose piece failed, bit for bit (recopy=False skips this path). A page damaged on
    one GPU only, still covered by its tags and with parity, is rebuilt from parity, bit for bit (parity=False skips
    this path). Every other page (damage on several GPUs, a skipped write, another sequence's KV, a new generation,
    a copy or rebuild that still fails) is recomputed in place from R's tokens, with R's cached KV before it as the
    prefix. With force, the span of `pages`/`pos` is recomputed whatever the check found: the fault-free control of
    this path.
    With suffix (the release rule's mend, and the check right after the prefill for every request), the repair also
    covers what read the damage: the positions of R's prefill computed in a later forward than the one that computed
    the first damaged page, or all it computed if that page came from the prefix cache or a host load
    (suspect_suffix; with force, as if R had no last good check). Every page from the suffix's first one to the end is
    recomputed in place, after the damaged pages before it, in pieces of at most one prefill chunk: their bytes and
    tags agree, so neither a rebuild from parity nor a copy from host memory would undo what the damage did. Host
    copies (HiCache) made from those bytes are freed (_drop_backups). `read`: pages that failed an earlier check of R
    in this step but that another request's repair has mended since; they repair nothing more, but what read them is
    found and recomputed all the same."""
    g, n, me = _group(sched)
    pages, expected, tk = _reader_view(st, sched, req)
    block_of = {int(p): j for j, p in enumerate(pages.tolist())}
    rebuild, redo, recopy = [], set(), []
    if a.get("force"):
        start, length = _span(req, a)
        redo = {int(pages[j]) for j in range(start // PAGE, (start + length) // PAGE)}
        tgt = sched.req_to_token_pool.req_to_token[req.req_pool_idx, start:start + length].to(torch.int64).to(st.device)
        st.saved_span = (tgt.clone(), torch.stack([torch.stack([st.pool.k_buffer[l][tgt], st.pool.v_buffer[l][tgt]])
                                                   for l in range(st.layers)]).clone())
    elif not st.found and not a.get("read"):
        return dict(mended=False, reason="the last check found nothing")
    else:
        by_page = {}
        for r, l, p, why in st.found:
            by_page.setdefault(p, []).append((r, l, why))
        host = _host_pool(sched) if a.get("recopy", True) else None
        for p, fs in by_page.items():
            h, one_gpu = int(st.holder[p]), len({r for r, _, _ in fs}) == 1
            # a page loaded from host memory: its host copy is what its tags describe, on every GPU (each GPU keeps
            # its own host copy), so the GPUs whose pieces fail copy them again
            if host is not None and all(w == "tag" for _, _, w in fs) and st.host_copy(p) is not None:
                recopy += [(r, l, p) for r, l, _ in fs]
            # parity across GPUs rebuilds damage on one GPU; a stripe within each GPU rebuilds it on any number
            elif (a.get("parity", True) and all(w == "tag" for _, _, w in fs) and bool(st.encoded[p])
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
    stored = len(req.origin_input_ids) + max(len(req.output_ids) - 1, 0)
    rec = dict(mended=True, rebuilt=[], recomputed=[], recopied=[])
    backups = []
    if a.get("suffix"):
        bad = {block_of[p] for p in redo | {x[2] for x in rebuild} | {x[2] for x in recopy} | set(a.get("read", ()))
               if p in block_of}
        rec["suffix"] = suspect_suffix(st, req, bad, stored, use_last_good=not a.get("force"))
        if rec["suffix"] is not None:
            later = {int(p) for p in pages[rec["suffix"]["start"] // PAGE:].tolist()}
            rebuild = [x for x in rebuild if x[2] not in later]
            recopy = [x for x in recopy if x[2] not in later]
            redo |= later
            backups = _suspect_backups(st, getattr(sched, "tree_cache", None), req, redo)
    if not (rebuild or redo or recopy):  # only `read` pages, and nothing R computed after them: nothing to repair
        return dict(mended=False, reason="nothing to repair: the pages were mended by another request's repair, and "
                                         "nothing R computed read them", suffix=rec.get("suffix"))
    t_mend = _start(st, g)
    if recopy:
        rec["recopied"] = _recopy(st, host, me, recopy)
        # a copied piece that still fails R's check (the host copy is damaged too) is recomputed
        rp = sorted({p for _, _, p in recopy if p in block_of})
        js = torch.tensor([block_of[p] for p in rp], dtype=torch.int64, device=st.device)
        failed = _gather_found(st, g, _reader_failures(st, me, pages[js], expected[js], tk[js]) if rp else [])
        if failed:
            rec["recopy_failed_pages"] = sorted({p for _, _, p, _ in failed})
            redo |= set(rec["recopy_failed_pages"])
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
    spans, piece = [], _recompute_piece(sched)
    for j in blocks:
        if spans and spans[-1][1] == j:
            spans[-1][1] = j + 1
        else:
            spans.append([j, j + 1])
    for j0, j1 in spans:
        s, e = j0 * PAGE, min(j1 * PAGE, stored)
        for p0 in range(s, e, piece or e - s):  # each piece reads the ones before it, recomputed, from the pool
            p1 = min(p0 + (piece or e - s), e)
            t0 = _start(st, g)
            _recompute_range(st, sched, req, p0, p1)
            st.sync()
            rec["recomputed"].append(dict(start=p0, end=p1, tokens=p1 - p0, seconds=time.perf_counter() - t0))
    if backups:
        rec["host_copies_dropped"] = _drop_backups(sched.tree_cache, backups)
    st.sync()
    rec["mend_seconds"] = time.perf_counter() - t_mend  # the whole repair, before the test's own checks below
    rec["mended_pages"] = sorted({p for _, _, p in rebuild} | {p for _, _, p in recopy} | redo)
    if blocks:
        idx = torch.tensor(blocks, device=st.device)
        rec["still_failing_here"] = len(_reader_failures(st, me, pages[idx], expected[idx], tk[idx]))
    if st.saved_span is not None and (rec["recomputed"] or rec["rebuilt"] or rec["recopied"]):
        tgt, clean = st.saved_span
        now = torch.stack([torch.stack([st.pool.k_buffer[l][tgt], st.pool.v_buffer[l][tgt]]) for l in range(st.layers)])
        rec.update(max_abs_diff_vs_clean=float((now.float() - clean.float()).abs().max()),
                   bit_equal_share_vs_clean=float((now.view(torch.int16) == clean.view(torch.int16)).float().mean()))
    st.found, st.damaged = [], []
    return rec


# ---------------------------------------------------------------------------------------------------------------
# Loads from host memory (SGLang's hierarchical cache, HiCache): the tags travel with the host copy, every loaded
# layer is checked on arrival before the forward reads it, and a piece that did not land is copied again

def _host_pool(sched):
    """HiCache's host pool of this GPU, or None without HiCache."""
    return getattr(getattr(sched, "tree_cache", None), "token_to_kv_pool_host", None)


def _host_kv(host, layer, host_slots):
    """K and V of the given host slots in one layer, for either of HiCache's host layouts."""
    if getattr(host, "layout", "layer_first") == "page_first":  # [size, layers, heads, dim]
        return host.k_buffer[host_slots, layer], host.v_buffer[host_slots, layer]
    return host.k_buffer[layer][host_slots], host.v_buffer[layer][host_slots]  # [layers, size, heads, dim]


def _copy_from_host(st, host, layer, dev_slots, host_slots):
    """Copy one layer's K and V of the given slots from the host copy; True if the device bytes now equal it."""
    k, v = _host_kv(host, layer, host_slots.cpu())
    st.pool.k_buffer[layer][dev_slots] = k.to(st.device)
    st.pool.v_buffer[layer][dev_slots] = v.to(st.device)
    return (torch.equal(st.pool.k_buffer[layer][dev_slots].cpu(), k)
            and torch.equal(st.pool.v_buffer[layer][dev_slots].cpu(), v))


def _arm_partial(st, sched, a):
    """Arm a partial load: the next load from host memory lands on every GPU but `rank`, for `pages` pages from
    position `pos` of the load (default: its last pages). On that GPU those device pages keep the bytes they held
    (mode stale: a transfer that never arrived) or read zeros (mode zeros), and nothing reports an error, as after a
    failed per-GPU transfer. Stale bytes can be the very bytes the load brings (SGLang hands the pages an eviction
    just freed out first); the check's record counts the layers the fault changed. With check_on_arrival false, that
    load is not checked on arrival, so the loading request's reader check must find it. Armed on every GPU, so that
    every GPU names the same pages."""
    g, n, me = _group(sched)
    st.partial = dict(rank=min(int(a.get("rank", 1)), n - 1), me=me, pages=int(a.get("pages", 4)),
                      pos=int(a.get("pos", -1)), mode=a.get("mode", "stale"),
                      check_on_arrival=bool(a.get("check_on_arrival", True)))
    if st.partial["mode"] not in ("stale", "zeros"):
        raise ValueError(f"unknown partial-load mode {st.partial['mode']}")
    return dict(kind="partial_load", armed=True, **{k: v for k, v in st.partial.items() if k != "me"})


def _tree_nodes(tree):
    stack, out = [tree.root_node], []
    while stack:
        node = stack.pop()
        stack += list(node.children.values())
        out.append(node)
    return out


def _settle_backups(tree, timeout=120.0):
    """HiCache: wait until the backup of every node of the current tree that is still being written has landed, then
    let SGLang release those nodes on every GPU together (its writing_check agrees on the count across GPUs); called
    at the same point on every GPU. Only nodes of the current tree are waited for: SGLang's reset (every cache flush)
    clears the queue of finished backups but not its list of backups in flight, so that list can hold nodes of an
    earlier tree for ever. Returns the seconds waited, the nodes waited for and those still pending."""
    t0 = time.perf_counter()
    writing = {n.id for n in _tree_nodes(tree) if tree.ongoing_write_through.get(n.id) is n}
    acks = tree.cache_controller.ack_write_queue
    while time.perf_counter() < t0 + timeout:
        with acks.mutex:  # the write thread adds to the queue; read it under its lock
            landed = set(acks.queue)
        if writing <= landed:
            break
        time.sleep(0.001)
    waited = time.perf_counter() - t0
    tree.writing_check()
    return waited, len(writing), sum(1 for i in writing if i in tree.ongoing_write_through)


def act_offload(st, sched, a, req):
    """Leave the prefix cache's KV in host memory only (HiCache): once every backup in flight has landed and SGLang
    has released those nodes (_settle_backups), evict the device copy of every unlocked node. A node with a host copy
    keeps it, so the next request with that prefix loads it back into new device pages; a node without one is
    dropped."""
    tree = sched.tree_cache
    if getattr(tree, "cache_controller", None) is None:
        return dict(offloaded=False, reason="no hierarchical cache")
    t0 = time.perf_counter()
    waited, writing, pending = _settle_backups(tree, float(a.get("timeout", 120)))
    before = int(tree.evictable_size())
    tree.evict(before)
    host_only = sum(len(n.host_value) for n in _tree_nodes(tree)  # tokens now held in host memory only
                    if n is not tree.root_node and n.value is None and getattr(n, "host_value", None) is not None)
    return dict(offloaded=True, waited_s=waited, writes_waited_for=writing, pending_writes=pending,
                evictable_before=before, evictable_after=int(tree.evictable_size()), host_only_tokens=host_only,
                seconds=time.perf_counter() - t0)


def _locked_below(node):
    return node.lock_ref > 0 or any(_locked_below(c) for c in node.children.values())


def _on_gpu_below(node):
    return any(c.value is not None or _on_gpu_below(c) for c in node.children.values())


def _clean_host_copy(st, node):
    """A node's host copy (HiCache) is clean if its record is valid and older than every device page of the node:
    it was made before those pages were written, so it does not hold what they hold."""
    hv = getattr(node, "host_value", None)
    if hv is None or st is None or st.hrec is None:
        return False
    hp, dp = _whole_pages(hv.cpu().to(torch.int64)), _whole_pages(node.value.cpu().to(torch.int64))
    if hp is None or dp is None or len(hp) != len(dp):
        return False
    hp, dp = hp.to(st.device), dp.to(st.device)
    return bool(st.hrec.valid[hp].all()) and bool((st.hrec.wstep[hp] < st.wstep[dp]).all())


def _remove_subtree(tree, node, hicache, out):
    """A node of the prefix cache and everything below it, leaves first, with their device and host copies."""
    for child in list(node.children.values()):
        _remove_subtree(tree, child, hicache, out)
    on_gpu = node.value is not None
    if on_gpu:
        tree.token_to_kv_pool_allocator.free(node.value)
        out["device_tokens"] += len(node.value)
    hv = getattr(node, "host_value", None)
    if hicache and hv is not None:
        tree.cache_controller.mem_pool_host.free(hv)
        out["host_tokens"] += len(hv)
    if on_gpu:
        tree._delete_leaf(node)  # also leaves the evictable count, as in RadixCache's own eviction
    else:  # an evicted node is not in the evictable count
        for k, v in list(node.parent.children.items()):
            if v is node:
                del node.parent.children[k]


def _drop_cached(st, tree, pages, last_node=None):
    """Drop from the prefix cache the KV of `pages`: device pages that a rolled-back prefill wrote, and that SGLang
    cached right after that prefill (cache_unfinished_req), before any check ran. Kept on the GPU, such a page would
    serve the next prefill of the same tokens: SGLang's insert frees the page that prefill computes and points the
    request at the cached one. Each node holding one of the pages (on the request's own path when last_node is given)
    goes once nothing holds it, nor anything below it. A HiCache node whose host copy is clean (_clean_host_copy) and
    which has nothing on the GPU below it loses only its device copy, as in SGLang's own eviction; the next prefill
    then computes the KV again, and the node takes that. Any other node goes with everything below it and its host
    copies, which were made from the dropped pages or on top of them."""
    out = dict(nodes=0, device_tokens=0, host_tokens=0, host_kept_tokens=0, locked_tokens=0)
    if not pages or tree is None or not hasattr(tree, "root_node"):
        return out
    hicache = getattr(tree, "cache_controller", None) is not None
    if hicache:
        _settle_backups(tree)  # a node locked only by its own backup is released once that backup has landed
    bad = torch.tensor(sorted(pages), dtype=torch.int64)
    if last_node is not None:  # the request's path, deepest node first
        nodes, node = [], last_node
        while node is not None and node is not tree.root_node:
            nodes.append(node)
            node = node.parent
    else:  # the whole tree, children before their parents
        nodes = list(reversed(_tree_nodes(tree)))
    for node in nodes:
        if (node is tree.root_node or node.value is None or not len(node.value) or not bool(torch.isin(
                torch.div(node.value.cpu().to(torch.int64), PAGE, rounding_mode="floor"), bad).any())):
            continue
        if _locked_below(node):
            out["locked_tokens"] += len(node.value)
            continue
        out["nodes"] += 1
        if hicache and _clean_host_copy(st, node) and not _on_gpu_below(node):
            out["host_kept_tokens"] += len(node.host_value)
            out["device_tokens"] += tree._evict_backuped(node)
        else:
            _remove_subtree(tree, node, hicache, out)
    return out


def _recopy(st, host, me, items):
    """Copy damaged pieces [(rank, layer, page)] again from the host copy their pages were loaded from, each on the
    GPU that holds it; no other GPU takes part. Each copied piece must match its tag again."""
    st.sync()
    t0 = time.perf_counter()
    mine, exact = [(l, p) for r, l, p in items if r == me], {}
    for l, p in mine:
        exact[(l, p)] = _copy_from_host(st, host, l, torch.arange(p * PAGE, (p + 1) * PAGE, device=st.device),
                                        st.host_copy(p))
    st.sync()
    dt = time.perf_counter() - t0
    out = []
    for r, l, p in items:
        rec = dict(rank=r, layer=l, page=p, holder="host", bytes_read=st.piece_bytes())
        if r == me:
            rec.update(recopied=True, seconds=dt / max(1, len(mine)), equals_host_copy=exact[(l, p)],
                       tag_matches=bool(st.check(l, torch.tensor([p], device=st.device)).all()))
        out.append(rec)
    return out


def _load_view(st, host_slots, dev_slots):
    """The load thread's view of the load it copies, made at layer 0 and kept for its other layers: the device and
    host slot of each token, the loaded pages, the slots an armed partial load strikes here, and the running
    totals of the check on arrival. In SGLang's "kernel" I/O both index tensors are on the GPU; in "direct" both are
    on the CPU."""
    ds = dev_slots.to(st.device, torch.int64)
    t = st.partial_target
    hit = None
    if t is not None:
        sel = torch.isin(ds, t["slots"])
        hit = ds[sel] if bool(sel.any()) else None
    skip = hit is not None and not t["check"]
    return dict(ds=ds, hs=host_slots.cpu().to(torch.int64), dpage=torch.div(ds, PAGE, rounding_mode="floor"),
                pages=torch.unique(torch.div(ds, PAGE, rounding_mode="floor")), target=t if hit is not None else None,
                hit=hit, check=LOAD_CHECK_ON and TAGS_ON and not skip, t0=time.time(),
                found=[], uncovered=[], recopied=0, still=[], exact=True, check_s=0.0, recopy_s=0.0)


def load_layer(st, host, host_slots, dev_slots, layer, copy):
    """One layer of a load from host memory, in HiCache's load thread on its load stream, before the layer is
    released to the forward (the thread synchronizes the stream and counts the layer done after this returns, and
    attention waits for that count): copy() does SGLang's copy; with an armed partial load on this GPU, the target
    slots then hold their earlier bytes again, or zeros; then the check on arrival tests every loaded piece of this
    layer against the tag its page left with, and copies each piece that fails again from the host copy, which must
    then pass. Never raises: an exception would end the load thread, and the forward would wait for ever."""
    b = stash = None
    try:
        if layer == 0 or st.load_batch is None:
            st.load_batch = _load_view(st, host_slots, dev_slots)
        b = st.load_batch
        if b["hit"] is not None and b["target"]["here"] and b["target"]["mode"] == "stale":
            stash = (st.pool.k_buffer[layer][b["hit"]].clone(), st.pool.v_buffer[layer][b["hit"]].clone())
    except Exception:
        st.load_events.append(dict(op="error", where="load_layer, before the copy", trace=traceback.format_exc()))
        b = st.load_batch = None
    copy()
    if b is None:
        return
    try:
        t = b["target"]
        if t is not None and t["here"]:  # the copy of the target pages did not land on this GPU
            k, v = st.pool.k_buffer[layer][b["hit"]], st.pool.v_buffer[layer][b["hit"]]  # what landed
            if stash is not None:
                changed = not (torch.equal(k, stash[0]) and torch.equal(v, stash[1]))
                st.pool.k_buffer[layer][b["hit"]] = stash[0]
                st.pool.v_buffer[layer][b["hit"]] = stash[1]
            else:
                changed = bool(k.any() or v.any())
                st.pool.k_buffer[layer][b["hit"]] = 0
                st.pool.v_buffer[layer][b["hit"]] = 0
            # SGLang hands freed pages out first, so a load right after an eviction can land in the very slots just
            # freed, whose stale bytes are the ones it brings: then the fault changes nothing in that layer
            t["changed_layers"] = t.get("changed_layers", 0) + int(changed)
            if layer == 0:
                t["stale_was_zero"] = stash is None or bool((stash[0] == 0).all() and (stash[1] == 0).all())
        if b["check"]:
            t0 = time.perf_counter()
            codes = st.arrival_codes(layer, b["pages"])
            bad = b["pages"][codes == 2]
            b["uncovered"] += [(layer, p) for p in b["pages"][codes == 1].tolist()]
            b["check_s"] += time.perf_counter() - t0
            if bad.numel():
                bad_list = bad.tolist()
                b["found"] += [(layer, p) for p in bad_list]
                t1 = time.perf_counter()
                sel = torch.isin(b["dpage"], bad)
                b["exact"] &= _copy_from_host(st, host, layer, b["ds"][sel], b["hs"][sel.cpu()])
                again = st.arrival_codes(layer, bad)
                b["still"] += [(layer, p) for p, c in zip(bad_list, again.tolist()) if c]
                b["recopied"] += len(bad_list)
                b["recopy_s"] += time.perf_counter() - t1
        if layer == st.layers - 1:
            found_pages = sorted({p for _, p in b["found"]})
            rec = dict(op="load_check", step=st.step, t0=b["t0"], tokens=int(b["ds"].numel()),
                       pages=int(b["pages"].numel()), layers=st.layers, checked=b["check"],
                       found_pieces=len(b["found"]), found_pages=found_pages,
                       found_layers=len({l for l, _ in b["found"]}), uncovered_pieces=len(b["uncovered"]),
                       uncovered_pages=sorted({p for _, p in b["uncovered"]})[:64], recopied_pieces=b["recopied"],
                       still_failing=b["still"][:64], still_failing_pieces=len(b["still"]),
                       recopy_equals_host_copy=b["exact"] if b["recopied"] else None,
                       check_seconds=b["check_s"], recopy_seconds=b["recopy_s"])
            if t is not None:
                rec["fault"] = dict(kind="partial_load", rank=t["rank"], struck_here=t["here"], mode=t["mode"],
                                    pages=t["pages"], load_positions=t["load_positions"],
                                    stale_was_zero=t.get("stale_was_zero"), changed_layers=t.get("changed_layers"))
                if st.partial_target is t:  # spent; a fault armed for a later load stays
                    st.partial_target = None
            st.load_events.append(rec)
            st.load_batch = None
    except Exception:
        st.load_events.append(dict(op="error", where="load_layer, after the copy", trace=traceback.format_exc()))
        st.load_batch = None


def drain_load_events(st, sched):
    """Move the records of the HiCache hooks (backups, loads issued, the load thread's checks on arrival, errors) to
    the log, with the requests of the batch that just ran."""
    if not st.load_events:
        return
    cur = getattr(sched, "cur_batch", None)
    rids = [r.rid for r in cur.reqs] if cur is not None else []
    while st.load_events:
        rec = st.load_events.pop(0)
        _log(sched, dict(rec, step=rec.get("step", st.step), rids=rids))


ACTIONS = dict(encode=act_encode, scrub=act_scrub, inject=act_inject, repair=act_repair, retract=act_retract,
               dump=act_dump, repair_share=act_repair_share, evict=act_evict, abort=act_abort,
               check=act_check, fault=act_fault, mend=act_mend, offload=act_offload, digest=act_digest)


# ---------------------------------------------------------------------------------------------------------------
# The release rule: a strict request's tokens leave only after a check covers what they read

_PATTERNS = [None, []]  # the STRICT table the list below was made from, and its "prefix*" ids, longest first


def _rule_id(rid):
    """The KVMEND_STRICT id that names request `rid`: the rid itself, else the longest "prefix*" it starts with."""
    rid = str(rid)
    if rid in STRICT:
        return rid
    if _PATTERNS[0] is not STRICT:
        _PATTERNS[:] = [STRICT, sorted((k for k in STRICT if k.endswith("*")), key=len, reverse=True)]
    return next((k for k in _PATTERNS[1] if rid.startswith(k[:-1])), None)


def is_strict(req):
    return bool(STRICT) and not str(req.rid).startswith("kvmend-") and _rule_id(req.rid) is not None


def strict_every(req):
    return STRICT.get(_rule_id(req.rid)) or STRICT.get("*") or STRICT_EVERY


def on_fail(req):
    """What a failed check does to a strict request: "mend" or "abort"."""
    return ON_FAIL.get(_rule_id(req.rid), ON_FAIL_DEFAULT)


def _may_read(other, req):
    """`other` holds KV that a check of its pages can cover: a live request other than `req`, past its prefill (a
    request in the middle of a chunked prefill has positions with no slots yet)."""
    return (other is not req and other.req_pool_idx is not None and not other.finished()
            and getattr(other, "is_chunked", 0) <= 0)


def release_check(st, sched, req, finishing=False):
    """Check a strict request's pages before its next tokens leave. On a pass, every output token so far may be
    released. On a failure, the pages are mended, with what read the damage: a check right after the prefill bounds
    nothing, so the positions the prefill computed in a later forward than the one that wrote the first damaged page
    are recomputed too (act_mend with suffix; suffix_tokens). Then the tokens chosen since the last pass, which were
    never released, are dropped and generated again. A request that is finishing is ended with an abort instead,
    which releases only its checked tokens (the proof of concept does not regenerate at finish). With on_fail
    "abort", nothing is mended: the request is aborted (abort_on_fail)."""
    rec = dict(op="release_check", rid=req.rid, n_out=len(req.output_ids), finishing=finishing)
    rec.update(act_check(st, sched, {}, req))
    if not st.found:
        st.checked_upto[req.rid] = len(req.output_ids)
        rec["releasable"] = len(req.output_ids)
        return rec
    if on_fail(req) == "abort":
        rec.update(abort_on_fail(st, sched, req, finishing))
        return rec
    keep = st.checked_upto.get(req.rid, 0)
    since = _since(st, req)  # the damage struck after this request's last passing check
    rec["mend"] = act_mend(st, sched, dict(suffix=True), req)
    rec["suffix_tokens"] = (rec["mend"].get("suffix") or {}).get("tokens", 0)
    mended = set(rec["mend"].get("mended_pages", []))
    if finishing:
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        req.finished_reason = FINISH_ABORT(message="KV damage found at the end; only checked tokens were released")
        rec["aborted_at_finish"] = True
    else:
        rec["regenerate"] = _redo(st, sched, req, keep, (mended, since))
    rec.update(_after_mend(st, sched, req, mended, since))
    return rec


def released_suspect(st, sched, req, read=()):
    """The check right after the prefill (KVMEND_CHECK_AFTER_PREFILL=all) found damage in pages that a request outside
    the release rule reads (`read`: those pages, as the batched check found them). Its first token, chosen by the
    prefill, has left already and may have read the damage: the record (op released_suspect) reports it. What the
    request's own check still finds is mended as for a strict request (copied again from host memory, rebuilt from
    parity, or recomputed), with what read the damage: the positions its
    prefill computed in a later forward than the one that wrote the first damaged page (act_mend with suffix; the
    record's suffix_tokens and suffix give the size and place of that suffix), so that the tokens it chooses from now
    on read none of it. The strict readers of a mended page and the cached KV that may have read the damage are
    handled as release_check handles them. Nothing of this request is held back, generated again or aborted, and this
    check is not its last good check: its own check may pass because another request's repair in this step has
    mended the page already, and its prefill read the damage all the same, so what it computed after that page is
    recomputed even then."""
    rec = dict(op="released_suspect", rid=req.rid, n_out=len(req.output_ids), released_tokens=len(req.output_ids),
               reason="damage found right after the prefill: the released tokens were chosen before this check")
    since = _since(st, req)
    rec.update(act_check(st, sched, {}, req, mark_good=False))
    rec["mend"] = act_mend(st, sched, dict(suffix=True, read=sorted(read)), req)
    if not rec["mend"].get("mended"):
        rec["mend"] = dict(mended=False, reason="its own check passes: another request's repair in this step mended "
                                                "the page, or the page failed only another reader's check; nothing "
                                                "its prefill computed after that page is suspect")
        rec.update(suffix_tokens=0, suffix=None)
        return rec
    suffix = rec["mend"].get("suffix")
    rec.update(suffix_tokens=suffix["tokens"] if suffix else 0,
               suffix={k: suffix[k] for k in ("first", "start", "end", "damage_in")} if suffix else None)
    rec.update(_after_mend(st, sched, req, set(rec["mend"].get("mended_pages", [])), since))
    return rec


def _since(st, req):
    """evict_suspect's bound for damage found by a check of `req`, which struck after the request's last passing check
    (good_step): the stamp of the last write that cannot have read it. on_step sets the step before that step's checks,
    and the next forward stamps its writes with that same step (wstep), so the first forward after a check at step g
    already writes KV stamped g: since is g - 1. A request with no passing check bounds nothing, and since is 0: only
    the engine's first forward, before any scheduler step, stamps 0, and it read nothing cached by another."""
    good = st.good_step.get(req.rid)
    return 0 if good is None else good - 1


def _after_mend(st, sched, req, mended, since):
    """After a repair that `req`'s check found: every other strict reader of a mended page drops its unreleased tokens
    too, since they may have read the damage, and the cached KV that finished readers computed after the damage leaves
    the prefix cache."""
    others = []
    for other in _batch_reqs(sched):
        if (not is_strict(other) or not _may_read(other, req) or not mended
                or not mended & set(_reader_view(st, sched, other)[0].tolist())):
            continue
        others.append(dict(rid=other.rid,
                           **_redo(st, sched, other, st.checked_upto.get(other.rid, 0), (mended, since))))
    return dict(other_readers=others, evicted_tokens=evict_suspect(st, sched.tree_cache, mended, since))


def _redo(st, sched, req, keep, suspect=None):
    """Drop a strict request's tokens after `keep` and generate them again, now, or at its next step if it is not in
    the running batch yet (it has just finished its prefill); until then, nothing past `keep` is released. suspect:
    the repair's (mended pages, since). A request whose rollback waits still holds its cached pages, so the eviction
    after the repair (evict_suspect) kept them: its prompt KV, computed on top of the damage if it arrived after it.
    They are evicted when the rollback lets them go, or the re-prefill would reuse them."""
    st.checked_upto[req.rid] = keep
    rec = act_retract(st, sched, dict(keep=keep), req)
    if not rec.get("retracted"):
        st.pending_redo[req.rid] = keep
        if suspect is not None:
            st.pending_suspect[req.rid] = suspect
        rec["pending"] = True
    else:
        forget(st, req.rid)
    return rec


def abort_on_fail(st, sched, req, finishing=False):
    """on_fail "abort": the damage is not mended. The request ends with an abort, which releases only the tokens its
    last check covered, and so does every other strict reader of a damaged page whose rule is also "abort": they may
    have read the damage. The client resubmits them. The damaged pages (those whose tags or coverage failed) leave the
    prefix cache with everything cached on top of them, once no running request holds them (drop_poisoned), so a
    resubmission recomputes from the first damaged page on and reuses the clean prefix before it."""
    bad = {p for _, _, p, why in st.found if why in ("tag", "coverage")}
    for p in bad:
        st.poisoned.setdefault(p, int(st.gen[p]))
    _abort(req, finishing)
    others = []
    for other in _batch_reqs(sched):
        if (is_strict(other) and on_fail(other) == "abort" and _may_read(other, req)
                and not getattr(other, "to_abort", False) and bad & set(_reader_view(st, sched, other)[0].tolist())):
            _abort(other)
            others.append(dict(rid=other.rid, aborted=True))
    st.found, st.damaged = [], []
    return dict(aborted=True, poisoned_pages=sorted(bad), other_readers=others,
                dropped_tokens=drop_poisoned(st, sched.tree_cache))


def _abort(req, finishing=False):
    """End a strict request with an abort; its unchecked tokens are never released. A request that is finishing (the
    check in cache_finished_req) gets its finish reason now; any other is marked as SGLang's own abort marks it, and
    ends at its next step."""
    message = "KV damage found; resubmit the request"
    if finishing:
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        req.finished_reason = FINISH_ABORT(message=message)
    else:
        req.to_abort, req.to_abort_message = True, message


def drop_poisoned(st, tree):
    """Drop the damaged pages of aborted requests from the prefix cache, with all the KV cached on top of them, which
    was computed after them and may have read them. A node that holds a damaged page is split at the first one: the
    clean part before it stays cached, the part from it on goes with everything below it. A node that a running
    request holds stays until that request ends (abort_on_fail aborts every reader). A damaged page that has been
    freed meanwhile (a new generation) is done. Returns the tokens dropped."""
    for p, gen in list(st.poisoned.items()):
        if int(st.gen[p]) != gen:
            del st.poisoned[p]
    if not st.poisoned or tree is None or not hasattr(tree, "root_node"):
        return 0
    bad = torch.tensor(sorted(st.poisoned), dtype=torch.int64, device=st.device)
    dropped, todo = 0, [tree.root_node]
    while todo:
        for child in list(todo.pop().children.values()):
            if child.value is None or not len(child.value):
                continue
            hit = torch.isin(torch.div(child.value.to(st.device, torch.int64), PAGE, rounding_mode="floor"),
                             bad).nonzero().flatten()
            if not hit.numel():
                todo.append(child)
            elif child.lock_ref == 0:  # nothing running holds it, nor anything below it (locks reach the root)
                cut = int(hit[0]) // PAGE * PAGE
                if cut:
                    tree._split_node(child.key, child, cut)  # a new parent keeps the clean head; child keeps the rest
                dropped += _drop_subtree(tree, child)
    return dropped


def _drop_subtree(tree, node):
    """Free a node of the prefix cache and everything below it, leaves first, as RadixCache.evict frees leaves."""
    n = sum(_drop_subtree(tree, c) for c in list(node.children.values()))
    tree.token_to_kv_pool_allocator.free(node.value)
    tree._delete_leaf(node)
    return n + len(node.value)


def evict_suspect(st, tree, damaged_pages, since):
    """Drop from the prefix cache the KV that may have read damaged pages: unlocked nodes, leaves first, in the
    subtrees of nodes that hold a damaged page, any of whose pages carries a write stamp (wstep) above `since`. Callers
    pass the stamp of the last write that cannot have read the damage: one less than the step of the check that bounds
    it (_since). The damaged pages themselves are repaired in place and stay; a later request recomputes
    what was dropped instead of reusing it."""
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


def _due(st, req):
    """Is a check of `req` due after this step? A strict request: right after its prefill (its first output token) and
    every strict_every(req) output tokens, while it has tokens no check covers. Any other request, with
    KVMEND_CHECK_AFTER_PREFILL=all: once, right after its prefill."""
    n = len(req.output_ids)
    if is_strict(req):
        due = (n % strict_every(req) == 0 or n == 1) and st.checked_upto.get(req.rid, 0) < n
    else:
        due = (CHECK_AFTER_PREFILL == "all" and n == 1 and req.rid not in st.prefill_checked
               and not str(req.rid).startswith("kvmend-"))
    return (due and not req.finished() and req.req_pool_idx is not None and not getattr(req, "to_abort", False)
            and req.rid not in st.pending_redo)


def release_checks(st, sched):
    """After a step: check every strict request that has unreleased tokens, right after its prefill (its first
    output token) and every strict_every(req) output tokens, and, with KVMEND_CHECK_AFTER_PREFILL=all, every other
    request once, right after its prefill. The check after the prefill gives a request a last good check before any
    decode step reads its KV: without it, damage found at a later check leaves no bound on what read it, so all of the
    request's cached KV is suspect and its whole prompt is computed again. A request outside the release rule that
    passes gets the same records as a strict one (checked_upto, good_step, its pages' generations); one that fails is
    mended and reported (released_suspect), nothing more. First, a reader of a mended page that could not regenerate
    yet does so now, and its cached KV that may have read the damage leaves the prefix cache. A request being aborted
    is not checked again."""
    if not STRICT and CHECK_AFTER_PREFILL != "all":
        return
    reqs = _batch_reqs(sched)
    running = set(map(id, getattr(sched.running_batch, "reqs", []) if getattr(sched, "running_batch", None) else []))
    for req in reqs:
        if req.rid in st.pending_redo and id(req) in running and req.req_pool_idx is not None:
            keep = st.pending_redo.pop(req.rid)
            rec = act_retract(st, sched, dict(keep=keep), req)
            if rec.get("retracted"):
                forget(st, req.rid)
                if req.rid in st.pending_suspect:
                    rec["evicted_tokens"] = evict_suspect(st, sched.tree_cache, *st.pending_suspect.pop(req.rid))
            _log(sched, dict(op="pending_redo", step=st.step, rid=req.rid, t0=time.time(), **rec))
    due = [req for req in _batch_reqs(sched) if _due(st, req)]
    if not due:
        return
    # One check for every request that is due, strict or not: one pass over all their pages, one collective. Only when
    # something fails does each affected request go through the full path (mend, regenerate, other readers, cache; for
    # a request outside the release rule, mend and report).
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
    passed, redone, not_strict = [], set(), []
    for req, pages, _, _ in views:
        if req.rid in redone or req.req_pool_idx is None or req.rid in st.pending_redo:
            continue  # already regenerated this step, as a reader of a page another request's check mended
        strict = is_strict(req)
        if not strict:  # its one check, right after the prefill
            not_strict.append(req.rid)
            st.prefill_checked.add(req.rid)
        if bad and bad & set(pages.tolist()):
            rec = (release_check(st, sched, req) if strict
                   else released_suspect(st, sched, req, read=bad & set(pages.tolist())))
            redone |= {req.rid} | {o["rid"] for o in rec.get("other_readers", [])}
            _log(sched, dict(step=st.step, t0=time.time(), **rec))
        else:
            st.checked_upto[req.rid] = len(req.output_ids)
            _mark_good(st, req, pages)
            passed.append(req.rid)
        if not strict:
            st.view_cache.pop(req.rid, None)  # checked once: its token keys need not stay on the GPU
    st.sync()
    _log(sched, dict(op="release_checks", step=st.step, t0=time.time(), requests=len(due), passed=passed,
                     not_strict=not_strict, pages=sum(int(v[1].numel()) for v in views),
                     seconds=time.perf_counter() - t0, found_pieces=len(found)))


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
# Checks on read inside the decode attention (KVMEND_CHECK_IN_ATTN=1)

def attn_verdict(st, sched, batch):
    """After a decode step, before SGLang frees the KV of the requests that finished in it: compare what the step's
    attention read with the stored tags and coverage (attn_check.AttnCheck.verdict), record this rank's failures as
    _reader_failures does, (rank, layer, page, reason), and zero the accumulators. The record's step is this
    scheduler iteration's (on_step increments st.step later in the same iteration). Waits for the GPU once."""
    t0 = time.perf_counter()
    found, read, layout = st.attn.verdict()
    me = sched.tp_rank
    fails = [(me, l, p, why) for l, p, why in found] + ([(me, -1, -1, "layout")] if layout else [])
    st.attn.stats["seconds"] += time.perf_counter() - t0
    if not fails:
        return fails
    step = st.step + 1
    st.attn_found += [(step,) + f for f in fails]
    del st.attn_found[:-10000]  # keep the newest
    if st.attn.stats["logged"] < ATTN_LOG_MAX:
        st.attn.stats["logged"] += 1
        _log(sched, dict(op="attn_found", step=step, t0=time.time(), pieces_read=read, layout_anomalies=layout,
                         found_pieces=len(fails), found=fails[:64], summary=_summary(fails),
                         rids=_readers_of(st, sched, batch, {p for _, _, p, _ in fails if p >= 0})))
    return fails


def _readers_of(st, sched, batch, pages):
    """The requests of the batch whose KV includes one of `pages` (for the log only; None if it cannot tell)."""
    if not pages:
        return []
    try:
        want = torch.tensor(sorted(pages), dtype=torch.int64, device=st.device)
        out = []
        for req in batch.reqs:
            n = len(req.origin_input_ids) + len(req.output_ids)  # what the step read: the new token's KV included
            slots = sched.req_to_token_pool.req_to_token[req.req_pool_idx, :n].to(torch.int64)
            if bool(torch.isin(torch.div(slots, PAGE, rounding_mode="floor"), want).any()):
                out.append(req.rid)
        return out
    except Exception:
        return None


def _attn_target(q, k_buffer, v_buffer):
    """The pool state and layer whose KV buffers these are, if the check on read covers them; else None."""
    for st in reversed(_STATES):
        if st.attn is not None and k_buffer.data_ptr() in st.attn.layer_of:
            layer = st.attn.target(q, k_buffer, v_buffer)
            return None if layer is None else (st, layer)
    return None


def _attn_decode(orig):
    """TritonAttnBackend.decode_attention_fwd with the check on read for the KV pools KVMend watches. It runs at
    capture time inside CUDA graphs: it only launches kernels on fixed buffers."""
    def decode_attention_fwd(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
                             max_kv_splits, sm_scale, logit_cap=0.0, sinks=None, xai_temperature_len=-1):
        hit = _attn_target(q, k_buffer, v_buffer)
        if hit is None:
            return orig(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
                        max_kv_splits, sm_scale, logit_cap=logit_cap, sinks=sinks,
                        xai_temperature_len=xai_temperature_len)
        st, layer = hit
        st.attn.decode(layer, q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
                       max_kv_splits, sm_scale, logit_cap=logit_cap, sinks=sinks,
                       xai_temperature_len=xai_temperature_len)

    return decode_attention_fwd


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
                if (rid, i) in self.done or _before_forward(a):
                    continue
                if len(req.output_ids) < a["at"]:
                    break
                yield rid, i, a, req

    def before_forward(self, batch):
        """The actions due right before this prefill forward (_before_forward): those of the requests it extends."""
        for req in batch.reqs:
            for i, a in enumerate(self.items.get(req.rid, ())):
                if ((req.rid, i) not in self.done and _before_forward(a) and len(req.output_ids) >= a["at"]
                        and req.req_pool_idx is not None):
                    yield req.rid, i, a, req


def _before_forward(a):
    """A plan action that runs right before a prefill forward of its request, not after a step: a skipped store."""
    return a.get("op") == "fault" and a.get("kind") == "skipped_store"


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
    drain_load_events(st, sched)
    for rid, i, a, req in list(PLAN.due(sched)):
        PLAN.done.add((rid, i))
        rec = dict(step=st.step, rid=rid, at=a["at"], op=a["op"], n_out=len(req.output_ids), t0=time.time())
        rec.update(ACTIONS[a["op"]](st, sched, a, req))
        _log(sched, rec)
    release_checks(st, sched)
    if st.poisoned:  # damaged pages of aborted requests that a running request still held at the last try
        _drop_and_log(st, sched, sched.tree_cache, "step")


def _drop_and_log(st, sched, tree, after):
    t0 = time.time()
    n = drop_poisoned(st, tree)
    if n:
        _log(sched, dict(op="drop_poisoned", step=st.step, t0=t0, after=after, dropped_tokens=n,
                         pending_pages=sorted(st.poisoned)))


# ---------------------------------------------------------------------------------------------------------------
# Patches, applied by sitecustomize when each module is imported

def patch_memory_pool(mod):
    cls = mod.MHATokenToKVPool
    orig_init, orig_set = cls.__init__, cls.set_kv_buffer

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._kvm = PoolState(self) if self.page_size == PAGE else None

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
        st = getattr(self, "_kvm", None)
        if st is not None and st.skip is not None:  # a test fault armed for this forward only (_arm_skip)
            return _store_with_skip(st, self, orig_set, layer, loc, cache_k, cache_v, k_scale, v_scale,
                                    layer_id_override)
        orig_set(self, layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override)
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
        st = _state(self)
        if st is not None and st.attn is not None and batch.forward_mode.is_decode():
            # Checks on read: before SGLang's own processing, which frees the tail pages of finished requests (and
            # with them their tags), so every page this step read still holds the tag it was read against.
            try:
                attn_verdict(st, self, batch)
            except Exception:
                _log(self, dict(op="error", trace=traceback.format_exc()), now=True)
                raise
        out = orig_result(self, batch, result, *args, **kwargs)
        try:
            if st is not None and batch.forward_mode.is_extend():
                note_fills(st, batch)  # before this step's checks, which read it
            if PLAN.items and batch.forward_mode.is_extend():
                for r in batch.reqs:
                    if r.rid in PLAN.items:
                        # prefix_len is read after SGLang cached the prefill (cache_unfinished_req sets prefix_indices
                        # to all it cached); cached_prefix is the prefix the prefill started from
                        _log(self, dict(op="extend", rid=r.rid, extend_input_len=int(r.extend_input_len),
                                        prefix_len=len(r.prefix_indices), n_out=len(r.output_ids),
                                        cached_prefix=len(r.fill_ids) - int(r.extend_input_len),
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
            drain_load_events(st, self)
            if st.stats["batch_steps"]:
                _log(self, dict(op="periodic", label=st.ctrl.get("label", ""), tags=TAGS_ON,
                                encode_every=st.ctrl["encode_every"], scrub_pieces_per_step=st.ctrl["scrub_pieces"],
                                piece_bytes=st.piece_bytes(), pool_pages=st.pages - 1, layers=st.layers, **st.stats))
            if st.attn is not None and st.attn.stats["steps"]:
                _log(self, dict(op="attn_check", label=st.ctrl.get("label", ""), reuse=st.attn.reuse,
                                unsupported=dict(st.attn.unsupported), piece_bytes=st.piece_bytes(), **st.attn.stats))
            st.reset()
            _read_ctrl(st)
        _flush_log(self)
        return ok

    orig_run = cls.run_batch

    def run_batch(self, batch, *args, **kwargs):
        """Right before a prefill forward, the plan's skipped stores due in it are armed (before_forward); right after
        it, they are disarmed and logged (after_forward)."""
        st = _state(self) if PLAN.items and batch is not None and batch.forward_mode.is_extend() else None
        armed = []
        if st is not None:
            try:
                armed = before_forward(st, self, batch)
            except Exception:
                _log(self, dict(op="error", trace=traceback.format_exc()), now=True)
                raise
        try:
            return orig_run(self, batch, *args, **kwargs)
        finally:
            if st is not None and (armed or st.skip is not None):
                after_forward(st, self, armed)

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
    cls.run_batch = run_batch
    cls.stream_output_generation = stream_output_generation


def patch_radix_cache(mod):
    """The release rule's last check: before a finished strict request's KV goes to the prefix cache, check what its
    final tokens read, so that they too leave only after a check. A request that ends with an abort releases nothing
    more and is not checked. Once SGLang has cached the request and released its lock, damaged pages of aborted
    requests that nothing holds any more leave the prefix cache (drop_poisoned), before the client hears of the
    abort. With KVMEND_LOG_READS, every finished request's pages are logged (read_record)."""
    cls = mod.RadixCache
    orig_finished = cls.cache_finished_req

    def cache_finished_req(self, req, *args, **kwargs):
        st = getattr(self.token_to_kv_pool_allocator.get_kvcache(), "_kvm", None) if STRICT or LOG_READS else None
        if (STRICT and st is not None and st.sched is not None and is_strict(req) and req.req_pool_idx is not None
                and not getattr(req, "to_abort", False) and st.checked_upto.get(req.rid, 0) < len(req.output_ids)):
            _log(st.sched, dict(step=st.step, t0=time.time(), **release_check(st, st.sched, req, finishing=True)))
        if LOG_READS and st is not None and st.sched is not None and req.req_pool_idx is not None:
            _log(st.sched, dict(step=st.step, **read_record(st, self, req)))
        out = orig_finished(self, req, *args, **kwargs)
        for state in _STATES:  # the request has ended: its record of prefill forwards goes
            getattr(state, "fills", {}).pop(req.rid, None)
        if st is not None and st.sched is not None and st.poisoned:
            _drop_and_log(st, st.sched, self, req.rid)
        return out

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


def _sync_current(device):
    if device.type == "cuda":
        torch.cuda.current_stream(device).synchronize()


def patch_cache_controller(mod):
    """HiCache's copies between GPU and host memory bypass set_kv_buffer, so the tags travel with them: a backup
    records the tags of the pages it copies under their host pages, and a load gives the new device pages those
    tags. load is SGLang 0.5.1's HiCacheController.load step for step, with the tags carried in before the operation
    is queued: the load thread, which checks every layer on arrival, never sees the new pages without them."""
    cls, op = mod.HiCacheController, mod.CacheOperation
    orig_write = cls.write

    def write(self, device_indices, priority=None, node_id=0):
        host_indices = orig_write(self, device_indices, priority, node_id)
        st = getattr(self.mem_pool_device, "_kvm", None)
        if host_indices is not None and st is not None and TAGS_ON:
            st.carry_out(device_indices, host_indices, self.mem_pool_host.size)
        return host_indices

    def load(self, host_indices, priority=None, node_id=0):
        device_indices = self.mem_pool_device_allocator.alloc(len(host_indices))
        if device_indices is None:
            return None
        self.mem_pool_host.protect_load(host_indices)
        st = getattr(self.mem_pool_device, "_kvm", None)
        if st is not None and TAGS_ON:
            st.carry_in(host_indices, device_indices)
        _sync_current(device_indices.device)  # SGLang: the indices must be ready before the load stream reads them
        self.load_queue.put(op(host_indices, device_indices, node_id, priority))
        return device_indices

    cls.write, cls.load = write, load


def patch_memory_pool_host(mod):
    """HiCache's host pool: each layer of a load is checked on arrival (load_layer), and freed host slots lose
    their tag records."""
    cls, base = mod.MHATokenToKVPoolHost, mod.HostKVCache
    orig_layer, orig_free = cls.load_to_device_per_layer, base.free

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        st = getattr(device_pool, "_kvm", None)

        def copy():
            orig_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend)

        if st is None or not TAGS_ON:
            return copy()
        # layer_id is the local layer (the load thread counts 0 .. layer_num - 1; SGLang indexes k_buffer with it)
        load_layer(st, self, host_indices, device_indices, layer_id, copy)

    def free(self, indices):
        out = orig_free(self, indices)
        st = getattr(getattr(self, "device_pool", None), "_kvm", None)
        if st is not None:
            st.forget_host(indices)
        return out

    cls.load_to_device_per_layer, base.free = load_to_device_per_layer, free


def patch_triton_backend(mod):
    """Checks on read (KVMEND_CHECK_IN_ATTN=1): each TritonAttnBackend's decode attention also adds up the tags of the
    pieces it reads. SGLang's forward_decode stores the new token's K and V, then calls self.decode_attention_fwd,
    which __init__ sets; the patch wraps that attribute, so forward_decode itself is SGLang's. A backend with sliding
    windows reads part of a page, which the coverage test would flag, so it keeps SGLang's kernel."""
    if not CHECK_IN_ATTN:
        return
    cls = mod.TritonAttnBackend
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        window = getattr(self, "sliding_window_size", None)
        if window is None or window <= 0:
            self.decode_attention_fwd = torch.compiler.disable(_attn_decode(self.decode_attention_fwd))

    cls.__init__ = __init__


PATCHES = {
    "sglang.srt.mem_cache.memory_pool": patch_memory_pool,
    "sglang.srt.mem_cache.allocator": patch_allocator,
    "sglang.srt.managers.scheduler": patch_scheduler,
    "sglang.srt.layers.radix_attention": patch_radix_attention,
    "sglang.srt.mem_cache.radix_cache": patch_radix_cache,
    "sglang.srt.managers.schedule_batch": patch_schedule_batch,
    "sglang.srt.managers.cache_controller": patch_cache_controller,
    "sglang.srt.mem_cache.memory_pool_host": patch_memory_pool_host,
    "sglang.srt.layers.attention.triton_backend": patch_triton_backend,
}
