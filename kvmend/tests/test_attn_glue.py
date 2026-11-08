"""The scheduler and backend glue of the checks on read (KVMEND_CHECK_IN_ATTN), on CPU: a stand-in replaces the GPU
accumulators of attn_check.AttnCheck, so no Triton or SGLang is needed. The kernels themselves are tested on a GPU in
tests_gpu/test_attn_check.py."""
import types

import torch

from kvmend import hook

PAGE = 16


class _Check:
    """attn_check.AttnCheck's interface: layer_of, target, decode, verdict, reset, stats."""

    def __init__(self, pool):
        self.layer_of = {b.data_ptr(): l for l, b in enumerate(pool.k_buffer)}
        self.calls, self.result, self.unsupported, self.reuse = [], ([], 10, 0), {}, True
        self.reset()

    def reset(self):
        self.stats = dict(steps=0, pieces=0, failures=0, layout=0, seconds=0.0, logged=0)

    def target(self, q, k, v):
        return self.layer_of.get(k.data_ptr())

    def decode(self, layer, *args, **kwargs):
        self.calls.append(layer)

    def verdict(self):
        self.stats["steps"] += 1
        return self.result


def _setup():
    pool = types.SimpleNamespace(page_size=PAGE, start_layer=0,
                                 k_buffer=[torch.zeros(8 * PAGE, 1, 64, dtype=torch.bfloat16) for _ in range(2)],
                                 v_buffer=[torch.zeros(8 * PAGE, 1, 64, dtype=torch.bfloat16) for _ in range(2)])
    st = hook.PoolState(pool)
    assert st.attn is None, "a CPU pool gets no check on read"
    st.attn = _Check(pool)
    table = torch.zeros(2, 64, dtype=torch.int32)
    table[0, :32] = torch.arange(2 * PAGE, 4 * PAGE)
    table[1, :32] = torch.arange(5 * PAGE, 7 * PAGE)
    reqs = [types.SimpleNamespace(rid="A", req_pool_idx=0, origin_input_ids=list(range(20)), output_ids=[1, 2]),
            types.SimpleNamespace(rid="B", req_pool_idx=1, origin_input_ids=list(range(17)), output_ids=[1])]
    sched = types.SimpleNamespace(tp_rank=1, req_to_token_pool=types.SimpleNamespace(req_to_token=table))
    return pool, st, sched, types.SimpleNamespace(reqs=reqs)


def test_the_verdict_records_and_logs_failures_with_their_readers():
    saved = hook._STATES[:], hook._BUF[:]
    hook._STATES.clear()
    hook._BUF.clear()
    try:
        pool, st, sched, batch = _setup()
        assert hook.attn_verdict(st, sched, batch) == [] and not hook._BUF, "a clean step logs nothing"
        st.attn.result = ([(1, 3, "tag"), (0, 6, "coverage")], 10, 2)
        st.step = 41
        fails = hook.attn_verdict(st, sched, batch)
        assert fails == [(1, 1, 3, "tag"), (1, 0, 6, "coverage"), (1, -1, -1, "layout")], fails
        rec = hook._BUF[-1]
        # the step is this iteration's: on_step increments st.step after the verdict
        assert rec["op"] == "attn_found" and rec["step"] == 42 and rec["rank"] == 1, rec
        # A read page 3 (its positions 16-21), B page 6 (its positions 16-17)
        assert rec["rids"] == ["A", "B"], rec["rids"]
        assert st.attn_found[0] == (42, 1, 1, 3, "tag") and rec["summary"]["3"]["tag"]["ranks"] == [1], rec
        st.reset()
        assert st.attn_found == [] and st.attn.stats["steps"] == 0, "a cache flush starts the check over"
    finally:
        hook._STATES[:], hook._BUF[:] = saved


def test_the_backend_patch_routes_watched_buffers_to_the_check():
    saved = hook._STATES[:], hook.CHECK_IN_ATTN
    hook._STATES.clear()
    calls = []

    def sglang_kernel(*args, **kwargs):
        calls.append("sglang")

    class Backend:  # as TritonAttnBackend: __init__ sets decode_attention_fwd
        sliding_window_size = None

        def __init__(self):
            self.decode_attention_fwd = sglang_kernel

    class Windowed(Backend):
        sliding_window_size = 1024

    try:
        pool, st, _, _ = _setup()
        hook.CHECK_IN_ATTN = False
        hook.patch_triton_backend(types.SimpleNamespace(TritonAttnBackend=Backend))
        assert Backend().decode_attention_fwd is sglang_kernel, "patched without KVMEND_CHECK_IN_ATTN"
        hook.CHECK_IN_ATTN = True
        hook.patch_triton_backend(types.SimpleNamespace(TritonAttnBackend=Backend))
        hook.patch_triton_backend(types.SimpleNamespace(TritonAttnBackend=Windowed))
        assert Windowed().decode_attention_fwd is sglang_kernel, "a sliding-window backend must keep its kernel"
        b = Backend()
        assert b.decode_attention_fwd is not sglang_kernel
        q = torch.zeros(2, 4, 64)
        rest = (torch.zeros(2, 4, 64), torch.zeros(3, dtype=torch.int32), torch.zeros(5, dtype=torch.int64),
                torch.zeros(2, 4, 8, 64), torch.zeros(2, 4, 8), torch.zeros(2, dtype=torch.int32), 8, 0.125)
        b.decode_attention_fwd(q, pool.k_buffer[1], pool.v_buffer[1], *rest, logit_cap=0.0, sinks=None,
                               xai_temperature_len=-1)
        assert st.attn.calls == [1] and calls == [], (st.attn.calls, calls)
        b.decode_attention_fwd(q, torch.zeros_like(pool.k_buffer[0]), pool.v_buffer[0], *rest)
        assert calls == ["sglang"], "buffers of no watched pool must go to SGLang's kernel"
    finally:
        hook._STATES[:] = saved[0]
        hook.CHECK_IN_ATTN = saved[1]
