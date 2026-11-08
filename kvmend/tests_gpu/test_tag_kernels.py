"""The fused Triton tag kernels give exactly the PyTorch tags (GPU only; skipped elsewhere)."""
import pytest
import torch

from kvmend import hook
from kvmend import tags as T

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
PAGE = 16


class Pool:
    def __init__(self, pages=40, layers=3, heads=2, dim=64, dtype=torch.bfloat16, seed=0):
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.page_size, self.start_layer = PAGE, 0
        self.k_buffer = [torch.randn(pages * PAGE, heads, dim, device="cuda", generator=g).to(dtype)
                         for _ in range(layers)]
        self.v_buffer = [torch.randn(pages * PAGE, heads, dim, device="cuda", generator=g).to(dtype)
                         for _ in range(layers)]


def pair(pool):
    """Two pool states over the same bytes: the PyTorch path and the fused path."""
    ref, fused = hook.PoolState(pool), hook.PoolState(pool)
    ref.fused = False
    fused.use_fused()
    return ref, fused


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_page_tags_equal_pytorch(dtype):
    from kvmend import tag_kernels as K

    pool = Pool(dtype=dtype)
    st = hook.PoolState(pool)
    pages = torch.tensor([1, 2, 5, 9, 17, 33], device="cuda")
    g = torch.Generator(device="cuda").manual_seed(1)
    tk = torch.randint(0, 2**32, (pages.numel(), PAGE), device="cuda", generator=g, dtype=torch.int64)
    for mask in (torch.full((6,), 0xFFFF), torch.tensor([1, 3, 0x7FFF, 0, 0x8001, 0xFFFF])):
        mask = mask.to(torch.int32).cuda()
        for layer in range(len(pool.k_buffer)):
            want = T.page_tags(st.kp(layer)[pages], st.vp(layer)[pages], mask, st.keys, tk)
            got = K.page_tags(K.words(pool.k_buffer[layer]), K.words(pool.v_buffer[layer]), pages, mask, tk, st.keys)
            assert T.same(got, want).all()


@cuda
def test_fused_store_update_equals_pytorch():
    pool = Pool()
    ref, fused = pair(pool)

    def store(layer, loc, tk=None):
        for st in (ref, fused):
            st.update(layer, loc, tk)

    for layer in range(3):
        loc = torch.arange(1 * PAGE, 4 * PAGE + 5, device="cuda")  # a prefill: three pages and part of a fourth
        store(layer, loc, T.token_keys(torch.arange(loc.numel(), device="cuda") + 7, torch.arange(loc.numel(), device="cuda")))
        store(layer, torch.tensor([4 * PAGE + 5, 9 * PAGE], device="cuda"))  # decode steps; page 9 starts afresh
        store(layer, torch.arange(2 * PAGE, 3 * PAGE, device="cuda"))  # page 2 rewritten from its first slot
    for layer in range(3):
        assert T.same(fused.tags[layer], ref.tags[layer]).all(), layer
        assert torch.equal(fused.mask[layer], ref.mask[layer]), layer
    assert torch.equal(fused.tkey, ref.tkey)
    assert torch.equal(fused.encoded[1:], ref.encoded[1:])
    idx = (ref.mask[0] != 0).nonzero().flatten()
    assert fused.check(0, idx).all() and ref.check(0, idx).all()
    pool.k_buffer[0].view(torch.int16)[2 * PAGE + 3, 1, 7] ^= 1 << 14  # one flipped bit on page 2
    assert fused.check(0, idx).tolist() == [p != 2 for p in idx.tolist()]


@cuda
def test_fused_store_update_in_a_cuda_graph():
    """The store-time kernels wait for nothing on the host, so a CUDA graph can replay them with new inputs."""
    pool = Pool()
    ref, fused = pair(pool)
    loc = torch.zeros(2, dtype=torch.int64, device="cuda")
    tk = torch.zeros(2, dtype=torch.int64, device="cuda")
    fused.update(0, loc, tk)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.update(0, loc, tk)
    for step in range(PAGE):  # two requests decode on pages 5 and 7, one token per step
        loc.copy_(torch.tensor([5 * PAGE + step, 7 * PAGE + step]))
        tk.copy_(T.token_keys(torch.tensor([100 + step, 200 + step]), torch.tensor([step, step])))
        graph.replay()
        ref.update(0, loc.clone(), tk.clone())
    torch.cuda.synchronize()
    for p in (5, 7):
        assert T.same(fused.tags[0][p], ref.tags[0][p]) and int(fused.mask[0][p]) == 0xFFFF


class _Layer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


@cuda
@pytest.mark.parametrize("fused", [False, True])
def test_graph_replays_key_each_steps_own_tokens(fused, monkeypatch):
    """SGLang runs a decode batch eagerly, then captures it in a CUDA graph with the same forward batch object, then
    replays the graph with new ids and positions in the same buffers. Every replay must key its own tokens: keys kept
    from the eager run would be baked into the graph (the bug of job 326397)."""
    import types

    from kvmend import tag_kernels

    class GraphPool(Pool):
        def __init__(self, *args, **kwargs):
            super().__init__(pages=12, layers=2, heads=1, dim=64)

        def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
            self.k_buffer[layer.layer_id][loc] = cache_k
            self.v_buffer[layer.layer_id][loc] = cache_v

    hook.patch_memory_pool(types.SimpleNamespace(MHATokenToKVPool=GraphPool))
    pool = GraphPool()
    st = pool._kvm
    if fused:
        st.use_fused()
    ids, pos, loc = (torch.zeros(2, dtype=torch.int64, device="cuda") for _ in range(3))  # the graph's buffers
    k = torch.zeros(2, 1, 64, dtype=torch.bfloat16, device="cuda")
    v = torch.zeros_like(k)
    fb = types.SimpleNamespace(out_cache_loc=loc, input_ids=ids, positions=pos)
    monkeypatch.setattr(hook, "_FB", [fb])

    def step():
        for layer in (_Layer(0), _Layer(1)):
            pool.set_kv_buffer(layer, loc, k, v)

    ids.copy_(torch.tensor([7, 8]))
    step()  # the eager run before the capture, with the warm-up's tokens
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    g = torch.Generator(device="cuda").manual_seed(3)
    for t in range(PAGE):  # two requests decode on pages 5 and 7
        loc.copy_(torch.tensor([5 * PAGE + t, 7 * PAGE + t]))
        ids.copy_(torch.tensor([100 + t, 200 + t]))
        pos.copy_(torch.tensor([t, 32 + t]))
        k.copy_(torch.randn(2, 1, 64, device="cuda", generator=g).to(torch.bfloat16))
        v.copy_(torch.randn(2, 1, 64, device="cuda", generator=g).to(torch.bfloat16))
        graph.replay()
    torch.cuda.synchronize()
    for page, first, start in ((5, 100, 0), (7, 200, 32)):
        want = T.token_keys(torch.arange(first, first + PAGE), torch.arange(start, start + PAGE)).cuda()
        assert torch.equal(st.tkey[page * PAGE:(page + 1) * PAGE], want), page
        for layer in (0, 1):
            assert st.check(layer, torch.tensor([page], device="cuda"), tk=want.view(1, PAGE)).all(), (page, layer)
    assert st.unkeyed == 0


@cuda
def test_one_launch_check_of_every_layer_equals_the_per_layer_check():
    """The fused reader's check of every layer in one launch gives the codes of the per-layer check: a changed byte
    (tag), a missing offset (coverage), and another sequence's token keys (tag)."""
    pool = Pool(pages=40, layers=3, heads=2, dim=64)
    ref, fused = pair(pool)
    n_tok = 8 * PAGE + 5  # pages 1-8 full, page 9 partly written
    loc = torch.arange(PAGE, PAGE + n_tok, device="cuda")
    tk = T.token_keys(torch.arange(n_tok, device="cuda") + 11, torch.arange(n_tok, device="cuda"))
    for layer in range(3):
        for st in (ref, fused):
            st.update(layer, loc, tk)
    pages = torch.arange(1, 10, device="cuda")
    expected = torch.full((9,), 0xFFFF, dtype=torch.int32, device="cuda")
    expected[-1] = (1 << 5) - 1
    tkp = torch.zeros(9 * PAGE, dtype=torch.int64, device="cuda")
    tkp[:n_tok] = tk
    tkp = tkp.view(9, PAGE)
    assert not fused.check_codes(pages, expected, tkp).any() and not ref.check_codes(pages, expected, tkp).any()
    pool.k_buffer[1].view(torch.int16)[3 * PAGE + 2, 0, 5] ^= 1 << 9  # a changed byte in page 3, layer 1
    wrong = expected.clone()
    wrong[4] = 0x7FFF  # the reader expects page 5 to cover 15 offsets: coverage differs
    other = tkp.clone()
    other[6] = T.token_keys(torch.arange(16, device="cuda"), torch.arange(16, device="cuda"))  # page 7: other tokens
    a, b = fused.check_codes(pages, wrong, other), ref.check_codes(pages, wrong, other)
    assert torch.equal(a, b)
    assert int(a[1, 2]) == 2 and int(a[0, 2]) == 0 and (a[:, 4] == 1).all() and (a[:, 6] == 2).all()
