"""CPU tests of the tag and parity math, and of PoolState's running tags (no SGLang, no GPU)."""
import pytest
import torch

from kvmend import hook
from kvmend import tags as T

PAGE, HEADS, DIM = 16, 1, 64


def pages(n, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(n, PAGE, HEADS, DIM, generator=g).to(dtype)
    v = torch.randn(n, PAGE, HEADS, DIM, generator=g).to(dtype)
    return k, v


def keys(dtype=torch.bfloat16):
    return T.position_keys(PAGE, HEADS * DIM * torch.tensor([], dtype=dtype).element_size() // 4)


def test_running_tag_equals_page_tag():
    k, v, ks = *pages(5), keys()
    running = torch.zeros(5, dtype=torch.int64)
    for p in range(5):
        for t in range(PAGE):
            running[p] += T.token_shares(k[p, t:t + 1], v[p, t:t + 1], torch.tensor([t]), ks)[0]
    full = T.page_tags(k, v, torch.full((5,), 0xFFFF, dtype=torch.int32), ks)
    assert T.same(running, full).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_every_single_bit_flip_is_detected(dtype):
    k, v = pages(1, seed=1, dtype=dtype)
    ks, mask = keys(dtype), torch.tensor([0xFFFF], dtype=torch.int32)
    clean = T.page_tags(k, v, mask, ks)
    words = k.view(torch.int16).view(-1)
    for elem in range(0, words.numel(), 97):
        for bit in range(16):
            bad = words.clone()
            bad[elem] ^= torch.tensor(1 << bit, dtype=torch.int32).to(torch.int16)
            assert not T.same(T.page_tags(bad.view(dtype).view_as(k), v, mask, ks), clean).item()


def test_partial_page_counts_only_written_tokens():
    k, v, ks = *pages(1, seed=2), keys()
    mask = torch.tensor([(1 << 5) - 1], dtype=torch.int32)  # tokens 0-4 written
    written = T.page_tags(k, v, mask, ks)
    k2 = k.clone()
    k2[0, 9] = 7.0  # an unwritten token changes: not part of the tag
    assert T.same(T.page_tags(k2, v, mask, ks), written).all()
    k2[0, 3] = 7.0  # a written token changes: caught
    assert not T.same(T.page_tags(k2, v, mask, ks), written).any()


def test_xor_parity_rebuilds_bit_for_bit():
    ranks = [pages(1, seed=s)[0].view(torch.int16).view(-1) for s in range(4)]
    parity = ranks[0].clone()
    for r in ranks[1:]:
        parity ^= r
    rebuilt = parity.clone()
    for i, r in enumerate(ranks):
        if i != 2:
            rebuilt ^= r
    assert torch.equal(rebuilt, ranks[2])


class FakePool:
    def __init__(self, n_pages=6, layers=2):
        self.page_size, self.start_layer = PAGE, 0
        self.k_buffer = [torch.zeros(n_pages * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(layers)]
        self.v_buffer = [torch.zeros(n_pages * PAGE, HEADS, DIM, dtype=torch.bfloat16) for _ in range(layers)]


def write(st, pool, layer, loc, seed):
    g = torch.Generator().manual_seed(seed)
    pool.k_buffer[layer][loc] = torch.randn(len(loc), HEADS, DIM, generator=g).to(torch.bfloat16)
    pool.v_buffer[layer][loc] = torch.randn(len(loc), HEADS, DIM, generator=g).to(torch.bfloat16)
    st.update(layer, loc)


@pytest.mark.parametrize("graph", [False, True])
def test_pool_state_running_tags_match_a_full_check(graph, monkeypatch):
    monkeypatch.setattr(hook.PoolState, "_capturing", lambda self: graph)
    pool = FakePool()
    st = hook.PoolState(pool)
    if graph:  # decode: one token per request per step, each request on its own page
        for t in range(PAGE):
            write(st, pool, 0, torch.tensor([1 * PAGE + t, 3 * PAGE + t]), seed=t)
    else:  # prefill: whole pages at once, then a partial page
        write(st, pool, 0, torch.arange(1 * PAGE, 3 * PAGE), seed=1)
        write(st, pool, 0, torch.arange(3 * PAGE, 3 * PAGE + 5), seed=2)
    idx = (st.mask[0] != 0).nonzero().flatten()
    assert st.check(0, idx).all()
    pool.k_buffer[0].view(torch.int16)[1 * PAGE + 4, 0, 9] ^= 1 << 14  # one flipped bit on page 1
    assert st.check(0, idx).tolist() == [p != 1 for p in idx.tolist()]
    # page 1 reused from token 0 is clean again; in a captured graph the page refills one token per step
    if graph:
        for t in range(PAGE):
            write(st, pool, 0, torch.tensor([1 * PAGE + t]), seed=100 + t)
    else:
        write(st, pool, 0, torch.arange(1 * PAGE, 2 * PAGE), seed=9)
    assert st.check(0, idx).all()


@pytest.mark.parametrize("graph", [False, True])
def test_seal_tracks_full_pages_without_tags(graph, monkeypatch):
    """Parity alone (KVMEND_TAGS=0) tracks only which pages are full, from the last layer, and computes no tags."""
    monkeypatch.setattr(hook.PoolState, "_capturing", lambda self: graph)
    pool = FakePool()
    st = hook.PoolState(pool)
    last = st.layers - 1
    if graph:  # decode: one token per request per step
        for t in range(PAGE):
            st.seal(torch.tensor([1 * PAGE + t, 3 * PAGE + t]))
        for t in range(5):
            st.seal(torch.tensor([4 * PAGE + t]))
    else:  # prefill: two whole pages, then one whole page and a partial one
        st.seal(torch.arange(1 * PAGE, 3 * PAGE))
        st.seal(torch.arange(3 * PAGE, 4 * PAGE + 5))
    full = (st.mask[last] == hook.FULL).nonzero().flatten().tolist()
    assert full == ([1, 3] if graph else [1, 2, 3])
    assert int(st.mask[last][4]) == (1 << 5) - 1
    assert int(st.tags.abs().sum()) == 0 and int(st.mask[:last].abs().sum()) == 0
    st.encoded[1] = True
    st.seal(torch.tensor([1 * PAGE]))  # page 1 reused from token 0: not full, and its parity no longer counts
    assert not bool(st.encoded[1]) and int(st.mask[last][1]) == 1


def test_token_keys_bind_a_page_to_its_tokens():
    """A page's tag, keyed by its tokens' ids and positions, fails a reader that expects other tokens or other
    positions, although the page's bytes and tag agree with each other."""
    k, v, ks = *pages(1, seed=3), keys()
    mask = torch.tensor([0xFFFF], dtype=torch.int32)
    ids, pos = torch.arange(500, 516), torch.arange(1024, 1040)
    mine = T.token_keys(ids, pos).view(1, PAGE)
    tag = T.page_tags(k, v, mask, ks, mine)
    assert T.same(T.page_tags(k, v, mask, ks, mine), tag).all()
    other_ids = T.token_keys(ids + 1, pos).view(1, PAGE)
    other_pos = T.token_keys(ids, pos + 80).view(1, PAGE)
    assert not T.same(T.page_tags(k, v, mask, ks, other_ids), tag).any()
    assert not T.same(T.page_tags(k, v, mask, ks, other_pos), tag).any()
    running = torch.zeros(1, dtype=torch.int64)
    for t in range(PAGE):
        running += T.token_shares(k[0, t:t + 1], v[0, t:t + 1], torch.tensor([t]), ks, mine[0, t:t + 1])[0]
    assert T.same(running, tag).all()


class FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id


def test_store_keys_come_from_the_running_forward_batch(monkeypatch):
    """set_kv_buffer keys each stored token by the id and position of the forward batch that stores it (the batch
    RadixAttention.forward remembered); a store from elsewhere keeps the slots' keys and is counted as unkeyed."""
    import types

    class Pool(FakePool):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, layer_id_override=None):
            self.k_buffer[layer.layer_id][loc] = cache_k
            self.v_buffer[layer.layer_id][loc] = cache_v

    mod = types.SimpleNamespace(MHATokenToKVPool=Pool)
    hook.patch_memory_pool(mod)
    pool = Pool()
    st = pool._kvm
    loc = torch.arange(PAGE, 2 * PAGE)
    ids, pos = torch.arange(700, 716), torch.arange(32, 48)
    fb = types.SimpleNamespace(out_cache_loc=loc, input_ids=ids, positions=pos)
    monkeypatch.setattr(hook, "_FB", [fb])
    g = torch.Generator().manual_seed(5)
    kk = torch.randn(PAGE, HEADS, DIM, generator=g).to(torch.bfloat16)
    vv = torch.randn(PAGE, HEADS, DIM, generator=g).to(torch.bfloat16)
    pool.set_kv_buffer(FakeLayer(0), loc, kk, vv)
    want = T.token_keys(ids, pos)
    assert torch.equal(st.tkey[loc], want) and st.unkeyed == 0
    page = torch.tensor([1])
    assert st.check(0, page).all()
    assert st.check(0, page, tk=want.view(1, PAGE)).all()
    assert not st.check(0, page, tk=T.token_keys(ids, pos + 16).view(1, PAGE)).any()
    pool.set_kv_buffer(FakeLayer(1), loc.clone(), kk, vv)  # not the batch's own loc tensor: keys unknown
    assert st.unkeyed == 1 and st.check(1, page).all()


def test_free_moves_pages_to_a_new_generation():
    pool = FakePool()
    st = hook.PoolState(pool)
    before = st.gen.clone()
    st.drop(torch.tensor([2, 3]))
    assert (st.gen - before).tolist() == [0, 0, 1, 1, 0, 0]
    st.reset()
    assert (st.gen - before).tolist() == [1, 1, 2, 2, 1, 1]
