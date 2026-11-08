#!/usr/bin/env python3
"""How fast the fused Triton checker is (step 2 of decision 20), on one GPU, with synthetic KV of a model's shape.

For each shape (layers, KV heads per GPU, head size), it measures, after a warm-up, with the GPU synchronized before
each clock read, the median of --repeats runs:
  scan     check every page of a pool of --pool-gib of KV: the checker's bandwidth, as a share of the GPU's
  prefix   check one request's --prefix-tokens tokens in every layer: the check before reuse or before release
  store    the store-time tag update of one decode step of --batch tokens in every layer
each with the fused kernels and with the PyTorch operations of the proof of concept.

  uv run python bench_checker.py --out runs/bench-checker
"""
import argparse
import json
import os
import statistics
import time

import torch

SHAPES = {  # name: (layers, KV heads on one GPU, head size)
    "llama31_8b_tp8": (32, 1, 128),
    "qwen32b_tp4": (64, 2, 128),
}
PAGE = 16


def parse():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shapes", default=",".join(SHAPES))
    p.add_argument("--pool-gib", type=float, default=8.0, help="KV (K and V, every layer) of the scanned pool")
    p.add_argument("--prefix-tokens", type=int, default=32768)
    p.add_argument("--batch", type=int, default=64, help="tokens per decode step")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--hbm-gbps", type=float, default=1555.0, help="the GPU's memory bandwidth, for the share")
    p.add_argument("--out", required=True)
    return p.parse_args()


def timed(fn, repeats):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return statistics.median(out), out


def bench_shape(name, layers, heads, dim, a):
    from kvmend import hook
    from kvmend import tags as T

    piece = 2 * PAGE * heads * dim * 2  # bytes of K and V of one page in one layer
    pool_pages = max(int(a.pool_gib * 2**30 / (layers * piece)), a.prefix_tokens // PAGE + 2)
    pool = type("Pool", (), {})()
    pool.page_size, pool.start_layer = PAGE, 0
    pool.k_buffer = [torch.randn(pool_pages * PAGE, heads, dim, device="cuda").to(torch.bfloat16) for _ in range(layers)]
    pool.v_buffer = [torch.randn(pool_pages * PAGE, heads, dim, device="cuda").to(torch.bfloat16) for _ in range(layers)]
    states = {}
    for mode in ("fused", "pytorch"):
        st = hook.PoolState(pool)
        st.fused = False
        if mode == "fused":
            st.use_fused()
        states[mode] = st
    fused, ref = states["fused"], states["pytorch"]
    every = torch.arange(1, pool_pages, device="cuda")
    loc_all = torch.arange(PAGE, pool_pages * PAGE, device="cuda")
    keys = T.token_keys(loc_all % 50000, loc_all)
    for layer in range(layers):  # tags for the whole pool, through the fused store path
        fused.update(layer, loc_all, keys)
    for st in (ref,):
        st.tags.copy_(fused.tags)
        st.mask.copy_(fused.mask)
        st.tkey.copy_(fused.tkey)
    res = dict(shape=name, layers=layers, kv_heads_per_gpu=heads, head_dim=dim, piece_bytes=piece,
               pool_pages=pool_pages, pool_bytes=layers * pool_pages * piece)
    prefix = torch.arange(1, a.prefix_tokens // PAGE + 1, device="cuda")
    batch = torch.arange(PAGE, PAGE + a.batch * PAGE, PAGE, device="cuda") + 5  # one token on each of --batch pages
    btk = T.token_keys(batch, batch)
    for mode, st in states.items():
        ok = all(bool(st.check(l, every[:4096]).all()) for l in range(layers))

        def scan(st=st):
            for l in range(layers):
                for chunk in every.split(16384):
                    st.check(l, chunk)

        def check_prefix(st=st):
            for l in range(layers):
                st.check(l, prefix)

        def store(st=st):
            for l in range(layers):
                st.update(l, batch, btk)

        full = torch.full((prefix.numel(),), 0xFFFF, dtype=torch.int32, device="cuda")
        ptk = st.tkey.view(st.pages, PAGE)[prefix]

        def reader(st=st):  # the reader's check of the prefix in every layer (one launch when fused)
            st.check_codes(prefix, full, ptk)

        reps = a.repeats if mode == "fused" else max(3, a.repeats // 5)
        t_scan, _ = timed(scan, max(3, reps // 4))
        t_prefix, _ = timed(check_prefix, reps)
        t_reader, _ = timed(reader, reps)
        t_store, _ = timed(store, reps)
        scanned = layers * every.numel() * piece
        res[mode] = dict(tags_pass=ok, scan_s=t_scan, scan_gb_per_s=scanned / t_scan / 1e9,
                         scan_share_of_hbm=scanned / t_scan / 1e9 / a.hbm_gbps,
                         prefix_s=t_prefix, prefix_tokens=a.prefix_tokens,
                         prefix_bytes=layers * prefix.numel() * piece, reader_check_s=t_reader,
                         store_s_per_step=t_store, store_tokens_per_step=a.batch, repeats=reps)
    del pool.k_buffer, pool.v_buffer
    torch.cuda.empty_cache()
    return res


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    from kvmend.meta import run_meta

    meta = run_meta()
    meta["device"] = torch.cuda.get_device_name(0)
    rows = [bench_shape(s, *SHAPES[s], a) for s in a.shapes.split(",")]
    with open(os.path.join(a.out, "bench_checker.json"), "w") as fh:
        json.dump(dict(meta=meta, args=vars(a), rows=rows), fh, indent=1)
    print(json.dumps(meta))
    print("| Shape | Mode | Tags pass | Scan GB/s (share of HBM) | 32K prefix check | Reader's check, all layers | Store per decode step |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        for mode in ("fused", "pytorch"):
            m = r[mode]
            print(f"| {r['shape']} | {mode} | {m['tags_pass']} | {m['scan_gb_per_s']:.0f} ({100 * m['scan_share_of_hbm']:.1f}%) "
                  f"| {1e3 * m['prefix_s']:.2f} ms | {1e3 * m['reader_check_s']:.2f} ms | {1e3 * m['store_s_per_step']:.3f} ms |")


if __name__ == "__main__":
    main()
