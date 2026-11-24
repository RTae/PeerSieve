"""CLI runner for Phase 1 TrustGraph-DFL CIFAR-10 experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import threading
import time

import numpy as np
import torch

from experiments.phase1.suites import SUITES, build_suite_runs
from src.aggregators import AggregatorConfig, aggregate_trustgraph
from src.models import (
    build_model,
    compute_validation_gradient,
    model_to_vector,
    train_local_epoch,
    vector_to_model,
)
from src.simulator import ExperimentConfig, prepare_dataset, run_experiment, seed_everything

__all__ = [
    "SUITES",
    "build_suite_runs",
    "run_transfer_benchmark",
    "run_multinode_benchmark",
    "parse_args",
    "main",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run single, suite, TCP transfer, or multi-node benchmarks.")
    p.add_argument("--mode", choices=["single", "suite", "transfer_bench", "multinode_bench"], default="single")
    p.add_argument("--suite", choices=sorted(SUITES), default="smoke")
    p.add_argument("--dataset", choices=["cifar10", "synthetic"], default="cifar10")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-nodes", type=int, default=20)
    p.add_argument("--samples-per-node", type=int, default=500)
    p.add_argument("--alpha-dir", type=float, default=0.5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-split", choices=["local", "iid"], default="local")
    p.add_argument(
        "--model",
        default="cnn2x2",
        choices=["cnn2x2", "cnn2x2_gn", "resnet18_pretrained", "resnet18_scratch"],
    )
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--grad-clip-norm", type=float, default=0.0)
    p.add_argument("--val-grad-weight-decay", action="store_true")
    p.add_argument("--byz-local-epochs", type=int, default=0)
    p.add_argument("--topology", choices=["ring", "random", "small_world"], default="random")
    p.add_argument("--er-p", type=float, default=0.3)
    p.add_argument("--ws-k", type=int, default=4)
    p.add_argument("--ws-p", type=float, default=0.3)
    p.add_argument("--byzantine-frac", type=float, default=0.3)
    p.add_argument(
        "--attack",
        choices=["none", "label_flip", "label_flip_all", "model_poison", "random_noise"],
        default="label_flip",
    )
    p.add_argument("--poison-lambda", type=float, default=10.0)
    p.add_argument("--noise-sigma", type=float, default=0.5)
    p.add_argument(
        "--aggregator",
        choices=["trustgraph"],
        default="trustgraph",
    )
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--val-grad-sign", choices=["descent_neg", "paper_pos"], default="descent_neg")
    p.add_argument("--no-relu", action="store_true")
    p.add_argument("--trustgraph-metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--trustgraph-score-threshold", type=float, default=0.0)
    p.add_argument("--trustgraph-clip-norm", action="store_true")
    p.add_argument("--trustgraph-clip-mode", choices=["update", "model"], default="update")
    p.add_argument("--trustgraph-clip-factor", type=float, default=2.0)
    p.add_argument("--trustgraph-delta-mode", choices=["sender", "receiver"], default="sender")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true", help="Skip runs that already have summary.json.")
    p.add_argument("--output-root", default="outputs/phase1")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--bench-role", choices=["auto", "server", "client"], default="auto")
    p.add_argument("--bench-host", default="127.0.0.1")
    p.add_argument("--bench-port", type=int, default=29501)
    p.add_argument("--bench-payload-bytes", type=int, default=11181642 * 4)
    p.add_argument("--bench-repeats", type=int, default=25)
    p.add_argument("--bench-burst-repeats", type=int, default=10)
    p.add_argument("--bench-out", default=None)
    p.add_argument("--bench-iface", default="enp68s0f0")
    p.add_argument("--bench-e2e-rounds", type=int, default=10)
    p.add_argument("--bench-deg6-rounds", type=int, default=5)
    return p.parse_args()


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        download_data=not args.no_download,
        n_nodes=args.n_nodes,
        samples_per_node=args.samples_per_node,
        alpha_dir=args.alpha_dir,
        val_frac=args.val_frac,
        val_split=args.val_split,
        model_name=args.model,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        augment=args.augment,
        grad_clip_norm=args.grad_clip_norm,
        val_grad_weight_decay=args.val_grad_weight_decay,
        byz_local_epochs=args.byz_local_epochs,
        topology=args.topology,
        er_p=args.er_p,
        ws_k=args.ws_k,
        ws_p=args.ws_p,
        byzantine_frac=args.byzantine_frac,
        attack=args.attack,
        poison_lambda=args.poison_lambda,
        noise_sigma=args.noise_sigma,
        aggregator=args.aggregator,
        alpha=args.alpha,
        val_grad_sign=args.val_grad_sign,
        trustgraph_use_relu=not args.no_relu,
        trustgraph_metric=args.trustgraph_metric,
        trustgraph_score_threshold=args.trustgraph_score_threshold,
        trustgraph_clip_norm=args.trustgraph_clip_norm,
        trustgraph_clip_mode=args.trustgraph_clip_mode,
        trustgraph_clip_factor=args.trustgraph_clip_factor,
        trustgraph_delta_mode=args.trustgraph_delta_mode,
        device=args.device,
        eval_every=args.eval_every,
        verbose=not args.quiet,
    )


def _run_single(args: argparse.Namespace, base_cfg: ExperimentConfig, out_root: Path) -> None:
    tag = (
        f"{args.dataset}_{args.topology}_{args.attack}_byz{int(round(args.byzantine_frac * 100))}"
        f"_{args.aggregator}"
    )
    if args.aggregator == "trustgraph":
        tag += f"_{args.val_grad_sign}"

    for seed in args.seeds:
        out_dir = out_root / "single" / tag / f"seed_{seed}"
        if args.skip_existing and (out_dir / "summary.json").exists():
            print(f"=== Skipping completed {tag} (seed={seed}) ===", flush=True)
            continue
        cfg = replace(base_cfg, seed=seed, output_dir=str(out_dir))
        print(f"=== Running {tag} (seed={seed}) -> {out_dir} ===", flush=True)
        res = run_experiment(cfg)
        s = res["summary"]
        print(
            f"DONE {tag} seed={seed}: final_acc={100.0 * float(s['final_benign_test_acc_mean']):.2f}% "
            f"best_acc={100.0 * float(s['best_benign_test_acc_mean']):.2f}% "
            f"fpr_cum={100.0 * float(s['fpr_cumulative']):.2f}% "
            f"wall={float(s['total_wall_sec']):.1f}s",
            flush=True,
        )


def _run_suite(args: argparse.Namespace, base_cfg: ExperimentConfig, out_root: Path) -> None:
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"Invalid shard {args.shard_index}/{args.num_shards}")

    suite_root = out_root / args.suite
    suite_root.mkdir(parents=True, exist_ok=True)

    pending_jobs: list[tuple[int, str, ExperimentConfig]] = []
    skipped = 0
    for seed in args.seeds:
        seed_base = replace(base_cfg, seed=seed)
        for run_name, cfg in build_suite_runs(args.suite, seed_base):
            out_dir = suite_root / run_name / f"seed_{seed}"
            if args.skip_existing and (out_dir / "summary.json").exists():
                skipped += 1
                continue
            pending_jobs.append((seed, run_name, cfg))

    shard_jobs = [job for idx, job in enumerate(pending_jobs) if idx % args.num_shards == args.shard_index]
    print(
        f"=== Suite '{args.suite}': shard {args.shard_index}/{args.num_shards} "
        f"running {len(shard_jobs)}/{len(pending_jobs)} pending jobs (skipped {skipped} completed) ===",
        flush=True,
    )

    summaries: list[dict[str, object]] = []
    data_cache: dict[tuple[int, str, float, int, int, float], object] = {}
    for job_idx, (seed, run_name, cfg) in enumerate(shard_jobs, start=1):
        cache_key = (seed, cfg.val_split, cfg.alpha_dir, cfg.n_nodes, cfg.samples_per_node, cfg.val_frac)
        if cache_key not in data_cache:
            data_cache[cache_key] = prepare_dataset(cfg)
        out_dir = suite_root / run_name / f"seed_{seed}"
        run_cfg = replace(cfg, output_dir=str(out_dir))
        print(
            f"\n=== [{args.suite} shard {args.shard_index} ({job_idx}/{len(shard_jobs)})] "
            f"{run_name} (seed={seed}) ===",
            flush=True,
        )
        res = run_experiment(run_cfg, preloaded_data=data_cache[cache_key])
        s = dict(res["summary"])
        s["run_name"] = run_name
        summaries.append(s)

    summary_name = (
        "suite_summaries.json"
        if args.num_shards == 1
        else f"suite_summaries_shard{args.shard_index}.json"
    )
    with open(suite_root / summary_name, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n=== Suite '{args.suite}' shard {args.shard_index}/{args.num_shards} summary ===", flush=True)
    print(
        f"{'Run':44s} {'Seed':>5s} {'FinalAcc':>9s} {'ConsAcc':>9s} "
        f"{'FPR(cum)':>9s} {'ByzAcc':>8s} {'CosBen':>8s} {'CosByz':>8s}",
        flush=True,
    )
    for s in summaries:
        cb = f"{float(s['mean_benign_score']):+.3f}" if s["mean_benign_score"] is not None else "   n/a"
        cm = f"{float(s['mean_byz_score']):+.3f}" if s["mean_byz_score"] is not None else "   n/a"
        print(
            f"{str(s['run_name']):44s} {int(s['config']['seed']):5d} "
            f"{100.0 * float(s['final_benign_test_acc_mean']):8.2f}% "
            f"{100.0 * float(s['final_consensus_test_acc']):8.2f}% "
            f"{100.0 * float(s['fpr_cumulative']):8.2f}% "
            f"{100.0 * float(s['byz_accept_cumulative']):7.2f}% "
            f"{cb:>8s} {cm:>8s}",
            flush=True,
        )


def _recv_exact(conn: socket.socket, view: memoryview) -> None:
    received = 0
    total = len(view)
    while received < total:
        n = conn.recv_into(view[received:])
        if n == 0:
            raise ConnectionError(f"Socket closed after {received}/{total} bytes.")
        received += n


def _probe_link_info(peer_host: str) -> dict[str, object]:
    """Query route interface and Linux sysfs link speed toward peer_host when available."""
    info: dict[str, object] = {"hostname": socket.gethostname(), "peer_host": peer_host}
    try:
        peer_ip = socket.gethostbyname(peer_host)
        info["peer_ip"] = peer_ip
    except OSError:
        return info
    try:
        out = subprocess.check_output(["ip", "route", "get", peer_ip], text=True, stderr=subprocess.DEVNULL)
        info["ip_route"] = out.strip().splitlines()[0]
        tokens = info["ip_route"].split()
        if "dev" in tokens:
            dev = tokens[tokens.index("dev") + 1]
            info["interface"] = dev
            speed_file = Path(f"/sys/class/net/{dev}/speed")
            if speed_file.exists():
                info["link_speed_mbps"] = speed_file.read_text(encoding="utf-8").strip()
            mtu_file = Path(f"/sys/class/net/{dev}/mtu")
            if mtu_file.exists():
                info["mtu"] = mtu_file.read_text(encoding="utf-8").strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def _serve_transfer_workloads(
    conn: socket.socket,
    workloads: list[tuple[str, int, int, int]],
) -> None:
    max_bytes = max(n_bytes for _, n_bytes, _, _ in workloads)
    buf = bytearray(max_bytes)
    view = memoryview(buf)
    for _, n_bytes, n_warm, n_rep in workloads:
        slice_view = view[:n_bytes]
        for _ in range(n_warm + n_rep):
            conn.sendall(b"G")
            _recv_exact(conn, slice_view)
            conn.sendall(b"D")


def _time_transfer_workloads(
    sock: socket.socket,
    host: str,
    workloads: list[tuple[str, int, int, int]],
) -> dict[str, dict[str, object]]:
    max_bytes = max(n_bytes for _, n_bytes, _, _ in workloads)
    buf = bytearray(max_bytes)
    view = memoryview(buf)
    go_buf = bytearray(1)
    done_buf = bytearray(1)
    results: dict[str, dict[str, object]] = {}
    for name, n_bytes, n_warm, n_rep in workloads:
        slice_view = view[:n_bytes]
        times: list[float] = []
        for step in range(n_warm + n_rep):
            _recv_exact(sock, memoryview(go_buf))
            t0 = time.perf_counter()
            sock.sendall(slice_view)
            _recv_exact(sock, memoryview(done_buf))
            dt = time.perf_counter() - t0
            if step >= n_warm:
                times.append(dt)
        med = float(np.median(times))
        mn = float(np.min(times))
        mx = float(np.max(times))
        mbps = (n_bytes * 8.0) / (med * 1e6)
        results[name] = {
            "bytes": n_bytes,
            "mb": round(n_bytes / 1e6, 3),
            "repeats": len(times),
            "warmups": n_warm,
            "median_sec": round(med, 6),
            "min_sec": round(mn, 6),
            "max_sec": round(mx, 6),
            "goodput_mbps": round(mbps, 2),
            "samples_sec": [round(t, 6) for t in times],
        }
        print(
            f"[client {socket.gethostname()} -> {host}] {name:20s} "
            f"({n_bytes / 1e6:6.2f} MB): median={med:.4f}s "
            f"(min={mn:.4f}s, max={mx:.4f}s, n={len(times)}) -> {mbps:.1f} Mb/s",
            flush=True,
        )
    return results


def run_transfer_benchmark(
    role: str,
    host: str,
    port: int,
    payload_bytes: int = 11181642 * 4,
    single_repeats: int = 25,
    burst_repeats: int = 10,
    warmups: int = 3,
    useful_bytes: int = 119100000,
    wasted_bytes: int = 149260000,
    out_path: str | Path | None = None,
    connected_sock: socket.socket | None = None,
) -> dict[str, object] | None:
    """Measure peer-to-peer TCP transfer durations for ResNet-18 FP32 payload sizes.

    The server receives payloads and replies with a 1-byte ack after reading the final byte.
    The client times full end-to-end delivery from send start to ack receipt.
    """
    if role == "auto":
        proc_id = int(os.environ.get("SLURM_PROCID", "0"))
        role = "server" if proc_id == 0 else "client"

    workloads: list[tuple[str, int, int, int]] = [
        ("single_update", payload_bytes, warmups, single_repeats),
        ("round_6_neighbors", payload_bytes * 6, min(warmups, 2), burst_repeats),
        ("model_poison_useful", useful_bytes, min(warmups, 2), burst_repeats),
        ("model_poison_wasted", wasted_bytes, min(warmups, 2), burst_repeats),
    ]

    if role == "server":
        if connected_sock is not None:
            _serve_transfer_workloads(connected_sock, workloads)
            return None
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(90.0)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        print(f"[server {socket.gethostname()}] listening on 0.0.0.0:{port}", flush=True)
        conn, addr = srv.accept()
        conn.settimeout(90.0)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[server {socket.gethostname()}] accepted connection from {addr[0]}:{addr[1]}", flush=True)
        try:
            _serve_transfer_workloads(conn, workloads)
        finally:
            conn.close()
            srv.close()
        return None

    if connected_sock is not None:
        results = _time_transfer_workloads(connected_sock, host, workloads)
    else:
        print(f"[client {socket.gethostname()}] connecting to {host}:{port}", flush=True)
        sock: socket.socket | None = None
        for attempt in range(180):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(90.0)
            try:
                s.connect((host, port))
                sock = s
                break
            except OSError:
                s.close()
                if attempt == 179:
                    raise
                time.sleep(0.5)
        assert sock is not None
        try:
            results = _time_transfer_workloads(sock, host, workloads)
        finally:
            sock.close()

    report: dict[str, object] = {
        "client_link": _probe_link_info(host),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "port": port,
        "workloads": results,
    }
    if out_path is not None:
        dst = Path(out_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _get_interface_info(iface: str) -> dict[str, str]:
    """Return IPv4 address, link speed, and MTU for a Linux network interface."""
    info = {"hostname": socket.gethostname(), "interface": iface, "ip": "127.0.0.1", "link_speed_mbps": "unknown", "mtu": "unknown"}
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show", "dev", iface], text=True, stderr=subprocess.DEVNULL)
        tokens = out.strip().split()
        if "inet" in tokens:
            info["ip"] = tokens[tokens.index("inet") + 1].split("/")[0]
    except (OSError, subprocess.SubprocessError):
        pass
    for key, fname in (("link_speed_mbps", "speed"), ("mtu", "mtu")):
        p = Path(f"/sys/class/net/{iface}/{fname}")
        if p.exists():
            info[key] = p.read_text(encoding="utf-8").strip()
    return info


def _build_tcp_mesh(
    rank: int,
    peers: list[dict[str, object]],
) -> tuple[dict[int, socket.socket], dict[int, socket.socket]]:
    """Open a full-duplex persistent TCP mesh where each ordered pair has a dedicated socket."""
    world_size = len(peers)
    my_info = peers[rank]
    recv_conns: dict[int, socket.socket] = {}
    accept_errors: list[BaseException] = []

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(90.0)
    srv.bind((str(my_info["ip"]), int(my_info["port"])))
    srv.listen(world_size)

    def _accept_all() -> None:
        try:
            for _ in range(world_size - 1):
                conn, _ = srv.accept()
                conn.settimeout(90.0)
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                hdr = bytearray(4)
                _recv_exact(conn, memoryview(hdr))
                src_rank = struct.unpack("!I", hdr)[0]
                recv_conns[src_rank] = conn
        except BaseException as exc:
            accept_errors.append(exc)
        finally:
            srv.close()

    acc_thread = threading.Thread(target=_accept_all, daemon=True)
    acc_thread.start()

    send_conns: dict[int, socket.socket] = {}
    hdr_me = struct.pack("!I", rank)
    for dst in range(world_size):
        if dst == rank:
            continue
        dst_ip = str(peers[dst]["ip"])
        dst_port = int(peers[dst]["port"])
        sock: socket.socket | None = None
        for attempt in range(180):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(90.0)
            try:
                s.connect((dst_ip, dst_port))
                s.sendall(hdr_me)
                sock = s
                break
            except OSError:
                s.close()
                if attempt == 179:
                    raise
                time.sleep(0.5)
        assert sock is not None
        send_conns[dst] = sock

    acc_thread.join(timeout=95.0)
    if accept_errors:
        raise accept_errors[0]
    if len(recv_conns) != world_size - 1:
        raise RuntimeError(f"Rank {rank} accepted {len(recv_conns)}/{world_size - 1} peer connections.")
    return send_conns, recv_conns


def _mesh_barrier(
    rank: int,
    world_size: int,
    send_conns: dict[int, socket.socket],
    recv_conns: dict[int, socket.socket],
) -> None:
    """Synchronize all ranks over the persistent TCP mesh."""
    buf = bytearray(1)
    if rank == 0:
        for src in range(1, world_size):
            _recv_exact(recv_conns[src], memoryview(buf))
        for dst in range(1, world_size):
            send_conns[dst].sendall(b"G")
    else:
        send_conns[0].sendall(b"B")
        _recv_exact(recv_conns[0], memoryview(buf))


def _exchange_all_to_all(
    send_conns: dict[int, socket.socket],
    recv_conns: dict[int, socket.socket],
    send_view: memoryview,
    recv_views: dict[int, memoryview],
) -> None:
    """Send send_view to all peers while concurrently receiving one payload from each peer."""
    errors: list[BaseException] = []

    def _do_send(conn: socket.socket) -> None:
        try:
            conn.sendall(send_view)
        except BaseException as exc:
            errors.append(exc)

    def _do_recv(conn: socket.socket, view: memoryview) -> None:
        try:
            _recv_exact(conn, view)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_do_send, args=(c,)) for c in send_conns.values()]
    threads.extend(threading.Thread(target=_do_recv, args=(recv_conns[r], recv_views[r])) for r in recv_conns)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def _fanin_to_rank0(
    rank: int,
    send_conns: dict[int, socket.socket],
    recv_conns: dict[int, socket.socket],
    send_view: memoryview,
    recv_views: dict[int, memoryview],
) -> None:
    """All non-zero ranks send send_view concurrently to rank 0."""
    if rank != 0:
        send_conns[0].sendall(send_view)
        return
    errors: list[BaseException] = []

    def _do_recv(conn: socket.socket, view: memoryview) -> None:
        try:
            _recv_exact(conn, view)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_do_recv, args=(recv_conns[r], recv_views[r])) for r in recv_conns]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def _summarize_samples(samples: list[float]) -> dict[str, object]:
    return {
        "repeats": len(samples),
        "median_sec": round(float(np.median(samples)), 6),
        "min_sec": round(float(np.min(samples)), 6),
        "max_sec": round(float(np.max(samples)), 6),
    }


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_multinode_benchmark(
    out_path: str | Path,
    iface: str = "enp68s0f0",
    base_port: int = 29600,
    data_dir: str = "data",
    e2e_rounds: int = 10,
    deg6_rounds: int = 5,
    seed: int = 42,
) -> dict[str, object] | None:
    """Run a 4-node Ethernet contention microbenchmark and end-to-end GPU TrustGraph-DFL rounds."""
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "4"))
    out_file = Path(out_path)
    coord_dir = out_file.parent / f"coord_{os.environ.get('SLURM_JOB_ID', 'local')}"
    coord_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    net_info = _get_interface_info(iface)
    my_record: dict[str, object] = {
        "rank": rank,
        "port": base_port + rank,
        "gpu_name": gpu_name,
        **net_info,
    }
    tmp_p = coord_dir / f"rank_{rank}.tmp"
    reg_p = coord_dir / f"rank_{rank}.json"
    tmp_p.write_text(json.dumps(my_record), encoding="utf-8")
    tmp_p.replace(reg_p)

    peers: list[dict[str, object]] = []
    for _ in range(180):
        if all((coord_dir / f"rank_{r}.json").exists() for r in range(world_size)):
            peers = [json.loads((coord_dir / f"rank_{r}.json").read_text(encoding="utf-8")) for r in range(world_size)]
            break
        time.sleep(0.5)
    if len(peers) != world_size:
        raise RuntimeError(f"Rank {rank} timed out waiting for {world_size} peer registrations.")

    send_conns, recv_conns = _build_tcp_mesh(rank, peers)
    _mesh_barrier(rank, world_size, send_conns, recv_conns)

    try:
        # Part 1: Multi-node Ethernet contention microbenchmarks on ResNet-18 FP32 payloads.
        payload_bytes = 11181642 * 4
        deg6_mult = max(1, 6 // max(world_size - 1, 1))
        max_bytes = payload_bytes * deg6_mult
        send_buf = bytearray(max_bytes)
        recv_bufs = {r: bytearray(max_bytes) for r in recv_conns}
        net_micro: dict[str, dict[str, object]] = {}

        for name, mode_kind, mult, n_warm, n_rep in (
            ("fanin_1x", "fanin", 1, 2, 8),
            ("fanin_deg6_equiv", "fanin", deg6_mult, 1, 5),
            ("all_to_all_1x", "all2all", 1, 2, 8),
            ("all_to_all_deg6_equiv", "all2all", deg6_mult, 1, 5),
        ):
            n_bytes = payload_bytes * mult
            s_view = memoryview(send_buf)[:n_bytes]
            r_views = {r: memoryview(recv_bufs[r])[:n_bytes] for r in recv_conns}
            times: list[float] = []
            for step in range(n_warm + n_rep):
                _mesh_barrier(rank, world_size, send_conns, recv_conns)
                t0 = time.perf_counter()
                if mode_kind == "fanin":
                    _fanin_to_rank0(rank, send_conns, recv_conns, s_view, r_views)
                else:
                    _exchange_all_to_all(send_conns, recv_conns, s_view, r_views)
                dt = time.perf_counter() - t0
                if step >= n_warm:
                    times.append(dt)
            rx_total_bytes = n_bytes * (world_size - 1) if (mode_kind == "all2all" or rank == 0) else 0
            med = float(np.median(times))
            goodput_mbps = (rx_total_bytes * 8.0) / (med * 1e6) if rx_total_bytes > 0 else 0.0
            net_micro[name] = {
                "rx_bytes": rx_total_bytes,
                "rx_mb": round(rx_total_bytes / 1e6, 3),
                "goodput_mbps": round(goodput_mbps, 2),
                **_summarize_samples(times),
            }
            if rank == 0:
                print(
                    f"[rank 0 {net_info['hostname']}] {name:24s} ({rx_total_bytes / 1e6:6.2f} MB rx): "
                    f"median={med:.4f}s (min={min(times):.4f}s, max={max(times):.4f}s, n={len(times)}) "
                    f"-> {goodput_mbps:.1f} Mb/s",
                    flush=True,
                )

        # Part 2: End-to-end distributed GPU TrustGraph-DFL rounds on real CIFAR-10 shards.
        seed_everything(seed)
        cfg = ExperimentConfig(
            dataset="cifar10",
            data_dir=data_dir,
            download_data=False,
            n_nodes=20,
            samples_per_node=2500,
            alpha_dir=0.5,
            val_frac=0.1,
            val_split="iid",
            model_name="resnet18_pretrained",
            rounds=max(e2e_rounds, deg6_rounds) + 2,
            lr=0.005,
            lr_schedule="cosine",
            augment=True,
            grad_clip_norm=10.0,
            val_grad_sign="descent_neg",
            trustgraph_clip_norm=True,
            trustgraph_clip_mode="model",
            trustgraph_clip_factor=2.0,
            trustgraph_score_threshold=0.003,
            trustgraph_delta_mode="receiver",
            seed=seed,
        )
        part = prepare_dataset(cfg)
        shard = part.nodes[rank]
        x_tr = shard.x_train.to(device)
        y_tr = shard.y_train.to(device)
        x_va = shard.x_val.to(device)
        y_va = shard.y_val.to(device)

        model = build_model("resnet18_pretrained", num_classes=10).to(device)
        init_vec = model_to_vector(model)
        param_count = int(init_vec.numel())
        assert param_count * 4 == payload_bytes
        agg_cfg = cfg.aggregator_config()
        is_byz = rank == world_size - 1
        train_gen = torch.Generator(device="cpu").manual_seed(seed + 100 + rank)

        def _run_e2e_suite(mult: int, n_rounds: int, suite_tag: str) -> dict[str, object]:
            cur_vec = init_vec.clone()
            send_host = torch.empty(param_count * mult, dtype=torch.float32, pin_memory=(device.type == "cuda"))
            recv_hosts = {
                r: torch.empty(param_count * mult, dtype=torch.float32, pin_memory=(device.type == "cuda"))
                for r in recv_conns
            }
            send_view = memoryview(send_host.view(torch.uint8).numpy())
            recv_views = {r: memoryview(recv_hosts[r].view(torch.uint8).numpy()) for r in recv_conns}

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            val_t, train_t, d2h_t, net_t, h2d_t, agg_t, tot_t = [], [], [], [], [], [], []
            rej_total = 0
            pairs_total = 0
            byz_rej_total = 0
            byz_pairs_total = 0

            for step in range(1 + n_rounds):
                _mesh_barrier(rank, world_size, send_conns, recv_conns)
                _sync_cuda(device)
                t_round0 = time.perf_counter()

                prev_vec = cur_vec.clone()
                vector_to_model(prev_vec, model)

                _sync_cuda(device)
                t0 = time.perf_counter()
                val_grad = compute_validation_gradient(model, x_va, y_va, batch_size=cfg.batch_size)
                _sync_cuda(device)
                t_val = time.perf_counter() - t0

                t0 = time.perf_counter()
                train_local_epoch(
                    model,
                    x_tr,
                    y_tr,
                    lr=cfg.lr,
                    momentum=cfg.momentum,
                    weight_decay=cfg.weight_decay,
                    batch_size=cfg.batch_size,
                    epochs=1,
                    augment=cfg.augment,
                    grad_clip_norm=cfg.grad_clip_norm,
                    generator=train_gen,
                )
                post_vec = model_to_vector(model)
                upd_vec = post_vec - prev_vec
                if is_byz:
                    # Send negated scaled update as a real model-poisoning payload.
                    post_vec = prev_vec - 10.0 * upd_vec
                _sync_cuda(device)
                t_tr = time.perf_counter() - t0

                t0 = time.perf_counter()
                for m_idx in range(mult):
                    send_host[m_idx * param_count : (m_idx + 1) * param_count].copy_(post_vec, non_blocking=False)
                _sync_cuda(device)
                t_d2h = time.perf_counter() - t0

                _mesh_barrier(rank, world_size, send_conns, recv_conns)
                t0 = time.perf_counter()
                _exchange_all_to_all(send_conns, recv_conns, send_view, recv_views)
                t_net = time.perf_counter() - t0

                _sync_cuda(device)
                t0 = time.perf_counter()
                nbr_ids: list[int] = []
                nbr_posts: dict[int, torch.Tensor] = {}
                nbr_updates: dict[int, torch.Tensor] = {}
                for src_r, h_ten in recv_hosts.items():
                    gpu_all = h_ten.to(device, non_blocking=False)
                    for m_idx in range(mult):
                        nid = src_r * 10 + m_idx
                        p_j = gpu_all[m_idx * param_count : (m_idx + 1) * param_count]
                        nbr_ids.append(nid)
                        nbr_posts[nid] = p_j
                        nbr_updates[nid] = p_j - prev_vec
                _sync_cuda(device)
                t_h2d = time.perf_counter() - t0

                t0 = time.perf_counter()
                if not is_byz:
                    res = aggregate_trustgraph(
                        self_post=post_vec,
                        val_grad=val_grad,
                        neighbor_ids=nbr_ids,
                        neighbor_posts=nbr_posts,
                        neighbor_updates=nbr_updates,
                        cfg=agg_cfg,
                        self_prev=prev_vec,
                        self_update=upd_vec,
                    )
                    cur_vec = res.new_model
                else:
                    cur_vec = prev_vec + upd_vec
                    res = None
                _sync_cuda(device)
                t_agg = time.perf_counter() - t0
                t_tot = time.perf_counter() - t_round0

                if step >= 1:
                    val_t.append(t_val)
                    train_t.append(t_tr)
                    d2h_t.append(t_d2h)
                    net_t.append(t_net)
                    h2d_t.append(t_h2d)
                    agg_t.append(t_agg)
                    tot_t.append(t_tot)
                    if res is not None:
                        for nid in nbr_ids:
                            pairs_total += 1
                            is_rej = res.neighbor_zero_weight_frac[nid] >= 0.5
                            if is_rej:
                                rej_total += 1
                            if nid // 10 == world_size - 1:
                                byz_pairs_total += 1
                                if is_rej:
                                    byz_rej_total += 1

            peak_mib = (
                round(float(torch.cuda.max_memory_allocated(device)) / float(1024 ** 2), 2)
                if device.type == "cuda"
                else None
            )
            rx_mb = round((payload_bytes * mult * (world_size - 1)) / 1e6, 2)
            suite_res = {
                "degree": (world_size - 1) * mult,
                "rx_mb_per_round": rx_mb,
                "recv_buf_mib": round((payload_bytes * mult * (world_size - 1)) / float(1024 ** 2), 2),
                "peak_gpu_alloc_mib": peak_mib,
                "wasted_transfer_pct": round(100.0 * rej_total / max(pairs_total, 1), 2) if not is_byz else None,
                "byz_blocked_pct": round(100.0 * byz_rej_total / max(byz_pairs_total, 1), 2) if not is_byz else None,
                "val_grad_sec": _summarize_samples(val_t),
                "train_sec": _summarize_samples(train_t),
                "d2h_sec": _summarize_samples(d2h_t),
                "net_exchange_sec": _summarize_samples(net_t),
                "h2d_sec": _summarize_samples(h2d_t),
                "agg_sec": _summarize_samples(agg_t),
                "total_round_sec": _summarize_samples(tot_t),
            }
            if rank == 0:
                print(
                    f"[rank 0 {suite_tag} deg={suite_res['degree']} ({rx_mb:.1f} MB)] "
                    f"val={suite_res['val_grad_sec']['median_sec']:.3f}s "
                    f"train={suite_res['train_sec']['median_sec']:.3f}s "
                    f"d2h={suite_res['d2h_sec']['median_sec']:.4f}s "
                    f"net={suite_res['net_exchange_sec']['median_sec']:.3f}s "
                    f"h2d={suite_res['h2d_sec']['median_sec']:.4f}s "
                    f"agg={suite_res['agg_sec']['median_sec']:.4f}s "
                    f"total={suite_res['total_round_sec']['median_sec']:.3f}s "
                    f"wasted={suite_res['wasted_transfer_pct']}% peak_gpu={peak_mib} MiB",
                    flush=True,
                )
            return suite_res

        e2e_base = _run_e2e_suite(mult=1, n_rounds=e2e_rounds, suite_tag=f"e2e_deg{world_size - 1}")
        e2e_deg6 = _run_e2e_suite(mult=deg6_mult, n_rounds=deg6_rounds, suite_tag="e2e_deg6_equiv")

        rank_out = {
            "node": my_record,
            "is_byzantine": is_byz,
            "net_micro": net_micro,
            "e2e_base": e2e_base,
            "e2e_deg6_equiv": e2e_deg6,
        }
        r_tmp = coord_dir / f"result_{rank}.tmp"
        r_fin = coord_dir / f"result_{rank}.json"
        r_tmp.write_text(json.dumps(rank_out, indent=2), encoding="utf-8")
        r_tmp.replace(r_fin)
        _mesh_barrier(rank, world_size, send_conns, recv_conns)

        if rank == 0:
            all_ranks = [json.loads((coord_dir / f"result_{r}.json").read_text(encoding="utf-8")) for r in range(world_size)]
            merged: dict[str, object] = {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "world_size": world_size,
                "interface": iface,
                "param_count": param_count,
                "payload_fp32_mb": round(payload_bytes / 1e6, 3),
                "peers": peers,
                "rank0_net_micro": all_ranks[0]["net_micro"],
                "ranks": all_ranks,
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            return merged
        return None
    finally:
        for c in list(send_conns.values()) + list(recv_conns.values()):
            try:
                c.close()
            except OSError:
                pass


def main() -> None:
    args = parse_args()
    if args.mode == "transfer_bench":
        run_transfer_benchmark(
            role=args.bench_role,
            host=args.bench_host,
            port=args.bench_port,
            payload_bytes=args.bench_payload_bytes,
            single_repeats=args.bench_repeats,
            burst_repeats=args.bench_burst_repeats,
            out_path=args.bench_out,
        )
        return
    if args.mode == "multinode_bench":
        run_multinode_benchmark(
            out_path=args.bench_out or "outputs/phase1_4node_bench/ja4node_v100_bench.json",
            iface=args.bench_iface,
            base_port=args.bench_port,
            data_dir=args.data_dir,
            e2e_rounds=args.bench_e2e_rounds,
            deg6_rounds=args.bench_deg6_rounds,
            seed=args.seeds[0],
        )
        return
    base_cfg = _config_from_args(args)
    out_root = Path(args.output_root)
    if args.mode == "single":
        _run_single(args, base_cfg, out_root)
    else:
        _run_suite(args, base_cfg, out_root)


if __name__ == "__main__":
    main()
