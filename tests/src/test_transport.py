from __future__ import annotations

from pathlib import Path
import socket
import threading

import pytest
import torch

from src.peer import PeerPayload
from src.transport import (
    LocalTransport,
    TcpPeerTransport,
    probe_interface_info,
    recv_exact,
    recv_json_frame,
    recv_tensor_frame,
    run_transfer_benchmark,
    send_json_frame,
    send_tensor_frame,
)


def test_local_transport_exchange_and_byte_accounting() -> None:
    tr = LocalTransport()
    p0 = PeerPayload(sender_id=0, post_model=torch.ones(10), update=torch.zeros(10))
    p1 = PeerPayload(sender_id=1, post_model=torch.full((10,), 2.0), update=torch.ones(10))
    tr.stage_round({0: p0, 1: p1})

    recv, stats = tr.exchange(peer_id=0, neighbors=[1], payload=p0)
    assert list(recv.keys()) == [1]
    assert torch.equal(recv[1].post_model, p1.post_model)
    assert stats.bytes_sent == 40
    assert stats.bytes_recv == 40
    assert probe_interface_info(None)["ip"] == "127.0.0.1"


def test_socket_framing_json_tensor_and_eof() -> None:
    s_a, s_b = socket.socketpair()
    try:
        send_json_frame(s_a, {"rank": 1, "ok": True})
        assert recv_json_frame(s_b) == {"rank": 1, "ok": True}

        t_out = torch.tensor([1.5, -2.25, 3.0], dtype=torch.float32)
        send_tensor_frame(s_a, t_out)
        t_in = recv_tensor_frame(s_b)
        assert torch.equal(t_in, t_out)

        s_a.close()
        with pytest.raises(ConnectionError, match="Socket closed"):
            recv_exact(s_b, memoryview(bytearray(4)))
    finally:
        s_a.close()
        s_b.close()


def test_tcp_peer_transport_primitives_over_socketpair() -> None:
    for send_both in (True, False):
        param_count = 8
        transports = TcpPeerTransport.build_socketpair_group(
            graph_neighbors={0: [1], 1: [0]},
            param_count=param_count,
            device=torch.device("cpu"),
            send_both_vectors=send_both,
        )
        expected_wire = param_count * (8 if send_both else 4)
        assert transports[0].wire_bytes_per_neighbor == expected_wire

        p0 = PeerPayload(sender_id=0, post_model=torch.arange(param_count, dtype=torch.float32), update=torch.ones(param_count))
        p1 = PeerPayload(sender_id=1, post_model=torch.full((param_count,), 5.0), update=torch.full((param_count,), -1.0))
        out_rx: list[dict[int, PeerPayload] | None] = [None, None]
        gathered_json: list[object] = [None, None]
        gathered_tensors: list[object] = [None, None]

        def _worker(rank: int, payload: PeerPayload) -> None:
            tr = transports[rank]
            rx, stats = tr.exchange(rank, tr.neighbors, payload)
            assert stats.bytes_sent == expected_wire and stats.bytes_recv == expected_wire
            out_rx[rank] = rx
            gathered_json[rank] = tr.gather_json({"rank": rank})
            gathered_tensors[rank] = tr.gather_tensors(payload.post_model)

        try:
            t0 = threading.Thread(target=_worker, args=(0, p0))
            t1 = threading.Thread(target=_worker, args=(1, p1))
            t0.start()
            t1.start()
            t0.join(timeout=10.0)
            t1.join(timeout=10.0)
        finally:
            for tr in transports:
                tr.close()

        assert out_rx[0] is not None and out_rx[1] is not None
        assert torch.equal(out_rx[0][1].post_model, p1.post_model)
        assert torch.equal(out_rx[1][0].post_model, p0.post_model)
        if send_both:
            assert torch.equal(out_rx[0][1].update, p1.update)
        else:
            assert torch.equal(out_rx[0][1].update, p1.post_model)
        assert gathered_json[0] == [{"rank": 0}, {"rank": 1}]
        assert gathered_json[1] is None
        assert gathered_tensors[0] is not None and len(gathered_tensors[0]) == 2
        assert gathered_tensors[1] is None


def test_transfer_benchmark_socketpair(tmp_path: Path) -> None:
    srv_sock, cli_sock = socket.socketpair()
    try:
        server_thread = threading.Thread(
            target=run_transfer_benchmark,
            kwargs={
                "role": "server",
                "host": "127.0.0.1",
                "port": 0,
                "payload_bytes": 4096,
                "single_repeats": 3,
                "burst_repeats": 2,
                "warmups": 1,
                "useful_bytes": 2048,
                "wasted_bytes": 3072,
                "connected_sock": srv_sock,
            },
            daemon=True,
        )
        server_thread.start()
        out_file = tmp_path / "transfer.json"
        result = run_transfer_benchmark(
            role="client",
            host="127.0.0.1",
            port=0,
            payload_bytes=4096,
            single_repeats=3,
            burst_repeats=2,
            warmups=1,
            useful_bytes=2048,
            wasted_bytes=3072,
            out_path=out_file,
            connected_sock=cli_sock,
        )
        server_thread.join(timeout=5.0)
    finally:
        srv_sock.close()
        cli_sock.close()
    assert result is not None and out_file.exists()
    w = result["workloads"]
    assert set(w.keys()) == {"single_update", "round_6_neighbors", "model_poison_useful", "model_poison_wasted"}
    assert w["single_update"]["repeats"] == 3 and w["single_update"]["median_sec"] > 0.0
    assert w["round_6_neighbors"]["bytes"] == 4096 * 6 and w["round_6_neighbors"]["goodput_mbps"] > 0.0
