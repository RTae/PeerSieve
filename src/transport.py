from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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

from src.logging_utils import get_logger
from src.peer import PeerPayload, sync_device

logger = get_logger()


@dataclass
class TransportStats:
    """Byte counts and wall-clock transfer timings for one peer's round exchange."""

    bytes_sent: int = 0
    bytes_recv: int = 0
    d2h_sec: float = 0.0
    net_sec: float = 0.0
    h2d_sec: float = 0.0


class PeerTransport(ABC):
    """Abstract peer-to-peer transport interface shared by local and distributed DFL drivers."""

    @abstractmethod
    def exchange(
        self,
        peer_id: int,
        neighbors: list[int],
        payload: PeerPayload,
    ) -> tuple[dict[int, PeerPayload], TransportStats]:
        """Send payload to neighbors and receive each neighbor's payload for this round.

        Args:
            peer_id: Sending peer ID.
            neighbors: List of adjacent peer IDs to exchange with.
            payload: Outgoing PeerPayload for peer_id.

        Returns:
            Tuple of (map from neighbor ID to PeerPayload, TransportStats for this round).
        """

    def barrier(self) -> None:
        """Optional synchronization barrier across peers.

        Returns:
            None.
        """

    def close(self) -> None:
        """Release any sockets or staging buffers held by the transport.

        Returns:
            None.
        """


class LocalTransport(PeerTransport):
    """In-process transport for single-GPU multi-peer sweeps with byte accounting."""

    def __init__(self) -> None:
        """Create an empty in-process payload table for synchronous rounds.

        Returns:
            None.
        """
        self._round_payloads: dict[int, PeerPayload] = {}

    def stage_round(self, payloads: dict[int, PeerPayload]) -> None:
        """Publish all peers' outgoing payloads for the current synchronous round.

        Args:
            payloads: Map from peer ID to outgoing PeerPayload.

        Returns:
            None.
        """
        self._round_payloads = payloads

    def exchange(
        self,
        peer_id: int,
        neighbors: list[int],
        payload: PeerPayload,
    ) -> tuple[dict[int, PeerPayload], TransportStats]:
        """Look up neighbor payloads staged for this round and count logical byte volume.

        Args:
            peer_id: Local peer ID.
            neighbors: Adjacent neighbor IDs to fetch.
            payload: Outgoing PeerPayload for peer_id.

        Returns:
            Tuple of (received neighbor payloads dict, TransportStats with byte counts).
        """
        received = {j: self._round_payloads[j] for j in neighbors}
        out_bytes = payload.payload_bytes() * len(neighbors)
        in_bytes = sum(p.payload_bytes() for p in received.values())
        return received, TransportStats(bytes_sent=out_bytes, bytes_recv=in_bytes)


def recv_exact(conn: socket.socket, view: memoryview) -> None:
    """Read len(view) bytes from conn directly into view without intermediate copies.

    Args:
        conn: Connected stream socket to read from.
        view: Writable memoryview receiving the exact byte count.

    Returns:
        None.
    """
    received = 0
    total = len(view)
    while received < total:
        n = conn.recv_into(view[received:])
        if n == 0:
            raise ConnectionError(f"Socket closed after {received}/{total} bytes.")
        received += n


def probe_interface_info(iface: str | None = None) -> dict[str, str]:
    """Query IPv4 address, link speed, and MTU for a Linux network interface or loopback.

    Args:
        iface: Linux interface name (e.g., 'enp68s0f0'), or None/'lo' for loopback.

    Returns:
        Dictionary with hostname, interface, ip, link_speed_mbps, and mtu strings.
    """
    info = {
        "hostname": socket.gethostname(),
        "interface": iface or "lo",
        "ip": "127.0.0.1",
        "link_speed_mbps": "unknown",
        "mtu": "unknown",
    }
    if not iface or iface == "lo":
        return info
    try:
        out = subprocess.check_output(
            ["ip", "-4", "-o", "addr", "show", "dev", iface],
            text=True,
            stderr=subprocess.DEVNULL,
        )
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


def probe_route_info(peer_host: str) -> dict[str, object]:
    """Query the active route interface, link speed, and MTU toward peer_host.

    Args:
        peer_host: Target hostname or IPv4 address.

    Returns:
        Dictionary with hostname, peer_host, and any resolved IP/interface/speed/MTU fields.
    """
    info: dict[str, object] = {"hostname": socket.gethostname(), "peer_host": peer_host}
    try:
        peer_ip = socket.gethostbyname(peer_host)
        info["peer_ip"] = peer_ip
    except OSError:
        return info
    try:
        out = subprocess.check_output(["ip", "route", "get", peer_ip], text=True, stderr=subprocess.DEVNULL)
        info["ip_route"] = out.strip().splitlines()[0]
        tokens = str(info["ip_route"]).split()
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


def send_json_frame(conn: socket.socket, obj: dict[str, object]) -> None:
    """Send a length-prefixed UTF-8 JSON object over a connected socket.

    Args:
        conn: Connected stream socket.
        obj: JSON-serializable dictionary to send.

    Returns:
        None.
    """
    raw = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!Q", len(raw)))
    conn.sendall(raw)


def recv_json_frame(conn: socket.socket) -> dict[str, object]:
    """Receive a length-prefixed UTF-8 JSON object from a connected socket.

    Args:
        conn: Connected stream socket.

    Returns:
        Decoded dictionary parsed from JSON.
    """
    hdr = bytearray(8)
    recv_exact(conn, memoryview(hdr))
    length = struct.unpack("!Q", hdr)[0]
    body = bytearray(length)
    recv_exact(conn, memoryview(body))
    return json.loads(body.decode("utf-8"))


def send_tensor_frame(conn: socket.socket, tensor: torch.Tensor) -> None:
    """Send a 1D FP32 tensor over a connected socket with an 8-byte length header.

    Args:
        conn: Connected stream socket.
        tensor: Tensor to convert to contiguous CPU FP32 and transmit.

    Returns:
        None.
    """
    cpu_flat = tensor.detach().contiguous().float().cpu()
    raw = memoryview(cpu_flat.view(torch.uint8).numpy())
    conn.sendall(struct.pack("!Q", len(raw)))
    conn.sendall(raw)


def recv_tensor_frame(conn: socket.socket) -> torch.Tensor:
    """Receive a length-prefixed 1D FP32 CPU tensor from a connected socket.

    Args:
        conn: Connected stream socket.

    Returns:
        1D FP32 CPU tensor reconstructed from incoming wire bytes.
    """
    hdr = bytearray(8)
    recv_exact(conn, memoryview(hdr))
    n_bytes = struct.unpack("!Q", hdr)[0]
    out = torch.empty(n_bytes // 4, dtype=torch.float32)
    recv_exact(conn, memoryview(out.view(torch.uint8).numpy()))
    return out


class TcpPeerTransport(PeerTransport):
    """Multi-process, multi-GPU, and multi-node TCP peer transport."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        neighbors: list[int],
        param_count: int,
        device: torch.device,
        send_conns: dict[int, socket.socket],
        recv_conns: dict[int, socket.socket],
        ctrl_send: dict[int, socket.socket],
        ctrl_recv: dict[int, socket.socket],
        send_both_vectors: bool = True,
    ) -> None:
        """Bind connected neighbor and control sockets and allocate host staging buffers.

        Args:
            rank: Global peer rank.
            world_size: Total peer count.
            neighbors: Graph neighbor IDs for this rank.
            param_count: Scalar parameter count of one model vector.
            device: Compute device assigned to this peer.
            send_conns: Outgoing data sockets keyed by neighbor ID.
            recv_conns: Incoming data sockets keyed by neighbor ID.
            ctrl_send: Outgoing rank-0 control sockets keyed by peer ID.
            ctrl_recv: Incoming rank-0 control sockets keyed by peer ID.
            send_both_vectors: True to transmit post_model and update; False for post_model only.

        Returns:
            None.
        """
        self.rank = rank
        self.world_size = world_size
        self.neighbors = list(neighbors)
        self.param_count = param_count
        self.device = device
        self.send_conns = send_conns
        self.recv_conns = recv_conns
        self.ctrl_send = ctrl_send
        self.ctrl_recv = ctrl_recv
        self.send_both_vectors = send_both_vectors

        vec_mult = 2 if send_both_vectors else 1
        pin = device.type == "cuda"
        self._send_host = torch.empty(param_count * vec_mult, dtype=torch.float32, pin_memory=pin)
        self._recv_hosts: dict[int, torch.Tensor] = {
            nbr: torch.empty(param_count * vec_mult, dtype=torch.float32, pin_memory=pin)
            for nbr in self.neighbors
        }
        self._send_view = memoryview(self._send_host.view(torch.uint8).numpy())
        self._recv_views = {
            nbr: memoryview(buf.view(torch.uint8).numpy())
            for nbr, buf in self._recv_hosts.items()
        }

    @property
    def wire_bytes_per_neighbor(self) -> int:
        """Return the number of bytes sent to each neighbor per round.

        Returns:
            Wire payload size in bytes for one directed neighbor transfer.
        """
        return len(self._send_view)

    @classmethod
    def connect_tcp(
        cls,
        rank: int,
        world_size: int,
        neighbors: list[int],
        param_count: int,
        device: torch.device,
        peers: list[dict[str, object]],
        send_both_vectors: bool = True,
        timeout_sec: float = 90.0,
    ) -> TcpPeerTransport:
        """Establish topology-scoped data sockets and rank-0 control sockets over TCP.

        Args:
            rank: Global peer rank.
            world_size: Total number of peers.
            neighbors: Graph neighbor IDs for this rank.
            param_count: Scalar parameter count of one model vector.
            device: Compute device for this peer.
            peers: Ordered list of peer endpoint dicts containing 'ip' and 'port'.
            send_both_vectors: True to send post_model and update; False for post_model only.
            timeout_sec: Handshake and socket timeout in seconds.

        Returns:
            Connected TcpPeerTransport ready for round exchanges and barriers.
        """
        my_ip = str(peers[rank]["ip"])
        my_port = int(peers[rank]["port"])

        # Each peer connects data sockets only to its graph neighbors, plus control sockets with rank 0.
        expected_in_data = set(neighbors)
        expected_in_ctrl = set(range(1, world_size)) if rank == 0 else {0}
        total_expected = len(expected_in_data) + len(expected_in_ctrl)

        recv_conns: dict[int, socket.socket] = {}
        ctrl_recv: dict[int, socket.socket] = {}
        accept_errors: list[BaseException] = []

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.settimeout(timeout_sec)
        srv.bind((my_ip, my_port))
        srv.listen(max(total_expected * 2, 8))

        def _accept_loop() -> None:
            """Accept incoming neighbor data sockets and rank-0 control sockets.

            Returns:
                None.
            """
            try:
                for _ in range(total_expected):
                    conn, _ = srv.accept()
                    conn.settimeout(timeout_sec)
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    hdr = bytearray(5)
                    recv_exact(conn, memoryview(hdr))
                    kind, src_rank = struct.unpack("!BI", hdr)
                    if kind == 0:
                        recv_conns[src_rank] = conn
                    else:
                        ctrl_recv[src_rank] = conn
            except BaseException as exc:
                accept_errors.append(exc)
            finally:
                srv.close()

        acc_thread = threading.Thread(target=_accept_loop, daemon=True)
        acc_thread.start()

        def _dial(dst: int, kind: int) -> socket.socket:
            """Connect to peer dst with retries and send the 5-byte (kind, rank) header.

            Args:
                dst: Target peer rank.
                kind: Channel type (0 for neighbor data, 1 for rank-0 control).

            Returns:
                Connected TCP socket to peer dst.
            """
            dst_ip = str(peers[dst]["ip"])
            dst_port = int(peers[dst]["port"])
            hdr = struct.pack("!BI", kind, rank)
            max_attempts = max(1, int(timeout_sec / 0.25))
            for attempt in range(max_attempts):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(timeout_sec)
                try:
                    s.connect((dst_ip, dst_port))
                    s.sendall(hdr)
                    return s
                except OSError:
                    s.close()
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(0.25)
            raise RuntimeError(f"Rank {rank} failed to connect to rank {dst}.")

        send_conns: dict[int, socket.socket] = {}
        ctrl_send: dict[int, socket.socket] = {}
        try:
            for nbr in neighbors:
                send_conns[nbr] = _dial(nbr, kind=0)
            ctrl_targets = list(range(1, world_size)) if rank == 0 else [0]
            for dst in ctrl_targets:
                ctrl_send[dst] = _dial(dst, kind=1)

            acc_thread.join(timeout=timeout_sec + 5.0)
            if accept_errors:
                raise accept_errors[0]
            if set(recv_conns.keys()) != expected_in_data or set(ctrl_recv.keys()) != expected_in_ctrl:
                raise RuntimeError(
                    f"Rank {rank} incomplete mesh: data={sorted(recv_conns)}, ctrl={sorted(ctrl_recv)}"
                )
        except BaseException:
            for sock_map in (send_conns, recv_conns, ctrl_send, ctrl_recv):
                for s in sock_map.values():
                    try:
                        s.close()
                    except OSError:
                        pass
            raise

        return cls(
            rank=rank,
            world_size=world_size,
            neighbors=neighbors,
            param_count=param_count,
            device=device,
            send_conns=send_conns,
            recv_conns=recv_conns,
            ctrl_send=ctrl_send,
            ctrl_recv=ctrl_recv,
            send_both_vectors=send_both_vectors,
        )

    @classmethod
    def build_socketpair_group(
        cls,
        graph_neighbors: dict[int, list[int]],
        param_count: int,
        device: torch.device,
        send_both_vectors: bool = True,
    ) -> list[TcpPeerTransport]:
        """Build an offline group of TcpPeerTransports backed by Unix socketpairs for tests.

        Args:
            graph_neighbors: Map from peer rank to list of neighbor ranks.
            param_count: Scalar parameter count of one model vector.
            device: Compute device for the test peers.
            send_both_vectors: True to send post_model and update; False for post_model only.

        Returns:
            List of connected TcpPeerTransport instances, indexed by rank.
        """
        world_size = len(graph_neighbors)
        send_maps: list[dict[int, socket.socket]] = [{} for _ in range(world_size)]
        recv_maps: list[dict[int, socket.socket]] = [{} for _ in range(world_size)]
        ctrl_send_maps: list[dict[int, socket.socket]] = [{} for _ in range(world_size)]
        ctrl_recv_maps: list[dict[int, socket.socket]] = [{} for _ in range(world_size)]

        for src, nbrs in graph_neighbors.items():
            for dst in nbrs:
                s_a, s_b = socket.socketpair()
                send_maps[src][dst] = s_a
                recv_maps[dst][src] = s_b
        for peer in range(1, world_size):
            c_up_a, c_up_b = socket.socketpair()
            ctrl_send_maps[peer][0] = c_up_a
            ctrl_recv_maps[0][peer] = c_up_b
            c_dn_a, c_dn_b = socket.socketpair()
            ctrl_send_maps[0][peer] = c_dn_a
            ctrl_recv_maps[peer][0] = c_dn_b

        return [
            cls(
                rank=r,
                world_size=world_size,
                neighbors=graph_neighbors[r],
                param_count=param_count,
                device=device,
                send_conns=send_maps[r],
                recv_conns=recv_maps[r],
                ctrl_send=ctrl_send_maps[r],
                ctrl_recv=ctrl_recv_maps[r],
                send_both_vectors=send_both_vectors,
            )
            for r in range(world_size)
        ]

    def barrier(self) -> None:
        """Synchronize all peer ranks over the rank-0 control sockets.

        Returns:
            None.
        """
        buf = bytearray(1)
        if self.rank == 0:
            for src in range(1, self.world_size):
                recv_exact(self.ctrl_recv[src], memoryview(buf))
            for dst in range(1, self.world_size):
                self.ctrl_send[dst].sendall(b"G")
        else:
            self.ctrl_send[0].sendall(b"B")
            recv_exact(self.ctrl_recv[0], memoryview(buf))

    def gather_json(self, local_obj: dict[str, object]) -> list[dict[str, object]] | None:
        """Gather one JSON dictionary from every rank onto rank 0.

        Args:
            local_obj: JSON-serializable dictionary produced by this rank.

        Returns:
            Ordered list of dictionaries from all ranks on rank 0, or None on other ranks.
        """
        if self.rank == 0:
            gathered = [local_obj]
            for src in range(1, self.world_size):
                gathered.append(recv_json_frame(self.ctrl_recv[src]))
            return gathered
        send_json_frame(self.ctrl_send[0], local_obj)
        return None

    def gather_tensors(self, local_tensor: torch.Tensor) -> list[torch.Tensor] | None:
        """Gather one 1D FP32 tensor from every rank onto rank 0 on CPU.

        Args:
            local_tensor: Local 1D parameter tensor to send.

        Returns:
            Ordered list of CPU FP32 tensors from all ranks on rank 0, or None on other ranks.
        """
        if self.rank == 0:
            out = [local_tensor.detach().float().cpu()]
            for src in range(1, self.world_size):
                out.append(recv_tensor_frame(self.ctrl_recv[src]))
            return out
        send_tensor_frame(self.ctrl_send[0], local_tensor)
        return None

    def exchange(
        self,
        peer_id: int,
        neighbors: list[int],
        payload: PeerPayload,
    ) -> tuple[dict[int, PeerPayload], TransportStats]:
        """Stage payload to pinned host memory, exchange concurrently with neighbors, and copy to GPU.

        Args:
            peer_id: Local peer rank.
            neighbors: Graph neighbor IDs to exchange payloads with.
            payload: Outgoing PeerPayload on device.

        Returns:
            Tuple of (map from neighbor ID to received PeerPayload on device, TransportStats with timings).
        """
        sync_device(self.device)
        t_d2h0 = time.perf_counter()
        if self.send_both_vectors:
            self._send_host[: self.param_count].copy_(payload.post_model, non_blocking=False)
            self._send_host[self.param_count :].copy_(payload.update, non_blocking=False)
        else:
            self._send_host.copy_(payload.post_model, non_blocking=False)
        sync_device(self.device)
        d2h_sec = time.perf_counter() - t_d2h0

        self.barrier()
        t_net0 = time.perf_counter()
        errors: list[BaseException] = []

        def _do_send(conn: socket.socket) -> None:
            """Send the staged host buffer over one neighbor socket.

            Args:
                conn: Outgoing neighbor socket.

            Returns:
                None.
            """
            try:
                conn.sendall(self._send_view)
            except BaseException as exc:
                errors.append(exc)

        def _do_recv(conn: socket.socket, view: memoryview) -> None:
            """Receive one neighbor's wire payload into its host buffer.

            Args:
                conn: Incoming neighbor socket.
                view: Writable memoryview of the destination host buffer.

            Returns:
                None.
            """
            try:
                recv_exact(conn, view)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_do_send, args=(self.send_conns[j],)) for j in neighbors]
        threads.extend(threading.Thread(target=_do_recv, args=(self.recv_conns[j], self._recv_views[j])) for j in neighbors)
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        net_sec = time.perf_counter() - t_net0

        sync_device(self.device)
        t_h2d0 = time.perf_counter()
        received: dict[int, PeerPayload] = {}
        for j in neighbors:
            gpu_buf = self._recv_hosts[j].to(self.device, non_blocking=False)
            if self.send_both_vectors:
                p_j = gpu_buf[: self.param_count]
                u_j = gpu_buf[self.param_count :]
            else:
                p_j = gpu_buf
                u_j = gpu_buf
            received[j] = PeerPayload(sender_id=j, post_model=p_j, update=u_j)
        sync_device(self.device)
        h2d_sec = time.perf_counter() - t_h2d0

        wire_bytes = len(self._send_view)
        return received, TransportStats(
            bytes_sent=wire_bytes * len(neighbors),
            bytes_recv=wire_bytes * len(neighbors),
            d2h_sec=d2h_sec,
            net_sec=net_sec,
            h2d_sec=h2d_sec,
        )

    def close(self) -> None:
        """Close all open neighbor data sockets and rank-0 control sockets.

        Returns:
            None.
        """
        for sock_map in (self.send_conns, self.recv_conns, self.ctrl_send, self.ctrl_recv):
            for s in sock_map.values():
                try:
                    s.close()
                except OSError:
                    pass


def _serve_transfer_workloads(conn: socket.socket, workloads: list[tuple[str, int, int, int]]) -> None:
    """Receive each benchmark payload from the client and send a 1-byte completion ack.

    Args:
        conn: Accepted client socket.
        workloads: List of (name, byte_count, warmups, repeats) tuples.

    Returns:
        None.
    """
    max_bytes = max(n_bytes for _, n_bytes, _, _ in workloads)
    view = memoryview(bytearray(max_bytes))
    for _, n_bytes, n_warm, n_rep in workloads:
        slice_view = view[:n_bytes]
        for _ in range(n_warm + n_rep):
            conn.sendall(b"G")
            recv_exact(conn, slice_view)
            conn.sendall(b"D")


def _time_transfer_workloads(
    sock: socket.socket,
    host: str,
    workloads: list[tuple[str, int, int, int]],
) -> dict[str, dict[str, object]]:
    """Send each benchmark payload over sock, wait for server ack, and summarize timings.

    Args:
        sock: Connected client socket to the benchmark server.
        host: Target server host string used in log output.
        workloads: List of (name, byte_count, warmups, repeats) tuples.

    Returns:
        Dictionary keyed by workload name containing median/min/max seconds and goodput in Mb/s.
    """
    max_bytes = max(n_bytes for _, n_bytes, _, _ in workloads)
    view = memoryview(bytearray(max_bytes))
    go_buf = bytearray(1)
    done_buf = bytearray(1)
    results: dict[str, dict[str, object]] = {}
    for name, n_bytes, n_warm, n_rep in workloads:
        slice_view = view[:n_bytes]
        times: list[float] = []
        for step in range(n_warm + n_rep):
            recv_exact(sock, memoryview(go_buf))
            t0 = time.perf_counter()
            sock.sendall(slice_view)
            recv_exact(sock, memoryview(done_buf))
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
        logger.info(
            "[client {} -> {}] {:20s} ({:6.2f} MB): median={:.4f}s (min={:.4f}s, max={:.4f}s, n={}) -> {:.1f} Mb/s",
            socket.gethostname(),
            host,
            name,
            n_bytes / 1e6,
            med,
            mn,
            mx,
            len(times),
            mbps,
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
    """Measure point-to-point TCP transfer durations for ResNet-18 FP32 payload sizes.

    Args:
        role: Benchmark role ('auto', 'server', or 'client').
        host: Server hostname or IPv4 address for the client to connect to.
        port: TCP listening port on the server.
        payload_bytes: Single-update payload size in bytes.
        single_repeats: Measured repeat count for single-update transfers.
        burst_repeats: Measured repeat count for multi-neighbor and useful/wasted bursts.
        warmups: Unmeasured warmup iterations before timing.
        useful_bytes: Model-poison useful-share payload size in bytes.
        wasted_bytes: Model-poison wasted-share payload size in bytes.
        out_path: Optional JSON output path written on the client.
        connected_sock: Optional pre-connected socket (used by offline socketpair tests).

    Returns:
        Benchmark report dictionary on the client, or None on the server.
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
        logger.info("[server {}] listening on 0.0.0.0:{}", socket.gethostname(), port)
        conn, addr = srv.accept()
        conn.settimeout(90.0)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info("[server {}] accepted connection from {}:{}", socket.gethostname(), addr[0], addr[1])
        try:
            _serve_transfer_workloads(conn, workloads)
        finally:
            conn.close()
            srv.close()
        return None

    if connected_sock is not None:
        results = _time_transfer_workloads(connected_sock, host, workloads)
    else:
        logger.info("[client {}] connecting to {}:{}", socket.gethostname(), host, port)
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
        "client_link": probe_route_info(host),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "port": port,
        "workloads": results,
    }
    if out_path is not None:
        dst = Path(out_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
