"""Round-level and system-cost metrics tracking for TrustGraph-DFL."""

from __future__ import annotations

from dataclasses import dataclass
import math

import networkx as nx
import numpy as np


@dataclass
class RoundAggregationStats:
    """Aggregated trust and timing counters collected during one DFL round."""

    benign_zero: float
    benign_pairs: int
    byz_accept: float
    byz_pairs: int
    byz_weight_share: float
    self_fallback_rate: float
    benign_score_mean: float | None
    byz_score_mean: float | None
    agg_sec_per_peer: float = 0.0
    d2h_sec_per_peer: float = 0.0
    net_sec_per_peer: float = 0.0
    h2d_sec_per_peer: float = 0.0
    wire_bytes_recv_per_peer: float = 0.0


def compute_rejection_streaks(seq: list[bool]) -> tuple[int, int, int, int]:
    """Return (total_rejected, max_consec_rejected, repeat_after_first, isolated_single_rejections)."""
    total = sum(1 for x in seq if x)
    max_streak = 0
    cur = 0
    isolated = 0
    for idx, rejected in enumerate(seq):
        if rejected:
            cur += 1
            if cur > max_streak:
                max_streak = cur
            prev_rej = seq[idx - 1] if idx > 0 else False
            next_rej = seq[idx + 1] if idx + 1 < len(seq) else False
            if not prev_rej and not next_rej:
                isolated += 1
        else:
            cur = 0
    repeat_after_first = max(0, total - 1) if total > 0 else 0
    return total, max_streak, repeat_after_first, isolated


def summarize_timings(vals: list[float]) -> dict[str, float | int]:
    """Return median, min, max, and repeat count for a list of timing samples in seconds."""
    if not vals:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "repeats": 0}
    return {
        "median": round(float(np.median(vals)), 6),
        "min": round(float(np.min(vals)), 6),
        "max": round(float(np.max(vals)), 6),
        "repeats": len(vals),
    }


class MetricsTracker:
    """Accumulates per-round accuracy, FPR, Byzantine acceptance, and system-cost metrics."""

    def __init__(
        self,
        graph: nx.Graph,
        benign_ids: list[int],
        byzantine_ids: set[int],
        param_count: int,
        rounds: int,
    ) -> None:
        """Initialize per-edge rejection history, timing lists, and cumulative counters."""
        self.graph = graph
        self.benign_ids = benign_ids
        self.byzantine_ids = byzantine_ids
        self.param_count = param_count
        self.rounds = rounds

        self.edge_rejected: dict[tuple[int, int], list[bool]] = {
            (i, j): [] for i in benign_ids for j in graph.neighbors(i)
        }
        self.train_sec_list: list[float] = []
        self.val_sec_list: list[float] = []
        self.agg_sec_list: list[float] = []
        self.d2h_sec_list: list[float] = []
        self.net_sec_list: list[float] = []
        self.h2d_sec_list: list[float] = []
        self.wire_bytes_recv_list: list[float] = []

        self.cum_benign_zero = 0.0
        self.cum_benign_pairs = 0
        self.cum_byz_accept = 0.0
        self.cum_byz_pairs = 0
        self.history: list[dict[str, object]] = []

    def record_edge_outcome(self, receiver_id: int, sender_id: int, rejected: bool) -> None:
        """Append one round's accept/reject decision for a directed benign-receiver edge."""
        key = (receiver_id, sender_id)
        if key in self.edge_rejected:
            self.edge_rejected[key].append(rejected)

    def summarize_benign_round(
        self,
        benign_reports: list[dict[str, object]],
        agg_sec_per_peer: float,
        d2h_sec_per_peer: float = 0.0,
        net_sec_per_peer: float = 0.0,
        h2d_sec_per_peer: float = 0.0,
        wire_bytes_recv_per_peer: float = 0.0,
    ) -> RoundAggregationStats:
        """Record per-edge outcomes from benign peer reports and build RoundAggregationStats."""
        benign_zero = 0.0
        benign_pairs = 0
        byz_accept = 0.0
        byz_pairs = 0
        ben_scores: list[float] = []
        byz_scores: list[float] = []
        byz_w_shares: list[float] = []
        self_fallbacks = 0

        for report in benign_reports:
            rx_id = int(report["rank"])
            if bool(report["self_fallback"]):
                self_fallbacks += 1
            byz_w_shares.append(float(report["byz_weight_share"]))
            for edge in report["edges"]:
                src_id = int(edge["sender_id"])
                z_frac = float(edge["zero_frac"])
                score = edge["score"]
                self.record_edge_outcome(rx_id, src_id, z_frac >= 0.5)
                if src_id in self.byzantine_ids:
                    byz_pairs += 1
                    byz_accept += 1.0 - z_frac
                    if score is not None:
                        byz_scores.append(float(score))
                else:
                    benign_pairs += 1
                    benign_zero += z_frac
                    if score is not None:
                        ben_scores.append(float(score))

        n_ben = max(len(benign_reports), 1)
        return RoundAggregationStats(
            benign_zero=benign_zero,
            benign_pairs=benign_pairs,
            byz_accept=byz_accept,
            byz_pairs=byz_pairs,
            byz_weight_share=float(np.mean(byz_w_shares)) if byz_w_shares else 0.0,
            self_fallback_rate=self_fallbacks / n_ben,
            benign_score_mean=float(np.mean(ben_scores)) if ben_scores else None,
            byz_score_mean=float(np.mean(byz_scores)) if byz_scores else None,
            agg_sec_per_peer=agg_sec_per_peer,
            d2h_sec_per_peer=d2h_sec_per_peer,
            net_sec_per_peer=net_sec_per_peer,
            h2d_sec_per_peer=h2d_sec_per_peer,
            wire_bytes_recv_per_peer=wire_bytes_recv_per_peer,
        )

    def record_round(
        self,
        round_idx: int,
        train_sec_per_peer: float,
        val_grad_sec_per_peer: float,
        agg_stats: RoundAggregationStats,
        eval_metrics: dict[str, float] | None = None,
        round_wall_sec: float = 0.0,
    ) -> dict[str, object]:
        """Store one round's metrics and timings, append to history, and return the JSON record."""
        self.train_sec_list.append(train_sec_per_peer)
        self.val_sec_list.append(val_grad_sec_per_peer)
        self.agg_sec_list.append(agg_stats.agg_sec_per_peer)
        if agg_stats.d2h_sec_per_peer > 0.0 or agg_stats.net_sec_per_peer > 0.0 or agg_stats.h2d_sec_per_peer > 0.0:
            self.d2h_sec_list.append(agg_stats.d2h_sec_per_peer)
            self.net_sec_list.append(agg_stats.net_sec_per_peer)
            self.h2d_sec_list.append(agg_stats.h2d_sec_per_peer)
            self.wire_bytes_recv_list.append(agg_stats.wire_bytes_recv_per_peer)

        self.cum_benign_zero += agg_stats.benign_zero
        self.cum_benign_pairs += agg_stats.benign_pairs
        self.cum_byz_accept += agg_stats.byz_accept
        self.cum_byz_pairs += agg_stats.byz_pairs

        round_total_pairs = agg_stats.benign_pairs + agg_stats.byz_pairs
        round_rej_pairs = agg_stats.benign_zero + (agg_stats.byz_pairs - agg_stats.byz_accept)
        record: dict[str, object] = {
            "round": round_idx,
            "fpr_round": (agg_stats.benign_zero / agg_stats.benign_pairs) if agg_stats.benign_pairs > 0 else 0.0,
            "fpr_cumulative": (self.cum_benign_zero / self.cum_benign_pairs) if self.cum_benign_pairs > 0 else 0.0,
            "byz_accept_rate": (agg_stats.byz_accept / agg_stats.byz_pairs) if agg_stats.byz_pairs > 0 else 0.0,
            "wasted_transfer_rate": (round_rej_pairs / round_total_pairs) if round_total_pairs > 0 else 0.0,
            "byz_weight_share": agg_stats.byz_weight_share,
            "self_fallback_rate": agg_stats.self_fallback_rate,
            "benign_score_mean": agg_stats.benign_score_mean,
            "byz_score_mean": agg_stats.byz_score_mean,
            "train_sec_per_peer": round(train_sec_per_peer, 6),
            "val_grad_sec_per_peer": round(val_grad_sec_per_peer, 6),
            "agg_sec_per_peer": round(agg_stats.agg_sec_per_peer, 6),
        }
        if agg_stats.net_sec_per_peer > 0.0:
            record["d2h_sec_per_peer"] = round(agg_stats.d2h_sec_per_peer, 6)
            record["net_sec_per_peer"] = round(agg_stats.net_sec_per_peer, 6)
            record["h2d_sec_per_peer"] = round(agg_stats.h2d_sec_per_peer, 6)
        if eval_metrics:
            record.update(eval_metrics)
        record["round_wall_sec"] = round_wall_sec
        self.history.append(record)
        return record

    def build_system_metrics(
        self,
        peak_gpu_alloc_mib: float | None = None,
        peak_gpu_reserved_mib: float | None = None,
    ) -> dict[str, object]:
        """Compute bandwidth, receive-memory, link-time, and rejection-streak summaries."""
        payload_bytes = self.param_count * 4
        payload_mb = payload_bytes / 1e6
        payload_mib = payload_bytes / float(1024 ** 2)
        benign_degrees = [int(self.graph.degree(i)) for i in self.benign_ids]
        mean_ben_deg = float(np.mean(benign_degrees)) if benign_degrees else 0.0
        max_ben_deg = max(benign_degrees) if benign_degrees else 0

        benign_edges = [seq for (_, j), seq in self.edge_rejected.items() if j not in self.byzantine_ids]
        byz_edges = [seq for (_, j), seq in self.edge_rejected.items() if j in self.byzantine_ids]
        byz_edge_stats = [compute_rejection_streaks(seq) for seq in byz_edges]
        benign_edge_stats = [compute_rejection_streaks(seq) for seq in benign_edges]

        total_rx = self.cum_benign_pairs + self.cum_byz_pairs
        benign_rej = int(round(self.cum_benign_zero))
        byz_rej = int(round(self.cum_byz_pairs - self.cum_byz_accept))
        total_rej = benign_rej + byz_rej
        n_ben_rounds = max(len(self.benign_ids) * self.rounds, 1)
        rx_bytes_per_peer_round = (total_rx / n_ben_rounds) * payload_bytes
        wasted_bytes_per_peer_round = (total_rej / n_ben_rounds) * payload_bytes
        repeat_byz_tx = sum(s[2] for s in byz_edge_stats)

        med_train = float(np.median(self.train_sec_list)) if self.train_sec_list else 0.0
        med_val = float(np.median(self.val_sec_list)) if self.val_sec_list else 0.0
        med_agg = float(np.median(self.agg_sec_list)) if self.agg_sec_list else 0.0

        round_time_by_link: dict[str, dict[str, float]] = {}
        for link_label, bw_bps in (("10gbps", 10e9), ("1gbps", 1e9), ("100mbps", 100e6)):
            comm_sec = (rx_bytes_per_peer_round * 8.0) / bw_bps
            wasted_comm_sec = (wasted_bytes_per_peer_round * 8.0) / bw_bps
            total_r_sec = med_train + med_val + comm_sec + med_agg
            round_time_by_link[link_label] = {
                "comm_sec": round(comm_sec, 4),
                "wasted_comm_sec": round(wasted_comm_sec, 4),
                "total_round_sec": round(total_r_sec, 4),
                "comm_share_pct": round(100.0 * comm_sec / max(total_r_sec, 1e-9), 2),
            }

        metrics: dict[str, object] = {
            "payload_fp32_mb": round(payload_mb, 3),
            "payload_fp32_mib": round(payload_mib, 3),
            "mean_benign_degree": round(mean_ben_deg, 2),
            "max_benign_degree": max_ben_deg,
            "total_transfers": int(total_rx),
            "total_rejected_transfers": int(total_rej),
            "honest_rejected_transfers": int(benign_rej),
            "byz_rejected_transfers": int(byz_rej),
            "wasted_transfer_pct": round(100.0 * total_rej / max(total_rx, 1), 2),
            "rx_mb_per_peer_round": round(rx_bytes_per_peer_round / 1e6, 2),
            "useful_mb_per_peer_round": round((rx_bytes_per_peer_round - wasted_bytes_per_peer_round) / 1e6, 2),
            "wasted_mb_per_peer_round": round(wasted_bytes_per_peer_round / 1e6, 2),
            "wasted_byz_mb_per_peer_round": round((byz_rej / n_ben_rounds) * payload_mb, 2),
            "wasted_honest_mb_per_peer_round": round((benign_rej / n_ben_rounds) * payload_mb, 2),
            "total_rx_gb_all_benign": round((total_rx * payload_mb) / 1000.0, 3),
            "total_wasted_gb_all_benign": round((total_rej * payload_mb) / 1000.0, 3),
            "batched_recv_buf_mean_mib": round(mean_ben_deg * payload_mib, 2),
            "batched_recv_buf_max_mib": round(max_ben_deg * payload_mib, 2),
            "peak_gpu_alloc_mib": peak_gpu_alloc_mib,
            "peak_gpu_reserved_mib": peak_gpu_reserved_mib,
            "train_sec_per_peer": summarize_timings(self.train_sec_list),
            "val_grad_sec_per_peer": summarize_timings(self.val_sec_list),
            "agg_sec_per_peer": summarize_timings(self.agg_sec_list),
            "round_time_by_link": round_time_by_link,
            "byz_repeat_rejections": {
                "n_byz_edges": len(byz_edges),
                "mean_rejected_rounds": round(float(np.mean([s[0] for s in byz_edge_stats])), 2) if byz_edge_stats else 0.0,
                "min_rejected_rounds": min([s[0] for s in byz_edge_stats]) if byz_edge_stats else 0,
                "max_rejected_rounds": max([s[0] for s in byz_edge_stats]) if byz_edge_stats else 0,
                "mean_max_consec_streak": round(float(np.mean([s[1] for s in byz_edge_stats])), 2) if byz_edge_stats else 0.0,
                "repeat_transfers_after_first_rej": int(repeat_byz_tx),
                "repeat_wasted_gb_after_first_rej": round((repeat_byz_tx * payload_mb) / 1000.0, 3),
                "frac_edges_with_streak_ge3_pct": round(
                    100.0 * sum(1 for s in byz_edge_stats if s[1] >= 3) / max(len(byz_edge_stats), 1), 2
                )
                if byz_edge_stats
                else 0.0,
            },
            "honest_rejection_streaks": {
                "n_honest_edges": len(benign_edges),
                "mean_rejected_rounds": round(float(np.mean([s[0] for s in benign_edge_stats])), 2) if benign_edge_stats else 0.0,
                "mean_max_consec_streak": round(float(np.mean([s[1] for s in benign_edge_stats])), 2) if benign_edge_stats else 0.0,
                "isolated_single_rejections": int(sum(s[3] for s in benign_edge_stats)),
                "frac_edges_with_ge1_rej_pct": round(
                    100.0 * sum(1 for s in benign_edge_stats if s[0] >= 1) / max(len(benign_edge_stats), 1), 2
                )
                if benign_edge_stats
                else 0.0,
                "frac_edges_with_streak_ge3_pct": round(
                    100.0 * sum(1 for s in benign_edge_stats if s[1] >= 3) / max(len(benign_edge_stats), 1), 2
                )
                if benign_edge_stats
                else 0.0,
                "frac_edges_with_streak_ge5_pct": round(
                    100.0 * sum(1 for s in benign_edge_stats if s[1] >= 5) / max(len(benign_edge_stats), 1), 2
                )
                if benign_edge_stats
                else 0.0,
            },
        }
        if self.net_sec_list:
            mean_wire_mb = (
                round(float(np.mean(self.wire_bytes_recv_list)) / 1e6, 2)
                if self.wire_bytes_recv_list
                else 0.0
            )
            metrics["measured_transport_sec"] = {
                "wire_mb_per_peer_round": mean_wire_mb,
                "d2h_sec_per_peer": summarize_timings(self.d2h_sec_list),
                "net_sec_per_peer": summarize_timings(self.net_sec_list),
                "h2d_sec_per_peer": summarize_timings(self.h2d_sec_list),
            }
        return metrics

    def finalize_summary(
        self,
        config_dict: dict[str, object],
        resolved_device: str,
        topo_info: dict[str, object],
        total_wall_sec: float,
        had_nan: bool,
        peak_gpu_alloc_mib: float | None = None,
        peak_gpu_reserved_mib: float | None = None,
    ) -> dict[str, object]:
        """Assemble the final run summary dictionary matching the summary.json schema."""
        eval_records = [rec for rec in self.history if "benign_test_acc_mean" in rec]
        final_eval = eval_records[-1]
        best_eval = max(eval_records, key=lambda rec: float(rec["benign_test_acc_mean"]))

        ben_scores = [float(r["benign_score_mean"]) for r in self.history if r["benign_score_mean"] is not None]
        byz_scores = [float(r["byz_score_mean"]) for r in self.history if r["byz_score_mean"] is not None]
        mean_ben = float(np.mean(ben_scores)) if ben_scores else None
        mean_byz = float(np.mean(byz_scores)) if byz_scores else None
        if mean_ben is not None and math.isnan(mean_ben):
            had_nan = True

        return {
            "config": config_dict,
            "resolved_device": resolved_device,
            "param_count": self.param_count,
            "topology": topo_info,
            "total_wall_sec": total_wall_sec,
            "had_nan": had_nan,
            "final_benign_test_acc_mean": final_eval["benign_test_acc_mean"],
            "final_benign_test_acc_std": final_eval["benign_test_acc_std"],
            "final_consensus_test_acc": final_eval["consensus_test_acc"],
            "best_benign_test_acc_mean": best_eval["benign_test_acc_mean"],
            "best_round": best_eval["round"],
            "fpr_cumulative": (self.cum_benign_zero / self.cum_benign_pairs) if self.cum_benign_pairs > 0 else 0.0,
            "fpr_final_round": self.history[-1]["fpr_round"],
            "byz_accept_cumulative": (self.cum_byz_accept / self.cum_byz_pairs) if self.cum_byz_pairs > 0 else 0.0,
            "mean_benign_score": mean_ben,
            "mean_byz_score": mean_byz,
            "system_metrics": self.build_system_metrics(
                peak_gpu_alloc_mib=peak_gpu_alloc_mib,
                peak_gpu_reserved_mib=peak_gpu_reserved_mib,
            ),
        }
