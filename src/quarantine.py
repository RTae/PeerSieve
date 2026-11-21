"""Per-edge quarantine, EMA backoff, permanent hard-fault bans, and referrals.

Implements PeerSieve Section 3.4 (Eq. 5):
  R_{ij}^{(t)} = beta * R_{ij}^{(t-1)} + (1 - beta) * c_hat_{ij}^{(t)}   (beta = 0.7)
  T_backoff(k) = min(2^k, T_max)                                         (T_max = 16)

State machine per neighbor edge:
- ACTIVE: receives sketches every round; single failed sketch skips one transfer.
  Three consecutive rounds with R_{ij} < 0, or a soft audit failure, moves the
  edge to PROBATION.
- PROBATION: no full payload requests; scheduled sketch probe after T_backoff(k)
  rounds. A passing probe restores ACTIVE; a failed probe increments k and
  extends probation.
- BANNED: triggered immediately by any hard protocol fault (invalid encoding,
  size/header violation, nonfinite values, or sketch mismatch). Permanent for
  the rest of the run; banned peers are never reinstated via referrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.audit import AuditOutcome, AuditResult, ScreenResult


class LinkState(str, Enum):
    ACTIVE = "active"
    PROBATION = "probation"
    BANNED = "banned"


@dataclass(frozen=True)
class QuarantineConfig:
    beta: float = 0.7
    neg_streak_limit: int = 3
    t_max: int = 16
    min_active_degree: int = 2
    max_neighbors: int = 6
    enabled: bool = True


@dataclass
class LinkRecord:
    peer_id: int
    state: LinkState = LinkState.ACTIVE
    ema_score: float = 0.0
    neg_ema_streak: int = 0
    probation_count: int = 0
    next_probe_round: int | None = None
    last_reason: str = "init"
    ban_reason: str | None = None
    sketches_seen: int = 0
    payloads_requested: int = 0
    payloads_accepted: int = 0
    soft_faults: int = 0
    hard_faults: int = 0
    recoveries: int = 0


def backoff_rounds(k: int, t_max: int = 16) -> int:
    """Compute T_backoff(k) = min(2^k, T_max) for k >= 1."""
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    return min(1 << min(k, 30), int(t_max))


class PeerQuarantineTable:
    """Per-receiver neighbor state table enforcing probation, bans, and referrals."""

    def __init__(
        self,
        self_id: int,
        initial_neighbors: Iterable[int],
        cfg: QuarantineConfig | None = None,
    ) -> None:
        self.self_id = int(self_id)
        self.cfg = cfg if cfg is not None else QuarantineConfig()
        self.links: dict[int, LinkRecord] = {}
        self.banned_peers: set[int] = set()
        self.referrals_added: list[tuple[int, int]] = []  # (round_idx, peer_id)

        for nid in initial_neighbors:
            pid = int(nid)
            if pid == self.self_id or pid in self.links:
                continue
            if len(self.links) >= self.cfg.max_neighbors:
                raise ValueError(
                    f"Initial neighbors exceed max_neighbors={self.cfg.max_neighbors} for peer {self.self_id}"
                )
            self.links[pid] = LinkRecord(peer_id=pid, state=LinkState.ACTIVE)

    def active_peers(self) -> list[int]:
        return sorted(pid for pid, rec in self.links.items() if rec.state == LinkState.ACTIVE)

    def probation_peers(self) -> list[int]:
        return sorted(pid for pid, rec in self.links.items() if rec.state == LinkState.PROBATION)

    def live_neighbor_count(self) -> int:
        """Count of non-banned neighbor links currently occupying neighbor slots."""
        return sum(1 for rec in self.links.values() if rec.state != LinkState.BANNED)

    def is_banned(self, peer_id: int) -> bool:
        return int(peer_id) in self.banned_peers

    def get_record(self, peer_id: int) -> LinkRecord | None:
        return self.links.get(int(peer_id))

    def should_accept_sketch(self, peer_id: int, round_idx: int) -> bool:
        """Return True if receiver should read/screen a sketch from peer_id in round_idx."""
        pid = int(peer_id)
        if pid in self.banned_peers:
            return False
        rec = self.links.get(pid)
        if rec is None:
            return False
        if rec.state == LinkState.BANNED:
            return False
        if rec.state == LinkState.ACTIVE:
            return True
        # PROBATION: only probe on or after scheduled next_probe_round.
        return rec.next_probe_round is not None and int(round_idx) >= rec.next_probe_round

    def peers_to_screen(self, round_idx: int) -> list[int]:
        """Sorted list of neighbors whose sketches should be screened in round_idx."""
        return sorted(pid for pid in self.links if self.should_accept_sketch(pid, round_idx))

    def _ban_peer(self, peer_id: int, reason: str) -> None:
        pid = int(peer_id)
        self.banned_peers.add(pid)
        rec = self.links.get(pid)
        if rec is not None:
            rec.state = LinkState.BANNED
            rec.ban_reason = reason
            rec.last_reason = reason
            rec.next_probe_round = None
            rec.hard_faults += 1

    def _enter_probation(self, rec: LinkRecord, round_idx: int, reason: str) -> None:
        rec.state = LinkState.PROBATION
        rec.probation_count += 1
        wait = backoff_rounds(rec.probation_count, t_max=self.cfg.t_max)
        rec.next_probe_round = int(round_idx) + wait
        rec.last_reason = reason

    def on_screen_result(self, peer_id: int, round_idx: int, screen_res: ScreenResult) -> bool:
        """Update link state after pre-transfer sketch screening; return True to request payload."""
        pid = int(peer_id)
        if pid in self.banned_peers:
            return False
        rec = self.links.get(pid)
        if rec is None or rec.state == LinkState.BANNED:
            return False

        rec.sketches_seen += 1

        if screen_res.outcome == AuditOutcome.HARD_FAULT:
            self._ban_peer(pid, screen_res.reason)
            return False

        if rec.state == LinkState.PROBATION:
            # Must be a scheduled probe; no full payloads are requested during probation.
            if rec.next_probe_round is None or int(round_idx) < rec.next_probe_round:
                return False
            if screen_res.outcome != AuditOutcome.ZERO_SKIP:
                rec.ema_score = (
                    self.cfg.beta * rec.ema_score + (1.0 - self.cfg.beta) * float(screen_res.sketch_cosine)
                )
            if screen_res.request_payload:
                rec.state = LinkState.ACTIVE
                rec.neg_ema_streak = 0
                rec.probation_count = 0
                rec.next_probe_round = None
                rec.last_reason = "probe_recovered"
                rec.recoveries += 1
                # Passing probe restores link to ACTIVE; payload transfers resume next round.
                return False
            self._enter_probation(rec, round_idx, reason=f"probe_failed:{screen_res.reason}")
            return False

        # ACTIVE link handling.
        if screen_res.outcome == AuditOutcome.ZERO_SKIP:
            rec.last_reason = "zero_sketch_skip"
            return False

        rec.ema_score = self.cfg.beta * rec.ema_score + (1.0 - self.cfg.beta) * float(screen_res.sketch_cosine)
        if rec.ema_score < 0.0:
            rec.neg_ema_streak += 1
        else:
            rec.neg_ema_streak = 0

        if self.cfg.enabled and rec.neg_ema_streak >= self.cfg.neg_streak_limit:
            self._enter_probation(rec, round_idx, reason="neg_ema_streak")
            return False

        if screen_res.request_payload:
            rec.payloads_requested += 1
            rec.last_reason = "payload_requested"
            return True

        rec.last_reason = "single_sketch_skip"
        return False

    def on_audit_result(self, peer_id: int, round_idx: int, audit_res: AuditResult) -> None:
        """Update link state after Pass 1 / Pass 2 audit outcome."""
        pid = int(peer_id)
        rec = self.links.get(pid)
        if rec is None:
            if audit_res.outcome == AuditOutcome.HARD_FAULT:
                self.banned_peers.add(pid)
            return

        if audit_res.outcome == AuditOutcome.HARD_FAULT:
            self._ban_peer(pid, audit_res.reason)
            return
        if audit_res.outcome == AuditOutcome.SOFT_FAULT:
            rec.soft_faults += 1
            if self.cfg.enabled:
                self._enter_probation(rec, round_idx, reason=f"soft_audit:{audit_res.reason}")
            else:
                rec.last_reason = f"soft_audit_no_quarantine:{audit_res.reason}"
            return
        if audit_res.outcome == AuditOutcome.ZERO_SKIP:
            rec.last_reason = "zero_payload_skip"
            return
        if audit_res.outcome == AuditOutcome.ACCEPTED:
            rec.payloads_accepted += 1
            rec.probation_count = 0
            rec.last_reason = "accepted"

    def needs_referral(self) -> bool:
        """True when active neighbor degree falls below min_active_degree (default 2)."""
        return self.cfg.enabled and len(self.active_peers()) < self.cfg.min_active_degree

    def recommend_neighbor(self, requester_id: int, exclude: set[int] | None = None) -> int | None:
        """Select one active neighbor to recommend to requester_id."""
        excluded = {self.self_id, int(requester_id)}
        if exclude:
            excluded.update(int(x) for x in exclude)
        candidates = [
            rec for pid, rec in self.links.items() if rec.state == LinkState.ACTIVE and pid not in excluded
        ]
        if not candidates:
            return None
        # Prefer highest EMA score, deterministic tie-break on peer_id.
        candidates.sort(key=lambda r: (-r.ema_score, r.peer_id))
        return candidates[0].peer_id

    def can_consider_referral(self, candidate_id: int) -> bool:
        """Check whether candidate_id is eligible for referral vetting."""
        cid = int(candidate_id)
        if not self.cfg.enabled:
            return False
        if cid == self.self_id or cid in self.banned_peers:
            return False
        existing = self.links.get(cid)
        if existing is not None:
            return False
        return self.live_neighbor_count() < self.cfg.max_neighbors

    def admit_referral(self, candidate_id: int, round_idx: int, screen_res: ScreenResult) -> bool:
        """Admit a vetted referral candidate if its sketch passes; ban on hard fault."""
        cid = int(candidate_id)
        if not self.can_consider_referral(cid):
            return False
        if screen_res.outcome == AuditOutcome.HARD_FAULT:
            self.banned_peers.add(cid)
            return False
        if not screen_res.request_payload:
            return False

        # Respect original max_neighbors cap; if dict has banned entries, keep banned_peers set
        # authoritative and prune one banned record from links if len(links) >= max_neighbors.
        if len(self.links) >= self.cfg.max_neighbors:
            banned_in_table = sorted(pid for pid, rec in self.links.items() if rec.state == LinkState.BANNED)
            if banned_in_table:
                del self.links[banned_in_table[0]]
            else:
                return False

        self.links[cid] = LinkRecord(
            peer_id=cid,
            state=LinkState.ACTIVE,
            ema_score=float(screen_res.sketch_cosine),
            last_reason="admitted_referral",
            sketches_seen=1,
        )
        self.referrals_added.append((int(round_idx), cid))
        return True

    def summary_counts(self) -> dict[str, int]:
        return {
            "active": len(self.active_peers()),
            "probation": len(self.probation_peers()),
            "banned": len(self.banned_peers),
            "referrals_added": len(self.referrals_added),
        }
