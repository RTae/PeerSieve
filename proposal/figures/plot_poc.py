#!/usr/bin/env python3
"""Figure 12 of the KVMend proposal: measured results of the proof of concept.

Data: job 326326 on node octave (8x A100-PCIE-40GB), Llama-3.1-8B at TP=8, read from the proof of concept's records
in ../../kvmend/results/kvmend-cmp8-326326/ (compare/summary.json and overhead/overhead.json). The script writes the
numbers it plots to poc_results.csv, prints every one of them, and draws the figure from that file.

Run with a Python that has matplotlib (here /opt/homebrew/bin/python3.8):
    python3.8 -I -B plot_poc.py
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import figstyle as fs  # noqa: E402
from figroles import HATCH, SERIES  # noqa: E402

RUN = HERE.parent.parent / "kvmend" / "results" / "kvmend-cmp8-326326"
H7 = 0.03  # target: fault-free throughput within 3% of parity alone

# (run in the records, label in the figure, color role); top to bottom in panel (a)
RESPONSES = [
    ("silent", "No checks\n(GhostServe as published)", None),
    ("gs_gpu", "GhostServe-style,\ntold the GPU", "GhostServe, told the GPU"),
    ("gs_chunk", "GhostServe-style, told\nthe GPU and the chunk", "GhostServe, told the GPU and the chunk"),
    ("reprefill_all", "Detect, then\nre-prefill all of R", "re-prefill the whole request"),
    ("kvmend", "KVMend", "KVMend, damage on one GPU"),
]
# (mode, segment, label in the figure, color role) for panel (b)
MODES = [
    ("parity", "parity", "parity alone\n(GhostServe-\nstyle)", "parity alone (GhostServe-style)"),
    ("kvmend", "scrub0", "KVMend,\nno scrub", "KVMend, damage on one GPU"),
    ("kvmend", "scrub1024", "KVMend,\nscrub 1,024\npieces/step", "KVMend, damage on one GPU"),
    ("kvmend", "scrub4096", "KVMend,\nscrub 4,096\npieces/step", "KVMend, damage on one GPU"),
]


def size(n):
    for unit, k in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= k:
            return f"{n / k:.4g} {unit}"
    return f"{n:.0f} B"


def measurements():
    """Rows of (panel, item, metric, value) from the proof of concept's records."""
    comp = {r["run"]: r for r in json.loads((RUN / "compare" / "summary.json").read_text())["ghostserve_comparison"]}
    over = json.loads((RUN / "overhead" / "overhead.json").read_text())["rows"]
    rows = []
    for run, _, _ in RESPONSES:
        r = comp[run]
        rows += [("a", run, "notices", int(bool(r["found"]))),
                 ("a", run, "recover_s", r["recover_s"] if r["recover_s"] is not None else -1),
                 ("a", run, "bytes_read", r["bytes_read"]),
                 ("a", run, "tokens_recomputed", r["tokens_recomputed"]),
                 ("a", run, "correct", int(bool(r["equals_reference"])))]
    for x in over:
        key = f"{x['mode']}:{x['segment']}"
        rows += [("b", key, "out_tok_s", x["out_tok_s"]),
                 ("b", key, "vs_plain", x["vs_plain"] if x["vs_plain"] is not None else 0.0),
                 ("b", key, "vs_parity", x["vs_parity"] if x["vs_parity"] is not None else 0.0),
                 ("b", key, "scrub_bytes_per_s", x.get("scrub_bytes_per_s") or 0.0)]
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "item", "metric", "value"])
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return {(r["panel"], r["item"], r["metric"]): float(r["value"]) for r in csv.DictReader(fh)}


def main():
    data_file = HERE / "poc_results.csv"
    write_csv(measurements(), data_file)
    d = read_csv(data_file)

    fs.apply()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    wrong = HATCH["next tokens still wrong after recovery"]

    # (a) recovery time per response, log axis; hatched = the output is still wrong after the recovery
    ys = list(range(len(RESPONSES)))[::-1]
    for y, (run, label, role) in zip(ys, RESPONSES):
        t = d[("a", run, "recover_s")]
        ok = d[("a", run, "correct")] == 1
        read, toks = d[("a", run, "bytes_read")], d[("a", run, "tokens_recomputed")]
        if t < 0:  # never notices, so never recovers
            ax_a.text(0.012, y, "never notices the damage", va="center", fontsize=13, color=fs.PALETTE["dark"])
            continue
        ax_a.barh(y, t, height=0.6, color=SERIES[role], edgecolor=fs.PALETTE["ink"], lw=1.8,
                  hatch=None if ok else wrong, zorder=2)
        parts = [f"{t:.2f} s" if t < 10 else f"{t:.1f} s"]
        if read:
            parts.append(f"reads {size(read)}")
        if toks:
            parts.append(f"recomputes {toks:,.0f} tokens")
        ax_a.text(t * 1.25, y, ", ".join(parts), va="center", fontsize=13)
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.01, 2000)
    ax_a.set_ylim(-0.6, len(RESPONSES) - 0.4)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([label for _, label, _ in RESPONSES], fontsize=13)
    ax_a.set_xlabel("recovery time after detection (s, log scale)")
    ax_a.legend(handles=[Patch(facecolor="white", edgecolor=fs.PALETTE["ink"], hatch=wrong,
                               label="next tokens still wrong after recovery"),
                         Patch(facecolor="white", edgecolor=fs.PALETTE["ink"], label="next tokens match a healthy server")],
                loc="upper right", fontsize=12)
    ax_a.set_title("(a) one bit flip, five responses", loc="left", fontsize=16)

    # (b) fault-free throughput against plain SGLang, with H7's target
    xs = range(len(MODES))
    plain = d[("b", "plain:plain", "out_tok_s")]
    parity = d[("b", "parity:parity", "out_tok_s")]
    for x, (mode, seg, label, role) in zip(xs, MODES):
        v = 100 * d[("b", f"{mode}:{seg}", "vs_plain")]
        ax_b.bar(x, v, width=0.62, color=SERIES[role], edgecolor=fs.PALETTE["ink"], lw=1.8, zorder=2)
        ax_b.text(x, v - 0.25, f"{v:.1f}%".replace("-", "\u2212"), ha="center", va="top", fontsize=13)
    target = 100 * ((1 - H7) * parity / plain - 1)
    ax_b.axhline(target, color=fs.PALETTE["dark"], lw=1.6, linestyle=":", zorder=3,
                 label="H7 target: 3% below parity alone")
    ax_b.legend(loc="lower left", fontsize=12)
    ax_b.axhline(0, color=fs.PALETTE["ink"], lw=1.6)
    ax_b.text(-0.45, 0.25, f"plain SGLang: {plain:.1f} output tokens/s", fontsize=12, color=fs.PALETTE["dark"])
    ax_b.set_xticks(list(xs))
    ax_b.set_xticklabels([label for _, _, label, _ in MODES], fontsize=12)
    ax_b.set_xlim(-0.55, len(MODES) - 0.45)
    ax_b.set_ylim(-7.6, 1.2)
    ax_b.set_ylabel("output throughput vs plain SGLang (%)")
    ax_b.set_title("(b) cost without faults, earlier version", loc="left", fontsize=16)

    fig.subplots_adjust(left=0.175, right=0.985, bottom=0.22, top=0.92, wspace=0.3)
    fs.save(fig, HERE, "fig12_poc")

    for run, label, _ in RESPONSES:
        t = d[("a", run, "recover_s")]
        print(f"{label.replace(chr(10), ' ')}: notices={int(d[('a', run, 'notices')])}, "
              f"recovery={'never' if t < 0 else f'{t:.4f} s'}, read={size(d[('a', run, 'bytes_read')])}, "
              f"tokens={d[('a', run, 'tokens_recomputed')]:,.0f}, correct={int(d[('a', run, 'correct')])}")
    for key in ("plain:plain", "parity:parity", "kvmend:scrub0", "kvmend:scrub1024", "kvmend:scrub4096"):
        print(f"{key}: {d[('b', key, 'out_tok_s')]:.1f} output tok/s, vs plain {100 * d[('b', key, 'vs_plain')]:+.1f}%, "
              f"vs parity {100 * d[('b', key, 'vs_parity')]:+.1f}%, scrub {d[('b', key, 'scrub_bytes_per_s')] / 1e9:.2f} GB/s")
    print(f"H7 target line: {target:.1f}% vs plain")


if __name__ == "__main__":
    main()
