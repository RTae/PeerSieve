#!/usr/bin/env python3
"""Figure 13 of the KVMend proposal: KVMend against two retries, and bursts, on the main setup.

Data: Llama-3.1-8B at TP=8 on node octave (8x A100-PCIE-40GB), from the proof of concept's records:
  panel (a) job 326442, commit d33e5ea: ../../kvmend/results/kvmend-main5-326442/{incident,incident_early}/summary.json
  panel (b) job 326412: ../../kvmend/results/kvmend-main-326412/burst/summary.json
The script writes the numbers it plots to results_tp8.csv, prints every one of them, and draws the figure from that
file.

Run with a Python that has matplotlib (here /opt/homebrew/bin/python3.8):
    python3.8 -I -B plot_results.py
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import figstyle as fs  # noqa: E402
from figroles import DASH, HOLLOW, SERIES  # noqa: E402

RUN = HERE.parent.parent / "kvmend" / "results" / "kvmend-main-326412"
RUN_A = HERE.parent.parent / "kvmend" / "results" / "kvmend-main5-326442"

# (record folder, position label, fault kind, label in the figure)
GROUPS = [
    ("incident", "end", "unwritten", "never written,\nlast 4 pages"),
    ("incident", "end", "late_bypass", "late write,\nlast 4 pages"),
    ("incident_early", "page64", "unwritten", "never written,\npage 64"),
    ("incident_early", "page64", "late_bypass", "late write,\npage 64"),
]
KV_EVERY = "KVMend, damage on every GPU"
KEEP = "re-prefill from the damaged page"
RETRY = "re-prefill the whole request"
REBUILD = "KVMend, damage on one GPU"
RECOMPUTE = "KVMend recomputing pages"


def measurements():
    """Rows of (panel, item, metric, value) from the proof of concept's records."""
    burst = json.loads((RUN / "burst" / "summary.json").read_text())
    rows = []
    for folder, pos, kind, _ in GROUPS:
        inc = json.loads((RUN_A / folder / "summary.json").read_text())
        f = inc["faults"][kind]
        key = f"{pos}:{kind}"
        rows += [("a", key, "reference_s", inc["reference_seconds"]),
                 ("a", key, "kvmend_extra_s", f["kvmend_extra_s"]),
                 ("a", key, "kvmend_text_right", int(bool(f["kvmend_text_right"] or f["kvmend_equals_control"]))),
                 ("a", key, "recomputed_tokens", sum(x["tokens"] for x in f["recomputed"] or [])),
                 ("a", key, "keep_extra_s", f["retry_keep_extra_s"]),
                 ("a", key, "keep_from_position", f["retry_keep_cached_prefix"]),
                 ("a", key, "retry_extra_s", f["retry_extra_s"])]
    for r in burst["rows"]:
        if r["layers"] != 1:  # the sweep over pages in one layer; the 8-layer burst is in Table 15 only
            continue
        key = f"{r['pages']}x1"
        rows += [("b", key, "pieces", r["pieces"]),
                 ("b", key, "rebuild_ms", 1e3 * r["rebuild"]["rebuild_s"]),
                 ("b", key, "rebuild_bit_exact", int(bool(r["rebuild"]["bit_exact"]))),
                 ("b", key, "recompute_ms", 1e3 * r["recompute"]["recompute_s"]),
                 ("b", key, "recompute_tokens", r["recompute"]["tokens_recomputed"]),
                 ("b", key, "texts_right", int(bool(r["rebuild"]["final_text_right"]
                                                    and r["recompute"]["final_text_right"])))]
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "item", "metric", "value"])
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return {(r["panel"], r["item"], r["metric"]): float(r["value"]) for r in csv.DictReader(fh)}


def secs(t):
    return f"{t:.2f} s"


def millis(t):
    return f"{t:.1f} ms" if t < 1000 else f"{t / 1e3:.1f} s"


def main():
    data_file = HERE / "results_tp8.csv"
    write_csv(measurements(), data_file)
    d = read_csv(data_file)

    fs.apply()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 7.0), gridspec_kw=dict(width_ratios=[1.35, 1]))

    # (a) extra time to the right text: KVMend, a retry that keeps the clean prefix, and a flushed retry; log axis
    width = 0.27
    for x, (_, pos, kind, _) in enumerate(GROUPS):
        key = f"{pos}:{kind}"
        for dx, metric, role in ((-width, "kvmend_extra_s", KV_EVERY), (0.0, "keep_extra_s", KEEP),
                                 (width, "retry_extra_s", RETRY)):
            v = d[("a", key, metric)]
            ax_a.bar(x + dx, v, width, color=SERIES[role], edgecolor=fs.PALETTE["ink"], lw=1.8, zorder=2)
            ax_a.text(x + dx, v * 1.12, secs(v), ha="center", va="bottom", fontsize=10.5, rotation=90)
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.1, 200)
    ax_a.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax_a.set_xticks(range(len(GROUPS)))
    ax_a.set_xticklabels([label for _, _, _, label in GROUPS], fontsize=12.5)
    ax_a.set_ylabel("extra time to the right text (s, log scale)")
    ax_a.legend(handles=[Patch(facecolor=SERIES[KV_EVERY], edgecolor=fs.PALETTE["ink"], label="KVMend"),
                         Patch(facecolor=SERIES[KEEP], edgecolor=fs.PALETTE["ink"],
                               label="retry that keeps the clean prefix"),
                         Patch(facecolor=SERIES[RETRY], edgecolor=fs.PALETTE["ink"],
                               label="abort and retry, cache flushed")],
                loc="upper left", fontsize=12)
    ax_a.set_title("(a) KVMend against two retries", loc="left", fontsize=16)

    # (b) repair time against burst size on one GPU: rebuild from parity against recompute, log-log
    keys = sorted((k for (p, k, m) in d if p == "b" and m == "pieces"), key=lambda k: d[("b", k, "pieces")])
    pieces = [d[("b", k, "pieces")] for k in keys]
    rebuild = [d[("b", k, "rebuild_ms")] for k in keys]
    recompute = [d[("b", k, "recompute_ms")] for k in keys]
    ax_b.plot(pieces, rebuild, color=SERIES[REBUILD], lw=2.8, marker="o", ms=9, zorder=3)
    hollow = RECOMPUTE in HOLLOW
    ax_b.plot(pieces, recompute, color=SERIES[RECOMPUTE], lw=2.8, linestyle=DASH[RECOMPUTE], marker="o", ms=9,
              markerfacecolor="white" if hollow else SERIES[RECOMPUTE], markeredgewidth=2.2, zorder=3)
    for x, y in zip(pieces, rebuild):
        ax_b.text(x, y / 1.6, millis(y), ha="center", va="top", fontsize=11.5)
    for x, y in zip(pieces, recompute):
        ax_b.text(x, y * 1.45, millis(y), ha="center", va="bottom", fontsize=11.5)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.5, 2500)
    ax_b.set_ylim(0.1, 40000)
    ax_b.set_xticks(pieces)
    ax_b.set_xticklabels([f"{p:,.0f}" for p in pieces])
    ax_b.set_xlabel("damaged pieces, one layer (log scale)")
    ax_b.set_ylabel("repair time (ms, log scale)")
    ax_b.legend(handles=[Line2D([], [], color=SERIES[REBUILD], lw=2.8, marker="o", ms=9,
                                label="KVMend, rebuild from parity"),
                         Line2D([], [], color=SERIES[RECOMPUTE], lw=2.8, linestyle=DASH[RECOMPUTE], marker="o", ms=9,
                                markerfacecolor="white", markeredgewidth=2.2, label="KVMend, recompute the pages")],
                loc="upper left", fontsize=12)
    ax_b.set_title("(b) a burst on one GPU", loc="left", fontsize=16)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.93, wspace=0.22)
    fs.save(fig, HERE, "fig13_results")

    for _, pos, kind, label in GROUPS:
        key = f"{pos}:{kind}"
        print(f"{label.replace(chr(10), ' ')} (reference {secs(d[('a', key, 'reference_s')])}): "
              f"KVMend +{secs(d[('a', key, 'kvmend_extra_s')])} (recomputed {d[('a', key, 'recomputed_tokens')]:,.0f} "
              f"tokens; text right={int(d[('a', key, 'kvmend_text_right')])}), "
              f"retry keeping the prefix +{secs(d[('a', key, 'keep_extra_s')])} (from position "
              f"{d[('a', key, 'keep_from_position')]:,.0f}), flushed retry +{secs(d[('a', key, 'retry_extra_s')])}")
    for k in keys:
        print(f"burst {k} ({d[('b', k, 'pieces')]:,.0f} pieces): rebuild {millis(d[('b', k, 'rebuild_ms')])} "
              f"(bit for bit={int(d[('b', k, 'rebuild_bit_exact')])}), recompute {millis(d[('b', k, 'recompute_ms')])} "
              f"of {d[('b', k, 'recompute_tokens')]:,.0f} tokens, texts right={int(d[('b', k, 'texts_right')])}")


if __name__ == "__main__":
    main()
