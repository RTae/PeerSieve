#!/usr/bin/env python3
"""Figure 11 of the KVMend proposal: expected repair work and the scrub trade-off.

All numbers are estimates from the proposal's cost model (section 4.9), for the running example:
Llama-3.1-8B at TP=8, request R with a 32,768-token prompt, page 64 (tokens 1,024-1,039) damaged,
R decoding 40 tokens per second. The script writes the estimates to expected_costs.csv, prints
every number it plots, and draws the figure from that file.

Run with a Python that has matplotlib (here /opt/homebrew/bin/python3.8):
    python3.8 -I -B plot_expected.py
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt  # noqa: E402

import figstyle as fs  # noqa: E402
from figroles import DASH, SERIES  # noqa: E402

PROMPT, PAGE_START, PAGE_TOKENS = 32768, 1024, 16
DECODE_RATE = 40.0          # tokens per second for R
KV_PER_GPU_GB = 20.0        # KV cache per GPU covered by one scrub pass
HBM_GBPS = 1555.0           # A100-40GB memory bandwidth
DELAYS = [round(0.025 * k, 3) for k in range(0, 121)]          # 0 to 3 s
SHARES = [round(0.005 + 0.0025 * k, 4) for k in range(0, 19)]  # 0.5% to 5%


def estimates():
    """Rows of (panel, x, series, y)."""
    rows = []
    for d in DELAYS:
        tainted = DECODE_RATE * d
        rows += [("a", d, "KVMend, damage on one GPU", tainted),
                 ("a", d, "KVMend, damage on every GPU", PAGE_TOKENS + tainted),
                 ("a", d, "re-prefill from the damaged page", PROMPT - PAGE_START + tainted),
                 ("a", d, "re-prefill the whole request", PROMPT + tainted)]
    for s in SHARES:
        rows.append(("b", s, "scrub pass time (s)", KV_PER_GPU_GB / (s * HBM_GBPS)))
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "x", "series", "y"])
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return [(r["panel"], float(r["x"]), r["series"], float(r["y"])) for r in csv.DictReader(fh)]


def main():
    data_file = HERE / "expected_costs.csv"
    write_csv(estimates(), data_file)
    rows = read_csv(data_file)

    fs.apply()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.4))

    # (a) tokens recomputed against detection delay; log axis, so zero work is marked separately
    for name in SERIES:
        pts = [(x, y) for p, x, s, y in rows if p == "a" and s == name and y >= 1]
        xs, ys = zip(*pts)
        # the two re-prefill lines nearly coincide on a log axis: draw the gray one dashed on top
        style = (0, (6, 4)) if name == "re-prefill the whole request" else DASH.get(name, "solid")
        ax_a.plot(xs, ys, color=SERIES[name], lw=3 if name != "re-prefill the whole request" else 2.2,
                  linestyle=style, label=name, zorder=3 if name == "re-prefill the whole request" else 2)
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.8, 1e5)
    ax_a.set_xlim(-0.05, 3.05)
    ax_a.axvline(1.0, color=fs.PALETTE["dark"], lw=1.5, linestyle=":")
    ax_a.text(1.05, 5.0e4, "1 s scrub pass (1.3% of bandwidth)", fontsize=12, color=fs.PALETTE["dark"])
    ax_a.text(0.05, 1.3e4, "both re-prefill lines: about 32,000 tokens", fontsize=12, color=fs.PALETTE["dark"])
    ax_a.plot([0.0], [1.0], marker="o", ms=10, color=SERIES["KVMend, damage on one GPU"], clip_on=False)
    ax_a.annotate("0 tokens when a check before\nreuse catches it (drawn at 1:\nlog axis)", xy=(0.02, 1.0), xytext=(1.75, 1.05),
                  fontsize=12, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_a.set_xlabel("time from the last good check to detection (s)")
    ax_a.set_ylabel("tokens recomputed (log scale)")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.76), fontsize=12)
    ax_a.set_title("(a) repair work", loc="left", fontsize=16)

    # (b) scrub share against pass time, with suspect tokens per reader on the right axis
    pts = [(x, y) for p, x, s, y in rows if p == "b"]
    xs, ys = zip(*pts)
    pct = [100 * x for x in xs]
    ax_b.plot(pct, ys, color=SERIES["KVMend, damage on one GPU"], lw=3)
    for share, tx, ty in ((0.01, 1.8, 2.3), (0.02, 2.6, 0.95)):
        t = KV_PER_GPU_GB / (share * HBM_GBPS)
        ax_b.plot([100 * share], [t], marker="o", ms=9, color=SERIES["KVMend, damage on one GPU"])
        ax_b.annotate(f"{100 * share:.0f}%: {t:.2f} s, up to {DECODE_RATE * t:.0f} tokens", xy=(100 * share, t),
                      xytext=(tx, ty), fontsize=13, arrowprops=dict(arrowstyle="->", lw=1.2))
    # the running example: a 1 s pass, the one gold callout of this figure
    share_1s = KV_PER_GPU_GB / (1.0 * HBM_GBPS)
    ax_b.plot([100 * share_1s], [1.0], marker="o", ms=12, color=fs.PALETTE["highlight"],
              markeredgecolor=fs.PALETTE["ink"], zorder=4)
    ax_b.annotate(f"running example: {100 * share_1s:.1f}%,\n1.0 s, up to {DECODE_RATE:.0f} tokens",
                  xy=(100 * share_1s, 1.0), xytext=(2.3, 1.5), fontsize=13, arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_b.set_xlim(0.3, 5.2)
    ax_b.set_ylim(0, 2.8)
    ax_b.set_xlabel("scrub share of memory bandwidth (%)")
    ax_b.set_ylabel("one scrub pass over 20 GB (s)")
    sec = ax_b.secondary_yaxis("right", functions=(lambda t: DECODE_RATE * t, lambda n: n / DECODE_RATE))
    sec.set_ylabel("suspect tokens per reader, worst case")
    ax_b.set_title("(b) scrub trade-off", loc="left", fontsize=16)

    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.16, top=0.9, wspace=0.42)
    fs.save(fig, HERE, "fig11_expected")

    for d in (0.0, 0.5, 1.0, 2.0, 3.0):
        vals = {s: y for p, x, s, y in rows if p == "a" and x == d}
        print(f"delay {d:.1f} s: " + "; ".join(f"{k} = {v:,.0f}" for k, v in vals.items()))
    for share in (0.005, 0.01, 0.02, 0.05):
        t = [y for p, x, s, y in rows if p == "b" and abs(x - share) < 1e-9][0]
        print(f"scrub {100 * share:.1f}%: pass {t:.2f} s, up to {DECODE_RATE * t:.0f} suspect tokens per reader")


if __name__ == "__main__":
    main()
