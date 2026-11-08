#!/usr/bin/env python3
"""Build the KVMend figures from one Excalidraw scene.

The figures draw the mechanism itself, not boxes of text: KV pieces as cells and page
columns, tags as small badges, checks as a magnifier, the parity rebuild as an XOR gate
with its bits, the spread of damage as attention lines, and the repair as a rollback of
the suspect tokens. Every figure follows the running example: request R, page 64
(tokens 1,024-1,039), layer 12, GPU 2, last good check at step 9,000, detection at
step 9,040, 40 suspect tokens.

Colors keep one meaning in every figure:
  blue    KV on a GPU (dark blue: just written)     amber   host memory and parity
  green   repaired, or shown to be clean            red     damaged, or a failed check
  violet  tags and checks                           orange  suspect tokens, and what is done to them
  gray    free, retired or busy

A frame is 1,560 units wide, so at a 6.5-inch page width one unit prints at 0.3 pt.
Labels are at least 17-18 units (about 5.4 pt), notes 21 and panel titles 26.

Frames (one per proposal figure, in the order the text uses them):

  frame_story     (1) silent damage and what KVMend does
  frame_vocab     (2) pages, pieces, slots and stripes
  frame_timeline  (3) the running example on a timeline
  frame_normal    (4) normal operation: write, seal, encode
  frame_detect    (5) finding the damage: the scrub, two more checks, and the check on read
  frame_life      (6) one slot over time: generations catch stale access
  frame_xor       (7) rebuilding the piece bit by bit, or recomputing it
  frame_spread    (8) why damage spreads through attention
  frame_rollback  (9) rolling back and re-prefilling; evicting cached KV; what the user receives
  frame_lists    (10) when to repair, and the watch list

Each figure is a view of the scene, filtered by frame and piped to the Excalidraw
CLI. Build the scene and render every figure with:

  python3 -I -B build_kvmend_figure.py --render
"""

import json
import subprocess
import sys
from pathlib import Path

FONT = 5              # Excalifont
CHAR_W = 0.6          # rough glyph width per font-size unit, used to size and centre text
LINE_H = 1.25
INK, MUTED = "#1e1e1e", "#495057"
RED, GREEN, BLUE, ORANGE, VIOLET, AMBER = "#e03131", "#2f9e44", "#1971c2", "#e8590c", "#6741d9", "#f08c00"
T_TITLE, T_SUB, T_PANEL, T_NOTE, T_LABEL = 32, 20, 26, 21, 18

STYLE = {  # (stroke, fill)
    "panel": ("#1971c2", "#e7f5ff"),   # GPU memory
    "host": ("#f08c00", "#fff9db"),    # host memory
    "cell": ("#1971c2", "#a5d8ff"),    # KV on a GPU
    "new": ("#1864ab", "#4dabf7"),     # KV just written
    "empty": ("#adb5bd", "#ffffff"),   # unwritten position or slot
    "parity": ("#f08c00", "#ffec99"),
    "bad": ("#e03131", "#ffc9c9"),     # damaged
    "sus": ("#e8590c", "#ffd8a8"),     # suspect: computed after the last good check
    "good": ("#2f9e44", "#b2f2bb"),    # repaired, or shown to be clean
    "tag": ("#6741d9", "#e5dbff"),     # tags and generations
    "gray": ("#868e96", "#f1f3f5"),
    "white": ("#495057", "#ffffff"),
}

WIDE = 1560
GAP = 40
VIEWS = {
    "fig1_story.png": ["frame_story"],
    "fig2_vocab.png": ["frame_vocab"],
    "fig3_timeline.png": ["frame_timeline"],
    "fig4_normal.png": ["frame_normal"],
    "fig5_detect.png": ["frame_detect"],
    "fig6_life.png": ["frame_life"],
    "fig7_xor.png": ["frame_xor"],
    "fig8_spread.png": ["frame_spread"],
    "fig9_rollback.png": ["frame_rollback"],
    "fig10_lists.png": ["frame_lists"],
}
DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def order_key(n):
    """Strictly increasing fractional-index keys ("a0".."az", then "b00".."bzz") for the CLI's `index` field."""
    if n < 62:
        return "a" + DIGITS[n]
    n -= 62
    assert n < 62 * 62, "too many elements for two-digit keys"
    return "b" + DIGITS[n // 62] + DIGITS[n % 62]


def text_size(text, size):
    lines = text.split("\n")
    return max(len(line) for line in lines) * size * CHAR_W, len(lines) * size * LINE_H


class Scene:
    def __init__(self):
        self.elements = []
        self.overflow = []

    def _add(self, kind, eid, x, y, w, h, frame, **props):
        n = len(self.elements)
        element = {
            "id": eid, "type": kind, "x": x, "y": y, "width": w, "height": h, "angle": 0,
            "strokeColor": INK, "backgroundColor": "transparent", "fillStyle": "solid",
            "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "opacity": 100,
            "groupIds": [], "frameId": frame, "index": order_key(n), "roundness": None, "seed": n + 1,
            "version": 1, "versionNonce": n + 1, "isDeleted": False,
            "boundElements": [], "updated": 1, "link": None, "locked": False,
        }
        element.update(props)
        self.elements.append(element)
        return element

    def frame(self, eid, x, y, w, h, name):
        return self._add("frame", eid, x, y, w, h, None, name=name, strokeColor="#adb5bd", roughness=0)

    def text(self, eid, x, y, text, frame, size=T_LABEL, color=INK, align="left", container=None):
        w, h = text_size(text, size)
        return self._add(
            "text", eid, x, y, w, h, frame, strokeColor=color, strokeWidth=1, boundElements=None,
            text=text, originalText=text, fontSize=size, fontFamily=FONT, textAlign=align,
            verticalAlign="middle" if container else "top", containerId=container,
            lineHeight=LINE_H, autoResize=True)

    def ctext(self, eid, cx, y, text, frame, size=T_LABEL, color=INK):
        w, _ = text_size(text, size)
        return self.text(eid, cx - w / 2, y, text, frame, size=size, color=color, align="center")

    def rect(self, eid, x, y, w, h, frame, kind="cell", dashed=False, width=2, round_=True):
        stroke, fill = STYLE[kind]
        return self._add("rectangle", eid, x, y, w, h, frame, strokeColor=stroke, backgroundColor=fill,
                         strokeWidth=width, roundness={"type": 3} if round_ else None,
                         strokeStyle="dashed" if dashed else "solid")

    def outline(self, eid, x, y, w, h, frame, color, dashed=True, width=3):
        return self._add("rectangle", eid, x, y, w, h, frame, strokeColor=color, backgroundColor="transparent",
                         strokeWidth=width, roundness={"type": 3}, strokeStyle="dashed" if dashed else "solid")

    def box(self, eid, x, y, w, h, label, frame, kind, size=T_LABEL, dashed=False):
        r = self.rect(eid, x, y, w, h, frame, kind, dashed)
        tw, th = text_size(label, size)
        if tw > w - 8 or th > h - 4:
            self.overflow.append(f"{eid}: label {tw:.0f}x{th:.0f} in box {w}x{h}")
        t = self.text(f"{eid}_t", x + (w - tw) / 2, y + (h - th) / 2, label, frame, size=size, align="center",
                      container=eid)
        r["boundElements"].append({"type": "text", "id": t["id"]})
        return r

    def ellipse(self, eid, x, y, w, h, frame, stroke=INK, fill="transparent", width=2, dashed=False):
        return self._add("ellipse", eid, x, y, w, h, frame, strokeColor=stroke, backgroundColor=fill,
                         strokeWidth=width, strokeStyle="dashed" if dashed else "solid", roundness={"type": 2})

    def _poly(self, kind, eid, points, frame, color, width, dashed, heads=(None, None)):
        x0, y0 = points[0]
        rel = [[px - x0, py - y0] for px, py in points]
        xs, ys = [p[0] for p in rel], [p[1] for p in rel]
        return self._add(kind, eid, x0, y0, max(xs) - min(xs), max(ys) - min(ys), frame,
                         strokeColor=color, strokeWidth=width, strokeStyle="dashed" if dashed else "solid",
                         points=rel, lastCommittedPoint=None, startBinding=None, endBinding=None,
                         startArrowhead=heads[0], endArrowhead=heads[1], elbowed=False)

    def line(self, eid, points, frame, color=INK, width=2, dashed=False):
        return self._poly("line", eid, points, frame, color, width, dashed)

    def arrow(self, eid, points, frame, color=INK, width=2, dashed=False, head="arrow"):
        return self._poly("arrow", eid, points, frame, color, width, dashed, heads=(None, head))


# Icons -------------------------------------------------------------------------------------------

def xor_gate(s, p, cx, cy, f, r=22, crossed=False):
    s.ellipse(f"{p}_ring", cx - r, cy - r, 2 * r, 2 * r, f, stroke=INK, fill="#ffffff", width=3)
    s.line(f"{p}_h", [(cx - r * 0.6, cy), (cx + r * 0.6, cy)], f, width=3)
    s.line(f"{p}_v", [(cx, cy - r * 0.6), (cx, cy + r * 0.6)], f, width=3)
    if crossed:
        cross(s, f"{p}_x", cx - r - 4, cy - r - 4, f, size=2 * r + 8)


def magnifier(s, p, x, y, f, r=17, color=VIOLET):
    s.ellipse(f"{p}_lens", x, y, 2 * r, 2 * r, f, stroke=color, fill="#ffffff", width=3)
    s.line(f"{p}_handle", [(x + 1.7 * r, y + 1.7 * r), (x + 2.6 * r, y + 2.6 * r)], f, color=color, width=5)


def check(s, p, x, y, f, size=24):
    s.line(p, [(x, y + size * 0.5), (x + size * 0.38, y + size * 0.9), (x + size, y)], f, color=GREEN, width=4)


def cross(s, p, x, y, f, size=22):
    s.line(f"{p}_a", [(x, y), (x + size, y + size)], f, color=RED, width=4)
    s.line(f"{p}_b", [(x + size, y), (x, y + size)], f, color=RED, width=4)


def bolt(s, p, x, y, f, size=40):
    s.line(p, [(x + size * 0.62, y), (x + size * 0.18, y + size * 0.55), (x + size * 0.55, y + size * 0.55),
               (x + size * 0.12, y + size * 1.15)], f, color=RED, width=4)


def crack(s, p, x, y, w, h, f):
    s.line(p, [(x + w * 0.25, y + 3), (x + w * 0.55, y + h * 0.42), (x + w * 0.35, y + h * 0.58),
               (x + w * 0.72, y + h - 3)], f, color="#a61e1e", width=2)


def badge(s, p, x, y, text, f, kind="tag", size=T_LABEL):
    w, h = text_size(text, size)
    return s.box(p, x, y, w + 14, h + 8, text, f, kind, size=size)


def cbadge(s, p, cx, y, text, f, kind="tag", size=T_LABEL):
    w, _ = text_size(text, size)
    return badge(s, p, cx - (w + 14) / 2, y, text, f, kind, size)


def lock(s, p, x, y, f, size=26):
    s.rect(f"{p}_body", x, y + size * 0.45, size, size * 0.55, f, "tag", width=2)
    s.line(f"{p}_shackle", [(x + size * 0.22, y + size * 0.45), (x + size * 0.22, y + size * 0.18),
                            (x + size * 0.38, y + size * 0.02), (x + size * 0.62, y + size * 0.02),
                            (x + size * 0.78, y + size * 0.18), (x + size * 0.78, y + size * 0.45)],
           f, color=VIOLET, width=2)


def step_marker(s, p, cx, cy, n, f, r=17):
    s.ellipse(f"{p}_c", cx - r, cy - r, 2 * r, 2 * r, f, stroke=MUTED, fill="#495057", width=1)
    s.ctext(f"{p}_n", cx, cy - 12.5, str(n), f, size=20, color="#ffffff")


def pause_icon(s, p, x, y, f, size=30):
    s.rect(f"{p}_a", x, y, size * 0.32, size, f, "gray", width=2, round_=False)
    s.rect(f"{p}_b", x + size * 0.6, y, size * 0.32, size, f, "gray", width=2, round_=False)


def play_icon(s, p, x, y, f, size=30, color=GREEN):
    s.line(p, [(x, y), (x + size * 0.85, y + size / 2), (x, y + size), (x, y)], f, color=color, width=3)


def rewind_icon(s, p, x, y, f, size=30, color=ORANGE):
    s.line(f"{p}_a", [(x + size * 0.5, y), (x, y + size / 2), (x + size * 0.5, y + size), (x + size * 0.5, y)], f,
           color=color, width=3)
    s.line(f"{p}_b", [(x + size, y), (x + size * 0.5, y + size / 2), (x + size, y + size), (x + size, y)], f,
           color=color, width=3)


def trash_icon(s, p, x, y, f, size=30):
    s.line(f"{p}_lid", [(x - 3, y + 4), (x + size + 3, y + 4)], f, color=MUTED, width=3)
    s.line(f"{p}_knob", [(x + size * 0.35, y + 4), (x + size * 0.35, y), (x + size * 0.65, y), (x + size * 0.65, y + 4)],
           f, color=MUTED, width=2)
    s.rect(f"{p}_can", x + 2, y + 8, size - 4, size, f, "gray", width=2, round_=False)


def clock_icon(s, p, cx, cy, f, r=16):
    s.ellipse(f"{p}_face", cx - r, cy - r, 2 * r, 2 * r, f, stroke=MUTED, fill="#ffffff", width=2)
    s.line(f"{p}_h", [(cx, cy), (cx, cy - r * 0.7)], f, color=MUTED, width=2)
    s.line(f"{p}_m", [(cx, cy), (cx + r * 0.55, cy + r * 0.2)], f, color=MUTED, width=2)


def warning_icon(s, p, x, y, f, size=38):
    s.line(f"{p}_tri", [(x + size / 2, y), (x + size, y + size), (x, y + size), (x + size / 2, y)], f, color=RED,
           width=3)
    s.ctext(f"{p}_bang", x + size / 2, y + size * 0.28, "!", f, size=22, color=RED)


def person(s, p, x, y, label, f, kind="white", dashed=False, w=64):
    return s.box(p, x, y, w, 46, label, f, kind, size=22, dashed=dashed)


# Small structures ------------------------------------------------------------------------------

def page_col(s, p, x, y, f, filled=16, kind="cell", special=None, w=30, h=10, gap=2):
    """A page piece as a column of 16 token positions (top = token 0 of the page)."""
    special = special or {}
    return [s.rect(f"{p}_s{k}", x, y + k * (h + gap), w, h, f, special.get(k, kind if k < filled else "empty"),
                   width=1, round_=False) for k in range(16)]


def page_row(s, p, x, y, f, filled=16, kind="cell", special=None, w=16, h=26, gap=3):
    """A page piece as a row of 16 token positions (left = token 0 of the page)."""
    special = special or {}
    return [s.rect(f"{p}_s{k}", x + k * (w + gap), y, w, h, f, special.get(k, kind if k < filled else "empty"),
                   width=1, round_=False) for k in range(16)]


def grid(s, p, x, y, f, cols, rows, kinds, cw=40, ch=28, gx=8, gy=8, labels=True, size=T_LABEL):
    """Pieces of one GPU: rows are layers (top first), columns are pages; kinds(row, col) -> style.

    A row or column named "..." stands for the layers or pages left out."""
    cells = {}
    ry = y
    for ri, rname in enumerate(rows):
        if rname == "...":
            for ci, cname in enumerate(cols):
                if cname != "...":
                    s.ctext(f"{p}_rdots{ri}_{ci}", x + ci * (cw + gx) + cw / 2, ry - 6, "...", f, size=20, color=MUTED)
            ry += 26
            continue
        if labels:
            s.text(f"{p}_r{ri}", x - 44, ry + ch / 2 - 11, rname, f, size=size, color=MUTED)
        for ci, cname in enumerate(cols):
            if cname == "...":
                if ri == 0 and labels:
                    s.ctext(f"{p}_cdots", x + ci * (cw + gx) + cw / 2, y - 28, "...", f, size=size, color=MUTED)
                s.ctext(f"{p}_dots{ri}_{ci}", x + ci * (cw + gx) + cw / 2, ry + ch / 2 - 14, "...", f, size=20,
                        color=MUTED)
                continue
            cells[(rname, cname)] = s.rect(f"{p}_{rname}_{cname}", x + ci * (cw + gx), ry, cw, ch, f, kinds(rname, cname),
                                           width=2)
            if ri == 0 and labels:
                s.ctext(f"{p}_c{ci}", x + ci * (cw + gx) + cw / 2, y - 28, cname, f, size=size, color=MUTED)
        ry += ch + gy
    return cells


def strip(s, p, x, y, f, items, w=24, h=24, gap=5):
    """A request's tokens as squares; items are kinds, or "..." for skipped tokens. Returns square elements."""
    out = []
    cx = x
    for k, kind in enumerate(items):
        if kind == "...":
            s.ctext(f"{p}_gap{k}", cx + 13, y + h / 2 - 16, "...", f, size=20, color=MUTED)
            cx += 30
            continue
        out.append(s.rect(f"{p}_t{k}", cx, y, w, h, f, kind, width=2, round_=False))
        cx += w + gap
    return out


def center(el):
    return el["x"] + el["width"] / 2, el["y"] + el["height"] / 2


def frame_box(s, key, ox, oy, h, title=None, subtitle=None):
    f = f"frame_{key}"
    s.frame(f, ox, oy, WIDE, h, key)
    if title:
        s.text(f"{key}_title", ox + 24, oy + 16, title, f, size=T_TITLE)
    if subtitle:
        s.text(f"{key}_subtitle", ox + 24, oy + 60, subtitle, f, size=T_SUB, color=MUTED)
    return f


def panel_titles(s, key, ox, oy, h, f, titles, y, color=INK):
    for i, t in enumerate(titles):
        px = ox + 24 + i * 512
        s.text(f"{key}_p{i}_title", px, y, t, f, size=T_PANEL, color=color)
        if i:
            s.line(f"{key}_sep{i}", [(px - 18, y + 2), (px - 18, oy + h - 24)], f, color="#dee2e6", width=2)


# Figures -----------------------------------------------------------------------------------------

def fig_story(s, ox, oy):
    key, h = "story", 640
    f = frame_box(s, key, ox, oy, h)
    panel_titles(s, key, ox, oy, h, f, ["(a) R reads its KV cache", "(b) one piece goes bad",
                                        "(c) KVMend catches and mends it"], oy + 22)
    for i in range(3):
        px = ox + 24 + i * 512
        page = ["cell", "bad", "bad"][i]
        tail = [["empty"], ["sus"] * 6, ["good"] * 6][i]
        sq = strip(s, f"{key}_p{i}_strip", px + 10, oy + 110, f,
                   ["cell"] * 3 + ["..."] + [page] + ["..."] + ["cell"] * 2 + tail)
        s.ctext(f"{key}_p{i}_pg", center(sq[3])[0], oy + 142, "page 64", f, color=MUTED)
        s.text(f"{key}_p{i}_gpu", px + 10, oy + 180, "GPU 2", f, size=20, color=BLUE)
        gx, gy = px + 90, oy + 260
        cells = grid(s, f"{key}_p{i}_g", gx, gy, f, ["p0", "p63", "p64"], ["L32", "...", "L12", "...", "L1"],
                     lambda r, c, k=page: k if (r, c) == ("L12", "p64") else "cell", cw=60, ch=40, gx=12, gy=12)
        hit = cells[("L12", "p64")]
        hx, hy = center(hit)
        right = hit["x"] + hit["width"]
        tail_sq = sq[-len(tail):]
        t0, t1 = tail_sq[0]["x"], tail_sq[-1]["x"] + tail_sq[-1]["width"]
        tail_cx = (t0 + t1) / 2
        if i:
            crack(s, f"{key}_p{i}_crack", hit["x"], hit["y"], hit["width"], hit["height"], f)
        if i == 0:
            nxt = sq[-1]
            s.text(f"{key}_p0_next", nxt["x"] + 32, oy + 104, "next\ntoken", f, color=MUTED)
            top = gy - 48
            s.line(f"{key}_p0_br", [(gx - 4, top + 8), (gx - 4, top), (right + 4, top), (right + 4, top + 8)], f,
                   color="#868e96", width=2)
            s.arrow(f"{key}_p0_read", [(nxt["x"] + 12, oy + 138), ((gx + right) / 2, top - 2)], f, color="#868e96",
                    width=3)
            s.text(f"{key}_p0_readl", px + 270, oy + 160, "reads every page,\nevery layer", f, color=MUTED)
            b = badge(s, f"{key}_p0_tag", right + 20, hy - 16, "tag 08e4c1a7", f)
            check(s, f"{key}_p0_ok", b["x"] + b["width"] + 8, hy - 12, f, size=22)
            s.text(f"{key}_p0_note", px, oy + 548, "every piece matches its tag", f, size=T_NOTE, color=GREEN)
        if i == 1:
            bolt(s, f"{key}_p1_bolt", right + 2, hit["y"] - 52, f, size=42)
            s.ctext(f"{key}_p1_new", tail_cx, oy + 76, "40 new tokens (drawn as 6)", f, color=ORANGE)
            s.arrow(f"{key}_p1_read", [(tail_cx, oy + 138), (tail_cx, hy), (right + 8, hy)], f, color=RED, width=3)
            s.text(f"{key}_p1_readl", tail_cx + 10, oy + 168, "each new\ntoken reads it", f, color=RED)
            badge(s, f"{key}_p1_tag", right + 20, hy + 30, "tag 08e4c1a7", f)
            s.text(f"{key}_p1_stale", right + 20, hy + 68, "(no longer true)", f, color=RED)
            s.box(f"{key}_p1_bubble", px + 150, oy + 478, 290, 46, "HTTP 200, no error", f, "bad", size=22)
            s.text(f"{key}_p1_note", px, oy + 548, "R keeps answering from bad data", f, size=T_NOTE, color=RED)
        if i == 2:
            # (1) the tag check fails on the damaged piece
            magnifier(s, f"{key}_p2_mag", right + 10, hit["y"] - 4, f)
            cross(s, f"{key}_p2_x", right + 58, hit["y"] - 26, f, size=20)
            s.text(f"{key}_p2_fail", right + 84, hit["y"] - 30, "tag check\nfails", f, color=RED)
            step_marker(s, f"{key}_p2_m1", right + 30, hit["y"] - 42, 1, f)
            # (2) page 64 is recomputed in place: a short prefill of its 16 tokens over the intact pages before it
            bx, by = px + 312, oy + 430
            s.box(f"{key}_p2_rec", bx, by, 148, 58, "recompute\npage 64", f, "good", size=20)
            s.text(f"{key}_p2_recl", bx, by + 66, "from the intact\npages 0-63", f, color=MUTED)
            s.arrow(f"{key}_p2_out", [(bx - 2, by + 20), (right + 2, hit["y"] + hit["height"] + 2)], f, color=GREEN,
                    width=4)
            step_marker(s, f"{key}_p2_m2", bx + 162, by + 14, 2, f)
            # (3) the 40 suspect tokens are redone
            s.line(f"{key}_p2_br", [(t0 - 2, oy + 104), (t0 - 2, oy + 96), (t1 + 2, oy + 96), (t1 + 2, oy + 104)], f,
                   color=GREEN, width=2)
            s.ctext(f"{key}_p2_redo", tail_cx, oy + 66, "40 redone (drawn as 6)", f, color=GREEN)
            rewind_icon(s, f"{key}_p2_rw", px + 172, oy + 64, f, size=26)
            step_marker(s, f"{key}_p2_m3", px + 146, oy + 77, 3, f)
            s.text(f"{key}_p2_note", px, oy + 534, "(1) watch: the tag check fails\n(2) mend: recompute the page\n(3) mend: redo the 40 tokens", f,
                   size=T_NOTE, color=GREEN)


def fig_vocab(s, ox, oy):
    key, h = "vocab", 690
    f = frame_box(s, key, ox, oy, h, "Pages, pieces, slots and stripes",
                  "drawn for 4 GPUs; sizes for TP=8 (8 KV heads, FP16)")
    # request R as tokens grouped into pages
    sx, sy = ox + 470, oy + 140
    gpos = {}
    x = sx
    for name in ["p0", "p1", None, "p64", "p65"]:
        if name is None:
            s.ctext(f"{key}_sdots", x + 20, sy - 4, "...", f, size=22, color=MUTED)
            x += 50
            continue
        for k in range(4):
            s.rect(f"{key}_tok_{name}_{k}", x + k * 17, sy, 14, 24, f, "cell", width=1, round_=False)
        s.ctext(f"{key}_tokl_{name}", x + 32, sy + 30, name, f, color=MUTED)
        gpos[name] = (x + 32, sy + 56)
        x += 90
    s.text(f"{key}_rlabel", sx - 140, sy - 2, "request R:", f, size=20)
    p64 = gpos["p64"][0]
    s.line(f"{key}_brace", [(p64 - 34, sy - 8), (p64 - 34, sy - 16), (p64 + 34, sy - 16), (p64 + 34, sy - 8)], f,
           color=INK, width=2)
    s.ctext(f"{key}_bracel", p64, sy - 46, "page 64 = tokens 1,024-1,039", f)
    # the same page on four GPUs: one piece per layer, all in slot s
    cols, rows = ["p0", "p1", "...", "p64", "p65"], ["L1", "L2", "...", "L12"]
    stripe = []
    for g in range(4):
        gx, gy = ox + 24 + g * 306, oy + 250
        s.rect(f"{key}_gpu{g}", gx, gy, 290, 266, f, "panel", width=2)
        s.text(f"{key}_gpul{g}", gx + 12, gy + 10, f"GPU {g}", f, size=20, color=BLUE)
        cells = grid(s, f"{key}_g{g}", gx + 60, gy + 104, f, cols, rows, lambda r, c: "cell", cw=32, ch=30, gx=12, gy=12)
        piece, top = cells[("L12", "p64")], cells[("L1", "p64")]
        stripe.append(piece)
        s.outline(f"{key}_pc{g}", piece["x"] - 4, piece["y"] - 4, piece["width"] + 8, piece["height"] + 8, f, INK,
                  dashed=False, width=3)
        s.outline(f"{key}_slot{g}", top["x"] - 6, gy + 70, top["width"] + 12,
                  piece["y"] + piece["height"] + 10 - (gy + 70), f, MUTED, dashed=True, width=2)
        s.ctext(f"{key}_slotl{g}", top["x"] + top["width"] / 2, gy + 42, "slot s", f, color=MUTED)
        s.line(f"{key}_map{g}", [gpos["p64"], (top["x"] + top["width"] / 2, gy - 2)], f, color="#ced4da", width=1,
               dashed=True)
    host_y = oy + 580
    s.rect(f"{key}_host", ox + 24, host_y, 1220, 76, f, "host", width=2)
    s.text(f"{key}_hostl", ox + 36, host_y + 8, "host memory", f, size=20, color=AMBER)
    par = s.rect(f"{key}_parity", ox + 610, host_y + 24, 40, 32, f, "parity", width=2)
    s.text(f"{key}_parl", ox + 662, host_y + 28, "parity piece", f, color=AMBER)
    bus = host_y - 22
    for g, c in enumerate(stripe):
        cx = c["x"] + c["width"] / 2
        s.line(f"{key}_st{g}", [(cx, c["y"] + c["height"] + 6), (cx, bus), (ox + 630, bus), (ox + 630, par["y"] - 2)],
               f, color=AMBER, width=3, dashed=True)
    # legend
    lx, y = ox + 1268, oy + 250
    s.rect(f"{key}_lg0", lx + 2, y + 4, 22, 18, f, "cell", width=2, round_=False)
    s.outline(f"{key}_lg0o", lx - 2, y, 30, 26, f, INK, dashed=False, width=3)
    s.text(f"{key}_lgh0", lx + 38, y, "one piece", f, size=20)
    s.text(f"{key}_lgb0", lx + 38, y + 28, "1 page, 1 layer, 1 GPU\n= 8 KiB at TP=8", f, color=MUTED)
    y += 110
    s.outline(f"{key}_lg1", lx, y, 26, 26, f, MUTED, dashed=True, width=2)
    s.text(f"{key}_lgh1", lx + 38, y, "slot s", f, size=20)
    s.text(f"{key}_lgb1", lx + 38, y + 28, "page 64's place: one\nindex for every layer\non every GPU", f, color=MUTED)
    y += 130
    s.line(f"{key}_lg2", [(lx, y + 13), (lx + 26, y + 13)], f, color=AMBER, width=3, dashed=True)
    s.text(f"{key}_lgh2", lx + 38, y, "one stripe", f, size=20)
    s.text(f"{key}_lgb2", lx + 38, y + 28, "the 4 pieces + their\nparity piece", f, color=MUTED)


def fig_timeline(s, ox, oy):
    key, h = "timeline", 700
    f = frame_box(s, key, ox, oy, h, "The running example on a timeline",
                  "request R, page 64, layer 12, GPU 2; the scrub passes every page once per second (40 decode steps)")
    ya = oy + 380
    # three moments, each above its tick on the time axis
    events = [(150, "check: ok", GREEN), (420, "bit flips", RED), (820, "check: tag wrong", RED)]
    for k, (ex, label, color) in enumerate(events):
        mx = ox + ex
        s.ctext(f"{key}_evl{k}", mx + 30, oy + 166, label, f, size=20, color=color)
        s.rect(f"{key}_ev{k}", mx, oy + 210, 60, 42, f, "cell" if k == 0 else "bad", width=2)
        if k:
            crack(s, f"{key}_evc{k}", mx, oy + 210, 60, 42, f)
        if k == 1:
            bolt(s, f"{key}_bolt", mx + 68, oy + 196, f, size=40)
        else:
            magnifier(s, f"{key}_mag{k}", mx + 68, oy + 196, f)
            (check if k == 0 else cross)(s, f"{key}_res{k}", mx + 76, oy + 262, f, size=22)
        s.line(f"{key}_drop{k}", [(mx + 30, oy + 290), (mx + 30, ya - 42)], f, color="#ced4da", width=2, dashed=True)
    # the repair, step by step
    s.text(f"{key}_rt", ox + 990, oy + 160, "repair, step by step:", f, size=20, color=GREEN)
    cy = oy + 228
    steps = [(1030, "hold R"), (1145, "recompute\nthe page"), (1260, "roll back\n40 tokens"),
             (1375, "re-prefill\n40 tokens"), (1495, "resume R")]
    pause_icon(s, f"{key}_s0", ox + 1030 - 14, cy - 15, f)
    s.rect(f"{key}_s1", ox + 1145 - 20, cy - 13, 40, 26, f, "good", width=2, round_=False)
    s.ctext(f"{key}_s1t", ox + 1145, cy - 12, "16", f, size=17, color=GREEN)
    rewind_icon(s, f"{key}_s2", ox + 1260 - 15, cy - 15, f)
    for k in range(3):
        s.rect(f"{key}_s3_{k}", ox + 1375 - 25 + k * 17, cy - 12, 14, 24, f, "good", width=1, round_=False)
    play_icon(s, f"{key}_s4", ox + 1495 - 12, cy - 15, f)
    for k, (sx, label) in enumerate(steps):
        s.ctext(f"{key}_sl{k}", ox + sx, cy + 30, label, f)
        if k < len(steps) - 1:
            s.arrow(f"{key}_sa{k}", [(ox + sx + 30, cy), (ox + steps[k + 1][0] - 32, cy)], f, color=MUTED, width=2)
    # the time axis
    s.arrow(f"{key}_axis", [(ox + 60, ya), (ox + 1530, ya)], f, color=INK, width=3)
    s.text(f"{key}_time", ox + 1506, ya + 12, "time", f, color=MUTED)
    ticks = [(180, "step 9,000", GREEN), (450, "damage (unknown time)", RED), (850, "step 9,040", RED)]
    for k, (tx, label, color) in enumerate(ticks):
        s.line(f"{key}_tick{k}", [(ox + tx, ya - 12), (ox + tx, ya + 12)], f, color=color, width=3)
        s.ctext(f"{key}_tickl{k}", ox + tx, ya - 38, label, f, size=19, color=color)
    # the 40 tokens R decoded meanwhile, and the same 40 after the repair
    n, x0, x1 = 10, ox + 196, ox + 834
    pitch = (x1 - x0 - 44) / (n - 1)
    for k in range(n):
        s.rect(f"{key}_sus{k}", x0 + k * pitch, ya + 40, 44, 36, f, "sus", width=2, round_=False)
    s.ctext(f"{key}_susl", (x0 + x1) / 2, ya + 86, "40 tokens decoded after step 9,000 (drawn as 10): all suspect", f,
            size=19, color=ORANGE)
    gy2 = ya + 166
    for k in range(n):
        s.rect(f"{key}_rep{k}", x0 + k * pitch, gy2, 44, 36, f, "good", width=2, round_=False)
    s.line(f"{key}_br", [(x0 - 2, gy2 - 6), (x0 - 2, gy2 - 14), (x1 + 2, gy2 - 14), (x1 + 2, gy2 - 6)], f, color=GREEN,
           width=3)
    s.ctext(f"{key}_repl", (x0 + x1) / 2, gy2 + 46, "after the repair: re-prefilled in one pass", f, size=19,
            color=GREEN)
    s.arrow(f"{key}_rpa", [(ox + 1375, oy + 306), (ox + 1375, gy2 - 14), (x1 + 12, gy2 - 14)], f, color="#adb5bd",
            width=2, dashed=True)
    nx = ox + 1470
    s.rect(f"{key}_next", nx, gy2, 44, 36, f, "cell", width=2, round_=False)
    s.ctext(f"{key}_nextl", nx + 22, gy2 + 46, "next token", f, color=BLUE)
    s.arrow(f"{key}_resa", [(ox + 1495, oy + 286), (nx + 22, gy2 - 6)], f, color="#adb5bd", width=2, dashed=True)
    s.text(f"{key}_cost", ox + 24, oy + 640,
           "tokens recomputed:   KVMend 56 (40 with parity)   |   re-prefill all of R 32,808   |   abort and retry 32,808", f, size=22)


def fig_normal(s, ox, oy):
    key, h = "normal", 700
    f = frame_box(s, key, ox, oy, h, "Normal operation: tag every piece; parity per page is optional",
                  "page 64, layer 12, drawn for 4 GPUs; each column is one GPU's piece: 16 token positions, top = token 1,024")
    panel_titles(s, key, ox, oy, h, f, ["(a) write: token 1,030 arrives", "(b) seal: the 16th token",
                                        "(c) optional: parity per page"], oy + 108)
    tags_a = ["c41d77e2", "9e0713ab", "2b5a6c04", "f310d9b8"]
    tags_b = ["3f2a5e61", "b71c0d93", "08e4c1a7", "d59f2a40"]
    # (a) each GPU writes its heads' K,V into position 6 and folds them into the piece's running tag
    px = ox + 24
    s.text(f"{key}_a_kv", px, oy + 158, "K,V of token 1,030\n(one segment per head)", f, color=MUTED)
    segs = [s.rect(f"{key}_a_seg{g}", px + 250 + g * 50, oy + 166, 44, 28, f, "new", width=2, round_=False)
            for g in range(4)]
    for g in range(4):
        cx = px + 40 + g * 115
        col = page_col(s, f"{key}_a_col{g}", cx, oy + 250, f, filled=6, special={6: "new"}, w=30, h=12, gap=2)
        y6 = col[6]["y"] + 6
        sx = segs[g]["x"] + 22
        s.arrow(f"{key}_a_in{g}", [(sx, oy + 196), (cx + 52, oy + 218), (cx + 52, y6), (cx + 34, y6)], f, color=BLUE,
                width=2)
        cbadge(s, f"{key}_a_tag{g}", cx + 15, oy + 488, tags_a[g], f, size=17)
        s.ctext(f"{key}_a_gl{g}", cx + 15, oy + 528, f"GPU {g}", f, color=BLUE)
        s.arrow(f"{key}_a_up{g}", [(cx - 2, y6), (cx - 18, y6), (cx - 18, oy + 484)], f, color=VIOLET, width=2)
    s.text(f"{key}_a_note", px, oy + 590, "each GPU writes position 6 of its\npiece and updates its running tag", f,
           size=T_NOTE, color=VIOLET)
    # (b) the 16th token seals the page; the tags are final
    px = ox + 536
    for g in range(4):
        cx = px + 40 + g * 115
        lock(s, f"{key}_b_lock{g}", cx + 2, oy + 204, f)
        page_col(s, f"{key}_b_col{g}", cx, oy + 250, f, filled=16, special={15: "new"}, w=30, h=12, gap=2)
        cbadge(s, f"{key}_b_tag{g}", cx + 15, oy + 488, tags_b[g], f, size=17)
        s.ctext(f"{key}_b_gl{g}", cx + 15, oy + 528, f"GPU {g}", f, color=BLUE)
    s.text(f"{key}_b_note", px, oy + 590, "16 of 16 written: each tag is final;\na tag: 4 bytes per 8 KiB piece (0.05%)", f,
           size=T_NOTE, color=VIOLET)
    # (c) this batch's encoder, GPU 1, gathers the sealed pieces and sends one parity piece to host memory
    px = ox + 1048
    for g in range(4):
        cx = px + 30 + g * 105
        page_col(s, f"{key}_c_col{g}", cx, oy + 170, f, filled=16, w=26, h=8, gap=2)
        s.ctext(f"{key}_c_gl{g}", cx + 13, oy + 336, f"GPU {g}", f, color=BLUE)
    ex, ey = px + 148, oy + 420
    ends = {0: (ex - 24, ey - 12), 1: (ex, ey - 30), 2: (ex + 22, ey - 12), 3: (ex + 24, ey - 4)}
    for g in range(4):
        s.arrow(f"{key}_c_in{g}", [(px + 43 + g * 105, oy + 362), ends[g]], f, color=MUTED, width=2)
    xor_gate(s, f"{key}_c_xor", ex, ey, f, r=26)
    s.text(f"{key}_c_enc", ex + 40, ey + 6, "GPU 1 encodes this batch;\nthe encoder rotates", f, color=MUTED)
    check(s, f"{key}_c_ok", ex + 40, ey + 66, f, size=20)
    s.text(f"{key}_c_okl", ex + 66, ey + 60, "tags checked\nwhile gathering", f, color=GREEN)
    s.arrow(f"{key}_c_down", [(ex, ey + 28), (ex, oy + 536)], f, color=AMBER, width=3)
    s.rect(f"{key}_c_host", px, oy + 540, 470, 66, f, "host", width=2)
    s.text(f"{key}_c_hostl", px + 10, oy + 546, "host memory", f, color=AMBER)
    page_row(s, f"{key}_c_par", px + 170, oy + 560, f, filled=16, kind="parity", w=14, h=28, gap=3)
    s.text(f"{key}_c_note", px, oy + 620, "one parity piece per page and layer,\nkeyed by slot and generation", f,
           size=T_NOTE, color=AMBER)


def fig_life(s, ox, oy):
    key, h = "life", 770
    f = frame_box(s, key, ox, oy, h, "One slot over time: generations catch stale access",
                  "a slot holds one page of the KV pool; its generation grows each time the slot is handed out")
    s.arrow(f"{key}_axis", [(ox + 60, oy + 130), (ox + 1520, oy + 130)], f, color=INK, width=3)
    states = [(200, "owned by A, gen 7", 16, "tag 6a91e3c5", "tag", BLUE),
              (720, "freed", 0, "tags dropped", "gray", MUTED),
              (1240, "owned by B, gen 8", 6, "tag 91cd04f8", "tag", BLUE)]
    for k, (x, label, filled, tag, tkind, color) in enumerate(states):
        sx = ox + x
        s.text(f"{key}_st{k}", sx - 10, oy + 148, label, f, size=20, color=color)
        page_col(s, f"{key}_col{k}", sx, oy + 190, f, filled=filled, w=34, h=8, gap=2)
        b = badge(s, f"{key}_tag{k}", sx + 50, oy + 250, tag, f, kind=tkind)
        if k == 1:
            cross(s, f"{key}_drop", b["x"] + b["width"] + 8, oy + 254, f)
        if k < 2:
            s.arrow(f"{key}_next{k}", [(sx + 240, oy + 266), (ox + states[k + 1][0] - 24, oy + 266)], f, width=2)
    s.text(f"{key}_head", ox + 24, oy + 392, "how each kind of stale access is caught", f, size=22)
    panel_titles(s, key, ox, oy, h, f, ["(1) late write", "(2) read after free", "(3) never written"], oy + 440,
                 color=RED)
    # (1) a late write from A breaks B's tag
    px = ox + 24
    person(s, f"{key}_a", px, oy + 540, "A", f, dashed=True)
    s.text(f"{key}_ag", px + 8, oy + 592, "gone", f, color=MUTED)
    s.text(f"{key}_lw", px + 82, oy + 512, "late write", f, color=RED)
    col = page_col(s, f"{key}_b1", px + 230, oy + 500, f, filled=6, special={4: "bad"}, w=34, h=8, gap=2)
    s.arrow(f"{key}_lwa", [(px + 70, oy + 563), (px + 226, col[4]["y"] + 4)], f, color=RED, width=3, dashed=True)
    badge(s, f"{key}_st1", px + 280, oy + 510, "stored 91cd04f8", f)
    badge(s, f"{key}_nw1", px + 280, oy + 552, "now 77e0b21d", f, kind="bad")
    cross(s, f"{key}_x1", px + 464, oy + 530, f)
    s.text(f"{key}_n1", px, oy + 690, "B's tag no longer matches", f, size=T_NOTE, color=RED)
    # (2) C still holds generation 7 and reads the slot in generation 8
    px = ox + 536
    person(s, f"{key}_c", px, oy + 540, "C", f)
    badge(s, f"{key}_cg", px - 4, oy + 596, "still holds gen 7", f)
    page_col(s, f"{key}_b2", px + 300, oy + 500, f, filled=6, w=34, h=8, gap=2)
    badge(s, f"{key}_g8", px + 350, oy + 510, "gen 8", f)
    s.arrow(f"{key}_ca", [(px + 70, oy + 563), (px + 294, oy + 563)], f, width=3)
    cross(s, f"{key}_x2", px + 236, oy + 550, f, size=26)
    s.text(f"{key}_n2", px, oy + 690, "C expects gen 7, finds gen 8: stopped", f, size=T_NOTE, color=RED)
    # (3) D reads a token that was never written
    px = ox + 1048
    person(s, f"{key}_d", px, oy + 540, "D", f)
    col = page_col(s, f"{key}_e3", px + 216, oy + 500, f, filled=0, w=34, h=8, gap=2)
    s.outline(f"{key}_t5", px + 211, col[5]["y"] - 4, 44, 16, f, RED, dashed=False, width=2)
    s.arrow(f"{key}_da", [(px + 70, oy + 563), (px + 206, col[5]["y"] + 4)], f, width=3)
    cross(s, f"{key}_x3", px + 146, oy + 546, f, size=26)
    badge(s, f"{key}_cov", px + 266, oy + 540, "tag covers 0 of 16", f, kind="gray", size=17)
    s.text(f"{key}_n3", px, oy + 690, "D reads token 5, which no tag\ncovers: stopped", f, size=T_NOTE, color=RED)


def fig_detect(s, ox, oy):
    key, h = "detect", 940
    f = frame_box(s, key, ox, oy, h, "Finding the damage", "GPU 2, layer 12, pages 62-66")
    panel_titles(s, key, ox, oy, 580, f, ["(a) a bit flips, silently", "(b) the scrub checks each tag",
                                          "(c) two more checks"], oy + 108)
    cols = ["p62", "p63", "p64", "p65", "p66"]
    kinds = lambda r, c: "bad" if c == "p64" else "cell"
    gy = oy + 270
    # (a) the bytes change, the stored tag does not
    px = ox + 24
    cells = grid(s, f"{key}_a", px + 60, gy, f, cols, ["L12"], kinds, cw=66, ch=50, gx=12, gy=12)
    hit = cells[("L12", "p64")]
    hx = hit["x"] + hit["width"] / 2
    crack(s, f"{key}_a_cr", hit["x"], hit["y"], hit["width"], hit["height"], f)
    bolt(s, f"{key}_a_bolt", hit["x"] + 14, oy + 180, f, size=46)
    s.line(f"{key}_a_lead", [(hx, hit["y"] + hit["height"] + 2), (hx, oy + 352)], f, color=VIOLET, width=2)
    cbadge(s, f"{key}_a_tag", hx, oy + 352, "stored tag 08e4c1a7", f)
    s.text(f"{key}_a_note", px, oy + 446, "the bytes change; the stored tag\nstill holds the old value", f, size=T_NOTE,
           color=RED)
    # (b) the scrub recomputes each tag from the bytes and compares
    px = ox + 536
    cells = grid(s, f"{key}_b", px + 60, gy, f, cols, ["L12"], kinds, cw=66, ch=50, gx=12, gy=12)
    row = [cells[("L12", c)] for c in cols]
    s.arrow(f"{key}_b_path", [(row[0]["x"] - 8, oy + 226), (row[-1]["x"] + 74, oy + 226)], f, color=VIOLET, width=3,
            dashed=True)
    for k in (0, 1):
        check(s, f"{key}_b_ok{k}", row[k]["x"] + 21, oy + 186, f)
    hit = row[2]
    hx = hit["x"] + hit["width"] / 2
    crack(s, f"{key}_b_cr", hit["x"], hit["y"], hit["width"], hit["height"], f)
    magnifier(s, f"{key}_b_mag", hit["x"] + 14, oy + 164, f, r=19)
    cross(s, f"{key}_b_x", hit["x"] + 78, oy + 180, f)
    s.line(f"{key}_b_lead", [(hx, hit["y"] + hit["height"] + 2), (hx, oy + 352)], f, color=VIOLET, width=2)
    cbadge(s, f"{key}_b_stored", hx, oy + 352, "stored 08e4c1a7", f)
    nb = cbadge(s, f"{key}_b_now", hx, oy + 392, "now 51b2f03e", f, kind="bad")
    cross(s, f"{key}_b_x2", nb["x"] + nb["width"] + 10, oy + 396, f)
    s.text(f"{key}_b_note", px, oy + 446, "mismatch names the piece:\npage 64, layer 12, GPU 2", f, size=T_NOTE, color=RED)
    s.text(f"{key}_b_speed", px, oy + 510, "one pass over 20 GB:\n1.0 s at 1.3% of HBM bandwidth", f, color=VIOLET)
    # (c) checks before reuse and after a risky event
    px = ox + 1048
    s.text(f"{key}_c1_t", px, oy + 160, "before reuse", f, size=22)
    person(s, f"{key}_c1_req", px, oy + 210, "new", f, w=70)
    s.text(f"{key}_c1_reql", px + 2, oy + 262, "request", f, color=MUTED)
    s.arrow(f"{key}_c1_a", [(px + 74, oy + 233), (px + 226, oy + 233)], f, color=MUTED, width=2)
    magnifier(s, f"{key}_c1_mag", px + 120, oy + 192, f, r=15)
    page_col(s, f"{key}_c1_col", px + 236, oy + 172, f, filled=16, w=30, h=6, gap=2)
    s.text(f"{key}_c1_cl", px + 278, oy + 190, "cached prefix,\nchecked first\n(0.5-2 ms)", f, color=MUTED)
    check(s, f"{key}_c1_ok", px + 440, oy + 214, f)
    s.text(f"{key}_c2_t", px, oy + 334, "after a risky event", f, size=22)
    warning_icon(s, f"{key}_c2_warn", px + 12, oy + 396, f)
    s.text(f"{key}_c2_wl", px, oy + 444, "request killed\nmid-write", f, color=MUTED)
    s.arrow(f"{key}_c2_a", [(px + 74, oy + 416), (px + 226, oy + 416)], f, color=MUTED, width=2)
    magnifier(s, f"{key}_c2_mag", px + 120, oy + 380, f, r=15)
    for k in range(3):
        s.rect(f"{key}_c2_s{k}", px + 236 + k * 42, oy + 398, 36, 38, f, "cell", width=2)
    s.text(f"{key}_c2_sl", px + 236, oy + 444, "its slots,\nchecked first", f, color=MUTED)
    # (d) the check on read: attention loads each piece once, and the same bytes give its tag
    top = oy + 566
    s.line(f"{key}_dsep", [(ox + 24, top), (ox + WIDE - 24, top)], f, color="#dee2e6", width=2)
    s.text(f"{key}_d_title", ox + 24, top + 22, "(d) checked on every read, inside attention", f, size=T_PANEL)
    dy, px = oy + 730, ox + 24
    cells = grid(s, f"{key}_d", px + 60, dy, f, cols, ["L12"], kinds, cw=56, ch=50, gx=10, gy=12)
    row = [cells[("L12", c)] for c in cols]
    hit = row[2]
    crack(s, f"{key}_d_cr", hit["x"], hit["y"], hit["width"], hit["height"], f)
    for k, c in enumerate(row):
        s.rect(f"{key}_d_tag{k}", c["x"] + 6, dy + 60, 44, 16, f, "tag", width=2)
    s.text(f"{key}_d_tagl", px + 4, dy + 56, "tags", f, size=17, color=VIOLET)
    right = row[-1]["x"] + row[-1]["width"]
    jx = right + 120
    s.line(f"{key}_d_load", [(right + 6, dy + 25), (jx, dy + 25)], f, color=BLUE, width=3)
    s.text(f"{key}_d_loadl", right + 14, dy - 30, "one load\nof K and V", f, size=18, color=BLUE)
    kx, kw = jx + 40, 330
    s.text(f"{key}_d_kl", kx, dy - 96, "attention kernel, one decode step", f, size=19, color=MUTED)
    s.outline(f"{key}_d_k", kx, dy - 64, kw, 172, f, MUTED, dashed=False, width=2)
    s.box(f"{key}_d_att", kx + 20, dy - 46, kw - 40, 54, "attention over K and V", f, "cell", size=18)
    s.box(f"{key}_d_chk", kx + 20, dy + 34, kw - 40, 54, "tags from the same bytes", f, "tag", size=18)
    s.arrow(f"{key}_d_ia", [(jx, dy + 25), (kx + 18, dy - 19)], f, color=BLUE, width=2)
    s.arrow(f"{key}_d_ib", [(jx, dy + 25), (kx + 18, dy + 61)], f, color=BLUE, width=2)
    # the step's token waits for the verdict on every piece it read
    tx0 = kx + kw + 64
    s.arrow(f"{key}_d_oa", [(kx + kw - 20, dy - 19), (tx0 - 6, dy - 19)], f, color=BLUE, width=2)
    s.rect(f"{key}_d_tok", tx0, dy - 34, 30, 30, f, "sus", width=2, round_=False)
    lock(s, f"{key}_d_lock", tx0 + 40, dy - 42, f, size=28)
    s.text(f"{key}_d_tokl", tx0 + 80, dy - 46, "the step's token:\nheld, not released", f, size=18, color=ORANGE)
    s.arrow(f"{key}_d_ob", [(kx + kw - 20, dy + 61), (tx0 - 6, dy + 61)], f, color=VIOLET, width=2)
    for k, c in enumerate(cols):
        x = tx0 + k * 48
        s.ctext(f"{key}_d_vl{k}", x + 14, dy + 26, c, f, size=17, color=MUTED)
        if k == 2:
            cross(s, f"{key}_d_vx", x + 2, dy + 52, f, size=24)
        else:
            check(s, f"{key}_d_vok{k}", x + 2, dy + 52, f, size=24)
    s.text(f"{key}_d_fail", tx0 + 240, dy + 30, "p64 fails: drop that token,\nrepair the piece,\nthen regenerate", f,
           size=18, color=RED)
    s.text(f"{key}_d_n1", px, dy + 120, "no extra memory traffic: the kernel already loads every byte it checks", f,
           size=19, color=VIOLET)
    s.text(f"{key}_d_n2", px, dy + 152,
           "found at the first read after the damage, so at most one step's tokens are suspect", f, size=19,
           color=ORANGE)


BITS = [("GPU 2 now", "0110101101110111", "0x6B77 = 3,822", RED),
        ("GPU 0", "0011110000011010", "0x3C1A = 1.025", INK),
        ("GPU 1", "0100000111110000", "0x41F0 = 2.969", INK),
        ("GPU 3", "0000111001010101", "0x0E55 = 0.0004", INK),
        ("parity", "0101100011001000", "0x58C8", AMBER),
        ("rebuilt GPU 2", "0010101101110111", "0x2B77 = 0.058", GREEN)]


def fig_xor(s, ox, oy):
    key, h = "xor", 690
    f = frame_box(s, key, ox, oy, h, "With the optional parity: rebuilding the piece, XOR bit by bit",
                  "stripe (page 64, layer 12), drawn for 4 GPUs, sizes for TP=8; each row is one piece, each column one "
                  "16-bit FP16 value")
    rows = [("GPU 0", "cell"), ("GPU 1", "cell"), ("GPU 2", "cell"), ("GPU 3", "cell"), ("parity", "parity")]
    slots = [("0", 0), ("1", 30), ("2", 60), ("i", 132), ("4,095", 210)]
    x0, y0 = ox + 120, oy + 150
    for r, (name, kind) in enumerate(rows):
        y = y0 + r * 54
        s.text(f"{key}_rn{r}", ox + 24, y + 4, name, f, size=20, color=AMBER if r == 4 else BLUE)
        for sname, dx in slots:
            bad = r == 2 and sname == "i"
            s.rect(f"{key}_c{r}_{sname}", x0 + dx, y, 26, 34, f, "bad" if bad else kind, width=2, round_=False)
            if bad:
                crack(s, f"{key}_cr", x0 + dx, y, 26, 34, f)
        for dx in (106, 184):
            s.ctext(f"{key}_d{r}_{dx}", x0 + dx, y - 4, "...", f, size=20, color=MUTED)
    for sname, dx in slots:
        s.ctext(f"{key}_sl_{sname}", x0 + dx + 13, y0 - 32, sname, f, color=MUTED)
    s.outline(f"{key}_col", x0 + 126, y0 - 8, 38, 4 * 54 + 34 + 16, f, INK, dashed=True, width=3)
    # the good rows and the parity row feed the XOR gate; GPU 2's own row is not read
    gate_x, gate_y = ox + 530, y0 + 2 * 54 + 17
    for r in (0, 1, 3, 4):
        s.arrow(f"{key}_in{r}", [(x0 + 244, y0 + r * 54 + 17), (gate_x - 30, gate_y)], f,
                color=AMBER if r == 4 else MUTED, width=2)
    s.text(f"{key}_skip", x0 + 250, y0 + 2 * 54 + 6, "not read", f, color=RED)
    xor_gate(s, f"{key}_xor", gate_x, gate_y, f, r=28)
    s.arrow(f"{key}_outa", [(gate_x + 30, gate_y), (gate_x + 60, gate_y)], f, color=GREEN, width=3)
    s.rect(f"{key}_out", gate_x + 64, gate_y - 17, 26, 34, f, "good", width=2, round_=False)
    s.text(f"{key}_outl", gate_x + 42, gate_y + 26, "same slot\nin GPU 2", f, color=GREEN)
    # the bits of column i: the damaged value stands apart; the four inputs XOR to the original
    bx, bit_w, nib = ox + 700, 26, 12

    def bit_x(b):
        return bx + 175 + b * bit_w + (b // 4) * nib

    val_x = bit_x(15) + 34
    s.text(f"{key}_coll", bx, oy + 108, "column i (dashed box), bit by bit:", f, size=20)
    s.text(f"{key}_bith", bit_x(0) - 6, oy + 146, "bit 15", f, size=17, color=MUTED)
    s.text(f"{key}_bitl", bit_x(15) - 20, oy + 146, "bit 0", f, size=17, color=MUTED)
    ys = [oy + 176, oy + 262, oy + 308, oy + 354, oy + 400, oy + 466]
    for r, (label, bits, value, color) in enumerate(BITS):
        y = ys[r]
        if r in (2, 3, 4):
            xor_gate(s, f"{key}_bx{r}", bx + 11, y + 13, f, r=10)
        s.text(f"{key}_bl{r}", bx + (30 if r in (2, 3, 4) else 0), y, label, f, size=20, color=color)
        for b, ch in enumerate(bits):
            s.text(f"{key}_b{r}_{b}", bit_x(b), y - 2, ch, f, size=22, color=color)
        s.text(f"{key}_bv{r}", val_x, y, value, f, size=20, color=color)
    s.outline(f"{key}_flip", bit_x(1) - 5, ys[0] - 6, 24, 36, f, RED, dashed=False, width=2)
    s.text(f"{key}_flipl", bit_x(1) - 10, ys[0] + 34, "bit 14 flipped", f, size=17, color=RED)
    s.text(f"{key}_unused", val_x, ys[0] + 30, "not an input", f, size=17, color=RED)
    s.line(f"{key}_div", [(bx, oy + 238), (val_x + 180, oy + 238)], f, color="#ced4da", width=2, dashed=True)
    s.line(f"{key}_sum", [(bx + 160, oy + 448), (bit_x(15) + 20, oy + 448)], f, width=2)
    check(s, f"{key}_ok", val_x + 180, ys[5] + 2, f, size=22)
    s.text(f"{key}_note", bx, oy + 530,
           "XOR of rows GPU 0, GPU 1, GPU 3 and parity = the original value.\n"
           "The same XOR runs over all 4,096 values of the piece.\n"
           "At TP=8 a rebuild reads 7 pieces + 1 parity piece = 64 KiB.", f, size=19)
    # when every GPU's piece is bad there is nothing clean to XOR: recompute the page's tokens and the suspect ones
    ix = ox + 24
    s.text(f"{key}_inset_t", ix, oy + 450, "without parity, or when every GPU's piece is damaged:", f, size=20, color=RED)
    for g in range(4):
        s.rect(f"{key}_ins{g}", ix + 10 + g * 32, oy + 494, 26, 34, f, "bad", width=2, round_=False)
    xor_gate(s, f"{key}_ixor", ix + 178, oy + 511, f, r=20, crossed=True)
    s.text(f"{key}_ino", ix + 214, oy + 500, "nothing clean to XOR", f, color=RED)
    ry = oy + 600
    for k in range(3):
        s.rect(f"{key}_old{k}", ix + 10 + k * 28, ry, 24, 26, f, "cell", width=2, round_=False)
    s.ctext(f"{key}_oldd", ix + 108, ry - 6, "...", f, size=20, color=MUTED)
    s.text(f"{key}_oldl", ix + 4, ry + 34, "0-1,023 intact", f, color=BLUE)
    s.arrow(f"{key}_ia", [(ix + 130, ry + 13), (ix + 180, ry + 13)], f, color=BLUE, width=3)
    for k in range(8):
        s.rect(f"{key}_new{k}", ix + 190 + k * 28 + (12 if k >= 4 else 0), ry, 24, 26, f, "good", width=2, round_=False)
    g0, g1 = ix + 188, ix + 424
    s.line(f"{key}_ibr", [(g0, ry - 6), (g0, ry - 14), (g1, ry - 14), (g1, ry - 6)], f, color=GREEN, width=3)
    s.ctext(f"{key}_ibrl", (g0 + g1) / 2, ry - 44, "recomputed: 16 + 40 = 56 tokens in total", f, color=GREEN)
    s.ctext(f"{key}_ip", ix + 244, ry + 34, "page 64", f, color=GREEN)
    s.ctext(f"{key}_is", ix + 368, ry + 34, "40 suspect", f, color=GREEN)


def fig_spread(s, ox, oy):
    key, h = "spread", 650
    f = frame_box(s, key, ox, oy, h, "Why damage spreads: attention mixes it into later tokens",
                  "request R; columns are tokens, rows are layers; each cell is one token's KV in one layer")
    y0, ch, cw = oy + 200, 44, 58
    rows = [("L32", y0), ("L13", y0 + 88), ("L12", y0 + 150), ("L11", y0 + 212), ("L1", y0 + 300)]
    dots = [y0 + 50, y0 + 262]
    for name, y in rows:
        s.text(f"{key}_row{name}", ox + 40, y + 10, name, f, color=MUTED)
    for d, dy in enumerate(dots):
        s.text(f"{key}_rowd{d}", ox + 44, dy, "...", f, size=20, color=MUTED)
    spec = [("e0", 120, "old"), ("e1", 184, "old"), ("pg", 290, "page"), ("l0", 396, "old"), ("l1", 460, "old"),
            ("n0", 572, "new"), ("n1", 690, "new"), ("n2", 754, "new"), ("n3", 818, "new")]
    col = {}
    for cid, x, kind in spec:
        col[cid] = {}
        for name, y in rows:
            if kind == "page":
                k = "bad" if name == "L12" else "cell"
            elif kind == "new":
                k = "sus" if name in ("L32", "L13") else "good"
            else:
                k = "cell"
            col[cid][name] = s.rect(f"{key}_{cid}_{name}", ox + x, y, cw, ch, f, k, width=2)
        for d, dy in enumerate(dots):
            s.ctext(f"{key}_{cid}_dots{d}", ox + x + cw / 2, dy, "...", f, size=20, color=MUTED)
    labels = [(181, "tokens\n0-1,023", INK), (319, "page 64\n1,024-1,039", INK), (457, "tokens\n1,040-32,767", INK),
              (724, "R's 40 suspect tokens\n(drawn as 4)", ORANGE)]
    for k, (cx, text, color) in enumerate(labels):
        s.ctext(f"{key}_gl{k}", ox + cx, oy + 128, text, f, color=color)
    pg = col["pg"]["L12"]
    crack(s, f"{key}_cr", pg["x"], pg["y"], cw, ch, f)
    # one new token's layer-12 attention reads every earlier token's layer-12 KV (the dot) ...
    bus_y = y0 + 141
    fx = ox + 572 + cw / 2
    for k, cid in enumerate(["e0", "e1", "pg", "l0", "l1"]):
        c = col[cid]["L12"]
        cx = c["x"] + cw / 2
        s.line(f"{key}_up{k}", [(cx, c["y"] - 1), (cx, bus_y)], f, color=RED if cid == "pg" else "#868e96",
               width=4 if cid == "pg" else 2)
    s.line(f"{key}_bus", [(ox + 149, bus_y), (fx - 10, bus_y)], f, color="#868e96", width=2)
    s.ellipse(f"{key}_dot", fx - 9, bus_y - 9, 18, 18, f, stroke=INK, fill=INK, width=1)
    # ... and its output is carried up into layers 13-32
    s.arrow(f"{key}_upflow", [(fx + 10, bus_y), (ox + 654, bus_y), (ox + 654, y0 + 4)], f, color=ORANGE, width=4)
    lx = ox + 960
    legend = [("bad", "damaged piece"), ("sus", "suspect: may be built on the damage"),
              ("good", "clean: layers 1-12 of the new tokens"), ("cell", "clean: computed before step 9,000")]
    for k, (kind, text) in enumerate(legend):
        s.rect(f"{key}_lg{k}", lx, oy + 192 + k * 44, 26, 22, f, kind, width=2)
        s.text(f"{key}_lgt{k}", lx + 38, oy + 189 + k * 44, text, f, size=19)
    y = oy + 380
    s.line(f"{key}_lgbus", [(lx, y + 12), (lx + 20, y + 12)], f, color="#868e96", width=2)
    s.ellipse(f"{key}_lgdot", lx + 18, y + 4, 16, 16, f, stroke=INK, fill=INK, width=1)
    s.text(f"{key}_lgbt", lx + 46, y, "one new token's layer-12 attention reads\nevery earlier token's layer-12 KV", f, size=19)
    y = oy + 446
    s.arrow(f"{key}_lgup", [(lx + 13, y + 40), (lx + 13, y)], f, color=ORANGE, width=4)
    s.text(f"{key}_lgut", lx + 46, y + 6, "its output is carried up into layers 13-32", f, size=19)
    s.text(f"{key}_rule", lx, oy + 516, "repair set = the damaged piece +\nlayers 13-32 of the 40 suspect tokens", f,
           size=T_NOTE)
    s.text(f"{key}_rule2", lx, oy + 582, "tokens computed before step 9,000 read\npage 64 while it was still correct", f,
           color=MUTED)


def fig_rollback(s, ox, oy):
    key, h = "rollback", 950
    f = frame_box(s, key, ox, oy, h, "Mending what the damage reached, and guarding the output",
                  "R's tokens at three moments under fast release, the cached KV of requests that shared page 64, and "
                  "what the user receives")

    def r_strip(p, y, page, tail, label, color):
        s.text(f"{p}_l", ox + 24, y - 48, label, f, size=20, color=color)
        items = ["cell"] * 3 + ["..."] + [page] + ["..."] + ["cell"] * 3 + tail
        return strip(s, p, ox + 40, y, f, items, w=30, h=30, gap=6)

    sq1 = r_strip(f"{key}_r1", oy + 168, "bad", ["sus"] * 10,
                  "(1) at step 9,040: page 64 damaged, 40 suspect tokens (drawn as 10)", RED)
    sq2 = r_strip(f"{key}_r2", oy + 312, "good", ["empty"] * 10, "(2) page 64 recomputed; the 40 tokens' KV dropped", ORANGE)
    sq3 = r_strip(f"{key}_r3", oy + 456, "good", ["good"] * 10 + ["cell"],
                  "(3) the 40 tokens re-prefilled in one pass; R resumes", GREEN)
    s.ctext(f"{key}_pg", center(sq1[3])[0], oy + 204, "page 64", f, color=MUTED)
    a, b = sq2[-10], sq2[-1]
    s.arrow(f"{key}_rw", [(b["x"] + 30, oy + 300), (a["x"], oy + 300)], f, color=ORANGE, width=3)
    s.text(f"{key}_rwl", b["x"] + 44, oy + 284, "roll back\nto step 9,000", f, size=19, color=ORANGE)
    a, b = sq3[-11], sq3[-2]
    s.line(f"{key}_br", [(a["x"] - 2, oy + 452), (a["x"] - 2, oy + 444), (b["x"] + 32, oy + 444), (b["x"] + 32, oy + 452)],
           f, color=GREEN, width=3)
    s.ctext(f"{key}_brl", (a["x"] + b["x"] + 30) / 2, oy + 496, "one prefill pass over 40 tokens", f, color=GREEN)
    nxt = sq3[-1]
    s.ctext(f"{key}_nextl", nxt["x"] + 15, oy + 496, "next", f, color=BLUE)
    play_icon(s, f"{key}_play", nxt["x"] + 44, oy + 458, f, size=26)
    # cached KV: every request that read page 64 lies below its radix node
    tx = ox + 900
    s.text(f"{key}_tt", tx, oy + 120, "(4) cached KV in the radix tree", f, size=20)
    s.ellipse(f"{key}_root", tx + 296, oy + 164, 28, 28, f, stroke=MUTED, fill="#ffffff")
    s.box(f"{key}_node", tx + 180, oy + 226, 260, 50, "prompt incl. page 64", f, "cell", size=19)
    s.line(f"{key}_e0", [(tx + 310, oy + 192), (tx + 310, oy + 226)], f, color=MUTED, width=2)
    kids = [(tx, "R: running", "cell", False), (tx + 210, "S: finished\nat step 9,020", "sus", True),
            (tx + 420, "T: finished\nat step 8,950", "cell", False)]
    for k, (kx, label, kind, dashed) in enumerate(kids):
        s.box(f"{key}_kid{k}", kx, oy + 340, 180, 64, label, f, kind, dashed=dashed)
        s.line(f"{key}_e{k + 1}", [(tx + 310, oy + 276), (kx + 90, oy + 340)], f, color=MUTED, width=2)
    cross(s, f"{key}_sx", tx + 376, oy + 324, f, size=26)
    check(s, f"{key}_tok", tx + 586, oy + 322, f, size=26)
    s.text(f"{key}_rl", tx, oy + 414, "rolled back\n(rows 1-3)", f, color=BLUE)
    s.text(f"{key}_sl", tx + 210, oy + 414, "evicted: computed\nafter step 9,000", f, color=RED)
    s.text(f"{key}_tl", tx + 420, oy + 414, "kept: computed\nbefore step 9,000", f, color=GREEN)
    s.text(f"{key}_note", tx, oy + 490,
           "every request that read page 64 lies below\nits node, so one subtree walk finds all\nsuspect cached KV", f,
           size=19, color=MUTED)
    # (5) what the user receives: a token is released only after a check covers what it read
    top = oy + 600
    s.line(f"{key}_usep", [(ox + 24, top), (ox + WIDE - 24, top)], f, color="#dee2e6", width=2)
    s.text(f"{key}_ut", ox + 24, top + 18,
           "(5) what the user receives, under strict and under fast release", f, size=20)
    tokens = ["cell"] * 3 + ["..."] + ["sus"] * 10
    # strict release: the suspect tokens were never sent, so they are dropped and regenerated
    ya = oy + 712
    s.text(f"{key}_ual", ox + 24, ya - 44, "strict release", f, size=19, color=VIOLET)
    qa = strip(s, f"{key}_ua", ox + 40, ya, f, tokens, w=30, h=30, gap=6)
    first, last = qa[3], qa[-1]
    s.outline(f"{key}_uahold", first["x"] - 8, ya - 8, last["x"] + 38 - first["x"], 46, f, VIOLET, dashed=True, width=3)
    lock(s, f"{key}_ualock", last["x"] + 44, ya - 2, f, size=28)
    s.text(f"{key}_uahl", first["x"] - 8, ya + 46, "40 suspect tokens, held until checked", f, size=18, color=VIOLET)
    s.arrow(f"{key}_uaa", [(last["x"] + 82, ya + 15), (last["x"] + 182, ya + 15)], f, color=ORANGE, width=3)
    s.text(f"{key}_uaal", last["x"] + 80, ya - 44, "repair, drop,\nregenerate", f, size=18, color=ORANGE)
    ga = strip(s, f"{key}_uag", last["x"] + 196, ya, f, ["good"] * 10, w=30, h=30, gap=6)
    gl = ga[-1]
    s.arrow(f"{key}_uau", [(gl["x"] + 40, ya + 15), (gl["x"] + 110, ya + 15)], f, color=GREEN, width=3)
    person(s, f"{key}_uauser", gl["x"] + 118, ya - 8, "user", f, w=84)
    check(s, f"{key}_uaok", gl["x"] + 214, ya - 4, f, size=26)
    s.text(f"{key}_uaol", gl["x"] + 250, ya - 2, "gets the right text", f, size=18, color=GREEN)
    # fast release: the suspect tokens were already sent; they are kept and reported
    yb = oy + 846
    s.text(f"{key}_ubl", ox + 24, yb - 44, "fast release", f, size=19, color=ORANGE)
    qb = strip(s, f"{key}_ub", ox + 40, yb, f, tokens, w=30, h=30, gap=6)
    lb = qb[-1]
    s.arrow(f"{key}_uba", [(lb["x"] + 40, yb + 15), (gl["x"] + 110, yb + 15)], f, color=ORANGE, width=3)
    s.text(f"{key}_ubal", lb["x"] + 60, yb - 30, "sent at once, before any check", f, size=18, color=ORANGE)
    person(s, f"{key}_ubuser", gl["x"] + 118, yb - 8, "user", f, w=84)
    warning_icon(s, f"{key}_ubwarn", gl["x"] + 212, yb - 6, f, size=34)
    s.text(f"{key}_ubwl", gl["x"] + 254, yb - 10, "told which tokens\nmay be wrong", f, size=18, color=RED)
    s.text(f"{key}_ubn", lb["x"] + 60, yb + 48,
           "kept, re-prefilled as they were, and reported in the response's metadata", f, size=18, color=MUTED)


def fig_lists(s, ox, oy):
    key, h = "lists", 620
    f = frame_box(s, key, ox, oy, h, "When to repair, and the watch list",
                  "left: three damaged pieces with different next readers; right: what is watched after a repair")
    # repair list: one lane per damaged piece, time running right
    s.text(f"{key}_rlt", ox + 24, oy + 108, "repair list", f, size=T_PANEL)
    x0, x1, ya = ox + 200, ox + 920, oy + 176
    s.arrow(f"{key}_axis", [(x0, ya), (x1, ya)], f, color=MUTED, width=2)
    s.text(f"{key}_now", x0 - 10, ya - 30, "now", f, color=MUTED)
    s.text(f"{key}_later", x1 - 56, ya - 30, "later", f, color=MUTED)
    lanes = {"X": oy + 270, "Y": oy + 390, "Z": oy + 510}
    for name, y in lanes.items():
        s.box(f"{key}_p{name}", ox + 40, y - 22, 56, 44, name, f, "bad", size=22)
        start = x0 + 300 if name == "Z" else x0
        s.line(f"{key}_lane{name}", [(start, y), (x1, y)], f, color="#dee2e6", width=2, dashed=(name == "Z"))
    y = lanes["X"]
    s.box(f"{key}_xr", x0 + 10, y - 16, 96, 32, "repair", f, "good", size=17)
    person(s, f"{key}_xreader", x0 + 120, y - 23, "R", f, w=52)
    s.text(f"{key}_xt", x0 + 200, y - 64, "next read: the very next step,\nso repair at once; only R waits", f)
    y = lanes["Y"]
    s.box(f"{key}_yb1", x0 + 10, y - 13, 130, 26, "busy", f, "gray", size=17)
    s.box(f"{key}_yr", x0 + 150, y - 16, 110, 32, "repair", f, "good", size=17)
    s.box(f"{key}_yb2", x0 + 270, y - 13, 170, 26, "busy", f, "gray", size=17)
    person(s, f"{key}_yreader", x1 - 120, y - 23, "S", f, w=52)
    clock_icon(s, f"{key}_yclk", x1 - 150, y - 50, f)
    s.text(f"{key}_yclkl", x1 - 130, y - 74, "reads in 5 s", f, size=17, color=MUTED)
    s.text(f"{key}_yt", x0 + 10, y - 64, "repaired in idle GPU time, before the read", f)
    y = lanes["Z"]
    person(s, f"{key}_zown", x0 + 10, y - 23, "U done", f, w=96, dashed=True)
    trash_icon(s, f"{key}_ztrash", x0 + 140, y - 20, f)
    s.text(f"{key}_ztl", x0 + 184, y - 12, "page freed", f, size=17, color=MUTED)
    s.text(f"{key}_zt", x0 + 10, y - 64, "owner done, page not shared: freed, never repaired", f)
    # watch list: the repaired slot's neighbours and the owner's other pages
    wx = ox + 1000
    s.line(f"{key}_sep", [(wx - 40, oy + 100), (wx - 40, oy + h - 24)], f, color="#dee2e6", width=2)
    s.text(f"{key}_wt", wx, oy + 108, "watch list", f, size=T_PANEL, color=VIOLET)
    for k, nm in enumerate(["s-2", "s-1", "s", "s+1", "s+2"]):
        x = wx + k * 70
        s.rect(f"{key}_w{k}", x, oy + 200, 56, 44, f, "good" if nm == "s" else "cell", width=2)
        s.ctext(f"{key}_wl{k}", x + 28, oy + 172, nm, f, color=MUTED)
    s.outline(f"{key}_hood", wx + 62, oy + 164, 212, 92, f, VIOLET, dashed=True, width=3)
    magnifier(s, f"{key}_wm", wx + 262, oy + 120, f, r=16)
    s.text(f"{key}_wlab", wx, oy + 264, "layer 12: repaired slot s and its neighbours", f, color=MUTED)
    for k, nm in enumerate(["p0", "p63", "p65"]):
        x = wx + k * 66
        s.rect(f"{key}_o{k}", x, oy + 344, 56, 44, f, "cell", width=2)
        s.ctext(f"{key}_ol{k}", x + 28, oy + 316, nm, f, color=MUTED)
    s.outline(f"{key}_own", wx - 10, oy + 308, 208, 92, f, VIOLET, dashed=True, width=3)
    s.text(f"{key}_owl", wx + 216, oy + 352, "R's other pages", f, color=MUTED)
    s.text(f"{key}_wn", wx, oy + 412, "all checked more often for a while", f, size=19, color=VIOLET)
    # a slot that fails twice is retired: the whole page moves
    ry = oy + 470
    s.text(f"{key}_rt", wx, ry, "a slot that fails twice is retired:", f, size=19)

    def mini_page(p, x, y, kind):
        for g in range(4):
            for layer in range(3):
                s.rect(f"{p}_{g}_{layer}", x + g * 20, y + layer * 15, 16, 11, f, kind, width=1, round_=False)

    mini_page(f"{key}_old", wx, ry + 44, "gray")
    cross(s, f"{key}_ox1", wx + 86, ry + 36, f, size=16)
    cross(s, f"{key}_ox2", wx + 106, ry + 36, f, size=16)
    s.text(f"{key}_oldl", wx, ry + 96, "old slot: retired", f, size=17, color=MUTED)
    s.arrow(f"{key}_ra", [(wx + 136, ry + 66), (wx + 206, ry + 66)], f, color=MUTED, width=3)
    mini_page(f"{key}_new", wx + 216, ry + 44, "cell")
    s.text(f"{key}_newl", wx + 306, ry + 44, "whole page moved:\nall layers, all GPUs", f, size=17, color=MUTED)


# Scene -------------------------------------------------------------------------------------------

def build():
    s = Scene()
    y = 0
    for draw, height in ((fig_story, 640), (fig_vocab, 690), (fig_timeline, 700), (fig_normal, 700), (fig_detect, 940),
                         (fig_life, 770), (fig_xor, 690), (fig_spread, 650), (fig_rollback, 950), (fig_lists, 620)):
        draw(s, 0, y)
        y += height + GAP
    if s.overflow:
        print("label overflow warnings:\n  " + "\n  ".join(s.overflow))
    return {
        "type": "excalidraw", "version": 2, "source": "https://excalidraw.com",
        "elements": s.elements,
        "appState": {"viewBackgroundColor": "#ffffff", "exportBackground": True, "exportWithDarkMode": False,
                     "gridSize": None, "frameRendering": {"enabled": True, "name": False, "outline": True, "clip": True}},
        "files": {},
    }


def view(scene, frames):
    keep = set(frames)
    return {**scene, "elements": [e for e in scene["elements"] if e["id"] in keep or e.get("frameId") in keep]}


def render(scene, folder, only=None):
    for name, frames in VIEWS.items():
        if only and name not in only:
            continue
        subprocess.run(["excalidraw", "render", "-", "--out", str(folder / name), "--scale", "2",
                        "--max-width", "4096", "--max-height", "4096"],
                       input=json.dumps(view(scene, frames)), text=True, check=True, capture_output=True)
        print(f"rendered {folder / name}")


if __name__ == "__main__":
    folder = Path(__file__).parent
    scene = build()
    out = folder / "kvmend.excalidraw"
    out.write_text(json.dumps(scene, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({len(scene['elements'])} elements)")
    if "--render" in sys.argv:
        only = [a for a in sys.argv[1:] if a.endswith(".png")]
        render(scene, folder, only or None)
