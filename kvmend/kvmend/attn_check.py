"""Checks on read: KVMend's tag check inside SGLang 0.5.1's Triton decode attention (KVMEND_CHECK_IN_ATTN=1).

A decode step loads every K and V byte of every page that each request reads, so the check rides on those loads and
adds no pass over memory. The two stage-1 kernels below are copies of SGLang 0.5.1's _fwd_kernel_stage1 (one query
head per KV head) and _fwd_grouped_kernel_stage1 (grouped-query attention), from
sglang/srt/layers/attention/triton_ops/decode_attention.py. Every line of the originals is kept, in order
(tests_gpu/test_attn_check.py checks this against the installed SGLang), so the attention output is SGLang's bit for
bit. Under the constexpr CHECK, each program also does this for every block of BLOCK_N tokens it reads:
- forms the 32-bit words of its KV head's K and V of those tokens, from the tiles the attention just loaded (REUSE),
  or, with REUSE=False (KVMEND_ATTN_RELOAD=1), by loading the same bytes again as int32 words, a fallback that leaves
  the attention's own tiles untouched;
- XORs each word with the position key of its place in the piece, pkey[kv, offset, j] with
  j = head * (head_dim / 2) + word within the head, and with the token key of its slot (tags.py), mixes it with
  fmix32 and adds a token's words up: the token's share, for this head, of its piece's tag;
- adds the shares of each page's tokens (16 consecutive positions of one request) to a per-(layer, page) int64
  accumulator with one atomic add, sets the bits of the offsets read in a per-(layer, page) coverage accumulator with
  one atomic or, and, in the program of KV head 0, counts one reader of the page.
A piece's tag is a sum of independent parts, so the programs that read different heads, and the splits that read
different pages, just add their parts. Programs that read the same KV head (more than BLOCK_H query heads per KV
head) add only once. Entries of a CUDA-graph batch that are padding (sequence length 1, SGLang's fill value for the
Triton backend) add nothing: they read the first slot of a stale request-pool row, which may be a live page.

After the step, verdict() checks every (layer, page) read: the offsets read must equal the offsets written (the
stored mask; the open page is read up to its last token, and so was written), and the accumulated tag must equal
readers x the stored tag, modulo 2^32, since a shared prefix page is read once by each request that holds it. Then it
zeroes the accumulators. Page 0 is SGLang's padding page and is skipped.

The token keys used are the slots' own (st.tkey, set when the slot was written), so this check finds damaged bytes
and slots read but never written; KV of another sequence whose bytes, tags and keys agree is the reader's check's
job (hook.act_check), which uses the reader's own token keys.
"""
import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.triton_ops.decode_attention import _MIN_BLOCK_KV, _decode_softmax_reducev_fwd
from sglang.srt.layers.attention.triton_ops.decode_attention import tanh

from kvmend.tag_kernels import _fmix32

PAGE = 16
REASONS = {1: "coverage", 2: "tag"}
VBLOCK = 1024  # (layer, page) entries per program of the verdict


# ---------------------------------------------------------------------------------------------------------------
# The check's part of a block of BLOCK_N tokens

@triton.jit
def _bit_or(a, b):
    return a | b


@triton.jit
def _pair_words(x, N: tl.constexpr, D: tl.constexpr):
    """[N, D] fp16 or bf16 values -> [N, D // 2] uint32 words: word w holds element 2w in its low half and element
    2w + 1 in its high half, as torch's view(torch.int32) reads them on a little-endian GPU."""
    u = x.to(tl.uint16, bitcast=True)
    lo, hi = tl.split(tl.reshape(u, [N, D // 2, 2]))
    return lo.to(tl.uint32) | (hi.to(tl.uint32) << 16)


@triton.jit
def _load_words(Words, kv_loc, head, stride_bs, stride_h, valid, WH: tl.constexpr, BW: tl.constexpr):
    """REUSE=False: the head's words of the tokens at slots kv_loc, loaded again as int32: [N, BW] uint32."""
    w = tl.arange(0, BW)
    ptr = Words + kv_loc[:, None] * stride_bs + head * stride_h + w[None, :]
    return tl.load(ptr, mask=valid[:, None] & (w < WH)[None, :], other=0).to(tl.uint32, bitcast=True)


@triton.jit
def _add_shares(kw, vw, kv_loc, valid, head, tag_me, count_me, Tkey, Pkey, Acc_Tag, Acc_Mask, Acc_Cnt, Stat,
                W: tl.constexpr, WK: tl.constexpr, WV: tl.constexpr, BWK: tl.constexpr, BWV: tl.constexpr,
                BLOCK_N: tl.constexpr, PG: tl.constexpr):
    """Add the tag shares of one block of BLOCK_N tokens (positions start_n + 0 .. BLOCK_N - 1 of one request, where
    start_n is a multiple of PG) to the accumulators of their pages. kw [BLOCK_N, BWK], vw [BLOCK_N, BWV]: uint32
    words of KV head `head`; kv_loc: the tokens' slots; valid: the tokens this program reads."""
    rows = tl.arange(0, BLOCK_N)
    off = rows % PG  # a token's offset in its page, as SGLang's paged allocator lays out a request page by page
    wk = tl.arange(0, BWK)
    wv = tl.arange(0, BWV)
    pkk = tl.load(Pkey + off[:, None] * W + head * WK + wk[None, :], mask=(wk < WK)[None, :], other=0)
    pkv = tl.load(Pkey + PG * W + off[:, None] * W + head * WV + wv[None, :], mask=(wv < WV)[None, :], other=0)
    tk = tl.load(Tkey + kv_loc, mask=valid, other=0).to(tl.uint32)  # the slot's token key, < 2^32
    hk = _fmix32(kw ^ pkk.to(tl.uint32, bitcast=True) ^ tk[:, None])
    hv = _fmix32(vw ^ pkv.to(tl.uint32, bitcast=True) ^ tk[:, None])
    hk = tl.where((wk < WK)[None, :], hk, tl.zeros_like(hk))
    hv = tl.where((wv < WV)[None, :], hv, tl.zeros_like(hv))
    share = tl.sum(hk, axis=1) + tl.sum(hv, axis=1)  # [BLOCK_N] uint32: each token's share for this head
    # The block's tokens in groups of PG: group g is one page. Sums wrap modulo 2^32, as the tags do.
    G: tl.constexpr = BLOCK_N // PG
    grp = tl.arange(0, G)
    sel = ((rows // PG)[None, :] == grp[:, None]) & valid[None, :]  # [G, BLOCK_N]
    gsum = tl.sum(tl.where(sel, share[None, :], tl.zeros([G, BLOCK_N], dtype=tl.uint32)), axis=1)
    page = kv_loc // PG
    slot_off = (kv_loc - page * PG).to(tl.int32)
    bits = tl.full([BLOCK_N], 1, tl.int32) << slot_off
    gbits = tl.reduce(tl.where(sel, bits[None, :], tl.zeros([G, BLOCK_N], dtype=tl.int32)), 1, _bit_or)
    gmax = tl.max(tl.where(sel, page[None, :], tl.full([G, BLOCK_N], -1, tl.int64)), axis=1)
    gmin = tl.min(tl.where(sel, page[None, :], tl.full([G, BLOCK_N], 1 << 62, tl.int64)), axis=1)
    has = gmax >= 0
    # Layout anomalies (a group spread over two pages, or a slot at another offset than its position): counted, and
    # the host reports them; the shares are then attributed to the wrong place, so the page's check fails too.
    nbad = (tl.sum((has & (gmin != gmax)).to(tl.int32), axis=0)
            + tl.sum((valid & (slot_off != off)).to(tl.int32), axis=0))
    add = has & tag_me
    tl.atomic_add(Acc_Tag + gmax, gsum.to(tl.int64), mask=add, sem="relaxed")
    tl.atomic_or(Acc_Mask + gmax, gbits, mask=add, sem="relaxed")
    tl.atomic_add(Acc_Cnt + gmax, tl.full([G], 1, tl.int32), mask=add & count_me, sem="relaxed")
    tl.atomic_add(Stat + 2, nbad, mask=nbad > 0, sem="relaxed")


# ---------------------------------------------------------------------------------------------------------------
# SGLang 0.5.1's stage-1 kernels, with the check. Lines marked "# KVMend" are added; every other line is SGLang's.

@triton.jit
def _fwd_kernel_stage1_check(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    K_Words, V_Words, stride_kw_bs, stride_kw_h, stride_vw_bs, stride_vw_h,  # KVMend
    Tkey, Pkey, Acc_Tag, Acc_Mask, Acc_Cnt, Stat,  # KVMend
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    CHECK: tl.constexpr, REUSE: tl.constexpr, W: tl.constexpr, PG: tl.constexpr,  # KVMend
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        if CHECK:  # KVMend
            tag_me = (cur_head % kv_group_num) == 0  # KVMend: one program per KV head adds its shares
            count_me = cur_head == 0  # KVMend: and one per (request, page) counts the reader
            real = cur_batch_seq_len >= 2  # KVMend: a padding entry of a CUDA-graph batch reads 1 token
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

            if CHECK:  # KVMend: the same tokens' tag shares
                valid = (offs_n < split_kv_end) & real  # KVMend
                if REUSE:  # KVMend
                    kw = _pair_words(k, BLOCK_N, BLOCK_DMODEL)  # KVMend
                    vw = _pair_words(v, BLOCK_N, BLOCK_DV)  # KVMend
                else:  # KVMend
                    kw = _load_words(K_Words, kv_loc, cur_kv_head, stride_kw_bs, stride_kw_h, valid, Lk // 2,  # KVMend
                                     BLOCK_DMODEL // 2)  # KVMend
                    vw = _load_words(V_Words, kv_loc, cur_kv_head, stride_vw_bs, stride_vw_h, valid, Lv // 2,  # KVMend
                                     BLOCK_DV // 2)  # KVMend
                _add_shares(kw, vw, kv_loc, valid, cur_kv_head, tag_me, count_me, Tkey, Pkey, Acc_Tag,  # KVMend
                            Acc_Mask, Acc_Cnt, Stat, W, Lk // 2, Lv // 2, BLOCK_DMODEL // 2, BLOCK_DV // 2,  # KVMend
                            BLOCK_N, PG)  # KVMend

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


@triton.jit
def _fwd_grouped_kernel_stage1_check(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    K_Words, V_Words, stride_kw_bs, stride_kw_h, stride_vw_bs, stride_vw_h,  # KVMend
    Tkey, Pkey, Acc_Tag, Acc_Mask, Acc_Cnt, Stat,  # KVMend
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    xai_temperature_len: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    CHECK: tl.constexpr, REUSE: tl.constexpr, W: tl.constexpr, PG: tl.constexpr,  # KVMend
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
        if BLOCK_DPE > 0:
            qpe = tl.load(
                Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
            )
        if CHECK:  # KVMend
            tag_me = (cur_head_id % tl.cdiv(kv_group_num, BLOCK_H)) == 0  # KVMend: one program per KV head adds
            count_me = cur_head_id == 0  # KVMend: and one per (request, page) counts the reader
            real = cur_batch_seq_len >= 2  # KVMend: a padding entry of a CUDA-graph batch reads 1 token
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg[:, None]

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

            if CHECK:  # KVMend: the same tokens' tag shares
                valid = (offs_n < split_kv_end) & real  # KVMend
                if REUSE:  # KVMend: k is [BLOCK_DMODEL, BLOCK_N] here, v is [BLOCK_N, BLOCK_DV]
                    kw = _pair_words(tl.trans(k), BLOCK_N, BLOCK_DMODEL)  # KVMend
                    vw = _pair_words(v, BLOCK_N, BLOCK_DV)  # KVMend
                else:  # KVMend
                    kw = _load_words(K_Words, kv_loc, cur_kv_head, stride_kw_bs, stride_kw_h, valid, Lk // 2,  # KVMend
                                     BLOCK_DMODEL // 2)  # KVMend
                    vw = _load_words(V_Words, kv_loc, cur_kv_head, stride_vw_bs, stride_vw_h, valid, Lv // 2,  # KVMend
                                     BLOCK_DV // 2)  # KVMend
                _add_shares(kw, vw, kv_loc, valid, cur_kv_head, tag_me, count_me, Tkey, Pkey, Acc_Tag,  # KVMend
                            Acc_Mask, Acc_Cnt, Stat, W, Lk // 2, Lv // 2, BLOCK_DMODEL // 2, BLOCK_DV // 2,  # KVMend
                            BLOCK_N, PG)  # KVMend

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


# ---------------------------------------------------------------------------------------------------------------
# Launchers: SGLang 0.5.1's _decode_att_m_fwd and _decode_grouped_att_m_fwd (CUDA branches), with the check's
# arguments. The check runs on CUDA only, so the HIP settings are left out.

def _decode_att_m_fwd_check(q, k_buffer, v_buffer, att_out, att_lse, kv_indptr, kv_indices, num_kv_splits,
                            max_kv_splits, sm_scale, logit_cap, xai_temperature_len, chk):
    BLOCK = 64
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1_check[grid](
        q, k_buffer, v_buffer, sm_scale, kv_indptr, kv_indices, att_out, att_lse, num_kv_splits,
        q.stride(0), q.stride(1), k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        *chk["args"],
        kv_group_num=kv_group_num, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DV=BLOCK_DV, BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV, logit_cap=logit_cap, xai_temperature_len=xai_temperature_len,
        num_warps=num_warps, num_stages=2, Lk=Lk, Lv=Lv,
        CHECK=chk["check"], REUSE=chk["reuse"], W=chk["w"], PG=PAGE,
    )


def _decode_grouped_att_m_fwd_check(q, k_buffer, v_buffer, att_out, att_lse, kv_indptr, kv_indices, num_kv_splits,
                                    max_kv_splits, sm_scale, logit_cap, xai_temperature_len, chk):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    BLOCK_H = 16
    MAX_KV_SPLITS = max_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        MAX_KV_SPLITS,
    )

    _fwd_grouped_kernel_stage1_check[grid](
        q, k_buffer, v_buffer, sm_scale, kv_indptr, kv_indices, att_out, att_lse, num_kv_splits,
        q.stride(0), q.stride(1), k_buffer.stride(0), k_buffer.stride(1), v_buffer.stride(0), v_buffer.stride(1),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        *chk["args"],
        kv_group_num=kv_group_num, q_head_num=head_num, BLOCK_DMODEL=BLOCK_DMODEL, BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV, BLOCK_N=BLOCK, BLOCK_H=BLOCK_H, MIN_BLOCK_KV=_MIN_BLOCK_KV, logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len, num_warps=4, num_stages=2, Lk=Lk, Lv=Lv,
        CHECK=chk["check"], REUSE=chk["reuse"], W=chk["w"], PG=PAGE,
    )


def decode_attention_fwd(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
                         max_kv_splits, sm_scale, logit_cap=0.0, sinks=None, xai_temperature_len=-1, *, chk):
    """SGLang 0.5.1's decode_attention_fwd: stage 1 with the check (chk from AttnCheck.launch), then SGLang's own
    stage 2."""
    assert max_kv_splits == attn_logits.shape[2]
    assert q.shape[0] <= kv_indptr.shape[0] - 1
    assert q.shape[0] <= attn_logits.shape[0]

    kv_group_num = q.shape[1] // v_buffer.shape[1]
    stage1 = _decode_att_m_fwd_check if kv_group_num == 1 else _decode_grouped_att_m_fwd_check
    stage1(q, k_buffer, v_buffer, attn_logits, attn_lse, kv_indptr, kv_indices, num_kv_splits, max_kv_splits,
           sm_scale, logit_cap, xai_temperature_len, chk)
    _decode_softmax_reducev_fwd(attn_logits, attn_lse, q, o, v_buffer, kv_indptr, num_kv_splits, max_kv_splits,
                                sinks)


# ---------------------------------------------------------------------------------------------------------------
# The verdict after a step

@triton.jit
def _verdict_kernel(acc_tag, acc_mask, acc_cnt, tags, mask, codes, stat, total, n_pages, BLOCK: tl.constexpr):
    """One program per BLOCK (layer, page) entries. Code 0: not read, or read and good; 1: the offsets read differ
    from the offsets written (or nobody was counted); 2: the tag differs. Zeroes what the step accumulated; adds the
    failures to stat[0] and the entries read to stat[1]."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    inb = offs < total
    am = tl.load(acc_mask + offs, mask=inb, other=0)
    touched = am != 0
    read = touched & ((offs % n_pages) != 0)  # page 0 is the padding page
    at = tl.load(acc_tag + offs, mask=read, other=0)
    ac = tl.load(acc_cnt + offs, mask=read, other=0)
    tg = tl.load(tags + offs, mask=read, other=0)
    mk = tl.load(mask + offs, mask=read, other=0)
    want = tg.to(tl.uint32) * ac.to(tl.uint32)  # readers x the stored tag, modulo 2^32
    code = tl.where(read & ((mk != am) | (ac <= 0)), 1, tl.where(read & (at.to(tl.uint32) != want), 2, 0))
    tl.store(codes + offs, code.to(tl.int8), mask=inb)
    tl.store(acc_tag + offs, tl.zeros_like(at), mask=touched)
    tl.store(acc_mask + offs, tl.zeros_like(am), mask=touched)
    tl.store(acc_cnt + offs, tl.zeros_like(ac), mask=touched)
    nf = tl.sum((code != 0).to(tl.int32), axis=0)
    nr = tl.sum(read.to(tl.int32), axis=0)
    tl.atomic_add(stat + 0, nf, mask=nf > 0, sem="relaxed")
    tl.atomic_add(stat + 1, nr, mask=nr > 0, sem="relaxed")


def _as_int32(keys: torch.Tensor) -> torch.Tensor:
    """Position keys (int64 holding values in [0, 2^32)) as int32 with the same 32 bits."""
    k = keys & 0xFFFFFFFF
    return torch.where(k >= 2**31, k - 2**32, k).to(torch.int32).contiguous()


def new_stats():
    return dict(steps=0, pieces=0, failures=0, layout=0, seconds=0.0, logged=0)


class AttnCheck:
    """The check on read for one rank's KV pool (a hook.PoolState): accumulators, allocated once, and the verdict.

    acc_tag int64 [layers, pages]: the step's tag shares per piece; acc_mask int32: the offsets read; acc_cnt int32:
    the readers counted; stat int32 [4]: this step's failures, pieces read and layout anomalies. Nothing here
    allocates or waits for the host inside a decode step, so CUDA graphs capture and replay the kernels as they are;
    the verdict runs between steps."""

    def __init__(self, st, reuse=True):
        self.st, self.reuse = st, reuse
        L, P, dev = st.layers, st.pages, st.device
        self.acc_tag = torch.zeros(L, P, dtype=torch.int64, device=dev)
        self.acc_mask = torch.zeros(L, P, dtype=torch.int32, device=dev)
        self.acc_cnt = torch.zeros(L, P, dtype=torch.int32, device=dev)
        self.codes = torch.zeros(L, P, dtype=torch.int8, device=dev)
        self.stat = torch.zeros(4, dtype=torch.int32, device=dev)
        self.pk32 = _as_int32(st.keys)
        self.layer_of = {b.data_ptr(): l for l, b in enumerate(st.pool.k_buffer)}
        self.unsupported = {}  # reason -> calls that fell back to SGLang's kernel
        self.stats = new_stats()

    def reset(self):
        for t in (self.acc_tag, self.acc_mask, self.acc_cnt, self.codes, self.stat):
            t.zero_()
        self.stats = new_stats()

    def target(self, q, k_buffer, v_buffer):
        """The layer whose buffers these are, if this pool owns them and the check supports them; else None."""
        layer = self.layer_of.get(k_buffer.data_ptr())
        if layer is None:
            return None
        st, why = self.st, None
        if torch.version.hip:
            why = "hip"
        elif k_buffer.dtype not in (torch.float16, torch.bfloat16) or v_buffer.dtype != k_buffer.dtype:
            why = f"dtype {k_buffer.dtype}"
        elif v_buffer.data_ptr() != st.pool.v_buffer[layer].data_ptr():
            why = "v buffer of another layer"
        elif tuple(k_buffer.shape[1:]) != (st.heads, st.dim) or tuple(v_buffer.shape[1:]) != (st.heads, st.dim):
            why = f"shape {tuple(k_buffer.shape)}"
        elif st.dim % 2 or st.dim in (576, 288) or st.words != st.heads * st.dim // 2:
            why = f"head dim {st.dim}"
        elif not (k_buffer.is_contiguous() and v_buffer.is_contiguous()):
            why = "not contiguous"
        elif q.shape[1] % k_buffer.shape[1]:
            why = "query heads"
        if why:
            self.unsupported[why] = self.unsupported.get(why, 0) + 1
            return None
        return layer

    def launch(self, layer, k_buffer, v_buffer, check=True, reuse=None):
        """The check's kernel arguments for one layer: views of fixed buffers only, so a captured graph keeps them."""
        kw, vw = k_buffer.view(torch.int32), v_buffer.view(torch.int32)  # [slots, heads, dim / 2] words
        args = (kw, vw, kw.stride(0), kw.stride(1), vw.stride(0), vw.stride(1), self.st.tkey, self.pk32,
                self.acc_tag[layer], self.acc_mask[layer], self.acc_cnt[layer], self.stat)
        return dict(args=args, check=bool(check), reuse=self.reuse if reuse is None else bool(reuse), w=self.st.words)

    def decode(self, layer, q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
               max_kv_splits, sm_scale, logit_cap=0.0, sinks=None, xai_temperature_len=-1, check=True, reuse=None):
        """SGLang's decode attention for one layer, with the check unless check=False."""
        decode_attention_fwd(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, attn_logits, attn_lse, num_kv_splits,
                             max_kv_splits, sm_scale, logit_cap=logit_cap, sinks=sinks,
                             xai_temperature_len=xai_temperature_len,
                             chk=self.launch(layer, k_buffer, v_buffer, check, reuse))

    def verdict(self):
        """Check what the last decode step read, then zero the accumulators: ([(layer, page, reason)], pieces read,
        layout anomalies). Waits for the GPU once (a 16-byte copy)."""
        L, P = self.acc_tag.shape
        total = L * P
        _verdict_kernel[(triton.cdiv(total, VBLOCK),)](self.acc_tag, self.acc_mask, self.acc_cnt, self.st.tags,
                                                     self.st.mask, self.codes, self.stat, total, P, BLOCK=VBLOCK)
        fails, read, layout, _ = self.stat.tolist()
        self.stat.zero_()
        found = []
        if fails:
            nz = self.codes.nonzero()
            codes = self.codes[nz[:, 0], nz[:, 1]].tolist()
            found = [(l, p, REASONS.get(c, str(c))) for (l, p), c in zip(nz.tolist(), codes)]
        s = self.stats
        s["steps"] += 1
        s["pieces"] += read
        s["failures"] += len(found)
        s["layout"] += layout
        return found, read, layout
