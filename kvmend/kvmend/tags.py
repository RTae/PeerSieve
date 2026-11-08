"""KVMend tags: one 32-bit checksum per piece, updated one token at a time.

A piece is one 16-token page of one layer on one GPU. Its tag is the sum, modulo 2^32, of one mixed value for each
32-bit word that the piece's tokens wrote: fmix32(word XOR key[position] XOR token_key). fmix32 is MurmurHash3's
32-bit finalizer, a bijection, so a change confined to one 32-bit word (one flipped bit, or any burst inside the word)
always changes the tag. A change spread over several words escapes with probability about 2^-32.

The token key binds a token's words to the token itself: it mixes the token id and its position in the sequence. A
reader that checks a page with its own token ids and positions therefore rejects a page that holds another
sequence's KV, even when that page's bytes and tag agree with each other (a misplaced page, or a late write that went
through the store path).

The sum makes the tag "running": the engine adds one token's share as it writes the token, with a plain scatter-add,
and a check recomputes the same sum from the stored bytes, in any order, so GPU threads that each read part of a
piece can just add their parts.
"""
import torch

M32 = 0xFFFFFFFF


def fmix32(h: torch.Tensor) -> torch.Tensor:
    """MurmurHash3's finalizer on int64 tensors holding values in [0, 2^32)."""
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & M32
    return h ^ (h >> 16)


def position_keys(page: int, words: int, device=None, seed: int = 0x9E3779B9) -> torch.Tensor:
    """One key per word position of a piece: [2 (K, V), page, words], int64."""
    idx = torch.arange(2 * page * words, dtype=torch.int64, device=device)
    return fmix32((idx * 0x9E3779B1 + seed) & M32).view(2, page, words)


def token_keys(ids: torch.Tensor, pos: torch.Tensor, seed: int = 0x7F4A7C15) -> torch.Tensor:
    """One 32-bit key per token from its id and its position in the sequence, as int64 in [0, 2^32)."""
    ids = ids.to(torch.int64) & M32
    pos = pos.to(torch.int64) & M32
    return fmix32(((ids * 0x9E3779B1) ^ ((pos * 0x85EBCA77) & M32) ^ seed) & M32)


def as_words(x: torch.Tensor) -> torch.Tensor:
    """[T, heads, dim] fp16 or bf16 -> [T, words] int64 with values in [0, 2^32)."""
    return x.reshape(x.shape[0], -1).contiguous().view(torch.int32).to(torch.int64) & M32


def token_shares(k: torch.Tensor, v: torch.Tensor, off: torch.Tensor, keys: torch.Tensor,
                 tk: torch.Tensor = None) -> torch.Tensor:
    """Each token's share of its piece's tag. k, v: [T, heads, dim]; off: [T] offsets within the page; tk: [T] token
    keys, or None for none."""
    kk = keys[0].index_select(0, off)
    vk = keys[1].index_select(0, off)
    if tk is not None:
        t = tk.to(torch.int64).unsqueeze(1)
        kk, vk = kk ^ t, vk ^ t
    return fmix32(as_words(k) ^ kk).sum(1) + fmix32(as_words(v) ^ vk).sum(1)


def page_tags(k_pages: torch.Tensor, v_pages: torch.Tensor, mask: torch.Tensor, keys: torch.Tensor,
              tk: torch.Tensor = None) -> torch.Tensor:
    """Tags recomputed from stored pieces. k_pages, v_pages: [n, page, heads, dim]; mask: [n] bits of the written
    offsets; tk: [n, page] token keys, or None for none. Only the written tokens count, as in the running tag."""
    n, page = k_pages.shape[:2]
    kw = as_words(k_pages.reshape(n * page, *k_pages.shape[2:])).view(n, page, -1)
    vw = as_words(v_pages.reshape(n * page, *v_pages.shape[2:])).view(n, page, -1)
    kk, vk = keys[0].unsqueeze(0), keys[1].unsqueeze(0)
    if tk is not None:
        t = tk.to(torch.int64).unsqueeze(2)
        kk, vk = kk ^ t, vk ^ t
    share = fmix32(kw ^ kk).sum(2) + fmix32(vw ^ vk).sum(2)
    covered = (mask.to(torch.int64).unsqueeze(1) >> torch.arange(page, device=mask.device)) & 1
    return (share * covered).sum(1)


def same(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Tags are compared modulo 2^32."""
    return (a & M32) == (b & M32)
