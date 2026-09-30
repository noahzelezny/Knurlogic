"""The cache key: what the prompt cache walks when a prompt has images.

The key is the prompt's token ids, the same length as the KV, with each
image token replaced by a sentinel

    ("img", sha, proc_hash, k)        k = 0 .. n_tokens-1

One sentinel per TOKEN because the scheduler does prefix arithmetic on the
prompt. `mlx_lm.models.cache.PromptTrie` accepts any hashable
(tests/test_vision_key.py runs the real LRUPromptCache). With plain pad ids
two same-size images collide and the cache returns KV from the wrong
picture; with sentinels they diverge at k=0, and proc_hash keeps a
processor change from hitting stale features. A sentinel that reaches
mx.array raises, so it can never be read as a wrong id.
Stdlib only. Design: docs/design/vision.md (cache key).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

from . import ImageRef, KeyMismatch

TAG = "img"

Sentinel = Tuple[str, str, str, int]


def sentinel(ref: ImageRef, k: int) -> Sentinel:
    return (TAG, ref.sha, ref.proc_hash, k)


def is_sentinel(x: Any) -> bool:
    return type(x) is tuple and len(x) == 4 and x[0] == TAG


def has_image(key: Iterable[Any]) -> bool:
    """Does this key hold any image token? The test that decides whether a
    row needs `Family.positions` on EVERY step, not just the ones whose
    new suffix has an image."""
    return any(type(x) is tuple for x in key)


def expand_pads(ids: Sequence[int], refs: Sequence[ImageRef],
                image_token_id: int) -> List[int]:
    """The template's output -> the model's ids: each single image_token_id
    (one per image, Family.placeholder_text's contract) becomes ref.n_tokens
    copies, refs taken in order. Raises KeyMismatch if the count of pads is
    not the count of refs -- a user who typed the pad token as text."""
    out: List[int] = []
    i = 0
    for t in ids:
        if t == image_token_id:
            if i >= len(refs):
                raise KeyMismatch(
                    f"image token {image_token_id} appears more often than "
                    f"there are images ({len(refs)}); was it typed as text?")
            out.extend([image_token_id] * refs[i].n_tokens)
            i += 1
        else:
            out.append(t)
    if i != len(refs):
        raise KeyMismatch(f"{len(refs)} images but {i} image placeholders "
                          f"in the tokenized prompt")
    return out


def expand(ids: Sequence[int], refs: Sequence[ImageRef],
           image_token_id: int) -> List[Any]:
    """Expanded ids (every image already n_tokens long) -> the key.

    Refs are consumed in order; each takes exactly its n_tokens consecutive
    image tokens, so two images back to back with no separator still split
    right. Anything else -- a run too short, a stray image token after the
    last ref, a ref left over -- raises KeyMismatch."""
    key: List[Any] = []
    i, n = 0, len(ids)
    r = 0
    while i < n:
        t = ids[i]
        if t != image_token_id:
            key.append(t)
            i += 1
            continue
        if r >= len(refs):
            raise KeyMismatch(f"image token at {i} with no image left to "
                              f"assign it ({len(refs)} images)")
        ref = refs[r]
        end = i + ref.n_tokens
        if end > n or any(ids[j] != image_token_id for j in range(i, end)):
            raise KeyMismatch(f"image {r} needs {ref.n_tokens} image tokens "
                              f"from {i}; the run is shorter")
        key.extend(sentinel(ref, k) for k in range(ref.n_tokens))
        i = end
        r += 1
    if r != len(refs):
        raise KeyMismatch(f"{len(refs)} images but {r} image runs")
    return key


def expand_segments(segments: Sequence[Sequence[int]],
                    refs: Sequence[ImageRef],
                    image_token_id: int) -> Tuple[List[Any], List[List[Any]]]:
    """The prompt stage returns (prompt, segments, ...), and the segments
    feed `insert_segments` and the checkpoints. Rewriting only the prompt
    would leave the segments naming the wrong tokens.

    Takes the UNexpanded segments (their concatenation is the template's
    prompt) and returns (key, segment keys): the same pad expansion and
    sentinel assignment, one ref cursor across the segment boundaries, and
    `sum(len(s)) == len(key)` checked, not assumed."""
    flat = [t for s in segments for t in s]
    key = expand(expand_pads(flat, refs, image_token_id), refs,
                 image_token_id)
    # One pass over the flat, unexpanded ids, tracking which segment each
    # id came from: simpler than per-segment arithmetic and cannot drift.
    pos = 0
    out = [[] for _ in segments]
    seg_of = [si for si, s in enumerate(segments) for _ in s]
    r = 0
    for idx, t in enumerate(flat):
        si = seg_of[idx]
        if t == image_token_id:
            n = refs[r].n_tokens
            out[si].extend(key[pos:pos + n])
            pos += n
            r += 1
        else:
            out[si].append(key[pos])
            pos += 1
    assert pos == len(key) and sum(len(s) for s in out) == len(key), (
        "segment keys do not add up to the key")
    return key, out


def to_ids(key: Iterable[Any], image_token_id: int) -> List[int]:
    """The key -> what the model is fed: every sentinel back to the pad."""
    return [image_token_id if type(x) is tuple else x for x in key]


@dataclass(frozen=True)
class Span:
    """One image's run in a key: [start, end), and whether the run starts at
    k=0 and ends at k=n-1 is NOT promised -- a key sliced by a prefix hit
    begins mid-image. `k0` is the sentinel index at `start`."""
    start: int
    end: int
    sha: str
    proc_hash: str
    k0: int


def image_spans(key: Sequence[Any]) -> List[Span]:
    """Every image run in the key, in order. A new span starts where the
    image changes or k does not follow on -- so the same image twice in a
    row is two spans, as it is two images."""
    spans: List[Span] = []
    s = None
    for i, x in enumerate(key):
        if type(x) is tuple:
            if (s is not None and x[1] == s[2] and x[2] == s[3]
                    and x[3] == s[5] + 1):
                s[1], s[5] = i + 1, x[3]
                continue
            if s is not None:
                spans.append(Span(s[0], s[1], s[2], s[3], s[4]))
            s = [i, i + 1, x[1], x[2], x[3], x[3]]
        elif s is not None:
            spans.append(Span(s[0], s[1], s[2], s[3], s[4]))
            s = None
    if s is not None:
        spans.append(Span(s[0], s[1], s[2], s[3], s[4]))
    return spans


def images_in(key: Sequence[Any]) -> List[Tuple[str, str]]:
    """(sha, proc_hash) of each image run in the key, in order, including a
    run the slice cuts into -- what a store pin must hold for embed, and,
    over a full key, the refs positions() needs."""
    return [(s.sha, s.proc_hash) for s in image_spans(key)]
