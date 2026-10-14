"""The control-token state machine: which part of an answer a token is in
(normal / reasoning / tool) and which token sequence ends the row.

Matched on TOKEN ids, one automaton per state. mlx-lm 0.31 shipped this as
`generate.SequenceStateMachine`; 0.32 dropped it for a stop-only automaton
(`StopSequences`) plus a TEXT-matching machine for its own server.
knurlogic's batch engine decides finishes per token, inside the drafting
loop (`MTPBatchGenerator._finish_at` must say which drafted token ends a
row before it is committed), and request.py strips the markers by their
token sequences, so the token-level machine lives here now.

Two builders: `control_machine(tokenizer)` (end of turn, think and tool
markers: what a request runs) and `stop_machine(stop_tokens)` (stops
only: the engine's default).

`ControlMachine(transitions, initial)`: transitions map a state to
[(token sequence, next state)]; a next state of None ends the row.
`make_state()` -> an opaque state; `match(state, token)` ->
(state, the sequence matched or None, the current state name).
"""
from __future__ import annotations

from collections import deque

_FAIL, _HIT = "__fail__", "__hit__"


def _automaton(sequences) -> dict:
    """An Aho-Corasick trie over token sequences: each node maps a token to
    a child, `_FAIL` to the longest proper suffix that is also a prefix,
    `_HIT` to (sequence, its index) when one ends there (or ends at a
    suffix of it)."""
    root: dict = {}
    for i, seq in enumerate(sequences):
        seq = tuple(seq)
        if not seq:
            continue
        node = root
        for t in seq:
            node = node.setdefault(t, {})
        node[_HIT] = (seq, i)
    queue: deque = deque()
    for k, child in root.items():
        if k in (_FAIL, _HIT):
            continue
        child[_FAIL] = root
        queue.append(child)
    while queue:
        parent = queue.popleft()
        for k, child in parent.items():
            if k in (_FAIL, _HIT):
                continue
            queue.append(child)
            f = parent[_FAIL]
            while k not in f and f is not root:
                f = f[_FAIL]
            child[_FAIL] = f[k] if k in f else root
            if _HIT not in child and _HIT in child[_FAIL]:
                child[_HIT] = child[_FAIL][_HIT]
    return root


class ControlMachine:
    """Immutable once built; the per-row position is the state tuple."""

    def __init__(self, transitions: dict | None = None,
                 initial: str = "normal"):
        self.initial = initial
        self._states: dict = {}
        for src, edges in (transitions or {}).items():
            seqs = [e[0] for e in edges]
            dsts = [e[1] for e in edges]
            self._states[src] = (_automaton(seqs), dsts)
        if initial not in self._states:
            self._states[initial] = (_automaton([]), [])

    def __deepcopy__(self, memo):
        return self                       # immutable: rows share it

    def make_state(self):
        return (self.initial, self._states[self.initial][0])

    def match(self, state, token):
        name, node = state
        root = self._states[name][0]
        while token not in node and node is not root:
            node = node[_FAIL]
        if token in node:
            node = node[token]
        hit = node.get(_HIT)
        if hit is None:
            return (name, node), None, name
        seq, i = hit
        name = self._states[name][1][i]
        node = self._states[name][0] if name is not None else None
        return (name, node), seq, name


def stop_machine(stop_tokens) -> ControlMachine:
    """A machine whose only edges end the row: the engine's default when an
    admission brings none (mlx-lm's `stop_tokens`, as sequences)."""
    seqs = [tuple(s) if isinstance(s, (list, tuple)) else (s,)
            for s in (stop_tokens or [])]
    return ControlMachine({"normal": [(s, None) for s in seqs]}
                          if seqs else None)


_MACHINES: dict = {}

#: The tags that close a think block. A model may end its thinking with
#: `</thinking>` where its template says `</think>` (seen on Qwen3.6); both
#: close the block, in the streamed and the non-streamed path alike. The
#: tokenizer's own think_end is one of these; the others are matched as the
#: token sequences their text encodes to.
THINK_CLOSE_TAGS = ("</think>", "</thinking>")
#: what may directly follow a close tag and merge with its last token
#: (">\n\n" is one token in some vocabularies)
_CLOSE_TAILS = ("", "\n", "\n\n", "\n\n\n", "\r\n")
#: what may directly precede it and merge with its first token (Qwen3.6
#: encodes " </" as one token, "</" as another)
_CLOSE_HEADS = ("", " ")


def _extra_think_closes(tokenizer) -> dict:
    """{token sequence: its text} for the alternative close tags, as the
    tokenizer encodes them bare and with a newline tail."""
    if getattr(tokenizer, "think_end", None) not in THINK_CLOSE_TAGS:
        return {}
    out: dict = {}
    for tag in THINK_CLOSE_TAGS:
        if tag == tokenizer.think_end:
            continue
        for head in _CLOSE_HEADS:
            for tail in _CLOSE_TAILS:
                try:
                    ids = tuple(tokenizer.encode(head + tag + tail,
                                                 add_special_tokens=False))
                except (AttributeError, TypeError, ValueError, KeyError):
                    continue        # a vocabulary that cannot encode it
                if ids:
                    out[ids] = head + tag + tail
    return out


def control_machine(tokenizer, initial: str = "normal"):
    """(ControlMachine, {token sequence: its text}) for the tokenizer's
    control tokens: end-of-turn, think markers, tool markers. The user's
    stop strings are NOT here -- they are text, matched by `Request`.
    Cached per tokenizer and initial state."""
    key = (id(tokenizer), initial)
    hit = _MACHINES.get(key)
    if hit is not None and hit[0] is tokenizer:
        return hit[1], hit[2]
    seqs: dict[tuple[int, ...], str] = {}
    ends: list = []
    for t in tokenizer.eos_token_ids:
        seqs[(t,)] = tokenizer.convert_ids_to_tokens(t)
        ends.append(((t,), None))
    edges: dict[str, list] = {"normal": list(ends)}
    if getattr(tokenizer, "has_thinking", False):
        ts, te = (tuple(tokenizer.think_start_tokens),
                  tuple(tokenizer.think_end_tokens))
        edges["normal"].append((ts, "reasoning"))
        extra = _extra_think_closes(tokenizer)
        edges["reasoning"] = [(te, "normal"),
                              *((q, "normal") for q in extra), *ends]
        seqs[ts], seqs[te] = tokenizer.think_start, tokenizer.think_end
        seqs.update(extra)
    if getattr(tokenizer, "has_tool_calling", False):
        ts = tuple(tokenizer.tool_call_start_tokens)
        te = tuple(tokenizer.tool_call_end_tokens or ())
        edges["normal"].append((ts, "tool"))
        edges["tool"] = ([(te, "normal")] if te else []) + ends
        seqs[ts] = tokenizer.tool_call_start
        if te:
            seqs[te] = tokenizer.tool_call_end
    sm = ControlMachine(edges, initial=initial)
    if len(_MACHINES) > 64:
        _MACHINES.clear()
    _MACHINES[key] = (tokenizer, sm, seqs)
    return sm, seqs
