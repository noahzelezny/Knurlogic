"""Drafting: an artifact's MTP head, used by mlx-lm's own server.

The sequential path swaps `stream_generate` for the drafting loop; the
batch path swaps the `BatchGenerator` the server builds for
`engine.mtp.batch_generator.MTPBatchGenerator` -- which a vision model gets
head or not (design D5).
"""

from __future__ import annotations

from . import state


# An artifact that ships a multi-token-prediction head carries weights mlx-lm
# will never run: it has no MTP path at all, and neither does upstream exo.
# knurlogic does, so a head beside the weights is simply used -- no flag, no
# environment variable, no mode file.



def drafting_status() -> dict:
    """What drafting is doing, for `/status.json` and the startup line.

    `engine_path` is here because it is the thing that explains a head that
    loaded and then never drafted. mlx-lm has two generators: a sequential
    one that calls `stream_generate`, and a BATCH one that does not. It picks
    the batch path whenever `is_batchable` -- which an MTP head does not
    affect, since that rule only asks about a separate draft MODEL. So an
    artifact whose caches can merge goes down a path the sequential swap
    never sees, and the only symptom is `requests: 0` next to `on: True`.
    """
    d = dict(state.DRAFT)
    d.pop("head", None)
    spec = d.pop("spec", None)
    d["family"] = getattr(spec, "name", "")
    d["acceptance"] = (d["accepted"] / d["steps"]) if d["steps"] else None
    prov = state.SERVED.get("provider")
    batchable = getattr(prov, "is_batchable", None) if prov else None
    d["batchable"] = batchable
    d["engine_path"] = ("batch" if batchable
                        else "sequential" if batchable is False else "")
    batch_on = bool(d.pop("batch_installed", False))
    if d["on"] and batchable and not batch_on:
        d["drafts_now"] = False
        d["blocked"] = ("this artifact's caches merge, so mlx-lm serves it "
                        "with the batch generator, and the drafting batch "
                        "generator was not installed. The head is loaded and "
                        "idle.")
    else:
        d["drafts_now"] = bool(d["on"])
        d["blocked"] = ""
    return d


def load_head(model_path: str):
    """Load the drafting head beside this artifact, if there is one.

    Absent is the ordinary case and not an error: sidecars are named outside
    mlx-lm's `model*.safetensors` glob precisely so a directory carrying one
    still loads normally through the stock loader.
    """
    from knurlogic.engine.mtp import find_head

    found = find_head(model_path)
    if found is None:
        state.DRAFT.update(on=False, why="no drafting head beside the weights")
        return None
    prov = state.SERVED.get("provider")
    model = getattr(prov, "model", None) if prov else None
    if model is None:
        state.DRAFT.update(on=False, why="model not loaded yet")
        return None
    try:
        from knurlogic.engine.mtp.loop import load_mtp_head
        head, spec = load_mtp_head(model, sidecar=found.path)
    except Exception as e:
        # A head that will not bind is a fact worth printing, not a crash:
        # the model serves perfectly well without one.
        state.DRAFT.update(on=False, why=f"{type(e).__name__}: {e}")
        return None
    state.DRAFT.update(head=head, spec=spec, on=True,
                  why=f"{found.path.name}, {found.gib:.2f} GiB")
    return head


def install(srv) -> bool:
    """Route the engine's own generation through the drafting loop.

    mlx-lm's server calls `stream_generate(...)` once per request and reads
    `.text`, `.token`, `.logprobs` and `.finish_reason` off what it yields.
    `mtp_stream_generate` yields all four, so this is a swap rather than a
    reimplementation -- knurlogic still does not have a second inference
    path, which is the rule this package is built on.

    THE SAMPLING PARAMETERS ARE THE AWKWARD PART. mlx-lm hands
    `stream_generate` a BUILT sampler, and the drafting loop needs the
    parameters themselves: verification is rejection sampling against the
    target distribution, so a callable that has already collapsed it is no
    use. They are available one frame up, in `_serve_single`, so that is
    wrapped to put them on a thread-local. Per request, per thread, and the
    server serves each request on its own thread.
    """
    if not state.DRAFT.get("on"):
        return False

    import threading

    from knurlogic.engine.mtp.loop import mtp_stream_generate

    local = threading.local()
    real_single = srv.ResponseGenerator._serve_single
    real_stream = srv.stream_generate

    def _serve_single(self, request):
        local.args = request[2]
        try:
            return real_single(self, request)
        finally:
            local.args = None

    def _stream(model, tokenizer, prompt, **kw):
        args = getattr(local, "args", None)
        head = state.DRAFT.get("head")
        # A draft model and a drafting head are two different mechanisms and
        # stacking them is not defined; the explicit one wins.
        if head is None or args is None or kw.get("draft_model") is not None:
            yield from real_stream(model=model, tokenizer=tokenizer,
                                   prompt=prompt, **kw)
            return
        s = args.sampling
        state.DRAFT["requests"] += 1
        last = None
        for r in mtp_stream_generate(
                model, tokenizer, prompt, head,
                max_tokens=kw.get("max_tokens", 256),
                temp=s.temperature, top_p=s.top_p, top_k=s.top_k,
                min_p=s.min_p, xtc_probability=s.xtc_probability,
                xtc_threshold=s.xtc_threshold,
                logits_processors=kw.get("logits_processors"),
                prefill_step_size=kw.get("prefill_step_size", 2048),
                prompt_cache=kw.get("prompt_cache"),
                want_logprobs=True):
            last = r
            if r.tail:          # detokenizer flush, not a new token
                continue
            yield r
        if last is not None:
            state.DRAFT["steps"] += last.steps
            state.DRAFT["accepted"] += last.accepted

    srv.ResponseGenerator._serve_single = _serve_single
    srv.stream_generate = _stream
    install_batch(srv)
    return True


def install_batch(srv) -> None:
    """The batch half: hand the server a drafting BatchGenerator.

    The server constructs `BatchGenerator(model, ...)` by the name it
    imported, once per batch, so swapping that name is the whole
    installation. It decides per construction: with no head bound to THIS
    model (a switch to an artifact without one, or `--no-draft`) the server
    gets mlx-lm's own, untouched.
    """
    if getattr(srv.BatchGenerator, "_knurlogic", False):
        return
    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator, tag_samplers

    tag_samplers(srv)
    real = srv.BatchGenerator

    def _factory(model, *a, **kw):
        head = state.DRAFT.get("head")
        prov = state.SERVED.get("provider")
        ours = getattr(prov, "model", None) is model
        vision = state.VISION.get("serve") if (
            ours and state.VISION.get("model") is model) else None
        if head is None or not state.DRAFT.get("on") or not ours:
            head = None
        if head is None and vision is None:
            return real(model, *a, **kw)
        if head is not None:
            try:
                return MTPBatchGenerator(model, head, stats=state.DRAFT,
                                         vision=vision, *a, **kw)
            except Exception as e:
                state.DRAFT.update(why=f"batch drafting refused: "
                                  f"{type(e).__name__}: {e}")
                if vision is None:
                    return real(model, *a, **kw)
        # Design D5: a vision model's batch is ALWAYS this engine, head or
        # not -- only its admit snaps prefill chunks to image blocks, and
        # only it reads a key. No fallback to mlx-lm's here: that one would
        # feed sentinels to mx.array.
        return MTPBatchGenerator(model, None, stats=state.VISION_STATS,
                                 vision=vision, *a, **kw)

    _factory._knurlogic = True
    srv.BatchGenerator = _factory
    state.DRAFT["batch_installed"] = True
