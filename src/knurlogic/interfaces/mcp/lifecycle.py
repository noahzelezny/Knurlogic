"""The MCP's tools that change what runs: load and unload.

`load` on this Mac checks the launch, the fit and the ready gate before it
spawns a server (interfaces/spawn.py); with `machines` it is the page's
Launch request, sent through page_client.py. `unload` stops one by port,
model, job or instance, on any machine the page sees.
"""

from __future__ import annotations

from typing import Any

from knurlogic.interfaces.mcp import inspection, page_client
from knurlogic.interfaces.mcp.inspection import GIB


def load(artifact: str = "", port: int = 0, tune: str = "default",
         sets: dict[str, str] | None = None, force: bool = False,
         draft: bool = True, machines: list[str] | None = None,
         split: str = "", link: str = "", cable: str = "",
         vision: bool = True, **_) -> dict[str, Any]:
    """Start a server for this artifact, after checking it can work.

    REFUSES rather than gambles: memory still moving or a model whose
    weights do not fit is a refusal with the reason attached. `force`
    overrides the moving-memory check only -- it will not make a model fit;
    `draft=false` (MTP off) may.

    `machines` names where: empty is this Mac. Another Mac, or several,
    go through the page on this Mac -- its Launch, the same request.

    `vision=false` launches without the vision tower, image store and
    image KV (KNURLOGIC_VISION=off): more headroom, and images get a 400.
    """
    from knurlogic.tuning.presets import preset_of
    if not vision:
        sets = {**dict(sets or {}), "KNURLOGIC_VISION": "off"}
    try:
        tune = preset_of(tune)
    except ValueError as e:
        return {"loaded": False, "refused": str(e),
                "note": "the tune is default or lean; nothing was started"}
    names = [str(m) for m in (machines or []) if str(m)]
    if names:
        return _load_on(names, artifact, port, tune, sets, force, draft,
                        split, link, cable)
    from knurlogic.interfaces import spawn
    from knurlogic.interfaces.load_checks import NotLoadable, resolve_name

    # a model named, never a directory (interfaces/load_checks.py): the same
    # rule a switch on a running server follows
    try:
        artifact = resolve_name(artifact, None)
    except NotLoadable as e:
        return {"loaded": False, "refused": "not a known artifact",
                "note": str(e)}
    # serve's deterministic refusals (bad settings, a context past the
    # model's maximum, ...), asked before a process is started: a server
    # that prints REFUSING and exits is a reason nobody sees
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning.checks import launch_fit, launch_refusal
    try:
        why = launch_refusal(Artifact.load(artifact), dict(sets or {}),
                            tune)
    except (OSError, ValueError, AttributeError, KeyError) as e:
        why = f"could not read the artifact: {type(e).__name__}: {e}"
    if why:
        return {"loaded": False, "refused": why,
                "note": "the launch settings (Settings -> Models) or the "
                        "artifact; nothing was started"}
    # before fit(), whose own "will not fit" would hide the way out: a
    # launch that cannot fit with MTP on but does with it off
    off = _mtp_off_doc(artifact, dict(sets or {}), tune, bool(draft))
    if off:
        return off
    # the launch's own check first: its refusal says the numbers, which
    # fit()'s verdict does not
    try:
        chk = launch_fit(Artifact.load(artifact), dict(sets or {}), tune,
                         bool(draft))
    except (OSError, ValueError, AttributeError, KeyError):
        chk = {"state": "fits"}
    if chk["state"] == "cannot":
        return {"loaded": False, "refused": "will not fit",
                "detail": chk["why"],
                "note": "no flag overrides this; it is arithmetic."}
    from knurlogic.tuning.knobs import vision_of
    f = inspection.fit(artifact=artifact, draft=bool(draft),
            vision=vision_of(dict(sets or {})))
    if not f["fits"]:
        return {"loaded": False, "refused": "will not fit",
                "detail": f,
                "note": "no flag overrides this; it is arithmetic."}
    r = inspection.ready()
    if not r["ready"] and not force:
        return {"loaded": False, "refused": "memory is about to move",
                "detail": r["blockers"],
                "note": "something is loading or unloading on this box, so "
                        "the fit above is stale. Poll `state` -- each server "
                        "says loading, serving or stalled -- then call "
                        "`ready` again. force=true loads anyway."}
    if not port:
        # none asked for: the first free one from the serve default up
        from knurlogic.machine.servers import free_port
        port = free_port(spawn.SERVE_PORT["n"])
    out = spawn.spawn(artifact, int(port), tune, dict(sets or {}),
                    draft=bool(draft))
    out["fit"] = f
    out["ready"] = r
    return out


def _mtp_off_doc(artifact: str, sets: dict, tune: str,
                 draft: bool) -> dict[str, Any] | None:
    """The refusal of a launch that cannot fit with MTP on but fits with it
    off, as a doc the page's mtpOffConfirm shows (turn MTP off, or cancel:
    it cannot fit as asked) and an agent reads as
    "retry with draft=false"; None when that is not the case."""
    if not draft:
        return None
    from knurlogic.machine.artifact import Artifact
    from knurlogic.tuning.checks import launch_fit
    try:
        a = Artifact.load(artifact)
        chk = launch_fit(a, sets, tune, True)
        if chk["state"] != "cannot" or not chk.get("head_bytes"):
            return None
        if launch_fit(a, sets, tune, False)["state"] == "cannot":
            return None
    except (OSError, ValueError, AttributeError, KeyError):
        return None
    why = chk["why"].split("; turn ")[0]
    reason = (f"{why}. It will not fit with MTP on; with MTP off the "
              f"{chk['head_bytes'] / GIB:.1f} GiB head is not loaded and "
              f"it fits.")
    if chk.get("vision_bytes"):
        alone = launch_fit(a, {**sets, "KNURLOGIC_VISION": "off"}, tune,
                           True)["state"] != "cannot"
        reason += (f" Vision off (VISION in Load model) frees "
                   f"{chk['vision_bytes'] / GIB:.1f} GiB"
                   + (" and fits with MTP on, if you will not send images."
                      if alone else " more."))
    return {"loaded": False, "refused": "will not fit",
            "mtp_off_fits": True, "reason": reason,
            "text": f"{reason} Nothing was started. Retry `load` with "
                    f"draft=false (MTP off); nothing else overrides "
                    f"this, it is arithmetic.",
            "what_to_do": "Tell the user, then retry `load` with "
                          "draft=false (MTP off), which fits."}


def _identity_of(artifact: str) -> tuple:
    """(identity, refusal): a model on other machines is named by identity
    (machine/artifact.identity), read from this Mac's copy; a 16-hex
    identity is taken as given, for a model this Mac does not hold."""
    import re

    from knurlogic.interfaces.load_checks import NotLoadable, resolve_name
    from knurlogic.machine.artifact import identity
    try:
        ident = identity(resolve_name(artifact, None))
    except NotLoadable as e:
        if re.fullmatch(r"[0-9a-f]{16}", str(artifact or "")):
            return artifact, None
        return "", {"loaded": False, "refused": "not a known artifact",
                    "note": f"{e}. A model this Mac does not hold is named "
                            f"by its identity (the page's /models.json on a "
                            f"Mac that has it)."}
    if not ident:
        return "", {"loaded": False, "refused": "no identity",
                    "note": f"{artifact} has no config.json"}
    return ident, None


def _artifact_name(artifact: str) -> str:
    """The directory name `load` was given the artifact by ("" for a bare
    identity): the name the other machines resolve it by."""
    import re
    a = str(artifact or "").rstrip("/")
    if re.fullmatch(r"[0-9a-f]{16}", a):
        return ""
    from pathlib import Path

    from knurlogic.interfaces.load_checks import NotLoadable, resolve_name
    try:
        # a path (or a pin's real path) -> the store's own name for it
        return Path(resolve_name(a, None)).name
    except NotLoadable:
        return Path(a).name


def _load_on(names, artifact, port, tune, sets, force, draft, split, link,
             cable) -> dict[str, Any]:
    """`load` on other machines: the page's Launch request, sent to the
    page on this Mac (interfaces/page/loads._load_fn), which forwards a
    one-peer load and coordinates a cluster (cluster/launch.launch)."""
    if len(set(names)) != len(names):
        return {"loaded": False, "refused": "a machine is named twice"}
    if len(names) >= 2:
        if split not in page_client.SPLITS:
            return {"loaded": False,
                    "refused": f"split is tensor | pipeline, not {split!r}"}
        if link and link not in page_client.LINKS:
            return {"loaded": False,
                    "refused": f"link is tcp | rdma, not {link!r}"}
    ident, no = _identity_of(artifact)
    if no:
        return no
    try:
        ids, no = page_client._node_ids(names)
        if no:
            return no
        from knurlogic.machine import identity
        if len(ids) == 1 and ids[0] == identity.identity().get("id"):
            # this Mac, named: the same as naming none
            return load(artifact=artifact, port=port, tune=tune, sets=sets,
                        force=force, draft=draft)
        # the name too: a machine holding two artifacts with one identity
        # loads the one called this, or refuses -- never picks
        req = {"action": "load", "identity": ident,
               "name": _artifact_name(artifact), "tune": tune,
               "sets": dict(sets or {})}
        if not draft:
            # one machine: its --no-draft; a cluster: KNURLOGIC_MTP=off on
            # every rank (cluster/launch)
            req["draft"] = False
        if port:
            req["port"] = int(port)
        if len(ids) == 1:
            req.update(node=ids[0], force=bool(force))
        else:
            req.update(nodes=ids, split=split, link=link)
            if cable:
                req["cable"] = cable
        out = page_client._page_post(req)
    except page_client.PageDown as e:
        return {"error": str(e)}
    no = page_client._refusal(out)
    if no:
        return no
    if len(ids) == 1:
        return dict(out, machines=names)
    plan = dict(out.get("placement") or {})
    plan.update(cable=out.get("cable"), cable_note=out.get("cable_note"))
    return {"starting": out.get("starting"), "artifact": artifact,
            "job": out.get("job"), "port": out.get("port"),
            "url": out.get("url"),
            "leader": out.get("leader"), "machines": out.get("machines"),
            "split": split, "link": page_client._link_name(out.get("link") or link),
            "placement": plan,
            **({"alerts": out["alerts"]} if out.get("alerts") else {}),
            "note": out.get("note", "") + " -- or `state`: the job is one "
                    "entry in `models`, with its phase."}


def unload(port: int | None = None, model: str = "", job: str = "",
           instance: str = "", machine: str = "", **_) -> dict[str, Any]:
    """Stop a model knurlogic started: by port on this Mac (as before), or
    by model name, job id or instance id on any machine this Mac's page
    sees. A cluster job stops on every machine -- the page's Unload, the
    same request. `instance` is the id `load` returned, or `state`'s
    `models[].instance` -- a single-Mac server's own 16-hex id, or a
    cluster job's id (its instance is its job id, so `instance` and `job`
    both find it)."""
    from knurlogic.interfaces import spawn
    if not (port or model or job or instance):
        return {"error": "name the port, the model, the job or the instance"}
    try:
        page = page_client._page_get("/loaded.json?peers=1")
    except page_client.PageDown as e:
        if port and not (model or job or instance or machine):
            return spawn.stop(int(port))      # this Mac, without its page
        return {"error": str(e)}
    here = page_client._me_name()
    rows = [r for r in page_client.models_across(page, here)
            if r.get("runtime") == "knurlogic"]
    if instance:
        hit = [r for r in rows if r.get("instance") == str(instance)]
    elif job:
        # a relaunched job is found by the id its load answered too
        hit = [r for r in rows if r.get("job") == str(job)
               or str(job) in ((r.get("recovery") or {}).get("jobs") or ())]
    else:
        hit = rows
        if model:
            hit = [r for r in hit if r.get("name") == model] or [
                r for r in hit if str(r.get("name") or "").lower().endswith(
                    str(model).lower())]
        if machine:
            hit = [r for r in hit if r.get("machine") == machine
                   or machine in (r.get("machines") or [])]
        elif port and not model:
            hit = [r for r in hit if r.get("machine") == here]
        if port:
            hit = [r for r in hit if r.get("port") == int(port)]
    if not hit:
        if port and not (model or job or instance or machine):
            return spawn.stop(int(port))      # e.g. a server still loading
        return {"error": "no knurlogic model matches that",
                "resident": [{k: r.get(k) for k in
                              ("name", "machine", "port", "job", "instance")}
                             for r in rows]}
    if len(hit) > 1:
        return {"error": "more than one model matches; name the job, the "
                         "instance, or the machine and port",
                "matches": [{k: r.get(k) for k in
                             ("name", "machine", "port", "job", "instance")}
                            for r in hit]}
    r = hit[0]
    running = [j for d in [page] + list(page.get("peers") or [])
               if isinstance(d, dict) for j in d.get("jobs") or []
               if isinstance(j, dict) and j.get("job") == r.get("job")
               and j.get("phase") != "stopped"]
    try:
        if r.get("job") and (not running or any(
                j.get("job") == r["job"] for j in page.get("jobs") or [])):
            # a rank of it runs here: this page stops it everywhere; or
            # it runs nowhere (failed, waiting to relaunch): the unload
            # clears its record on every Mac, and says what it cleared
            out = page_client._page_post({"action": "unload", "job": r["job"]}, 60)
        elif r.get("machine") == here:
            out = page_client._page_post({"action": "unload",
                              "target": str(r.get("port"))}, 60)
        else:
            if not r.get("port"):
                return {"error": f"{r.get('name')} on {r.get('machine')} "
                                 f"has no port yet; stop it there"}
            ids, no = page_client._node_ids([r["machine"]])
            if no:
                return {"error": no["refused"]}
            # the peer's rank 0 port: that page stops its job everywhere
            out = page_client._page_post({"action": "unload", "node": ids[0],
                              "port": int(r["port"])}, 60)
    except page_client.PageDown as e:
        return {"error": str(e)}
    return dict(out, model=r.get("name"), machine=r.get("machine"),
                machines=r.get("machines"), job=r.get("job"),
                instance=r.get("instance"))
