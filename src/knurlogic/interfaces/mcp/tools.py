"""The MCP's tool table: TOOLS maps each tool name to its function, the
description an agent reads and its JSON input schema; tool_list is what
tools/list answers."""

from __future__ import annotations

from typing import Any

from knurlogic.interfaces.mcp.inspection import (
    deps,
    drafting,
    fit,
    model_folders,
    models,
    ready,
    settings,
    state,
)
from knurlogic.interfaces.mcp.lifecycle import load, unload


def S(desc: str, typ: str = "string") -> dict[str, Any]:
    return {"type": typ, "description": desc}


def _schema(props: dict[str, Any], required: list[str] | None = None):
    return {"type": "object", "properties": props,
            "required": required or []}


TOOLS: dict[str, dict[str, Any]] = {
    "ready": {
        "fn": ready,
        "description": "Is it safe to load now? Not while another load "
                       "(a server `load` started, or any process holding "
                       "the model-load lock) is still moving memory. Every "
                       "blocker is named. Call this before loading.",
        "schema": _schema({}),
    },
    "state": {
        "fn": state,
        "description": "What is loaded, here and on every machine this "
                       "Mac's page (`knurlogic ui`) sees. `models`: one "
                       "entry per resident model -- name, machine, port, "
                       "machines, split (tensor | pipeline), link (tcp | "
                       "rdma), job, instance (a single-Mac server's own "
                       "16-hex id, or a cluster job's id), leader, phase, "
                       "and `requests` "
                       "(in_flight, pending, capacity, oldest_pending_s, "
                       "holding; null when its server does not report "
                       "them). A cluster job is ONE entry, on its leader, "
                       "never one per rank. `machines`: the peers asked, "
                       "with any that did not answer. The rest is this Mac "
                       "alone: every runtime (knurlogic, ollama, exo, any "
                       "OpenAI port), where the memory went, and "
                       "`started_here` -- each server `load` started with "
                       "its phase: serving, loading (elapsed time, last log "
                       "line), stalled (stop waiting, read the log), or "
                       "exited (exit code, log tail).",
        "schema": _schema({}),
    },
    "models": {
        "fn": models,
        "description": "Every model on this machine, with whether it fits, "
                       "whether it has a drafting head, and what "
                       "reasoning_effort (none minimal low medium high "
                       "xhigh) maps to on it.",
        "schema": _schema({"fits_only": {
            "type": "boolean",
            "description": "only models that can actually run here"}}),
    },
    "fit": {
        "fn": fit,
        "description": "Will this artifact fit NOW: verdict fits | fits, low "
                       "headroom | will not fit, with headroom and what low "
                       "headroom changes (it still loads, with a narrower "
                       "prompt chunk). Uses the same budget `settings` and "
                       "`load` use.",
        "schema": _schema({"artifact": S("path to the artifact"),
                           "draft": {"type": "boolean",
                                     "description": "count the MTP head "
                                                    "(default true); false "
                                                    "asks about MTP off"},
                           "vision": {"type": "boolean",
                                      "description": "count the vision "
                                                     "tower, image store "
                                                     "and image KV (default "
                                                     "true); false asks "
                                                     "about vision off"}},
                          ["artifact"]),
    },
    "settings": {
        "fn": settings,
        "description": "The resolved knobs for an artifact, each with the "
                       "measurement behind it and whether it can be changed "
                       "at runtime or only at launch.",
        "schema": _schema({"artifact": S("path to the artifact"),
                           "tune": S("default | lean")},
                          ["artifact"]),
    },
    "drafting": {
        "fn": drafting,
        "description": "Whether this artifact has a multi-token-prediction "
                       "head and what will happen to it.",
        "schema": _schema({"artifact": S("path to the artifact")},
                          ["artifact"]),
    },
    "load": {
        "fn": load,
        "description": "Start a model. With no `machines`, a server on "
                       "this Mac, its settings resolved against the load "
                       "budget; refused if it will not fit (no override) "
                       "or another load is still moving memory (force "
                       "overrides). With `machines`, through this Mac's "
                       "page (`knurlogic ui` must run), exactly as its "
                       "Launch button: one other Mac is a load there, "
                       "checked by that Mac; two or more is a cluster job "
                       "-- placement, leader and cable are chosen, never "
                       "asked for. Returns job, port (rank 0 serves the "
                       "model there, on the leader), leader, machines and "
                       "placement {order, leader, layers, cable, "
                       "cable_note}; the job loads in the background -- "
                       "poll `state`. Refused, with the reason and nothing "
                       "started, when a share does not fit, a machine is "
                       "not answering, the model is not on a machine, rdma "
                       "has no Thunderbolt 5 cable with RDMA up, or a "
                       "machine is already loading (one load at a time). "
                       "Nothing is ever evicted to make room: unload first.",
        "schema": _schema({
            "artifact": S("the model's name (or, for machines that are "
                          "not this Mac, its 16-hex identity)"),
            "port": S("port to serve on (a cluster job: rank 0's port on "
                      "the leader); omitted: the first free port from "
                      "8080 up", "integer"),
            "tune": S("default | lean"),
            "sets": {"type": "object",
                     "description": "launch-only knob overrides, KEY: VALUE"},
            "force": {"type": "boolean",
                      "description": "load while another load is still "
                                     "moving memory (one machine only)"},
            "draft": {"type": "boolean",
                      "description": "use a packed drafting head "
                                     "(default true; one machine only)"},
            "vision": {"type": "boolean",
                       "description": "load the vision tower (default "
                                      "true); false frees its memory "
                                      "(tower, image store, image KV) and "
                                      "image requests get a 400"},
            "machines": {"type": "array", "items": {"type": "string"},
                         "description": "machine names, as `state` lists "
                                        "them; empty: this Mac only"},
            "split": S("with two or more machines: tensor (every layer "
                       "split, same share each) | pipeline (layers in "
                       "runs, sized to each machine)"),
            "link": S("with two or more machines: tcp (the ring, any "
                      "link) | rdma (jaccl over Thunderbolt 5, every pair "
                      "cabled; beyond two machines experimental)"),
            "cable": S("optional, two machines: the Thunderbolt subnet to "
                       "use (e.g. 198.51.100); by default the fastest shared "
                       "one, moving to the next if link init fails; "
                       "ignored with three or more (each pair picks its own)"),
        }, ["artifact"]),
    },
    "model_folders": {
        "fn": model_folders,
        "description": "The model folders this Mac remembers beside every "
                       "tool's own store (e.g. a folder on an external "
                       "drive), each with whether it is mounted. `add` or "
                       "`remove` one; `models` lists what is in them.",
        "schema": _schema({
            "add": S("a folder to remember (it must exist)"),
            "remove": S("a remembered folder to forget (the models in it "
                        "are not touched)")}),
    },
    "deps": {
        "fn": deps,
        "description": "What this stack stands on: mlx, mlx-lm, mlx-vlm, "
                       "each marked stock or fork by what is installed, not "
                       "by version -- plus what each fork carries and why it "
                       "is or is not ported.",
        "schema": _schema({}),
    },
    "unload": {
        "fn": unload,
        "description": "Stop a model knurlogic started. `port` alone: the "
                       "server on that port of this Mac. `model`, `job` or "
                       "`instance`: on any machine this Mac's page sees "
                       "(`machine` and `port` narrow it when a name matches "
                       "twice). A cluster job stops on every machine it "
                       "runs on.",
        "schema": _schema({
            "port": S("port it serves on (this Mac, unless `machine`)",
                      "integer"),
            "model": S("its name, as `state` lists it"),
            "job": S("a cluster job's id, from `load` or `state`"),
            "instance": S("its instance id, from `load` or `state`'s "
                          "`models[].instance` -- a single-Mac server's own "
                          "16-hex id, or a cluster job's id"),
            "machine": S("the machine it runs on")}),
    },
}


def tool_list() -> list[dict[str, Any]]:
    return [{"name": n, "description": t["description"],
             "inputSchema": t["schema"]} for n, t in TOOLS.items()]
