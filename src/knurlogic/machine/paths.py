"""Where knurlogic keeps its files on this machine -- the one lookup.

  cache_dir()   $XDG_CACHE_HOME/knurlogic (default ~/.cache/knurlogic):
                what can be rebuilt -- servers.json, job markers, locks,
                logs, the fallback prompt cache
  config_dir()  $XDG_CONFIG_HOME/knurlogic (default ~/.config/knurlogic):
                settings a person chose (allowance, strategy, settings.json)
  state_dir()   $KNURLOGIC_HOME (default ~/.knurlogic): what this machine
                remembers -- peers, model folders, the request ledger

Each is read from the environment on every call (tests move them) and none
is created here: a caller that writes makes the directory it writes to.
"""
from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME",
                               Path.home() / ".cache")) / "knurlogic"


def config_dir() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME",
                               Path.home() / ".config")) / "knurlogic"


def state_dir() -> Path:
    return Path(os.environ.get("KNURLOGIC_HOME", Path.home() / ".knurlogic"))
