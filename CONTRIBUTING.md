# Contributing

## Setup

    git clone https://github.com/noahzelezny/Knurlogic
    cd Knurlogic
    python -m venv .venv && . .venv/bin/activate
    pip install -e . pytest

## Run the tests

    python -m pytest -q tests

The suite uses tiny random-weight models and needs no downloads. Tests that
need real weights, a second Mac or a Thunderbolt link skip themselves.

## Changes

* One branch and one pull request per change; keep unrelated edits apart.
* Add or update a test for what the change fixes or adds.
* Vendored files are pinned: the architectures under
  `src/knurlogic/engine/families/*/architecture/` and the VQ runtime in
  `src/knurlogic/engine/vq/` are recorded with their source and sha256 in
  the `PROVENANCE.md` beside them. Do not edit them in place; re-vendor
  from upstream and update the record in the same change.
* The layout and layer rules are in `docs/architecture.md`; a test enforces
  them. Tests live in a folder that mirrors the package they cover.
* Design notes for the larger pieces live in `docs/design/`.
