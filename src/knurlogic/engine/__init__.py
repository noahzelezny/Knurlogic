"""engine/ -- what runs a model. The only folder that may import mlx.

A test enforces that boundary: anything outside engine/ that imports mlx
fails the suite, so swapping the engine stays a change to this folder.
Subpackages: serve/ (what is served), runtime/ (the server's engine half),
mtp/ (drafting), families/ (one folder per model family), vision/, vq/;
modules arch.py, register.py, vendor.py, smoke.py.

Depends on nothing else in knurlogic. Importing this package imports no mlx;
only calling into serve/ or mtp's engine-side modules does.
Layout: docs/design/engine.md.
"""
