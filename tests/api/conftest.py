"""The API conformance suite runs against a LIVE server -- today's patched
mlx-lm server, and later knurlogic's own -- so the same tests judge both.

    KNURLOGIC_API_URL=http://127.0.0.1:8099 pytest tests/api

Without the variable every test here is skipped: the ordinary suite never
loads a real model (the package's hard rule), and this one needs one.
"""
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("KNURLOGIC_API_URL"):
        return
    skip = pytest.mark.skip(reason="set KNURLOGIC_API_URL to a running "
                                   "server to run the API conformance suite")
    for it in items:
        if "tests/api/" in str(it.fspath).replace("\\", "/"):
            it.add_marker(skip)
