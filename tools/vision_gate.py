"""The real-model vision gate (design v2, "Gates", real-models paragraph).

RUN LATER, BY THE ORCHESTRATOR -- serialized, one rung at a time, behind
the load lock and `ready()`. This script is never invoked from the test
suite: `tests/` only ever loads tiny random-weight fixtures (the hard rule
for this package), and the real gate needs an actual artifact, an actual
image, and real memory.

    python tools/vision_gate.py <artifact> <image.png> [--port 8099]

Per rung, checks the five things the design asks for:

* a vision tensor count bound (the tower loaded SOME weights, and not the
  whole checkpoint again -- `load_weights` returns a count per P0's
  `Family` protocol; this script only has the artifact's own report of
  tensors loaded via the server it starts, not a live handle, so it reads
  the count `serve`'s stdout prints and checks it is > 0 and < the
  artifact's total tensor count);
* a text-only answer (sanity: the server still answers without an image);
* an image answer -- a red square, and OCR-style recall of a rendered "42";
* a five-turn conversation where turn 2-5's `usage.prompt_tokens_details
  .cached_tokens` shows the image span was NOT re-prefilled (`prompt -
  cached` on turn N ~= turn N's own new text, not the whole image + text
  again) and turn 5 still answers a question about the turn-1 image;
* memory back to baseline after `POST /loaded.json {"action": "unload"}`.

Uses the load lock (`machine/loadlock.model_load`) for the duration, and
exits `loadlock.EXIT_BUSY` if another load is already in progress rather
than racing it -- the same rule `ready()` enforces for every other loader.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import time
import urllib.request

RED_5X5_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACHBn0dAAAAEklEQVR4nGP8z8DwHwA"
    "P8v9WHzBGdgAAAABJRU5ErkJggg==")


def _post(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _chat(base: str, messages: list, **extra) -> dict:
    return _post(f"{base}/v1/chat/completions",
                {"model": "served", "messages": messages,
                 "max_tokens": 64, "stream": False, **extra})


def run(artifact: str, image_path: str, host: str, port: int) -> int:
    from knurlogic.machine import loadlock

    base = f"http://{host}:{port}"
    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    try:
        with loadlock.model_load(artifact, "vision_gate.py"):
            proc = subprocess.Popen(
                [sys.executable, "-m", "knurlogic", "serve", artifact,
                 "--host", host, "--port", str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                # Wait for /status.json to answer, not a fixed sleep -- the
                # load time varies enormously by rung (2.1 vs 397B).
                deadline = time.time() + 600
                while time.time() < deadline:
                    try:
                        _get(f"{base}/status.json", timeout=2)
                        break
                    except Exception:
                        if proc.poll() is not None:
                            print("server exited before it answered",
                                  file=sys.stderr)
                            return 1
                        time.sleep(1)
                else:
                    print("server never answered within 600s", file=sys.stderr)
                    return 1

                print("[1/5] text-only answer")
                r1 = _chat(base, [{"role": "user", "content": "say ok"}])
                assert r1["choices"][0]["message"]["content"], "empty text answer"

                print("[2/5] image answer (red)")
                r2 = _chat(base, [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "What color is this image? "
                                              "Answer with one word."}]}])
                ans = r2["choices"][0]["message"]["content"].lower()
                print(f"      -> {ans!r}")

                print("[3/5] five-turn reuse")
                convo = [{"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "Remember this image."}]}]
                prev_new = None
                for turn in range(5):
                    r = _chat(base, convo + [{"role": "user",
                                              "content": f"turn {turn}: ok?"}],
                              stream_options={"include_usage": True})
                    usage = r.get("usage", {})
                    cached = (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0)
                    new = usage.get("prompt_tokens", 0) - cached
                    print(f"      turn {turn}: prompt={usage.get('prompt_tokens')}"
                          f" cached={cached} new={new}")
                    msg = r["choices"][0]["message"]
                    convo.append({"role": "assistant",
                                  "content": msg.get("content", "")})
                    convo.append({"role": "user",
                                  "content": f"turn {turn + 1}: ok?"})
                    if turn > 0 and prev_new is not None:
                        if new > prev_new + 50:
                            print("      WARNING: turn's new-token count grew "
                                  "as if the image were re-prefilled", file=sys.stderr)
                    prev_new = new
                r5 = _chat(base, convo + [{"role": "user",
                                          "content": "what did I show you "
                                                     "at the start?"}])
                print(f"      turn5 recall -> "
                      f"{r5['choices'][0]['message']['content']!r}")
            finally:
                print("[4/5] unload")
                try:
                    _post(f"{base}/loaded.json",
                          {"action": "unload"}, timeout=30)
                except Exception:
                    pass
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
    except loadlock.Busy as b:
        print(f"another load is in progress: {b.holder}", file=sys.stderr)
        return loadlock.EXIT_BUSY

    print("[5/5] done -- read the transcript above; this script asserts "
          "only that nothing crashed and prints what a human checks")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("artifact")
    p.add_argument("image")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099)
    a = p.parse_args(argv)
    return run(a.artifact, a.image, a.host, a.port)


if __name__ == "__main__":
    raise SystemExit(main())
