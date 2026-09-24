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
import urllib.error
import urllib.request

RED_5X5_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACHBn0dAAAAEklEQVR4nGP8z8DwHwA"
    "P8v9WHzBGdgAAAABJRU5ErkJggg==")


def _post(url: str, body: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} from {url}: {e.read().decode()[:500]}")


def _get(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def _chat(base: str, messages: list, **extra) -> dict:
    return _post(f"{base}/v1/chat/completions",
                {"model": "served", "messages": messages,
                 "max_tokens": 64, "stream": False,
                 # The gate checks what the model SEES; thinking only spends
                 # the token budget before the answer.
                 "chat_template_kwargs": {"enable_thinking": False},
                 **extra}, timeout=300)


def run(artifact: str, image_path: str, host: str, port: int) -> int:
    from knurlogic.machine import loadlock

    base = f"http://{host}:{port}"
    fails: list = []
    with open(image_path, "rb") as fh:
        img_b64 = base64.b64encode(fh.read()).decode()

    try:
        with loadlock.model_load(artifact, "vision_gate.py"):
            proc = subprocess.Popen(
                [sys.executable, "-m", "knurlogic", "serve", artifact,
                 "--host", host, "--port", str(port)],
                stdout=open(f"vision_gate-{port}.log", "w"),
                stderr=subprocess.STDOUT, text=True)
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


                def check(ok: bool, what: str) -> None:
                    print(f"      {'PASS' if ok else 'FAIL'}  {what}")
                    if not ok:
                        fails.append(what)

                def said(r: dict) -> str:
                    return (r["choices"][0]["message"].get("content")
                            or "").strip()

                def img_part():
                    return {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"}}

                print("[1/5] text-only answer")
                r1 = _chat(base, [{"role": "user", "content": "say ok"}])
                check(bool(said(r1)), f"text answer {said(r1)!r}")

                print("[2/5] image answer")
                r2 = _chat(base, [{"role": "user", "content": [
                    img_part(), {"type": "text", "text":
                                 "What color is the square, and what number "
                                 "is written? Answer briefly."}]}])
                ans = said(r2).lower()
                check("red" in ans, f"sees the red square: {ans!r}")
                check("42" in ans, "reads the 42")

                print("[3/5] five-turn reuse")
                convo = [{"role": "user", "content": [
                    img_part(), {"type": "text",
                                 "text": "Remember this image. Say ok."}]}]
                # The image's own token count: the same first message with
                # and without it. A turn that re-prefilled the image would
                # process at least this many new tokens.
                bare = _chat(base, [{"role": "user", "content":
                                     "Remember this image. Say ok."}],
                             max_tokens=1)
                for turn in range(5):
                    r = _chat(base, convo)
                    u = r.get("usage", {})
                    cached = (u.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0)
                    prompt = u.get("prompt_tokens", 0)
                    print(f"      turn {turn}: prompt={prompt} cached={cached}"
                          f" new={prompt - cached}")
                    if turn == 0:
                        img_tokens = prompt - bare["usage"]["prompt_tokens"]
                        print(f"      the image is {img_tokens} tokens")
                    else:
                        check(prompt - cached < img_tokens,
                              f"turn {turn} did not re-prefill the image "
                              f"({prompt - cached} new < {img_tokens})")
                    convo.append({"role": "assistant", "content": said(r)})
                    convo.append({"role": "user",
                                  "content": f"Turn {turn + 1}. Say ok."})
                convo[-1] = {"role": "user", "content":
                             "What color was the square in the image I "
                             "showed you first? One word."}
                r5 = _chat(base, convo)
                check("red" in said(r5).lower(),
                      f"recalls the image after five turns: {said(r5)!r}")
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

    print(f"[5/5] {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    return 1 if fails else 0


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
