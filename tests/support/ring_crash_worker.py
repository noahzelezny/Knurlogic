"""One rank of tests/cluster/test_ring_crash.py's two-process ring (not a
test module): the serving path (rank 0's TensorExecutor, the follower's
tensor.follow) on a tiny model, with one rank made to fail.

    argv: <out dir> <split> <fault>

`split`: tensor (the tiny qwen3_5_moe sharded, no head) or pipeline (the
tiny qwen3_5 split by layer runs 1,3, rank 0 drafting with its MTP head).
`fault` names the rank that fails and how:

    f_desync      the follower's step counter jumps: both ranks raise
                  Desync in the same exchange (the live failure), and the
                  follower then LINGERS alive -- as a process stuck in its
                  exit would, or a jaccl peer that never fails a collective
                  -- so a survivor that entered another collective would
                  wait on it
    f_raise_step  the follower raises inside its 3rd step (rank 0 is in
                  that step's collectives) and exits
    f_kill_step   the follower SIGKILLs itself inside its 3rd step
    f_raise_coord the follower raises in its 6th per-step broadcast
                  (pipeline.Coord: b0, the drafting b1 / b2) and exits
    f_kill_parked the follower SIGKILLs itself while parked on the bell;
                  rank 0 is idle, in no collective
    r0_raise_step rank 0 raises inside its 3rd step, after the exchange
                  (the follower is in that step's collectives), then leaves
    r0_kill_parked rank 0 SIGKILLs itself with the follower parked

Each rank writes <out dir>/rank<r>.json: what it did and when (time.time()),
the failing rank its `injected` time first."""
import json
import os
import signal
import sys
import time

FAULTS = ("f_desync", "f_raise_step", "f_kill_step", "f_raise_coord",
          "f_kill_parked", "r0_raise_step", "r0_kill_parked")


class Injected(RuntimeError):
    pass


def main(out_dir, split, fault):
    import faulthandler
    faulthandler.dump_traceback_later(float(os.environ.get(
        "PIPELINE_WORKER_DEADLINE", "120")), exit=True)
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.environ["KNURLOGIC_MTP_BATCH_MAX_ROWS"] = "8"
    assert fault in FAULTS, fault
    from knurlogic.engine.runtime import tensor as T
    link = T.init("ring")
    me = link.rank
    rec = {"rank": me, "fault": fault}

    def write():
        with open(os.path.join(out_dir, f"rank{me}.json"), "w") as f:
            json.dump(rec, f)

    def injected(what):
        rec["injected"] = time.time()
        rec["what"] = what
        write()

    def die():
        injected("SIGKILL")
        os.kill(os.getpid(), signal.SIGKILL)

    from pipeline_ring_worker import FakeTok, _tiny_with_head

    from knurlogic.engine.mtp.batch_generator import MTPBatchGenerator
    from knurlogic.engine.runtime import pipeline as PL
    from knurlogic.engine.runtime.executor import (
        Admission,
        LocalExecutor,
        Token,
    )
    from knurlogic.engine.runtime.request import control_machine
    tok = FakeTok()
    if split == "pipeline":
        model, head, prompts = _tiny_with_head(512)
        PL.split(model, link.group, PL.bounds_of([1, 3]))
        drafting = True
    else:
        from tensor_ring_worker import build
        model, head, drafting = build(), None, False
        prompts = [[5, 17, 3, 99, 42, 7, 64, 11], [23, 31, 104, 33, 9, 8, 7,
                                                   6]]
        T.shard(model, link.group)

    # ------------------------------------------------- the faults
    real_step = LocalExecutor.step
    steps = [0]

    def step_hook(kill, raise_):
        def step(self):
            steps[0] += 1
            if steps[0] == 3:
                if kill:
                    die()
                if raise_:
                    injected("raise in step")
                    raise Injected("injected failure inside a step")
            return real_step(self)
        LocalExecutor.step = step

    if me == 1 and fault == "f_desync":
        real_x, n = T.Link.exchange, [0]

        def exchange(self, over, payload=None):
            n[0] += 1
            if n[0] == 4:
                injected("step counter jump")
                self.step += 1
            return real_x(self, over, payload)
        T.Link.exchange = exchange
    elif me == 1 and fault in ("f_raise_step", "f_kill_step"):
        step_hook(fault == "f_kill_step", fault == "f_raise_step")
    elif me == 1 and fault == "f_raise_coord":
        calls = [0]
        for name in ("_bcast", "b0"):
            real = getattr(PL.Coord, name)

            def wrap(self, *a, _real=real, _name=name, **k):
                calls[0] += 1
                if calls[0] == 6:
                    injected(f"raise in Coord.{_name}")
                    raise Injected(f"injected failure in Coord.{_name}")
                return _real(self, *a, **k)
            setattr(PL.Coord, name, wrap)
    elif me == 1 and fault == "f_kill_parked":
        def sleep(self):
            die()
        T.Link.sleep = sleep
    elif me == 0 and fault == "r0_raise_step":
        step_hook(False, True)

    # ------------------------------------------------- the follower
    if me > 0:
        try:
            n = T.follow(model, tok, ("tiny", None, None), link,
                         prompt_cache_size=4, completion_batch_size=32,
                         prefill_step_size=16, working_set=0, split=split,
                         drafting=drafting)
            rec.update(outcome="returned", steps=n, down=link.why)
        except BaseException as e:
            rec.update(outcome="raised", error=f"{type(e).__name__}: {e}",
                       down=link.why)
        rec["done"] = time.time()
        write()
        if fault == "f_desync":
            time.sleep(60)           # linger: the test kills it
        if rec["outcome"] == "returned":
            T.leave_if_down(link)    # serve_follower's exit
            return 0
        return 3

    # ------------------------------------------------- rank 0
    gen = MTPBatchGenerator(model, head, stats={}, prefill_step_size=16,
                            completion_batch_size=32)
    if drafting:
        PL.coordinate(gen, link.group)
    ring = T.Ring(link, split=split)
    ex = T.TensorExecutor(gen, ring, over=lambda: 0)
    uids = []
    for p in prompts:
        sm, _ = control_machine(tok, "normal")
        uids.append(ex.insert(Admission(
            segments=[p[:5], p[5:]], max_tokens=24, sampling={"seed": 7},
            state_machine=sm, wire={"penalties": {}, "initial": "normal"})))
    toks = 0
    try:
        done = set()
        for _ in range(10_000):
            for e in ex.step():
                toks += isinstance(e, Token)
                if type(e).__name__ in ("Finished", "RowFailure"):
                    done.add(e.uid)
            if done >= set(uids):
                break
        rec["drained"] = True
        ring.park()                      # idle: the follower sleeps
        if fault == "r0_kill_parked":
            time.sleep(0.5)
            die()
        # idle, in no collective: told of a gone peer by the bell alone
        rec["down_seen"] = link.down.wait(30)
        rec["outcome"] = "down" if rec["down_seen"] else "never told"
    except BaseException as e:
        rec.update(outcome="raised", error=f"{type(e).__name__}: {e}")
    rec.update(tokens=toks, down=link.why)
    # what the scheduler's exit does: close the executor, stop the ring --
    # neither may enter a collective on a ring that is down
    try:
        ex.close()
        ring.stop()
        rec["stopped"] = True
    except BaseException as e:
        rec["stop_error"] = f"{type(e).__name__}: {e}"
    rec["done"] = time.time()
    write()
    T.leave_if_down(link)            # as watch_ring's exit: os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
