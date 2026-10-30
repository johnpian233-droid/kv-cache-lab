#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the real reporting path of ollama_probe.py against recorded VRAM levels.

No GPU, no Ollama, no network. Every measurement function is replaced by a stub, and
`main()` is executed as written. This is deliberately not a unit test of the arithmetic -
test_fit.py already does that. This is a test of the numbers that get *printed*, which is
where v0.2.1's worst bug lived: the step analysis computed the slope inflation as
(step slope - high-point slope) instead of (line slope - step slope). Both of those are
estimates of the same within-region slope, so the line printed "+0.00 KiB/token" on every
run, forever, while the correct answer was 1.89. The arithmetic underneath was right the
whole time and had an assertion on it; nothing checked the sentence.

The levels below are the q8_0 replication run, so the replay also re-derives the published
CSV from the raw levels: deltas of 1138 / 1173 / 1301 / 1441 / 1716 MiB, bracketed deltas
of 1138.0 / 1170.5 / 1305.5 / 1439.0 / 1714.0. If the script's differencing changes, these
stop matching.

    python tests/test_replay.py
"""

import contextlib
import importlib.util
import io
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# nvidia-smi levels, MiB, per visit. Index advances when a row's load is released.
IDLE = [1889, 1889, 1894, 1885, 1889, 1893]
LOADED = [3027, 3062, 3195, 3326, 3605, 3029]
SIZE_VRAM = [1031.1, 1062.8, 1198.8, 1333.8, 1603.8, 1031.1]

CHECKS = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    return bool(ok)


def load_probe():
    path = os.path.join(ROOT, "scripts", "ollama_probe.py")
    spec = importlib.util.spec_from_file_location("ollama_probe_replay", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeTime(object):
    """Enough of the time module to let the sampler spin without waiting."""

    def __init__(self):
        self.t = 1000.0

    def time(self):
        self.t += 0.05
        return self.t

    def sleep(self, _seconds):
        return None


class Stub(object):
    def __init__(self, probe, blind_from=None):
        self.probe = probe
        self.done = 0
        self.loaded = None
        # Once `done` reaches this row index, nvidia-smi "returns nothing" - the failure
        # mode that used to carry the previous row's baseline forward.
        self.blind_from = blind_from

    # -- measurement surface -------------------------------------------------
    def gpu_count(self):
        return 1

    def gpu_used(self, _idx=0):
        if self.blind_from is not None and self.done >= self.blind_from:
            return None
        if self.loaded is not None:
            return self.loaded
        return IDLE[min(self.done, len(IDLE) - 1)]

    def api(self, _host, path, payload=None, timeout=None):
        if path == "/api/version":
            return {"version": "0.40.6"}
        if path == "/api/generate":
            self.loaded = LOADED[min(self.done, len(LOADED) - 1)]
            return {"response": "hi"}
        return {}

    def api_ps(self, _host, _model):
        return True, SIZE_VRAM[min(self.done, len(SIZE_VRAM) - 1)]

    def unload(self, _model):
        if self.loaded is not None:
            self.loaded = None
            self.done += 1
        return True

    def wait_unloaded(self, *_a, **_k):
        return True, "api/ps"

    def server_kv_type(self):
        return "q8_0", "%LOCALAPPDATA%\\Ollama\\server.log"


def run(probe, argv, blind_from=None):
    stub = Stub(probe, blind_from)
    saved = (probe._gpu_used, probe.gpu_count, probe.api, probe.api_ps, probe.unload,
             probe.wait_unloaded, probe.server_kv_type, probe.time)
    probe._gpu_used = stub.gpu_used
    probe.gpu_count = stub.gpu_count
    probe.api = stub.api
    probe.api_ps = stub.api_ps
    probe.unload = stub.unload
    probe.wait_unloaded = stub.wait_unloaded
    probe.server_kv_type = stub.server_kv_type
    probe.time = FakeTime()

    cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="kv-replay-")
    buf = io.StringIO()
    try:
        os.chdir(tmp)  # the script writes its CSV into the working directory
        sys.argv = ["ollama_probe.py", "--kv-bytes", "1.0625", "--samples", "3",
                    "--tag", "replay"] + list(argv)
        with contextlib.redirect_stdout(buf):
            try:
                probe.main()
            except SystemExit as e:
                buf.write("\n[exited] %s\n" % e)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
        (probe._gpu_used, probe.gpu_count, probe.api, probe.api_ps, probe.unload,
         probe.wait_unloaded, probe.server_kv_type, probe.time) = saved
    return buf.getvalue()


def inflation_line(out):
    m = re.search(r"inflated by ([+-]?\d+\.\d+)", out)
    return float(m.group(1)) if m else None


def main():
    probe = load_probe()

    # ---------------------------------------------- the standard five-point run
    out = run(probe, ["--aba"])
    check("replay: five points, per-point column", "on 5 points (delta_mib)" in out)
    # the differencing still reproduces the published CSV
    for v in ("1138.0", "1173.0", "1301.0", "1441.0", "1716.0"):
        check("replay: delta %s MiB reproduced" % v.split(".")[0], v in out)

    # the bug this file exists for: the printed inflation must be the distance between
    # the straight line and the step model, not the distance between two estimates of
    # the same within-region slope (which is ~0.00 and looks like a rounding artefact).
    xs = [2048, 4096, 8192, 16384, 32768]
    ys = [float(l - i) for l, i in zip(LOADED[:5], IDLE[:5])]
    a, _b, _r2 = probe.linfit(xs, ys)
    st = probe.detect_step(xs, ys)
    expected = (a - st["slope"]) * 1024
    got = inflation_line(out)
    check("replay: inflation line is present", got is not None)
    if got is not None:
        check("replay: inflation = %.2f, not 0.00" % expected,
              abs(got - expected) < 0.02 and abs(got) > 1.0)

    # ------------------------------------------------- the bracket without --aba
    # With --bracket the bracketed column becomes the main fit, so the missing top point
    # shows up there rather than in the refit block. Four points, and the largest context
    # is gone - the point that constrains the slope hardest.
    out = run(probe, ["--bracket"])
    check("replay: unbracketed top point is announced", "Re-run with --aba" in out)
    check("replay: bracketed fit drops to 4 points",
          "on 4 points (delta_bracket_mib)" in out)
    check("replay: no step claimed on 4 bracketed points", "[step]" not in out)

    # ---------------------------------------------------- the bracket with --aba
    out = run(probe, ["--bracket", "--aba"])
    check("replay: --aba keeps all five bracketed",
          "on 5 points (delta_bracket_mib)" in out)
    check("replay: no drop warning when --aba is set", "Re-run with --aba" not in out)

    # ------------------------------------------------------------- four points
    # With n=4 the step model leaves one residual degree of freedom, so SSE collapses and
    # F diverges whatever the data says. stepfit must refuse rather than announce a step.
    out = run(probe, ["--seqs", "2048", "4096", "8192", "16384", "--aba"])
    check("replay: four-point fit is labelled as four points", "on 4 points (delta_mib)" in out)
    check("replay: no step claimed on four points", "[step]" not in out)

    # ------------------------------------- guards added in v0.2.2, each a silent nan
    # A repeated num_ctx makes the regression singular: the slope divides by zero, every
    # number prints as nan, and the process still exits 0 with a CSV on disk.
    out = run(probe, ["--seqs", "2048", "4096", "4096", "8192", "--aba"])
    check("replay: duplicate num_ctx is refused", "appears more than once" in out)

    # A --kv-bytes matching no known type cannot be compared with the server, and used to
    # be recorded as kv_type_verified=True: a claim of a check that never ran.
    out = run(probe, ["--kv-bytes", "1.2"])
    check("replay: unknown kv-bytes is recorded unverified",
          "kv_type_verified=False" in out)
    check("replay: unknown kv-bytes does not claim a match",
          "server agrees" not in out)

    # Zero geometry divides by zero in the theory ratio.
    out = run(probe, ["--layers", "0"])
    check("replay: zero geometry is refused", "geometry gives 0" in out)

    # --gpu beyond the machine used to fall back to GPU 0 without saying so.
    out = run(probe, ["--gpu", "3"])
    check("replay: nonexistent --gpu is refused", "--gpu 3 does not exist" in out)

    # nvidia-smi going blind for one row must drop that row. Carrying the previous row's
    # baseline forward produced delta = 0 - idle and fitted it like a measurement.
    out = run(probe, [], blind_from=4)
    check("replay: a blind row is dropped, not differenced",
          "FAILED  baseline: nvidia-smi returned nothing" in out)
    check("replay: the dropped row leaves four points", "on 4 points (delta_mib)" in out)
    # and the bracket note must blame the failed load, not call it the largest context
    check("replay: missing middle bracket is explained correctly",
          "the load after it failed" in out)

    bad = [n for n, ok in CHECKS if not ok]
    for n, ok in CHECKS:
        print(("  ok   " if ok else "  FAIL ") + n)
    if bad:
        print("\n%d of %d replay checks failed" % (len(bad), len(CHECKS)))
        return 1
    print("\nall %d replay checks passed - the reporting path prints what it claims to"
          % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
