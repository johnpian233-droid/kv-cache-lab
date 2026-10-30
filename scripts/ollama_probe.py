#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KV Cache Memory Probe for Ollama
================================

Measure how much GPU memory the KV cache *actually* costs on your machine,
one context length at a time.

Why this exists
---------------
Vendors keep publishing "we cut KV cache by 50%" claims. But on a consumer GPU,
KV cache is not the whole story: model weights and compute buffers are a fixed
cost that never discounts. This script separates the two, so you can work out
what *your* bill actually saves.

Method
------
For each num_ctx:
  1. unload the model        (`ollama stop`)
  2. confirm it is gone      (`ollama ps`, not a VRAM threshold - see ollama_ps)
  3. sample the idle floor   -> idle baseline (median of 5)
  4. load at that num_ctx    -> loaded VRAM (peak, not median)
  5. delta = loaded - idle

Weights are constant across runs, so the *slope* of delta vs num_ctx is the
per-token KV cost. The intercept is the model's fixed footprint.

Always use deltas, never absolute VRAM: desktop usage drifts by hundreds of MiB
between runs, and deltas absorb that automatically.

Requirements
------------
- Ollama running locally (any recent version)
- `nvidia-smi` on PATH
- Python 3.8+  (standard library only — no pip install needed)

Usage
-----
    # q8_0 KV cache
    python ollama_probe.py --tag q8_0 --kv-bytes 1.0625

    # f16 KV cache, 9 readings per point, plus a baseline drift probe at the end
    python ollama_probe.py --tag f16 --kv-bytes 2 --samples 9 --aba

    # custom model geometry
    python ollama_probe.py --model qwen2.5:7b --layers 28 --kv-heads 4 --head-dim 128

If --kv-bytes is omitted, it is inferred from $OLLAMA_KV_CACHE_TYPE.

Output
------
Print: fit result (slope = KiB/token, R^2) and an extrapolation table.
File:  ollama_probe_<tag>.csv
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

MIB = 1024 ** 2

# Bytes per element for common KV cache types (ggml block quantization included)
KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,   # 32 int8 + 1 fp16 scale per block -> 34/32
    "q4_0": 0.5625,   # 32 int4 + 1 fp16 scale per block -> 18/32
}


# --------------------------------------------------------------------- helpers
def _gpu_used(gpu=0):
    """Currently used VRAM (MiB) of the given GPU index."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        if not lines:
            return None
        idx = gpu if 0 <= gpu < len(lines) else 0
        return int(float(lines[idx]))
    except Exception:
        return None


def api(host, path, payload=None, timeout=900):
    url = host.rstrip("/") + path
    if payload is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def unload(model):
    """Unload model from VRAM. Returns True on success."""
    try:
        r = subprocess.run(["ollama", "stop", model], capture_output=True, timeout=180)
        return r.returncode == 0
    except Exception:
        return False


def server_kv_type():
    """The KV cache type the *server* is actually using, read from its own startup log.

    Why this exists, and what it cost to learn. `--kv-bytes` is a labelling flag: it
    tells this script what to print in the theory column and what to write into the
    CSV. It does not tell Ollama anything. The real type comes from
    OLLAMA_KV_CACHE_TYPE in the *server's* environment, set before `ollama serve`
    started, and nothing in the request path can override it.

    The consequence is not academic. During v0.2 development the server was on q8_0
    while the operator passed `--kv-bytes 2`, so a run was recorded as f16 whose
    measured slope was 19.0 KiB/token - which is 1.28x the q8_0 theory and 0.68x the
    f16 theory. It matched the older q8_0 run to within 20 MiB at all five contexts,
    which is how the mislabel was caught. A one-line flag had silently relabelled an
    entire experiment, and the CSV column that was supposed to prove otherwise was a
    copy of that same flag.

    Returns (type_string, source) or (None, None) when the log cannot be read.
    """
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Ollama", "server.log"),
        os.path.join(os.path.expanduser("~"), ".ollama", "logs", "server.log"),
    ]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        # The last "server config" line is the running instance's environment.
        hits = re.findall(r"OLLAMA_KV_CACHE_TYPE:([A-Za-z0-9_]+)", text)
        if hits:
            return hits[-1].lower(), path
    return None, None


def api_ps(host, model):
    """(running, size_vram_mib) from /api/ps, in exact bytes.

    Better than the CLI: `ollama ps` prints "1.7 GB", which is 102 MiB of rounding and
    therefore useless as a check on a 400 MiB effect. /api/ps reports bytes, and
    size_vram is the runner's own account of the VRAM it holds - an instrument that is
    independent of nvidia-smi and of the desktop entirely.
    """
    try:
        d = api(host, "/api/ps", timeout=20)
    except Exception:
        return None, None
    for m in d.get("models", []):
        if model is None or m.get("name") == model or m.get("model") == model:
            sv = m.get("size_vram")
            return True, (sv / MIB) if sv else None
    return False, None


def wait_unloaded(host, model, threshold=None, tries=60, gpu=0):
    """Wait until the model is out of VRAM. Returns (ok, how).

    `ollama ps` (via /api/ps) is authoritative. The VRAM threshold is only a fallback
    for machines where the API is unreachable, and it is a blunt instrument: it cannot
    tell a resident model from a desktop that grew.

    Why ask Ollama instead of the GPU. The obvious unload check is "wait until VRAM
    falls back to the baseline", and it is wrong. A desktop is not a constant: on the
    machine this repo was measured on the idle floor climbed ~150 MiB over ten minutes
    with nothing loaded at all (`ollama ps` empty the whole time). An absolute
    threshold then fires for a reason that has nothing to do with the model - it
    reports "model still resident" while the model has been gone for minutes. The
    only party that knows whether a model holds VRAM is the process that put it
    there, so ask that.
    """
    for _ in range(tries):
        running, _ = api_ps(host, model)
        if running is False:
            return True, "ollama ps"
        if running is None and threshold is not None:
            v = _gpu_used(gpu)
            if v is not None and v <= threshold:
                return True, "vram threshold"
        time.sleep(0.5)
    running, _ = api_ps(host, model)
    if running is False:
        return True, "ollama ps"
    return False, "timeout"


def theory_kv_kib(n_ctx, layers, kv_heads, head_dim, bytes_per_elem):
    """2 (K and V) * layers * kv_heads * head_dim * bytes / 1024"""
    return 2 * layers * kv_heads * head_dim * bytes_per_elem * n_ctx / 1024.0


def linfit(xs, ys):
    """Least squares y = a*x + b. Returns (a, b, r2)."""
    n = len(xs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if den == 0:
        return float("nan"), float("nan"), float("nan")
    a = (n * sxy - sx * sy) / den
    b = (sy - a * sx) / n
    ybar = sy / n
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2


# Two-sided 95% Student-t quantiles, df 1..30. Anything above 30 falls back to 1.96.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}

# Upper 5% points of F(1, df2). The step model spends exactly one extra parameter,
# so df1 is always 1 and the table only needs df2 = n - 3. Hard-coding a single
# number here would be wrong for any grid other than five points: at n=4 the
# critical value is 161, at n=7 it is 6.61, and a fixed 18.5 flips the verdict.
_F95_DF1_1 = {1: 161.448, 2: 18.513, 3: 10.128, 4: 7.709, 5: 6.608,
              6: 5.987, 7: 5.591, 8: 5.318, 9: 5.117, 10: 4.965,
              11: 4.844, 12: 4.747, 13: 4.667, 14: 4.600, 15: 4.543,
              16: 4.494, 17: 4.451, 18: 4.414, 19: 4.381, 20: 4.351}


def f_critical_95(df2):
    """5% critical value of F(1, df2), or None when df2 is outside the table."""
    return _F95_DF1_1.get(df2)


def fit_uncertainty(xs, ys, a, b):
    """Standard error of slope/intercept and 95% CI half-widths.

    Five points is a small sample. Reporting a bare slope of "32.11" implies a
    precision the data does not support, so the CI is printed next to it.
    Returns (se_a, se_b, half_a, half_b, df); NaNs when df < 1.
    """
    n = len(xs)
    df = n - 2
    nan = float("nan")
    if n < 3 or df < 1:
        return nan, nan, nan, nan, df
    sse = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    sx = sum(xs)
    sxx = sum(x * x for x in xs)
    sxx_centred = sxx - sx * sx / n
    if sxx_centred <= 0:
        return nan, nan, nan, nan, df
    mse = sse / df
    se_a = math.sqrt(mse / sxx_centred)
    se_b = math.sqrt(mse * (1.0 / n + (sx / n) ** 2 / sxx_centred))
    t = _T95.get(df, 1.96)
    return se_a, se_b, t * se_a, t * se_b, df


def _solve(matrix, rhs):
    """Gaussian elimination with partial pivoting on a tiny square system."""
    n = len(matrix)
    m = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            factor = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]
    out = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = m[r][n] - sum(m[r][c] * out[c] for c in range(r + 1, n))
        out[r] = acc / m[r][r]
    return out


def stepfit(xs, ys, step_at):
    """Least squares with a one-time level shift at `step_at`:

        y = b + a*x + c * 1[x >= step_at]

    Why this exists. A straight line drawn through a curve that contains a
    single step does not fail loudly - it quietly spreads the step across
    every token and reports it as per-token cost. Both the slope and the
    "cost over theory" come out too high by exactly (step / largest_ctx).

    The F statistic compares this model against the plain line. With n points
    the step model spends one extra parameter, so df goes from n-2 to n-3.

    Returns a dict, or None when there are too few points on either side.
    """
    n = len(xs)
    if n < 4:
        return None
    left = [y for x, y in zip(xs, ys) if x < step_at]
    right = [y for x, y in zip(xs, ys) if x >= step_at]
    if len(left) < 1 or len(right) < 2:
        return None

    design = [[1.0, x, 1.0 if x >= step_at else 0.0] for x in xs]
    k = 3
    ata = [[sum(design[i][p] * design[i][q] for i in range(n)) for q in range(k)]
           for p in range(k)]
    aty = [sum(design[i][p] * ys[i] for i in range(n)) for p in range(k)]
    coef = _solve(ata, aty)
    if coef is None:
        return None

    intercept, slope, step = coef
    sse_step = sum((ys[i] - sum(coef[p] * design[i][p] for p in range(k))) ** 2
                   for i in range(n))
    df_step = n - k
    lin_a, lin_b, _ = linfit(xs, ys)
    sse_lin = sum((y - (lin_a * x + lin_b)) ** 2 for x, y in zip(xs, ys))

    if df_step < 1 or sse_step <= 0:
        f_stat = float("inf")
    else:
        mse_step = sse_step / df_step
        f_stat = (sse_lin - sse_step) / mse_step

    # slope estimated from the points at or above the step only
    hx = [x for x in xs if x >= step_at]
    hy = [y for x, y in zip(xs, ys) if x >= step_at]
    mx = sum(hx) / len(hx)
    my = sum(hy) / len(hy)
    sxx_h = sum((x - mx) ** 2 for x in hx)
    slope_hi = (sum((x - mx) * (y - my) for x, y in zip(hx, hy)) / sxx_h
                if sxx_h > 0 else float("nan"))

    return {
        "step_at": step_at, "slope": slope, "intercept": intercept,
        "step": step, "sse": sse_step, "df": df_step, "f": f_stat,
        "sse_lin": sse_lin, "df_lin": n - 2, "slope_hi": slope_hi,
        "n_left": len(left), "n_right": len(right),
    }


def polysse(xs, ys, deg):
    """Least-squares polynomial of degree `deg`. Returns (sse, residuals) or None.

    Why this exists. A one-time step and a smooth curve both bend five points, and
    a step model with three parameters can look like a cheap way to fake curvature.
    If a quadratic - also three parameters - absorbs the same bump, then the "step"
    is curvature wearing a different name and the physical story behind it is wrong.
    The two models are non-nested but equally sized, so SSE can be compared directly
    with no penalty term. Fitting both is the only way to tell a jump from a bend.

    Fitted in scaled coordinates: raw `num_ctx` runs to 32768, and the normal
    equations for a quadratic on x^4 are hopelessly ill-conditioned unscaled.
    """
    n = deg + 1
    xm = float(max(xs)) or 1.0
    u = [x / xm for x in xs]
    a = [[sum(x ** (i + j) for x in u) for j in range(n)] for i in range(n)]
    rhs = [sum(y * x ** i for x, y in zip(u, ys)) for i in range(n)]
    coef = _solve(a, rhs)
    if coef is None:
        return None
    res = [y - sum(c * x ** i for i, c in enumerate(coef)) for x, y in zip(u, ys)]
    return sum(r * r for r in res), res


def detect_step(xs, ys):
    """Pick the step location that explains the data best, among interior points."""
    best = None
    for x in xs[1:-1]:
        r = stepfit(xs, ys, x)
        if r and (best is None or r["sse"] < best["sse"]):
            best = r
    return best


def infer_kv_bytes():
    t = os.environ.get("OLLAMA_KV_CACHE_TYPE", "").strip().lower()
    return KV_BYTES.get(t), t


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--seqs", type=int, nargs="+",
                    default=[2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--tag", default="run", help="suffix for the output CSV")
    ap.add_argument("--kv-bytes", type=float, default=None,
                    help="bytes per KV element (f16=2, q8_0=1.0625, q4_0=0.5625). "
                         "Defaults to $OLLAMA_KV_CACHE_TYPE.")
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--samples", type=int, default=7,
                    help="nvidia-smi readings per point, peak reported. 1 reproduces the "
                         "v0.1.x behaviour; 9-15 is what the v0.2 measurement used")
    ap.add_argument("--aba", action="store_true",
                    help="re-measure the first context at the end, as a baseline drift probe")
    ap.add_argument("--settle", type=int, default=32,
                    help="MiB tolerance for the fallback unload check, used only when "
                         "`ollama ps` is unavailable")
    ap.add_argument("--allow-kv-mismatch", action="store_true",
                    help="proceed even if the server's OLLAMA_KV_CACHE_TYPE disagrees "
                         "with --kv-bytes (the CSV will say the label was unverified)")
    ap.add_argument("--bracket", action="store_true",
                    help="also report deltas against the mean of the idle readings that "
                         "bracket each load (needs --aba to bracket the last point)")
    args = ap.parse_args()

    gpu_idx = args.gpu
    used = lambda: _gpu_used(gpu_idx)  # noqa: E731

    # ---- resolve KV byte size (this drives the theory column)
    kv_bytes = args.kv_bytes
    kv_type_env = None
    if kv_bytes is None:
        kv_bytes, kv_type_env = infer_kv_bytes()
        if kv_bytes is None:
            sys.exit(
                "[error] cannot infer KV cache type.\n"
                "        Pass --kv-bytes explicitly (f16=2, q8_0=1.0625, q4_0=0.5625),\n"
                "        or set OLLAMA_KV_CACHE_TYPE and restart `ollama serve`."
            )
    kv_label = f"{kv_bytes:g} B/elem"

    # ---- service health
    try:
        ver = api(args.host, "/api/version", timeout=15).get("version", "?")
    except Exception as e:
        sys.exit(f"[error] cannot reach Ollama at {args.host}\n"
                 f"        is `ollama serve` running?  ({e})")

    # ---- did the server actually agree to what we just labelled this run?
    srv_type, srv_src = server_kv_type()
    kv_verified = False
    declared_type = next((k for k, v in KV_BYTES.items()
                          if abs(v - kv_bytes) < 1e-9), None)
    if srv_type is None:
        print("[kv check] could not read the server's environment - the KV type below is")
        print("           whatever YOU declared, not what the server is doing. Any")
        print("           comparison against a run made under a different environment is")
        print("           unverified. Check `OLLAMA_KV_CACHE_TYPE` yourself.")
    elif declared_type and srv_type != declared_type:
        print("=" * 72)
        print(f"[kv check] MISMATCH. The server is running OLLAMA_KV_CACHE_TYPE={srv_type}")
        print(f"           but you passed bytes for {declared_type}. Ollama ignores this")
        print("           flag, so the run would be measured with the server's type and")
        print(f"           labelled with yours - and the theory column would be wrong by")
        print(f"           {kv_bytes / KV_BYTES.get(srv_type, kv_bytes):.2f}x.")
        print(f"           (source: {srv_src})")
        print("=" * 72)
        if not args.allow_kv_mismatch:
            sys.exit("[error] refusing to write a mislabelled CSV. Restart `ollama serve`")
    else:
        kv_verified = True

    print(f"[ollama]   {ver}    model: {args.model}")
    print(f"[kv type]  {kv_label}" + (f"  (declared via --kv-bytes)" if kv_bytes else ""))
    if kv_verified:
        print(f"[kv check] server agrees: OLLAMA_KV_CACHE_TYPE={srv_type}  ({srv_src})")
    print(f"[geometry] layers={args.layers} kv_heads={args.kv_heads} head_dim={args.head_dim}")

    # ---- idle baseline
    print("[setup]   unloading model, measuring idle baseline ...")
    if not unload(args.model):
        print("[warning] `ollama stop` failed — VRAM may not be released. "
              "Is the ollama CLI on PATH?")
    probe = used()
    if probe is None:
        sys.exit("[error] nvidia-smi returned nothing — check driver / PATH")
    ok, how = wait_unloaded(args.host, args.model, threshold=probe + args.settle,
                            tries=40, gpu=gpu_idx)
    if not ok:
        print("[warning] model still resident after 20 s; the baseline will be too high")
    idle = statistics.median([v for v in (_gpu_used(gpu_idx) for _ in range(5))
                             if v is not None] or [probe])
    print(f"[idle]    {idle} MiB (desktop + other apps, unload confirmed via {how})\n")

    rows = []
    hdr = (f"{'num_ctx':>8} | {'loaded MiB':>10} | {'delta MiB':>9} | "
           f"{'ps MiB':>7} | {'theory':>7} | {'load s':>6} | sampling")
    print(hdr)
    print("-" * len(hdr))

    # A-B-A: re-measure the smallest context at the end. The difference between the
    # first and last reading of the SAME point is the baseline drift of this run,
    # measured rather than assumed. It costs one extra load and settles a question
    # that otherwise has to be argued about (see v0.1.3 in the changelog).
    plan = [(L, False) for L in sorted(args.seqs)]
    if args.aba and plan:
        plan.append((plan[0][0], True))

    for L, is_probe in plan:
        unload(args.model)
        ok, how = wait_unloaded(args.host, args.model, threshold=idle + args.settle,
                                tries=60, gpu=gpu_idx)
        if not ok:
            print(f"[warning] num_ctx={L}: model still resident after 30 s. This row's")
            print("          baseline is contaminated and its delta will read too low.")
        # The baseline is sampled five times and taken as a median: one reading on a
        # busy desktop is a coin flip, and it was observed swinging 32 MiB peak-to-peak
        # over 8 seconds here. The median is the right statistic for a *quiet* state;
        # the peak below is the right statistic for a *loaded* state, and the difference
        # is not cosmetic - see the note at `loaded`.
        idle_series = []
        for _ in range(5):
            v = used()
            if v is not None:
                idle_series.append(v)
            time.sleep(0.3)
        if idle_series:
            cur_idle = statistics.median(idle_series)

        peak = [0]
        stop_flag = threading.Event()

        def sampler():
            while not stop_flag.is_set():
                v = used()
                if v is not None and v > peak[0]:
                    peak[0] = v
                time.sleep(0.15)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()

        t0 = time.time()
        err = None
        try:
            api(args.host, "/api/generate", {
                "model": args.model,
                "prompt": "hi",
                "stream": False,
                "keep_alive": "10m",
                "options": {"num_ctx": L, "num_predict": 1},
            })
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:160]}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        dt = time.time() - t0

        stop_flag.set()
        th.join(timeout=2)

        if err:
            print(f"{L:>8} | FAILED  {err}")
            rows.append(dict(num_ctx=L, loaded_mib=None, delta_mib=None,
                             theory_kv_mib=None, load_sec=None))
            continue

        time.sleep(1.0)
        if args.samples and args.samples > 1:
            series = []
            for _ in range(args.samples):
                v = used()
                if v is not None:
                    series.append(v)
                time.sleep(0.5)
        else:
            series = []

        # loaded is a PEAK, not a median, and getting this backwards silently wrecked a
        # full five-point run during v0.2 development: median shaved 300-600 MiB off the
        # high contexts, which looked exactly like a smaller KV cost and would have been
        # published as "the step is gone". The idle side wants a median (noise only), the
        # loaded side wants a peak (the state IS the allocation). They are not symmetric.
        after = used()
        loaded = max([peak[0], after or 0] + series) if (series or after) else peak[0]
        spread = (max(series) - min(series)) if len(series) > 1 else 0
        delta = loaded - cur_idle
        theory = theory_kv_kib(L, args.layers, args.kv_heads, args.head_dim, kv_bytes) / 1024
        _, ps_size = api_ps(args.host, args.model)

        tag = " (probe)" if is_probe else ""
        ps_txt = f"{ps_size:>7.0f}" if ps_size else f"{'-':>7}"
        print(f"{L:>8} | {loaded:>10.1f} | {delta:>9.1f} | {ps_txt} | {theory:>7.0f}"
              f" | {dt:>6.1f} | n={len(series)} spread {spread:.0f}{tag}")
        # idle_mib is the reading this row was differenced against. It used to be
        # thrown away, and that is how a 26 MiB baseline drift in one run went
        # unnoticed for two releases: `loaded - delta` was recoverable, but only if
        # somebody thought to compute it.
        rows.append(dict(num_ctx=L, loaded_mib=round(loaded, 1), delta_mib=round(delta, 1),
                         idle_mib=cur_idle, n_samples=len(series),
                         spread_mib=spread, is_probe=is_probe,
                         size_vram_mib=round(ps_size, 1) if ps_size else None,
                         theory_kv_mib=round(theory, 1), load_sec=round(dt, 1)))

    # ---- bracketed baselines
    # A load sits between two idle readings: the one taken just before it and the one
    # taken just before the NEXT load. Their mean cancels a monotone drift exactly,
    # which a per-point baseline cannot do - and a per-point baseline that drifts is
    # precisely what manufactures a fake step at whichever context the drift happens
    # to step over. --aba supplies the closing bracket for the last point.
    for i, r in enumerate(rows):
        nxt = rows[i + 1] if i + 1 < len(rows) else None
        if (nxt and r.get("idle_mib") is not None
                and nxt.get("idle_mib") is not None and r.get("loaded_mib") is not None):
            base = (r["idle_mib"] + nxt["idle_mib"]) / 2.0
            r["idle_mean_mib"] = round(base, 1)
            r["delta_bracket_mib"] = round(r["loaded_mib"] - base, 1)
    bracketed = [r for r in rows if r.get("delta_bracket_mib") is not None
                 and not r.get("is_probe")]
    if len(bracketed) >= 3:
        print()
        print("[bracket] same loads, baseline = mean of the idle readings either side")
        print(f"          {'num_ctx':>8} {'loaded':>8} {'idle pre':>9} {'idle post':>10}"
              f" {'mean':>7} {'delta':>8} {'vs point':>9}")
        for r in bracketed:
            j = rows.index(r)
            post = rows[j + 1]["idle_mib"] if j + 1 < len(rows) else None
            print(f"          {r['num_ctx']:>8} {r['loaded_mib']:>8.0f}"
                  f" {r['idle_mib']:>9} {post if post is not None else '-':>10}"
                  f" {r['idle_mean_mib']:>7.0f} {r['delta_bracket_mib']:>8.0f}"
                  f" {r['delta_bracket_mib'] - r['delta_mib']:>+9.0f}")

    # ---- A-B-A verdict: how far did the baseline move while we were measuring?
    probes = [r for r in rows if r.get("is_probe") and r["delta_mib"] is not None]
    first = next((r for r in rows if not r.get("is_probe") and r["num_ctx"] == probes[0]["num_ctx"]
                  and r["delta_mib"] is not None), None) if probes else None
    if probes and first:
        d = probes[0]["delta_mib"] - first["delta_mib"]
        print()
        print(f"[drift]   same num_ctx={first['num_ctx']} measured at the start and at the end:")
        print(f"          {first['delta_mib']:.1f} -> {probes[0]['delta_mib']:.1f} MiB"
              f"   ({d:+.1f} MiB)")
        if abs(d) <= 3:
            print("          The baseline held. Single measurements would have been fine.")
        else:
            print("          The baseline moved by more than the fit residual. Any structure in")
            print("          this curve smaller than that number is not interpretable - including")
            print("          a step, if the step is small relative to this.")

    # ---- second opinion: does the runner agree with what we differenced?
    # `ollama ps` reports what the runner allocated for itself. delta_mib should track it.
    # When the two disagree, the desktop baseline moved between the idle sample and the
    # loaded sample, and it is the delta - not the runner - that is lying. This check is
    # cheap, it needs no extra state, and it would have caught the peak/median mistake
    # that cost a full v0.2 run.
    agree = [(r["num_ctx"], r["delta_mib"] - r["size_vram_mib"]) for r in rows
             if r.get("size_vram_mib") and r.get("delta_mib") is not None]
    if agree:
        worst = max(abs(e) for _, e in agree)
        print(f"[check]   delta_mib vs runner's size_vram: worst gap {worst:.0f} MiB"
              f" across {len(agree)} points")
        if worst > 200:
            print("          WARNING: the two disagree. Either the model was not fully")
            print("          unloaded when the baseline was taken, or the desktop moved")
            print("          during the load. Treat these deltas as suspect.")
        elif max(e for _, e in agree) - min(e for _, e in agree) < 60:
            print("          The gap is constant, which is the signature of a fixed CUDA")
            print("          context (~100 MiB) rather than a drifting baseline.")

    # ---- fit BEFORE writing, so the CSV always lands
    # The A-B-A probe is not a data point: it exists to measure drift, and including it
    # in the fit would double-weight the smallest context.
    delta_key = "delta_bracket_mib" if (args.bracket and len(bracketed) >= 3) else "delta_mib"
    good = [r for r in rows if r.get(delta_key) is not None and not r.get("is_probe")]
    xs = [r["num_ctx"] for r in good]
    ys = [r[delta_key] for r in good]
    a = b = r2 = float("nan")
    if len(good) >= 2:
        a, b, r2 = linfit(xs, ys)

    csv_path = f"ollama_probe_{args.tag}.csv"
    meta = {
        "model": args.model,
        "ollama_version": ver,
        "kv_cache_type": (srv_type or declared_type or kv_label),
        "kv_bytes_per_elem": kv_bytes,
        "kv_type_verified": kv_verified,
        "server_env_source": srv_src or "unreadable",
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["num_ctx", "loaded_mib", "delta_mib", "delta_bracket_mib", "idle_mib",
                  "idle_mean_mib", "n_samples", "spread_mib", "is_probe", "size_vram_mib",
                  "theory_kv_mib", "load_sec", "model", "ollama_version", "kv_cache_type",
                  "kv_bytes_per_elem", "kv_type_verified", "server_env_source"]
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **meta})
    print(f"\n[saved]   {csv_path}")

    if len(good) < 2:
        unload(args.model)
        sys.exit("[error] fewer than 2 valid samples — cannot fit")

    print()
    print("=" * 72)
    print(f"[fit]  delta = {a * 1024:.2f} KiB/token x num_ctx + {b:.0f} MiB"
          f"     R^2 = {r2:.5f}")
    # theory_kv_kib() already returns KiB, so no /1024 here.
    theory_per_tok = theory_kv_kib(1, args.layers, args.kv_heads, args.head_dim, kv_bytes)
    print(f"       theory: {theory_per_tok:.2f} KiB/token  "
          f"({kv_label})    measured/theory = {a * 1024 / theory_per_tok:.2f}x")
    se_a, se_b, half_a, half_b, df = fit_uncertainty(xs, ys, a, b)
    if se_a == se_a:  # not NaN
        gap = a * 1024 - theory_per_tok
        print(f"       1 SE   : +/-{se_a * 1024:.2f} KiB/token, "
              f"95% CI [{a * 1024 - half_a * 1024:.2f}, {a * 1024 + half_a * 1024:.2f}]"
              f"   (df={df})")
        print(f"       toll   : {gap:+.2f} KiB/token over theory; "
              f"{'outside' if abs(gap) > half_a * 1024 else 'INSIDE'} the 95% CI")
    else:
        print("       (fewer than 3 samples - no interval; model still needs 3+ points)")
    # A slope far from theory usually means the label is wrong, not that physics changed.
    # The two candidate KV types on this machine differ by 1.88x, so a ratio near 0.68 or
    # near 1.28 is the fingerprint of a mislabel, not of a discovery.
    ratio_meas = a * 1024 / theory_per_tok
    if ratio_meas < 0.70 or ratio_meas > 1.60:
        print(f"       NOTE   : measured/theory = {ratio_meas:.2f}x is far from 1. Suspect the")
        print("                LABEL before suspecting the hardware: a server on q8_0 that")
        print("                is scored against f16 theory lands at 0.53x, and one on f16")
        print("                scored against q8_0 theory lands at 1.88x.")
    # A high R^2 is a formality, not evidence: five monotone points produce it whether
    # or not a line is the right model. Say so, instead of certifying the slope.
    if r2 > 0.999:
        print("       R^2 is very high - but that is a formality, not evidence of linearity.")
    elif r2 > 0.99:
        print("       R^2 near 1; again a formality. Read the residuals below, not this.")
    else:
        print("       WARNING: poor linearity. Check for leftover VRAM from other apps.")
    print("=" * 72)

    # ---- was the baseline stable? This check did not exist before v0.1.3, and its
    # absence is what let a 26 MiB drift hide inside delta_mib.
    idle_series = [r["idle_mib"] for r in good if r.get("idle_mib") is not None]
    drift = (max(idle_series) - min(idle_series)) if len(idle_series) > 1 else 0
    if idle_series:
        print(f"[idle]    baseline across the run: {idle_series[0]} -> {idle_series[-1]} MiB"
              f"   (spread {drift} MiB)")
        if drift > 5:
            print("          WARNING: the baseline moved while measuring. Every delta inherited")
            print("          that movement, and a monotone drift looks like curvature - it can")
            print("          be absorbed by a fit as a fake step or a fake bend. The constant-")
            print("          baseline refit below is the other end of the honest range.")
    print("=" * 72)

    # A high R^2 does not mean the line is the right model: five monotone points
    # can look linear whatever the truth is. So fit the alternative explicitly.
    st = detect_step(xs, ys)
    if st:
        fcrit = f_critical_95(st["df"])
        print()
        print("-" * 72)
        print(f"[step] testing a one-time level shift, best location num_ctx={st['step_at']}")
        print(f"       y = {st['intercept']:.0f} + {st['slope'] * 1024:.2f} KiB/token x N"
              f" + {st['step']:+.1f} MiB  for N >= {st['step_at']}")
        print(f"       SSE {st['sse_lin']:.1f} -> {st['sse']:.1f}   df {st['df_lin']} -> {st['df']}"
              f"   F(1,{st['df']}) = {st['f']:.1f}")
        if st["slope_hi"] == st["slope_hi"]:
            print(f"       slope from points >= {st['step_at']} only: "
                  f"{st['slope_hi'] * 1024:.2f} KiB/token")
        if fcrit is None:
            print(f"       -> F critical value unknown for df2={st['df']} (table covers 1..20);")
            print("          judge this one by eye, not by the default threshold.")
        elif st["f"] > fcrit:
            spread = (st["slope"] - st["slope_hi"]) * 1024
            print(f"       -> F {st['f']:.1f} > {fcrit:.3f} at 5%: the step wins.")
            print(f"          The straight-line slope is inflated by {spread:+.2f} KiB/token,")
            print("          i.e. the step divided by the largest context. Do not quote the")
            print("          straight-line slope as per-token cost.")
            # The location was chosen by searching, so the nominal 5% is optimistic.
            print("          Caveat: the step location was picked by minimising SSE, so this")
            print("          p-value is not adjusted for that search. With 3 candidate")
            print("          positions the effective false-positive rate is nearer 14%.")
        else:
            print(f"       -> F {st['f']:.1f} <= {fcrit:.3f} at 5%: the step does not earn")
            print("          its parameter. Straight line stands.")
        print("-" * 72)

        # If the baseline drifted, the slope depends on a convention the data does not
        # settle. Print both ends rather than one number.
        if drift > 5 and idle_series:
            const_ys = [r["loaded_mib"] - idle_series[0] for r in good]
            cst = stepfit(xs, const_ys, st["step_at"])
            if cst:
                lo = min(st["slope"], cst["slope"]) * 1024
                hi = max(st["slope"], cst["slope"]) * 1024
                print()
                print(f"[range]   constant baseline {idle_series[0]} MiB gives slope "
                      f"{cst['slope'] * 1024:.2f}, step {cst['step']:+.1f} MiB, SSE {cst['sse']:.1f}")
                print(f"          -> quote slope as {lo:.1f}-{hi:.1f} KiB/token. Two defensible")
                print("             conventions disagree by more than the effect being measured,")
                print("             so neither single value is a result.")
                print("-" * 72)

        # A step and a smooth curve both bend five points. Same parameter count,
        # so SSE compares directly with no penalty term.
        quad = polysse(xs, ys, 2)
        if quad and quad[0]:
            pred = [st["intercept"] + st["slope"] * x
                    + st["step"] * (1 if x >= st["step_at"] else 0) for x in xs]
            sres = [y - p for y, p in zip(ys, pred)]
            print()
            print("[curves] jump or curvature? same parameter count, so compare SSE directly")
            print(f"         {'model':<15}{'params':>7}{'SSE':>11}{'max|resid|':>12}")
            for label, k, sse, res in (("straight line", 2, st["sse_lin"], None),
                                       ("quadratic", 3, quad[0], quad[1]),
                                       ("line + step", 3, st["sse"], sres)):
                mx = f"{max(abs(r) for r in res):.2f}" if res else "-"
                print(f"         {label:<15}{k:>7}{sse:>11.1f}{mx:>12}")
            ratio = quad[0] / st["sse"] if st["sse"] else float("inf")
            if ratio > 5:
                print(f"         -> the step fits {ratio:.0f}x better than the quadratic:")
                print("            the bend is a jump, not a curve.")
            else:
                print(f"         -> only {ratio:.1f}x better than the quadratic. A smooth")
                print("            curve may be enough here; do not claim a step on this data.")
            print("-" * 72)

    # The decisive comparison. Same loads, second baseline convention. If the step
    # survives here it is the model; if it collapses, it was the desktop moving and the
    # per-point fit was reading the room, not the GPU.
    if len(bracketed) >= 3 and delta_key == "delta_mib":
        bx = [r["num_ctx"] for r in bracketed]
        by = [r["delta_bracket_mib"] for r in bracketed]
        ba, _bb, br2 = linfit(bx, by)
        bst = detect_step(bx, by)
        bq = polysse(bx, by, 2)
        print()
        print("-" * 72)
        print("[bracket] the same loads, drift-cancelled baseline -> refit")
        print(f"          slope {ba * 1024:.2f} KiB/token   R^2 {br2:.5f}")
        if bst:
            print(f"          step at {bst['step_at']}: {bst['step']:+.1f} MiB"
                  f"   SSE {bst['sse_lin']:.1f} -> {bst['sse']:.1f}"
                  f"   F(1,{bst['df']}) = {bst['f']:.1f}")
        if bq:
            print(f"          quadratic SSE {bq[0]:.1f}")
        print("          Compare against the per-point numbers above. A step that shrinks")
        print("          here was the baseline moving, not the model. A step that survives")
        print("          is real and worth quoting.")
        print("-" * 72)

    print()
    print("Extrapolate to real models (f16 KV, formula 2*layers*kv_heads*head_dim*2B)")
    print(f"{'model':<26}{'KV/token':>11}{'32K':>10}{'128K':>11}")
    refs = [
        ("Qwen2.5-1.5B (this box)", args.layers, args.kv_heads, args.head_dim),
        ("Llama-3-8B", 32, 8, 128),
        ("Qwen2.5-72B", 80, 8, 128),
    ]
    for name, nl, nkv, hd in refs:
        per_tok = 2 * nl * nkv * hd * 2
        print(f"{name:<26}{per_tok / 1024:>7.0f} KiB"
              f"{per_tok * 32768 / MIB:>8.0f} MiB"
              f"{per_tok * 131072 / MIB / 1024:>8.1f} GiB")

    unload(args.model)
    print("\n[done]    model unloaded")


if __name__ == "__main__":
    main()
