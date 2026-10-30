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
  2. wait for VRAM to fall   -> idle baseline (desktop / other apps)
  3. load at that num_ctx    -> loaded VRAM
  4. delta = loaded - idle

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

    # f16 KV cache
    python ollama_probe.py --tag f16 --kv-bytes 2

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


def wait_idle(threshold, tries=90, gpu=0):
    """Wait for VRAM to drop to `threshold` MiB. Returns (value, ok)."""
    for _ in range(tries):
        v = _gpu_used(gpu)
        if v is not None and v <= threshold:
            return v, True
        time.sleep(0.5)
    return _gpu_used(gpu), False


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
    ap.add_argument("--settle", type=int, default=32,
                    help="MiB tolerance when detecting a fully unloaded model")
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

    print(f"[ollama]   {ver}    model: {args.model}")
    print(f"[kv type]  {kv_label}" + (f"  (from OLLAMA_KV_CACHE_TYPE={kv_type_env})"
                                      if kv_type_env else "  (explicit)"))
    print(f"[geometry] layers={args.layers} kv_heads={args.kv_heads} head_dim={args.head_dim}")

    # ---- idle baseline
    print("[setup]   unloading model, measuring idle baseline ...")
    if not unload(args.model):
        print("[warning] `ollama stop` failed — VRAM may not be released. "
              "Is the ollama CLI on PATH?")
    time.sleep(1.0)
    probe = used()
    if probe is None:
        sys.exit("[error] nvidia-smi returned nothing — check driver / PATH")
    idle, ok = wait_idle(probe, tries=40, gpu=gpu_idx)
    if idle is None:
        sys.exit("[error] cannot read VRAM")
    if not ok:
        print(f"[warning] VRAM did not settle, using {idle} MiB as baseline")
    print(f"[idle]    {idle} MiB (desktop + other apps)\n")

    rows = []
    hdr = (f"{'num_ctx':>8} | {'loaded MiB':>10} | {'delta MiB':>9} | "
           f"{'theory MiB':>10} | {'load s':>6}")
    print(hdr)
    print("-" * len(hdr))

    for L in sorted(args.seqs):
        unload(args.model)
        time.sleep(1.2)
        cur_idle, ok = wait_idle(idle + args.settle, tries=40, gpu=gpu_idx)
        if not ok:
            print(f"[warning] num_ctx={L}: model may not be fully unloaded "
                  f"(VRAM at {cur_idle} MiB, expected <= {idle + args.settle})")
        if cur_idle is None:
            cur_idle = idle

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
        after = used() or 0
        loaded = max(peak[0], after)
        delta = loaded - cur_idle
        theory = theory_kv_kib(L, args.layers, args.kv_heads, args.head_dim, kv_bytes) / 1024

        print(f"{L:>8} | {loaded:>10} | {delta:>9} | {theory:>10.1f} | {dt:>6.1f}")
        rows.append(dict(num_ctx=L, loaded_mib=loaded, delta_mib=delta,
                         theory_kv_mib=round(theory, 1), load_sec=round(dt, 1)))

    # ---- fit BEFORE writing, so the CSV always lands
    good = [r for r in rows if r["delta_mib"] is not None]
    xs = [r["num_ctx"] for r in good]
    ys = [r["delta_mib"] for r in good]
    a = b = r2 = float("nan")
    if len(good) >= 2:
        a, b, r2 = linfit(xs, ys)

    csv_path = f"ollama_probe_{args.tag}.csv"
    meta = {
        "model": args.model,
        "ollama_version": ver,
        "kv_cache_type": (kv_type_env or kv_label),
        "kv_bytes_per_elem": kv_bytes,
    }
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["num_ctx", "loaded_mib", "delta_mib", "theory_kv_mib", "load_sec",
                  "model", "ollama_version", "kv_cache_type", "kv_bytes_per_elem"]
        w = csv.DictWriter(f, fieldnames=fields)
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
    if r2 > 0.999:
        print("       strictly linear — slope is the real per-token cost on this box.")
    elif r2 > 0.99:
        print("       near-linear (allocator rounding causes small deviation).")
    else:
        print("       WARNING: poor linearity. Check for leftover VRAM from other apps.")
    print("=" * 72)

    # A high R^2 does not mean the line is the right model: five monotone points
    # will look linear whatever the truth is. So fit the alternative explicitly.
    st = detect_step(xs, ys)
    if st:
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
        if st["f"] > 18.5:
            spread = (st["slope"] - st["slope_hi"]) * 1024
            print("       -> the step wins. The straight-line slope is inflated by")
            print(f"          {spread:+.2f} KiB/token, i.e. the step divided by the largest")
            print("          context. Do not quote the straight-line slope as per-token cost.")
        else:
            print("       -> the step does not earn its parameter. Straight line stands.")
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
