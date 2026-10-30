#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-derive every number quoted in README.md from data/*.csv.

No GPU, no Ollama, no third-party packages. This re-runs only the arithmetic the README
commits to, so a reader can verify the claims without repeating the experiment.

Run from anywhere:

    python tests/test_fit.py

Exit code 0 = every claim holds. Exit code 1 = at least one drifted; the failing line is
printed with its expected value so you can see which one moved.
"""

import csv
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TIGHT = 0.005     # exact arithmetic (formulas)
LOOSE = 0.05      # values rounded to 2 decimals in the README

CHECKS = []


def check(name, actual, expected, tol):
    ok = abs(actual - expected) <= tol
    CHECKS.append((name, actual, expected, ok))
    return ok


def load_probe():
    """Import the probe script without executing main()."""
    path = os.path.join(ROOT, "scripts", "ollama_probe.py")
    spec = importlib.util.spec_from_file_location("ollama_probe_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_csv(tag):
    path = os.path.join(ROOT, "data", "ollama_probe_%s.csv" % tag)
    xs, ys = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["num_ctx"]))
            ys.append(float(row["delta_mib"]))
    return xs, ys


def main():
    probe = load_probe()

    # ---------------------------------------------------------------- raw data
    fx, fy = read_csv("f16")
    qx, qy = read_csv("q8_0")

    check("data rows (f16)", len(fx), 5, 0)
    check("data rows (q8_0)", len(qx), 5, 0)
    check("num_ctx grid == 2048..32768", sum(fx), sum([2048, 4096, 8192, 16384, 32768]), 0)

    # ---------------------------------------------------------------- the fit
    fa, fb, fr2 = probe.linfit(fx, fy)
    qa, qb, qr2 = probe.linfit(qx, qy)

    check("f16 slope = 32.11 KiB/token", fa * 1024, 32.11, LOOSE)
    check("q8_0 slope = 19.08 KiB/token", qa * 1024, 19.08, LOOSE)
    check("f16 intercept = 1119 MiB", fb, 1119.0, 0.5)
    check("q8_0 intercept = 1118 MiB", qb, 1118.0, 0.5)
    check("f16 R^2 = 0.9956", fr2, 0.9956, 0.0001)
    check("q8_0 R^2 = 0.9910", qr2, 0.9910, 0.0001)

    # ---------------------------------------------------------------- uncertainty
    fse, _, fhalf, _, fdf = probe.fit_uncertainty(fx, fy, fa, fb)
    qse, _, qhalf, _, qdf = probe.fit_uncertainty(qx, qy, qa, qb)

    check("f16 1 SE = 1.24", fse * 1024, 1.24, LOOSE)
    check("q8_0 1 SE = 1.05", qse * 1024, 1.05, LOOSE)
    check("f16 CI low = 28.18", (fa - fhalf) * 1024, 28.18, LOOSE)
    check("f16 CI high = 36.05", (fa + fhalf) * 1024, 36.05, LOOSE)
    check("q8_0 CI low = 15.74", (qa - qhalf) * 1024, 15.74, LOOSE)
    check("q8_0 CI high = 22.43", (qa + qhalf) * 1024, 22.43, LOOSE)
    check("degrees of freedom = 3", fdf, 3, 0)

    # ---------------------------------------------------------------- theory
    t_f16 = probe.theory_kv_kib(1, 28, 2, 128, 2.0)
    t_q8 = probe.theory_kv_kib(1, 28, 2, 128, 1.0625)
    t_q4 = probe.theory_kv_kib(1, 28, 2, 128, 0.5625)

    check("theory f16 = 28.00 KiB/token", t_f16, 28.00, TIGHT)
    check("theory q8_0 = 14.88 KiB/token", t_q8, 14.88, 0.006)
    check("theory q4_0 = 7.88 KiB/token", t_q4, 7.88, 0.006)

    # ---------------------------------------------------------------- the toll
    check("f16 toll = +4.11", fa * 1024 - t_f16, 4.11, LOOSE)
    check("q8_0 toll = +4.20", qa * 1024 - t_q8, 4.20, LOOSE)
    check("tolls agree within 0.15", abs((fa * 1024 - t_f16) - (qa * 1024 - t_q8)), 0.0, 0.15)
    check("f16 theory outside 95% CI", 1.0 if (fa - fhalf) * 1024 > t_f16 else 0.0, 1.0, 0)
    check("q8_0 theory outside 95% CI", 1.0 if (qa - qhalf) * 1024 > t_q8 else 0.0, 1.0, 0)

    # ---------------------------------------------------------------- ratios
    check("measured/theory f16 = 1.15x", fa * 1024 / t_f16, 1.15, LOOSE)
    check("measured/theory q8_0 = 1.28x", qa * 1024 / t_q8, 1.28, LOOSE)

    # ---------------------------------------------------------------- 32K / 128K
    # slope is MiB/token, so cost at N tokens is slope * N (N = 32768 or 131072,
    # written as "x 32" / "x 128" in the README because KiB/token x K-tokens = MiB).
    v32_f = fb + fa * 32768
    v32_q = qb + qa * 32768
    v128_f = fb + fa * 131072
    v128_q = qb + qa * 131072

    check("32K f16 from fit = 2146.6 MiB", v32_f, 2146.6, 0.1)
    check("32K q8_0 from fit = 1728.4 MiB", v32_q, 1728.4, 0.1)
    check("32K saving = 418.2 MiB", v32_f - v32_q, 418.2, 0.1)
    check("32K saving = 418 MiB by raw data", fy[-1] - qy[-1], 418.0, 0.1)
    check("KV cache saving = 40.6%", (fa - qa) / fa * 100, 40.6, LOOSE)
    check("total VRAM saving @32K = 19.6%", (v32_f - v32_q) / fy[-1] * 100, 19.6, LOOSE)
    check("total VRAM saving @128K = 31.9%", (v128_f - v128_q) / v128_f * 100, 31.9, LOOSE)

    # ---------------------------------------------------------------- prediction
    q4_slope = t_q4 + (fa * 1024 - t_f16)
    check("predicted q4_0 slope = 12.0", q4_slope, 12.0, 0.1)
    check("predicted q4_0 @32K = 1503 MiB", fb + q4_slope / 1024 * 32768, 1503.0, 1.0)

    # ---------------------------------------------------------------- extrapolation
    def per_token(layers, kv_heads, head_dim):
        return 2 * layers * kv_heads * head_dim * 2 / 1024.0   # KiB

    check("Qwen2.5-1.5B = 28 KiB/token", per_token(28, 2, 128), 28.0, TIGHT)
    check("Llama-3-8B = 128 KiB/token", per_token(32, 8, 128), 128.0, TIGHT)
    check("Qwen2.5-72B = 320 KiB/token", per_token(80, 8, 128), 320.0, TIGHT)
    check("1.5B @128K = 3.5 GiB", per_token(28, 2, 128) * 131072 / 1024 / 1024, 3.5, LOOSE)
    check("Llama-3-8B @128K = 16 GiB", per_token(32, 8, 128) * 131072 / 1024 / 1024, 16.0, LOOSE)
    check("72B @128K = 40 GiB", per_token(80, 8, 128) * 131072 / 1024 / 1024, 40.0, LOOSE)

    # ---------------------------------------------------------------- report
    failed = 0
    width = max(len(c[0]) for c in CHECKS)
    for name, actual, expected, ok in CHECKS:
        mark = "  ok  " if ok else " FAIL "
        if not ok:
            failed += 1
        line = "%s %-*s  actual=%12.4f" % (mark, width, name, actual)
        if not ok:
            line += "  expected=%.4f" % expected
        print(line)

    print()
    if failed:
        print("%d of %d checks FAILED - README and data have drifted apart." % (failed, len(CHECKS)))
        return 1
    print("all %d checks passed - every number in README.md is reproducible from data/." % len(CHECKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
