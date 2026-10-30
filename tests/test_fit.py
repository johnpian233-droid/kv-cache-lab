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
LOOSE = 0.01      # values printed to 2 decimals in the README

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


def read_csv_full(tag):
    """Return (num_ctx, delta_mib, loaded_mib, implied_idle_mib) for one run.

    The fourth element is the point of the exercise: the CSV never stored the idle
    reading, only the difference against it, so the baseline has to be reconstructed
    as loaded - delta. Doing that is what exposed the q8_0 drift in v0.1.3.
    """
    path = os.path.join(ROOT, "data", "ollama_probe_%s.csv" % tag)
    xs, d, l = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["num_ctx"]))
            d.append(float(row["delta_mib"]))
            l.append(float(row["loaded_mib"]))
    return xs, d, l, [a - b for a, b in zip(l, d)]


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

    check("f16 plain-line slope = 32.11", fa * 1024, 32.11, LOOSE)
    check("q8_0 plain-line slope = 19.08", qa * 1024, 19.08, LOOSE)
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
    check("f16 plain-line toll = +4.11", fa * 1024 - t_f16, 4.11, LOOSE)
    check("q8_0 plain-line toll = +4.20", qa * 1024 - t_q8, 4.20, LOOSE)
    check("tolls agree within 0.15", abs((fa * 1024 - t_f16) - (qa * 1024 - t_q8)), 0.0, 0.15)
    check("f16 theory outside 95% CI", 1.0 if (fa - fhalf) * 1024 > t_f16 else 0.0, 1.0, 0)
    check("q8_0 theory outside 95% CI", 1.0 if (qa - qhalf) * 1024 > t_q8 else 0.0, 1.0, 0)

    # ---------------------------------------------------------------- the step
    # A straight line through five monotone points always fits well, so R^2 is not
    # evidence that a line is the right model. Fit the alternative and let F decide.
    fst = probe.detect_step(fx, fy)
    qst = probe.detect_step(qx, qy)

    check("f16 step detected at num_ctx", fst["step_at"], 8192, 0)
    check("q8_0 step detected at num_ctx", qst["step_at"], 8192, 0)
    check("f16 slope with step = 29.99", fst["slope"] * 1024, 29.99, LOOSE)
    check("f16 step size = +67.1 MiB", fst["step"], 67.1, 0.2)
    check("q8_0 slope with step = 17.31", qst["slope"] * 1024, 17.31, LOOSE)
    check("q8_0 step size = +56.2 MiB", qst["step"], 56.2, 0.2)
    check("f16 step beats line at 5%", 1.0 if fst["f"] > probe.f_critical_95(fst["df"]) else 0.0, 1.0, 0)
    check("q8_0 step beats line at 5%", 1.0 if qst["f"] > probe.f_critical_95(qst["df"]) else 0.0, 1.0, 0)
    check("f16 steady-state slope = 30.00", fst["slope_hi"] * 1024, 30.00, 0.01)
    check("q8_0 steady-state slope = 17.30", qst["slope_hi"] * 1024, 17.30, 0.01)

    # the statistics themselves, not just the coefficients derived from them:
    # if df2 were ever computed as n-2 instead of n-3, F would be 1832 and still
    # clear the threshold, so SSE and F each need their own assertion.
    check("f16 step-model SSE = 2.0", fst["sse"], 2.0, 0.1)
    check("q8_0 step-model SSE = 57.8", qst["sse"], 57.8, 0.1)
    check("f16 F = 2748", fst["f"], 2748.0, 1.0)
    check("q8_0 F = 66.2", qst["f"], 66.2, 0.1)
    check("f16 step-model df = 2", fst["df"], 2, 0)
    check("F(1,2) 5% critical value = 18.513", probe.f_critical_95(2), 18.513, 0.001)

    # a step and a smooth curve both bend five points; the quadratic has the same
    # three parameters as the step model, so SSE compares with no penalty term
    fq = probe.polysse(fx, fy, 2)
    qq = probe.polysse(qx, qy, 2)
    check("f16 quadratic SSE = 966.4", fq[0], 966.4, 0.1)
    check("q8_0 quadratic SSE = 380.5", qq[0], 380.5, 0.1)
    check("f16 step beats quadratic by >100x", 1.0 if fq[0] / fst["sse"] > 100 else 0.0, 1.0, 0)
    check("q8_0 step beats quadratic by >5x", 1.0 if qq[0] / qst["sse"] > 5 else 0.0, 1.0, 0)
    check("f16 quadratic still has a 24 MiB bump",
          max(abs(r) for r in fq[1]), 23.80, 0.05)

    # the step, expressed per token of the largest context, is what the naive
    # straight line was reporting as per-token cost
    f_step_per_tok = fst["step"] / 32768 * 1024
    check("f16 step per 32K token = +2.10", f_step_per_tok, 2.10, 0.02)
    check("inflation = step per token", (fa * 1024) - (fst["slope"] * 1024), f_step_per_tok, 0.05)

    # ------------------------------------------- the baseline nobody recorded (v0.1.3)
    ffx, ffd, ffl, ffidle = read_csv_full("f16")
    qqx, qqd, qql, qqidle = read_csv_full("q8_0")
    check("f16 implied idle spread = 1 MiB", max(ffidle) - min(ffidle), 1, 0)
    check("q8_0 implied idle spread = 26 MiB", max(qqidle) - min(qqidle), 26, 0)
    check("q8_0 implied idle is monotone down",
          1.0 if all(a >= b for a, b in zip(qqidle, qqidle[1:])) else 0.0, 1.0, 0)

    # f16 must be immune to the choice of baseline convention...
    f_const = probe.stepfit(ffx, [l - ffidle[0] for l in ffl], 8192)
    q_const = probe.stepfit(qqx, [l - qqidle[0] for l in qql], 8192)
    check("f16 slope unchanged by baseline choice",
          f_const["slope"] * 1024, fst["slope"] * 1024, 0.05)
    check("f16 toll unchanged by baseline choice",
          f_const["slope"] * 1024 - t_f16, fst["slope"] * 1024 - t_f16, 0.05)
    # ...and q8_0 must not be.
    check("q8_0 constant-baseline slope = 16.54", q_const["slope"] * 1024, 16.54, LOOSE)
    check("q8_0 constant-baseline step = 53.2", q_const["step"], 53.2, 0.1)
    check("q8_0 constant-baseline SSE = 0.66", q_const["sse"], 0.66, 0.01)

    # the retraction, as an assertion: f16's toll must fall inside q8_0's range,
    # which is exactly why "the toll differs between precisions" cannot be claimed
    q_lo = min(q_const["slope"], qst["slope"]) * 1024 - t_q8
    q_hi = max(q_const["slope"], qst["slope"]) * 1024 - t_q8
    f_toll_now = fst["slope"] * 1024 - t_f16
    check("f16 toll lies inside q8_0's range",
          1.0 if q_lo <= f_toll_now <= q_hi else 0.0, 1.0, 0)

    # ---------------------------------------------------------------- corrected toll
    f_toll = fst["slope"] * 1024 - t_f16
    q_toll = qst["slope"] * 1024 - t_q8
    check("f16 toll with step model = +1.99", f_toll, 1.99, LOOSE)
    check("q8_0 toll with step model = +2.43", q_toll, 2.43, LOOSE)
    check("toll gap between precisions = 0.44", abs(q_toll - f_toll), 0.44, LOOSE)

    # ---------------------------------------------------------------- ratios
    check("measured/theory f16 = 1.15x", fa * 1024 / t_f16, 1.15, LOOSE)
    check("measured/theory q8_0 = 1.28x", qa * 1024 / t_q8, 1.28, LOOSE)

    # ---------------------------------------------------------------- 32K / 128K
    # slope is MiB/token, so cost at N tokens is slope * N (N = 32768 or 131072,
    # written as "x 32" / "x 128" in the README because KiB/token x K-tokens = MiB).
    v32_f = fb + fa * 32768
    v32_q = qb + qa * 32768

    check("32K f16 plain line = 2146.6 MiB", v32_f, 2146.6, 0.1)
    check("32K q8_0 plain line = 1728.4 MiB", v32_q, 1728.4, 0.1)
    check("32K saving by plain line = 418.2 MiB", v32_f - v32_q, 418.2, 0.1)
    check("32K saving by raw data = 418 MiB", fy[-1] - qy[-1], 418.0, 0.1)

    # the same numbers under the step model; 32K is above the step so it pays it once
    s32_f = fst["intercept"] + fst["slope"] * 32768 + fst["step"]
    s32_q = qst["intercept"] + qst["slope"] * 32768 + qst["step"]
    s128_f = fst["intercept"] + fst["slope"] * 131072 + fst["step"]
    s128_q = qst["intercept"] + qst["slope"] * 131072 + qst["step"]

    check("32K f16 line+step = 2131.9 MiB", s32_f, 2131.9, 0.1)
    check("32K q8_0 line+step = 1716.1 MiB", s32_q, 1716.1, 0.1)
    check("32K saving by step model = 415.8 MiB", s32_f - s32_q, 415.8, 0.1)
    check("step model predicts 32K f16 within 0.2 MiB",
          1.0 if abs(s32_f - fy[-1]) <= 0.2 else 0.0, 1.0, 0)
    check("step model beats plain line at 32K (f16)", 1.0 if abs(s32_f - fy[-1]) < abs(v32_f - fy[-1]) else 0.0, 1.0, 0)

    # the numbers the README quotes: now RANGES, because everything with q8_0 under
    # it moves with the baseline convention
    kv_lo = (fst["slope"] - qst["slope"]) / fst["slope"] * 100
    kv_hi = (fst["slope"] - q_const["slope"]) / fst["slope"] * 100
    check("KV cache saving low end = 42.3%", kv_lo, 42.3, LOOSE)
    check("KV cache saving high end = 44.9%", kv_hi, 44.9, 0.05)
    check("total VRAM saving @32K low = 19.6%", (fy[-1] - qy[-1]) / fy[-1] * 100, 19.6, LOOSE)

    q32c_f = f_const["intercept"] + f_const["slope"] * 32768 + f_const["step"]
    q32c_q = q_const["intercept"] + q_const["slope"] * 32768 + q_const["step"]
    check("total VRAM saving @32K high = 20.8%",
          (1 - q32c_q / q32c_f) * 100, 20.8, LOOSE)
    check("total VRAM saving @128K low = 32.6%", (s128_f - s128_q) / s128_f * 100, 32.6, LOOSE)

    # ---------------------------------------------------------------- prediction
    # q4_0 is deliberately not predicted as a point value. Note the width of this
    # range is now driven by the baseline uncertainty, not by a claimed difference
    # between precisions - that claim was withdrawn in v0.1.3.
    q4_lo = t_q4 + min(f_toll, q_lo)
    q4_hi = t_q4 + max(f_toll, q_hi)
    check("q4_0 slope range low = 9.5", q4_lo, 9.54, 0.05)
    check("q4_0 slope range high = 10.3", q4_hi, 10.31, 0.05)

    # ---------------------------------------------------------------- replication
    # The q8_0 grid was re-measured under a different protocol, on a desktop carrying
    # 1078 MiB more VRAM. These checks exist because a replication described only in
    # prose rots: the claim is that the deltas moved by <= 4 MiB, and that is checkable.
    rep_path = os.path.join(ROOT, "data", "ollama_probe_q8_0_replication.csv")
    rx, rd, rb, rv, ridle, rver = [], [], [], [], [], False
    with open(rep_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["is_probe"] == "True":
                continue
            rx.append(int(row["num_ctx"]))
            rd.append(float(row["delta_mib"]))
            rb.append(float(row["delta_bracket_mib"]))
            rv.append(float(row["size_vram_mib"]))
            ridle.append(int(row["idle_mib"]))
            rver = rver or row["kv_type_verified"] == "True"

    q8_map = dict(zip(qx, qy))
    rep_gap = max(abs(dict(zip(rx, rd))[x] - q8_map[x]) for x in rx)
    check("replication vs original: worst delta gap = 4 MiB", rep_gap, 4, 0)

    rep_a, _, _ = probe.linfit(rx, rd)
    rep_ba, _, _ = probe.linfit(rx, rb)
    check("replication straight slope = 19.17", rep_a * 1024, 19.17, 0.02)
    check("replication bracketed slope = 19.09", rep_ba * 1024, 19.09, 0.02)

    vr_gaps = [a - b for a, b in zip(rd, rv)]
    check("delta - size_vram gap is flat (spread <= 10 MiB)",
          max(vr_gaps) - min(vr_gaps), 10, 0)
    check("replication idle spread = 9 MiB", max(ridle) - min(ridle), 9, 0)
    check("replication KV label was verified against the server",
          1.0 if rver else 0.0, 1.0, 0)

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
