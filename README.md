# kv-cache-lab

> **Scope note (2026.10.30)**: this repository measures **memory only**, on **one GPU**, with
> **one model**, at **one load per point**. It is not a throughput benchmark and it does not
> evaluate output quality. The method generalizes; the numbers do not. Read
> `## What this repo is not` before citing anything here.

**English** | [中文](README_zh.md)

## Introduction

`kv-cache-lab` is a one-script lab that answers a single question with arithmetic instead of
marketing: **how much VRAM does KV cache quantization actually save on a consumer GPU?**

Vendors advertise "50% off". On an 8 GB laptop GPU the measured answer is **19.5–20.8% off the
total VRAM bill at 32K context**. Separately, and on a different denominator, **42–45% comes off
the KV cache itself**. The difference is overhead that quantization cannot discount: about
**2.0 KiB/token** of running cost on f16, plus a **one-time ~67 MiB allocation step** at 8192.

Those two percentages are not two parts of one number. Read as "19.6% of which 42.3%" they
multiply out to roughly 8%, which is not what either figure means.

That step is the actual finding, and it is the part most KV benchmarks get wrong. Fitting one
line across the grid yields 32.11 KiB/token. The data rejects that model — **F = 2748 against
a critical value of 18.513** — in favour of **29.99 KiB/token plus a 67.1 MiB step**, and the
gap between the two is, to within 0.03 KiB/token, the step divided by the largest context.
Half of the "fixed 4.1 KiB/token **toll**" that circulates in KV discussions — *toll* throughout
this file meaning measured slope minus textbook slope — is an artifact of that mistake.

This repo contains the script that produces the numbers, the raw CSVs it produced, and the
derivation of both models, so that every claim can be recomputed rather than trusted.

No frameworks, no dependencies. One script, three CSVs, and the arithmetic.

## News

- **2026.10.30 v0.2.2 — the reporting path had no test, and it was printing `+0.00`.** Three
  defects changed a number on screen: the step analysis reported the straight-line slope as
  inflated by `+0.00 KiB/token` because it subtracted one estimate of the within-region slope
  from another (the answer is **1.89**); `--bracket` silently dropped the largest context from
  its fit whenever `--aba` was absent, which is the point with the longest lever arm; and
  `stepfit` accepted four points with a lone point on one side, where `df=1` makes F diverge
  whatever the data says. Six more could produce a number where there was no measurement: an
  idle sample that returned nothing carried the *previous* row's baseline forward, giving
  `delta = 0 − idle` and fitting it as data; the peak list was a closure cell, so a sampler
  thread left over from one row could write into the next row's peak; `--kv-bytes` matching no
  known type was recorded as `kv_type_verified=True`; a repeated `num_ctx` printed `nan` for
  every coefficient and still exited 0; `--model qwen2.5` never matched the `/api/ps` entry
  `qwen2.5:latest`, so an unload was "confirmed" for a model still holding VRAM; and `--gpu 3`
  on a one-GPU box measured GPU 0 without saying so. `tests/test_replay.py` runs the real
  reporting path against recorded VRAM levels, so a printed sentence can fail a test.
  Self-checks 89 → **93**, plus 23 replay checks. No measurement changed.
- **2026.10.30 v0.2.1 — two transcription errors, one of them reversing a conclusion.** The
  q8_0 step had been quoted as +67.1 MiB, which is the *f16* number; the replication section
  paired a slope from one Δ column with a step from the other, so a reader could not reproduce
  it from the table on the same page. The replication CSV also carried the machine owner's
  absolute local path in every row, in a public repository. Documentation only; self-checks
  89 → 89. *This entry was missing until v0.2.2 — the tag and the release existed, but neither
  README's changelog mentioned them.*
- **2026.10.30 v0.2.0 — the q8_0 grid was re-measured, and it reproduced to 4 MiB.** A second
  pass over the same five contexts, with median-sampled idles, peak-sampled loads, unload
  confirmed through `/api/ps` instead of a VRAM threshold, and the KV type read out of the
  server's own startup environment, lands on 19.08 → **19.17** KiB/token and +56.2 → **+59.8** MiB
  at the same step location — on a desktop carrying **1078 MiB** more VRAM. The runner's own
  `size_vram` is now recorded per row as an independent cross-check (it agrees with the delta to
  a constant 102-112 MiB), and `--kv-bytes` can no longer relabel a run the server disagrees with.
  One honest gap: **f16 was not re-run**, because the server's `OLLAMA_KV_CACHE_TYPE` is still
  q8_0. Self-checks 83 → 89. No number in the existing analysis changed.
- **2026.10.30 v0.1.3 — one claim withdrawn, one column found.** `loaded_mib − delta_mib`
  recovers the idle reading behind every row, and it shows the f16 run held a baseline to **1 MiB**
  while the q8_0 run drifted **26 MiB**. f16 is unaffected (both baseline conventions give 29.99,
  +67 MiB, toll 1.993). q8_0 is not: its slope is 16.5–17.3 and its toll 1.66–2.43, and f16's 1.99
  sits inside that range — so **v0.1.2's claim that the toll differs between precisions is
  withdrawn.** Percentages that rest on q8_0 are now ranges. Also in this release: the F test is
  labelled conditional rather than calibrated, "every KV benchmark makes this mistake" is softened
  to what was actually checked, and the CSV now records `idle_mib` so the next run cannot hide this.
  Self-checks 72 → 83. No measurement changed.
- **2026.10.30 v0.1.2 — the step result survived a second audit.** Added a quadratic
  comparison (same parameter count as the step model; the step wins by 486× on f16 and 6.6× on
  q8_0), downgraded the 32K agreement from "prediction" to "in-sample fit", corrected the
  leave-one-out comparison (it was measured against 1 SE instead of the 2 SE band), defined
  *toll* on first use, replaced "annualised" with "amortised", and added a Related work section
  and a measurement-protocol section. Self-checks 59 → 72. No measurement changed: the CSVs are
  byte-identical to v0.1.0.
- **2026.10.30 v0.1.0 — first public release.** Two configurations (f16, q8_0) across five
  context lengths each. A straight-line fit gives 32.11 / 19.08 KiB/token, then fails a
  residual check: modelling a one-time step at 8192 fits the same data to ±1 MiB and moves the
  slopes to **29.99 / 17.31**. 59 self-checks recompute every number quoted here without a GPU. 15-page
  Chinese write-up with charts and derivations (`docs/kv-cache-report-zh.pdf`).
- **Roadmap** — q4_0 actually measured (currently refused on principle, see
  `### The third rung`), repeated points at every grid position for a real error bar,
  output-quality comparison at q8_0, and the mechanism behind the 8192 step.
  Contributions welcome; see `## Contributing`.

## Performance

Measured on an RTX 4060 Laptop (8188 MiB), Qwen2.5-1.5B (Q4_K_M), Ollama 0.40.6,
`OLLAMA_NUM_PARALLEL=1`. Each point is one model load; the reported value is the VRAM delta
between a fully unloaded GPU and a load at that `num_ctx`.

### Measured slopes

| KV type | slope, plain line | slope, line + step | one-time step | slope from theory |
|---|---|---|---|---|
| **f16** (baseline) | 32.11 ± 1.24 | **29.99** | **+67.1 MiB** | 28.00 |
| **q8_0** | 19.08 ± 1.05 | **17.31** | **+56.2 MiB** | 14.88 |

The first column is what you get by fitting one line across the whole grid. The second is
what you get after fitting the model the data actually wants. **They disagree by 6.6% and 9.3%,
and the next section shows why the second one wins.**

The **intercept is the model's fixed footprint** — weights plus CUDA overhead — around
1110 MiB. It never discounts, no matter how you quantize the cache.

### A straight line hides a step

Five points, five monotone values. Of course a line fits: R² = 0.9956. That number is not
evidence of linearity, it is a formality — a smooth monotone five-point series usually produces it.
R² asks "does the line explain the variance", not "is the line the right model".

So look at the residuals of that fit instead:

```
num_ctx    Δ MiB    residual vs the plain line
   2048     1166        -17.2
   4096     1224        -23.5
   8192     1412        +36.1   <- one clean bump
  16384     1652        +19.2
  32768     2132        -14.6
```

Those residuals do not scatter. They run negative, negative, positive, positive, negative —
a smooth arc with its peak sitting exactly on 8192. If a straight line were the true model,
the signs would be in random order.

Add the obvious missing term, a one-time level shift:

```
ΔVRAM = b + a · N + c · 1[N ≥ 8192]
```

| | plain line | line + step | F(1,2) | verdict |
|---|---|---|---|---|
| f16 | SSE 2731.6 | **SSE 2.0** | **2748** | step wins |
| q8_0 | SSE 1972.9 | **SSE 57.8** | **66.2** | step wins |

The 5% critical value is 18.513. Both statistics are past it by a wide margin. On f16 the step
model leaves residuals of **±1 MiB** — the resolution of `nvidia-smi` itself, which reports
whole MiB. A three-parameter model that lands inside the instrument's own tick, on all five
points, is not an improvement over the line; it replaces it.

One caveat on that p-value, because it is the kind of thing that gets missed: the step
*location* was chosen by trying each interior point and keeping the smallest SSE, so this is
not a single clean comparison. With three candidate positions the effective false-positive
rate is nearer **14%** than 5%. It does not change the verdict here — F = 2748 is not a
14%-or-5% kind of number — but on a marginal result it would decide the answer, and the script
prints the warning alongside its own verdict.

Fit only the points at or above 8192 and the f16 slope is **30.000 KiB/token**, with three
points collinear to within a megabyte. That subset is *inside* the fit above, so it is not
independent evidence. It is still worth printing, because it is the one thing a curve cannot do:
three points on a straight line are what a step leaves behind, and what a parabola cannot.

### The q8_0 run had a drifting baseline, and that changes the q8_0 answer

The CSV stores `loaded_mib` and `delta_mib`. Subtract them and you recover the idle reading that
was used for each point — a column nobody ever printed:

```
num_ctx    f16 implied idle    q8_0 implied idle
   2048                1230                   811
   4096                1230                   805
   8192                1229                   805
  16384                1229                   789
  32768                1229                   785
```

The f16 run sits on a baseline that moves **1 MiB** across the whole grid. The q8_0 run moves
**26 MiB**, monotonically. Whether the desktop was genuinely releasing VRAM or the reading was
drifting does not matter for what follows: the subtraction puts all of it into `delta_mib`, and
26 MiB is about 4.5x the largest q8_0 residual (5.8 MiB) it was supposed to be testing against.

So fit q8_0 both ways — trust each idle reading, or assume one baseline for the run:

| q8_0, baseline convention | slope | step | SSE | toll over theory |
|---|---|---|---|---|
| per-point idle, as recorded | 17.31 | +56.2 MiB | 57.8 | +2.43 |
| constant idle = 811 | **16.54** | **+53.2 MiB** | **0.66** | **+1.66** |

Both are defensible, and nothing in the data picks between them. Three consequences, in order of
how much they hurt:

**The f16 result is immune.** Its baseline moves 1 MiB, and both conventions give a slope of
29.99, a step of 66–67 MiB, an SSE of 2.0 and a toll of 1.993. Everything this README leans on is
the f16 column, and the f16 column does not care.

**The q8_0 numbers have to be quoted as ranges.** Slope **16.5–17.3**, step **53–56 MiB**, 32K
figure **1688–1716 MiB**, toll **1.66–2.43**.

**A claim in v0.1.2 is withdrawn.** That version said the toll was 1.99 on f16 and 2.43 on q8_0,
called it "a 22% spread", and treated the difference as a finding. f16's 1.99 sits *inside*
q8_0's range. The gap is smaller than the uncertainty from one unrecorded column, so **this
repository has no evidence that the toll differs between precisions.**

Note also which way the error ran. With a constant baseline, q8_0's step model fits to an SSE of
**0.66** — better than f16's 2.0 — and beats a quadratic by 923×. The earlier reading, that q8_0
"does not fit tightly", was the baseline moving, not the model failing.

### Could it just be a smooth curve?

A reader is entitled to be suspicious here. A step term is a very flexible thing to bolt onto a
line, and with five points and three parameters the model has enough rope to hang itself. Maybe
the data is simply curved and the "step" is a parabola in disguise.

So fit the alternative rather than argue about it. A quadratic carries the same three parameters
as the step model, and the two are non-nested, so the SSEs compare directly with no penalty:

| model | params | f16 SSE | q8_0 SSE |
|---|---|---|---|
| straight line | 2 | 2731.6 | 1972.9 |
| quadratic | 3 | 966.4 | 380.5 |
| line + step | 3 | **2.0** | **57.8** |

The quadratic cuts f16's SSE by 2.8×. The step cuts it by 1375×. Put the two three-parameter
models against each other and the step wins by a factor of **486**.

The residuals explain why. The quadratic's are **+2.2, −16.3, +23.8, −11.2, +1.6** — still the
same negative-negative-positive-positive-negative wobble, only smaller. A parabola cannot put a
spike on one interior point without paying for it at both ends, and 8192 is where the spike is.
The data contains a jump, not a bend.

Two notes on the q8_0 column. First, its margin is **6.6×** rather than 486×, and if you use the
constant-baseline fitting from the previous section the same margin becomes **923×** — the number
moves by two orders of magnitude depending on a choice nothing in the data settles. Second, that
is exactly why the q8_0 comparison is worth less than the f16 one: the f16 column is stable under
that choice and the q8_0 column is not.

### Why the step matters: it was being charged as per-token cost

This is the mistake worth knowing about, because any KV benchmark that fits a single line across
a context grid will make it — including, in an earlier draft, this one. Whether specific published
numbers have made it is not something this repository checked, and it does not claim so.

A 67 MiB step at 8192, if you refuse to model it, gets spread across the whole range by the
regression. Divided by the largest context it becomes

```
67.1 MiB / 32768 token × 1024 = 2.10 KiB/token
```

and that is, to within 0.02, how much the plain slope is inflated: 32.11 − 29.99 = 2.12. The
arithmetic closes.

Now put that next to the gap between measurement and theory. Throughout this file **toll** means
*measured slope minus textbook slope* — the per-token cost the formula does not predict:

| | plain line | line + step |
|---|---|---|
| f16 toll over theory | +4.11 | **+1.99** (identical under both baseline conventions) |
| q8_0 toll over theory | +4.20 | **+1.66 … +2.43** |
| gap between the two precisions | 0.09 | **not established** |

Half of the "fixed 4.1 KiB/token toll" was never a per-token cost. It was a one-time allocation
step, amortised over 32K tokens by the fit. The real per-token overhead on f16 is **1.99
KiB/token**.

The obvious next sentence — "and it is not the same in both precisions" — is the one this
repository cannot support. q8_0's range brackets f16's value, so the data is equally consistent
with one shared toll and with two different ones. v0.1.2 of this file claimed a difference of
0.44 and called it "nearly five times" the plain line's 0.09. Both numbers are smaller than the
uncertainty from a single unrecorded column. The claim is gone.

What survives is narrower and still worth having: the toll is one number around 2 KiB/token, not
a 4.1 KiB/token constant, and the difference between those two answers is a step that was being
charged per token.

That does invert part of the original argument. The plain line made the toll look constant, and
treated that consistency as proof of a mechanism. Once the step is modelled the consistency is
gone — but so is the evidence for the mechanism that was supposed to replace it. This file
therefore reports the decomposition and not the mechanism.

### What it means for the VRAM bill

| Question | Answer |
|---|---|
| "How much does q8_0 save on the KV cache itself?" | **42–45%** (the plain line said 40.6%) |
| "How much does it save on total VRAM at 32K context?" | **19.5–20.8%** |
| "How much at 128K context?" | **32.6–34.6%** (extrapolated, no interval) |

Both "about 40%" and "about a fifth" are true. They answer different questions: the first
compares KV cache to KV cache, the second compares full VRAM bills. If you are quoting a
customer, you want the second number.

These are ranges rather than points for one reason, and it is not statistical: q8_0's baseline
wanders (see the section above), so every ratio that has q8_0 underneath it inherits that
wander. The low end of each range is the as-recorded fitting; the high end is the
constant-baseline fitting. **Quoting one decimal of any of them is false precision.**

Three footnotes so the ratios can be reproduced. The percentages are ratios of the two *measured*
slopes, and each slope already carries its own ~2 KiB/token overhead, so this is "KV cache vs KV
cache as billed", not "pure KV bytes" — against pure theory the ratio is 46.9%. The percentages do
not share a denominator: the 32K figure divides by the fitted value at 32K, while the 128K figure
divides by an extrapolation that has no interval. And the plain line's 40.6% is computed the same
way, so the comparison in the first row is at least like-for-like.

One thing the ranges do **not** cover: the step height. q8_0's is 53–56 MiB and f16's is 66–67,
and nothing here explains why they differ.

Model check at 32K — each model against the raw measurement at `num_ctx=32768`:

| | plain line | Δ | line + step | Δ |
|---|---|---|---|---|
| f16 | 1119 + 32.11 × 32 = 2146.6 MiB | 14.6 | 1105.02 + 29.993 × 32 + 67.10 = **2131.9 MiB** | **0.1** |
| q8_0 | 1118 + 19.08 × 32 = 1728.4 MiB | 14.4 | 1106.08 + 17.306 × 32 + 56.21 = **1716.1 MiB** | 2.1 |
| saved | 418.2 MiB | — | 415.8 MiB | — |

Note what this is: **in-sample**. 32768 is one of the five points being fitted, so a small
residual here is fit quality, not prediction, and no out-of-sample test has been run. What it
does establish is narrower and still useful — a three-parameter model reproduces a five-point
dataset to the resolution of the instrument, while the two-parameter line misses the same point
by 14.6 MiB. That 14.6 MiB is precisely the step the line refused to model.

### How sure is that number

The intervals below belong to the **plain line** — five points, three degrees of freedom,
Student-t at 95%:

| KV type | slope ± 1 SE | 95% CI | theory | theory vs CI |
|---|---|---|---|---|
| f16 | 32.11 ± 1.24 | [28.18, 36.05] | 28.00 | outside, by 0.18 |
| q8_0 | 19.08 ± 1.05 | [15.74, 22.43] | 14.88 | outside, by 0.86 |

On f16 the margin is **0.18 KiB/token — 0.6%**. One extra point could pull the theory inside
the interval and turn "there is a toll" into "no detectable difference". Nobody should build
an argument on a margin that thin, and this repository does not.

Note also what the CI does **not** say. It answers "if the line were true, how much would the
slope wobble?" It does not answer "is the line true?" — that is the question the F test
answered, and the answer was no. Leave-one-out confirms the interval is honest about *sampling*
noise: dropping any single point moves the plain slope between **31.6 and 35.2**, a 3.6
KiB/token swing, about 1.5× the ±2 SE band and comfortably inside the reported 95% interval.
The interval was never the problem. Sampling noise simply was not the dominant error here —
model choice was, and that error is 32.11 against 29.99, which is larger than any of these
wobbles.

The step model is stronger on the main claim and **weaker on precision**:

- **Stronger**: F = 2748 against a critical value of 18.513, and an in-sample residual at 32K of
  0.1 MiB.
- **Weaker**: the steady-state slope rests on three points that are collinear to within about a
  megabyte, so its standard error computes to **zero**. That is not a precise number, it is an
  unmeasurable one — three points cannot estimate their own scatter, and "collinear to 1 MiB" and
  "exactly collinear" are the same fact reported at two levels of charity.

A third asymmetry, which is the one a referee would press hardest: the F test's denominator is
the step model's own residual, and on f16 that residual (2.0, i.e. about 1 MiB per point) is the
`nvidia-smi` tick. So F = 2748 is **conditional** on the step model's form being right and on the
measurement being essentially noiseless. With this dataset there is no pure replicate, so the
denominator cannot be separated from lack of fit. Treat F as an upper bound on how strongly the
data prefer a step, not as a calibrated p-value.

So the honest statement is: **the slope is 30, not 32; the step is real; and neither has a
trustworthy error bar yet.** Repeating each grid point is the experiment that would fix it. The
q8_0 grid has since been repeated once, under a different protocol, and reproduced to 4 MiB (see
*Independent replication*) — that is one replicate, not a variance estimate, and the f16 grid has
not been repeated at all. It remains the first item in `## What this repo is not`.

### Independent replication

The q8_0 grid was re-measured after the fact with a different protocol: idle sampled five times
and taken as a median instead of once, loaded read nine times and taken as a peak, unload
confirmed through `/api/ps` instead of a VRAM threshold, and the KV cache type read out of the
server's own startup environment instead of assumed. The desktop was also busier — **1078 MiB**
more VRAM in use at the idle floor.

| num_ctx | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|
| first run (Δ MiB) | 1139 | 1177 | 1297 | 1445 | 1714 |
| replication (Δ MiB) | 1138 | 1173 | 1301 | 1441 | 1716 |
| difference | −1 | −4 | +4 | −4 | +2 |

Every point lands within 4 MiB of the original, on a desktop that had moved by more than a
gigabyte in absolute VRAM. Scored on the per-point baseline in the table above, the straight-line
slope is 19.08 before and **19.17** after, and the step is +56.2 before and **+59.8** after, at the
same location. Scored on the bracketed baseline (`delta_bracket_mib` — the mean of the idle
readings either side of each load) the same run gives **19.09** and **+65.0**. The two columns
differ by 0.08 KiB/token and 5 MiB, and neither is "the" answer: quote the column you actually
fitted, and do not mix a slope from one with a step from the other.

That is the strongest evidence in this repository, and it is evidence for the *method* rather
than for the model: a delta absorbs a 1078 MiB change in the desktop and a change of measurement
protocol, and comes back four MiB wide. It is also what lets the q8_0 baseline drift above be
called a limitation rather than a defect — the number it moved did not move.

The f16 leg was **not** re-run: the server's `OLLAMA_KV_CACHE_TYPE` had been changed to q8_0
between the two original runs and is still q8_0, so an f16 measurement now requires restarting
`ollama serve`. Its internal cross-check is that the f16 and q8_0 runs differ by 418 MiB of model
footprint at 32768, against a predicted `(28 − 14.875) × 32768 / 1024 = 420 MiB`. That is
consistent with the two runs having really been f16 and q8_0, but it is inference, not a
replicate.

Two defects the replication found, both in the tooling rather than the data:

- The unload check used to poll `nvidia-smi` against an absolute threshold. When the desktop grew
  150 MiB mid-session the threshold became unsatisfiable, and the script reported *"model may not
  be fully unloaded"* three times while `ollama ps` was empty throughout. It now asks Ollama,
  which is the only party that knows.
- `--kv-bytes` does not configure anything; it only **labels**. A run executed while the server
  was on q8_0, but passed `--kv-bytes 2`, was recorded as f16 with a slope of 19.0 KiB/token —
  1.28x the q8_0 theory and 0.68x the f16 theory. It was caught only because it matched the older
  q8_0 run to within 21 MiB at all five contexts. The script now reads `OLLAMA_KV_CACHE_TYPE` out of the server's
  startup log and refuses to write a CSV when the label disagrees.

### Raw measurements

```
num_ctx   f16 (Δ MiB)   q8_0 (Δ MiB)   f16 idle   q8_0 idle
  2048        1166          1139         1230         811
  4096        1224          1177         1230         805
  8192        1412          1297         1229         805
 16384        1652          1445         1229         789
 32768        2132          1714         1229         785
```

The last two columns are not in the CSV — they are `loaded_mib − delta_mib`, and they were the
most informative thing in this dataset once someone bothered to compute them.

### What the step might be

Three candidates. **None of them is confirmed, and this repository does not claim to know
which one it is.**

- **Attention kernel switch.** Ollama picks a different attention path above some `n_kv`
  threshold, and the new path allocates its own workspace. Qwen2.5-1.5B has a GQA ratio of 6,
  which is where such thresholds tend to sit. Testable in one afternoon: rerun the grid with
  `OLLAMA_FLASH_ATTENTION=0` and see whether the step moves or vanishes.
- **Allocator arena growth.** The GGML allocator crosses a size class at 8192 and rounds the
  KV buffer up to the next arena boundary.
- **A buffer outside the model.** Ollama's VRAM-based default context on this box is **4096**
  (`default_num_ctx=4096` in the server log). Something that only materialises once the
  request passes that default would land in exactly this place.

One observation constrains the search: the step appears at the **same position** in both
precisions but with **different heights** (+67.1 vs +56.2 MiB). A mechanism keyed to
`num_ctx` would do that. A mechanism keyed to the KV byte count would not.

### The third rung — deliberately not predicted

| KV type | slope | 32K VRAM on Qwen2.5-1.5B | status |
|---|---|---|---|
| f16 | 29.99 KiB/token + 67.1 MiB step | 2132.0 MiB | measured |
| q8_0 | 16.5-17.3 + 53-56 MiB step | 1688-1716 MiB | measured (baseline-sensitive) |
| q4_0 | 7.88 theory, toll unknown | — | **not predicted** |

An earlier draft of this file did predict q4_0: theory 7.88 plus the "same 4.1 toll" gives
12.0. That arithmetic is no longer available, and it is worth being explicit about why.

Two precisions, on the same machine, on the same day, with the same script, and the toll is
**1.99** on one and **1.66–2.43** on the other. That is not a constant, it is a bracket. Adding
it to a theory value for a third precision would produce a number with two significant figures
and no content — and the step height is worse still: 66–67 MiB on f16, 53–56 on q8_0, and nothing
says where q4_0 lands.

So the honest range is a slope somewhere near **9.5–10.3 KiB/token** and a 32K figure with an
uncertainty of tens of MiB that this dataset cannot narrow. **If you need the q4_0 number,
measure it**: `py scripts/ollama_probe.py --kv-bytes 0.5625 --tag q4_0`. That is one command
and about fifty seconds.

### Extrapolation to other models

KV cost is `2 × layers × kv_heads × head_dim × bytes`, so it scales predictably:

| model | KV/token (f16) | 32K | 128K |
|---|---|---|---|
| Qwen2.5-1.5B | 28 KiB | 896 MiB | 3.5 GiB |
| Llama-3-8B | 128 KiB | 4 GiB | **16 GiB** |
| Qwen2.5-72B | 320 KiB | 10 GiB | **40 GiB** |

A 72B model needs 40 GB just to remember the conversation — comparable to its own 4-bit weights,
and larger than them at 8-bit. That
single line explains why every model released in late 2026 is a KV cache optimization paper in
disguise.

Note what this table is: **pure theory, no overhead applied**. On this machine the measured
overhead at f16 was about 7% on top of theory, but that 7% is a machine fact, not a law, and
the step height is anybody's guess until measured. Treat these as floors.

Run the script on your own card and you get your own slope, from which you can compute the rest.

## Related work

This is a small measurement note, not a method paper. The things it leans on, and the things it
deliberately does not attempt:

- **llama.cpp / GGML** is the implementation under test here, via Ollama. Its
  `--cache-type-k` / `--cache-type-v` flags are what `OLLAMA_KV_CACHE_TYPE` sets, and its
  allocator is the most likely home of the 8192 step. The block layout that makes `q8_0` cost
  1.0625 B/element (32 values plus an fp16 scale) is documented in that codebase, not here.
- **PagedAttention / vLLM** (Kwon et al., SOSP 2023) is the reason KV cache is normally
  discussed in *blocks* rather than bytes per token. This note deliberately uses the cruder
  bytes-per-token accounting, because that is the number a practitioner multiplies by context
  length on a single GPU.
- **KIVI** (Liu et al., ICML 2024, arXiv:2402.02750) quantizes K per-channel and V per-token,
  and reports peak-memory reductions including weights. Its accounting is end-to-end; this note
  is the complement — a decomposition of the *overhead* that survives quantization.
- **QServe / QoQ** (Lin et al., MLSys 2025, arXiv:2405.04532) pushes to W4A8KV4 on datacenter
  GPUs and reports throughput rather than resident memory.

What is missing, and honestly so: no comparison against any of these on the same hardware. The
claim here is narrow — that a single straight line through a five-point context grid misreports
per-token cost by about 7% on this configuration, and that the error has a shape. Whether that
shape appears in vLLM, TensorRT-LLM or a datacenter GPU is untested and would be the obvious
follow-up.

## Requirements

| | |
|---|---|
| Python | 3.8+ (standard library only — no pip installs required) |
| GPU | any GPU visible to `nvidia-smi` |
| Inference server | [Ollama](https://ollama.com) reachable at `http://127.0.0.1:11434` |
| Model | any model already pulled locally (this repo used `qwen2.5:1.5b`) |
| OS | Windows, Linux, macOS (VRAM read via `nvidia-smi`) |

There is no CUDA toolchain requirement, no PyTorch, and no build step.

## Installation

The script is self-contained; "installation" is a clone and a model pull.

```bash
git clone https://github.com/johnpian233-droid/kv-cache-lab.git
cd kv-cache-lab

# optional: verify the arithmetic against the committed CSVs (no GPU needed)
python tests/test_fit.py
python tests/test_replay.py

# pull a model if you do not have one
ollama pull qwen2.5:1.5b
```

## Usage

### Measuring a configuration

```bash
# q8_0 KV cache
python scripts/ollama_probe.py --tag q8_0 --kv-bytes 1.0625

# f16 baseline
python scripts/ollama_probe.py --tag f16  --kv-bytes 2
```

Output: a per-`num_ctx` table, a least-squares fit with 1-SE and 95% CI, an extrapolation
table, and `ollama_probe_<tag>.csv`.

Key flags:

| flag | meaning |
|---|---|
| `--kv-bytes` | bytes per KV element: f16=2, q8_0=1.0625, q4_0=0.5625. **Labels the run; it does not configure Ollama** — the script reads the server's `OLLAMA_KV_CACHE_TYPE` and refuses to run if the two disagree |
| `--seqs` | context lengths to test (default `2048 4096 8192 16384 32768`) |
| `--layers --kv-heads --head-dim` | model geometry, used for the theory column |
| `--samples` | `nvidia-smi` readings per point, peak reported (default 7; the measurement used 9) |
| `--aba` | re-measure the first context at the end, as a baseline drift probe |
| `--bracket` | use the mean of the idle readings either side of each load, cancelling monotone drift |
| `--settle` | MiB tolerance for the fallback unload check, used only if `/api/ps` is unreachable |
| `--allow-kv-mismatch` | proceed despite a label/environment mismatch, and record that it was unverified |

### Switching the actual KV cache type

Set the environment variable **and restart the server** — the setting is read at load time.

```bash
export OLLAMA_FLASH_ATTENTION=1          # required for V-cache quantization
export OLLAMA_KV_CACHE_TYPE=q8_0         # or f16 / q4_0
ollama serve
```

> Windows: use `set` (applies to the current window immediately), not `setx` (applies only to
> windows opened later). `ollama serve` must be restarted for any of it to take effect.

### Measurement protocol

For each `num_ctx`, in order:

1. `ollama stop <model>`, then poll `/api/ps` until the model is gone. **Do not use a VRAM
   threshold for this.** An absolute threshold cannot tell a resident model from a desktop that
   grew, and on this machine an idle floor that rose 150 MiB over ten minutes produced three
   "model may not be fully unloaded" warnings while `ollama ps` was empty the whole time.
2. Sample `nvidia-smi` five times and take the **median**. That is **idle**. One reading is a
   coin flip — peak-to-peak swing was 32 MiB over 8 s on this desktop.
3. Load the model at that `num_ctx` with a single request, then read `nvidia-smi` nine times and
   take the **peak**. Loaded is a peak, not a median: the loaded state *is* the allocation, and
   taking a median here shaves 300–600 MiB off the high contexts while looking exactly like a
   smaller KV cost. That mistake was made and caught during v0.2 development.
4. **loaded − idle is the measurement.** Cross-check it against the runner's own `size_vram`
   from `/api/ps`. The two differ by 102-112 MiB of CUDA context (spread 10 MiB across five
   points); if the gap is *not*
   constant, the baseline moved during the load and the row is suspect.

Always compare **deltas**, never absolute VRAM. Desktop usage drifted by 1078 MiB between the
original run and its replication; deltas absorbed it completely and the two agreed to 4 MiB.

What is *not* controlled, and should be read as part of the error budget: GPU clock and power
state are left to the driver, driver and CUDA versions are not recorded in the CSV, ambient load
from the desktop is only reflected in the idle reading, and each configuration was loaded
**once** per run. `load_sec` is written to the CSV for provenance but is not analysed.

### CSV format

| column | meaning |
|---|---|
| `num_ctx` | requested context length |
| `loaded_mib` | absolute VRAM after load (for debugging only) |
| `delta_mib` | **the measurement** — loaded minus idle |
| `idle_mib` | the baseline that `delta_mib` was differenced against. Added in v0.1.3; the two historical CSVs predate it, so recover it as `loaded_mib - delta_mib` |
| `theory_kv_mib` | pure KV per textbook formula, same row |
| `load_sec` | load wall time |
| `model`, `ollama_version`, `kv_cache_type`, `kv_bytes_per_elem` | provenance |

`delta_mib` includes the weights. That is intentional: the fit's intercept absorbs them, which
is exactly how the weights get separated from the per-token cost.

## Repository layout

```
scripts/ollama_probe.py        the only script — standard library only
tests/test_fit.py              re-derives every number in this README from data/, no GPU needed
tests/test_replay.py           runs the reporting path on recorded levels, so printed claims are tested too
data/*.csv                     raw measurements, with provenance columns
docs/kv-cache-report-zh.pdf    15-page Chinese write-up (v0.1.1 snapshot)
docs/warehouse-story-zh.pdf    10-page plain-language version, warehouse-inventory analogy
docs/PUBLISHING-zh.md          how this repo was published, step by step (Chinese)
.github/ISSUE_TEMPLATE/        "report your own number" issue form
CITATION.cff                   GitHub's "Cite this repository" metadata
README_zh.md                   Chinese translation of this file
```

## Data availability

Everything needed to reproduce the numbers is in this repository, and none of it needs a GPU:

| | |
|---|---|
| Raw measurements | `data/ollama_probe_f16.csv`, `data/ollama_probe_q8_0.csv` — one row per load, with provenance columns |
| Replication | `data/ollama_probe_q8_0_replication.csv` — the q8_0 grid re-measured under the v0.2 protocol, with `idle_mib`, `size_vram_mib` and a verified `kv_type_verified` flag |
| Re-derivation | `python tests/test_fit.py` — 93 checks, standard library only |
| Reporting path | `python tests/test_replay.py` — 23 checks; runs `main()` against recorded VRAM levels, no GPU |
| Measurement | `python scripts/ollama_probe.py --kv-bytes <b> --tag <name> --samples 9 --aba` — needs Ollama and `nvidia-smi` |
| Plain language | `docs/warehouse-story-zh.pdf` — 10 pages, Chinese. Explains the three ledgers through a warehouse inventory analogy, starting from a stock row whose three fields imply negative stock. No statistics required. |
| Write-up | `docs/kv-cache-report-zh.pdf` — Chinese, 15 pages. **Snapshot of v0.1.1**: it predates the quadratic comparison and the audit corrections in v0.1.2. Where it disagrees with this README, the README and the CSVs are current. |

The CSVs are the primary record. If a number in this README disagrees with a CSV, the CSV is
right and the README is a bug.

The two original CSVs are deliberately left as they were, including the columns they do **not**
have. They carry no `idle_mib`, which is why the q8_0 baseline drift went unnoticed for two
releases; it had to be recovered as `loaded − delta`. Adding the column retroactively would have
erased the evidence that it was missing. The replication CSV has it.

## What this repo is not

- **Not a benchmark.** No tokens/s, no latency, no quality evaluation. Memory only.
- **No repeated measurements.** Every `num_ctx` was loaded **once**. All three degrees of
  freedom in the confidence intervals come from model residuals, not from pure error, so a
  lack-of-fit test is impossible and the error bars are optimistic. This is the single biggest
  weakness in the dataset and it is the first item on the roadmap.
- **One machine, one model.** Five points per configuration. The method generalizes; the
  numbers do not. The two runs here differ in idle baseline by **419 MiB** (1230 vs 811) because
  the desktop was busier for one of them — on a 32K span that alone moves a two-point slope by
  14 KiB/token.
- **Not proof that q8_0 is lossless.** Practitioner consensus says it is near-lossless; this
  repo did not test it. Do not cite it as evidence that it is.
- **The step mechanism is unknown.** Three candidates are listed above and none is confirmed.
  Do not cite this repo for *why* the step exists, only for *that* it exists on this
  configuration.
- **The confidence intervals are wide.** df=3. If you need tighter numbers, measure more
  points — the script will tell you the interval honestly.
- **The idle baseline was not recorded.** Until v0.1.3 the CSV stored `delta_mib` and
  `loaded_mib` but not the idle reading they were differenced against, which is how a 26 MiB
  drift in the q8_0 run went unnoticed for two releases. It is recorded now. The historical
  CSVs are left byte-identical on purpose: rewriting a primary record to add a column you
  reconstructed afterwards is worse than admitting the column is missing.

## Contributing

Different GPU, different model, different answer? **Open an issue with your number** — use the
"Report a measurement" template. Include:

- GPU and VRAM: `nvidia-smi --query-gpu=name,memory.total --format=csv`
- model tag + quant, Ollama version, `OLLAMA_KV_CACHE_TYPE`
- the CSV the script produced

Community measurements get added to a table with the card and the slope. The method is the
point — the numbers are supposed to disagree.

## Acknowledgement

The measurement discipline here — deltas over absolutes, intercept as a sanity check, error
bars printed next to every estimate — is borrowed from the systems-measurement tradition that
hardware kernel libraries like [FlashMLA](https://github.com/deepseek-ai/FlashMLA) practice:
publish the number, publish the command that reproduces it, and state the regime it holds in.

## Citation

```bibtex
@misc{kvcachelab2026,
      title={kv-cache-lab: VRAM accounting for KV cache quantization on consumer GPUs},
      author={Kong Xia},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/johnpian233-droid/kv-cache-lab}},
}
```

## License

MIT — see [LICENSE](LICENSE).
