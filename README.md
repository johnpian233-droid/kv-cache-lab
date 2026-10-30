# kv-cache-lab

> **Scope note (2026.10.30)**: this repository measures **memory only**, on **one GPU**, with
> **one model**, at **one load per point**. It is not a throughput benchmark and it does not
> evaluate output quality. The method generalizes; the numbers do not. Read
> `## What this repo is not` before citing anything here.

**English** | [中文](README_zh.md)

## Introduction

`kv-cache-lab` is a one-script lab that answers a single question with arithmetic instead of
marketing: **how much VRAM does KV cache quantization actually save on a consumer GPU?**

Vendors advertise "50% off". On an 8 GB laptop GPU the measured answer is **19.6% off the
total VRAM bill at 32K context**, of which **42.3% comes off the KV cache itself**. The
difference is overhead that quantization cannot discount: about **2.0 KiB/token** of running
cost, plus a **one-time ~67 MiB allocation step** at 8192.

That step is the actual finding, and it is the part most KV benchmarks get wrong. Fitting one
line across the grid yields 32.11 KiB/token. The data rejects that model — **F = 2748 against
a critical value of 18.513** — in favour of **29.99 KiB/token plus a 67.1 MiB step**, and the
gap between the two is, to within 0.02 KiB/token, the step divided by the largest context.
Half of the commonly quoted "fixed 4.1 KiB/token toll" is an artifact of that mistake.

This repo contains the script that produces the numbers, the raw CSVs it produced, and the
derivation of both models, so that every claim can be recomputed rather than trusted.

No frameworks, no dependencies. One script, two CSVs, and the arithmetic.

## News

- **2026.10.30 v0.1.2 — the step result survived a second audit.** Added a quadratic
  comparison (same parameter count as the step model; the step wins by 486× on f16 and 6.6× on
  q8_0), downgraded the 32K agreement from "prediction" to "in-sample fit", corrected the
  leave-one-out comparison (it was measured against 1 SE instead of the 2 SE band), defined
  *toll* on first use, replaced "annualised" with "amortised", and added a Related work section
  and a measurement-protocol section. Test count 67 → 72. No measurement changed: the CSVs are
  byte-identical to v0.1.0.
- **2026.10.30 v0.1.0 — first public release.** Two configurations (f16, q8_0) across five
  context lengths each. A straight-line fit gives 32.11 / 19.08 KiB/token, then fails a
  residual check: modelling a one-time step at 8192 fits the same data to ±1 MiB and moves the
  slopes to **29.99 / 17.31**. 72 self-checks recompute every number quoted here without a GPU. 15-page
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

Two honesty notes. First, the margin on q8_0 is **6.6×** rather than 486× — smaller, same
direction, and this repository would rather print the smaller number. Second, **q8_0 does not
fit tightly at all**: its step-model residuals run to 5.8 MiB against f16's 1.0 MiB, roughly six
times worse, and there is no established explanation in this work. The F test still rejects the
line for q8_0, but anyone citing that configuration should know the fit is looser than the f16 one.

### Why the step matters: it was being charged as per-token cost

This is the mistake worth knowing about, because every KV benchmark that fits a single line
across a context grid makes it.

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
| f16 toll over theory | +4.11 | **+1.99** |
| q8_0 toll over theory | +4.20 | **+2.43** |
| gap between the two precisions | 0.09 | **0.44** |

Half of the "fixed 4.1 KiB/token toll" was never a per-token cost. It was a one-time
allocation step, amortised over 32K tokens by the fit. The real per-token overhead is
**≈2.0 KiB/token** — and it is *not* the same in both precisions. The gap is 0.44, which is
nearly five times what the plain line reported.

That inverts the original argument. The plain line made the toll look constant, and treated
that consistency as proof. Once the step is modelled, the consistency is gone and has to be
replaced by a mechanism.

### What it means for the VRAM bill

| Question | Answer |
|---|---|
| "How much does q8_0 save on the KV cache itself?" | **42.3%** (the plain line said 40.6%) |
| "How much does it save on total VRAM at 32K context?" | **19.6%** |
| "How much at 128K context?" | **32.6%** (extrapolated, no interval) |

Both "about 40%" and "about a fifth" are true. They answer different questions: the first
compares KV cache to KV cache, the second compares full VRAM bills. If you are quoting a
customer, you want the second number.

Two footnotes so the ratios can be reproduced. **42.3%** is the ratio of the two *measured*
slopes, and each slope already carries its own ~2 KiB/token overhead, so it is "KV cache vs KV
cache as billed", not "pure KV bytes" — against pure theory the ratio is 46.9%. And the
percentages do not share a denominator: **19.6%** divides by the raw 32K measurement, while
**32.6%** divides by the fitted 128K value, which is an extrapolation with no interval.

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

- **Stronger**: F = 2748 against a critical value of 18.513, and a 32K prediction that lands
  on the raw measurement with Δ = 0.1 MiB.
- **Weaker**: the steady-state slope rests on three points that are exactly collinear, so its
  standard error computes to **zero**. That is not a precise number, it is an unmeasurable
  one — three points cannot estimate their own scatter.

So the honest statement is: **the slope is 30, not 32; the step is real; and neither has a
trustworthy error bar yet.** Repeating each grid point is the experiment that would fix it,
and it has not been run. It is listed as the first item in `## What this repo is not`.

### Raw measurements

```
num_ctx   f16 (Δ MiB)   q8_0 (Δ MiB)
  2048        1166          1139
  4096        1224          1177
  8192        1412          1297
 16384        1652          1445
 32768        2132          1714
```

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
| q8_0 | 17.31 + 56.2 MiB step | 1714.0 MiB | measured |
| q4_0 | 7.88 theory, toll unknown | — | **not predicted** |

An earlier draft of this file did predict q4_0: theory 7.88 plus the "same 4.1 toll" gives
12.0. That arithmetic is no longer available, and it is worth being explicit about why.

Once the step is modelled, the toll is not one number. It is **1.99** for f16 and **2.43**
for q8_0 — a 22% spread between two precisions measured on the same machine, the same day,
with the same script. Extrapolating a constant that is not constant to a third precision is
not a prediction, it is a guess wearing a decimal point. The step height is worse: 67.1 for
f16, 56.2 for q8_0, and nothing says where q4_0 lands.

So the honest range is a slope somewhere near **9.9–10.3 KiB/token** and a 32K figure with
an uncertainty of tens of MiB that this dataset cannot narrow. **If you need the q4_0 number,
measure it**: `py scripts/ollama_probe.py --kv-bytes 0.5625 --tag q4_0`. That is one command
and about fifty seconds.

### Extrapolation to other models

KV cost is `2 × layers × kv_heads × head_dim × bytes`, so it scales predictably:

| model | KV/token (f16) | 32K | 128K |
|---|---|---|---|
| Qwen2.5-1.5B | 28 KiB | 896 MiB | 3.5 GiB |
| Llama-3-8B | 128 KiB | 4 GiB | **16 GiB** |
| Qwen2.5-72B | 320 KiB | 10 GiB | **40 GiB** |

A 72B model needs 40 GB just to remember the conversation — more than its own weights. That
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
| `--kv-bytes` | bytes per KV element: f16=2, q8_0=1.0625, q4_0=0.5625 |
| `--seqs` | context lengths to test (default `2048 4096 8192 16384 32768`) |
| `--layers --kv-heads --head-dim` | model geometry, used for the theory column |
| `--settle` | MiB tolerance when detecting a fully unloaded GPU |

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

1. `ollama stop <model>` to evict the model from VRAM.
2. Poll `nvidia-smi --query-gpu=memory.used` every 0.5 s until it stops falling, up to 90 polls
   (`--settle`, default 32 MiB tolerance). The value read at that point is **idle**.
3. Load the model at that `num_ctx` with a single request, and wait for the load to return.
4. Read `nvidia-smi` again. **loaded − idle is the measurement**, and it is the only number the
   fit uses.

Always compare **deltas**, never absolute VRAM. Desktop usage drifted by 419 MiB between the two
runs in this repo; deltas absorbed it completely.

What is *not* controlled, and should be read as part of the error budget: GPU clock and power
state are left to the driver, driver and CUDA versions are not recorded in the CSV, ambient load
from the desktop is only reflected in the idle reading, and each configuration was loaded
**once**. `load_sec` is written to the CSV for provenance but is not analysed.

### CSV format

| column | meaning |
|---|---|
| `num_ctx` | requested context length |
| `loaded_mib` | absolute VRAM after load (for debugging only) |
| `delta_mib` | **the measurement** — loaded minus idle |
| `theory_kv_mib` | pure KV per textbook formula, same row |
| `load_sec` | load wall time |
| `model`, `ollama_version`, `kv_cache_type`, `kv_bytes_per_elem` | provenance |

`delta_mib` includes the weights. That is intentional: the fit's intercept absorbs them, which
is exactly how the weights get separated from the per-token cost.

## Repository layout

```
scripts/ollama_probe.py        the only script — standard library only
tests/test_fit.py              re-derives every number in this README from data/, no GPU needed
data/*.csv                     raw measurements, with provenance columns
docs/kv-cache-report-zh.pdf    15-page Chinese write-up (v0.1.1 snapshot)
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
| Re-derivation | `python tests/test_fit.py` — 72 checks, standard library only |
| Measurement | `python scripts/ollama_probe.py --kv-bytes <b> --tag <name>` — needs Ollama and `nvidia-smi` |
| Write-up | `docs/kv-cache-report-zh.pdf` — Chinese, 15 pages. **Snapshot of v0.1.1**: it predates the quadratic comparison and the audit corrections in v0.1.2. Where it disagrees with this README, the README and the CSVs are current. |

The CSVs are the primary record. If a number in this README disagrees with a CSV, the CSV is
right and the README is a bug.

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
