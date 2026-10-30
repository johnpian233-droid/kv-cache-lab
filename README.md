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
a critical value of 18.5** — in favour of **29.99 KiB/token plus a 67.1 MiB step**, and the
gap between the two is exactly the step divided by the largest context. Half of the commonly
quoted "fixed 4.1 KiB/token toll" is an artifact of that mistake.

This repo contains the script that produces the numbers, the raw CSVs it produced, and the
derivation of both models, so that every claim can be recomputed rather than trusted.

No frameworks, no dependencies. One script, two CSVs, and the arithmetic.

## News

- **2026.10.30 v0.1.0 — first public release.** Two configurations (f16, q8_0) across five
  context lengths each. A straight-line fit gives 32.11 / 19.08 KiB/token, then fails a
  residual check: modelling a one-time step at 8192 fits the same data to ±1 MiB and moves the
  slopes to **29.99 / 17.31**. 55 self-checks recompute every number without a GPU. 15-page
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
what you get after fitting the model the data actually wants. **They disagree by 7% and 10%,
and the next section shows why the second one wins.**

The **intercept is the model's fixed footprint** — weights plus CUDA overhead — around
1110 MiB. It never discounts, no matter how you quantize the cache.

### A straight line hides a step

Five points, five monotone values. Of course a line fits: R² = 0.9956. That number is not
evidence of linearity, it is a formality — **any** monotone five-point series produces it.
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

The 5% critical value is 18.5. Both statistics are past it by a wide margin. On f16 the step
model leaves residuals of **±1 MiB** — the resolution of `nvidia-smi` itself. A model that
reproduces every measurement to the last reported digit is not an improvement over the line;
it replaces it.

Fit only the points at or above 8192 and the f16 slope is **30.000 KiB/token**, with three
points collinear to within a megabyte. The bend was never noise.

### Why the step matters: it was being charged as per-token cost

This is the mistake worth knowing about, because every KV benchmark that fits a single line
across a context grid makes it.

A 67 MiB step at 8192, if you refuse to model it, gets spread across the whole range by the
regression. Divided by the largest context it becomes

```
67.1 MiB / 32768 token × 1024 = 2.10 KiB/token
```

and that is **exactly** how much the plain slope is inflated: 32.11 − 29.99 = 2.12. The
arithmetic closes.

Now put that next to the gap between measurement and theory:

| | plain line | line + step |
|---|---|---|
| f16 toll over theory | +4.11 | **+2.00** |
| q8_0 toll over theory | +4.20 | **+2.42** |
| gap between the two precisions | 0.09 | **0.42** |

Half of the "fixed 4.1 KiB/token toll" was never a per-token cost. It was a one-time
allocation step, annualised over 32K tokens by the fit. The real per-token overhead is
**≈2.0 KiB/token** — and it is *not* the same in both precisions. The gap is 0.42, which is
four and a half times what the plain line reported.

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

Model check at 32K — each model against the raw measurement at `num_ctx=32768`:

| | plain line | Δ | line + step | Δ |
|---|---|---|---|---|
| f16 | 1119 + 32.11 × 32 = 2146.6 MiB | 14.6 | 1105.0 + 29.99 × 32 + 67.1 = **2131.9 MiB** | **0.1** |
| q8_0 | 1118 + 19.08 × 32 = 1728.4 MiB | 14.4 | 1106.1 + 17.31 × 32 + 56.2 = **1716.1 MiB** | 2.1 |
| saved | 418.2 MiB | — | 415.8 MiB | — |

A model that predicts the raw measurement to **0.1 MiB** — the resolution of `nvidia-smi`
itself — is not a coincidence you can argue with. The plain line was off by 14.6 MiB here, and
that error is precisely the step it refused to model.

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
answered, and the answer was no. Leave-one-out makes the point: dropping any single point
moves the plain slope between **31.6 and 35.2**, a spread four times wider than the reported
±1.24. The nominal error bar was measuring the wrong thing.

The step model is stronger on the main claim and **weaker on precision**:

- **Stronger**: F = 2748 against a critical value of 18.5, and a 32K prediction that lands
  on the raw measurement with Δ = 0.
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
| q8_0 | 17.31 + 56.2 MiB step | 1715.6 MiB | measured |
| q4_0 | 7.88 theory, toll unknown | — | **not predicted** |

An earlier draft of this file did predict q4_0: theory 7.88 plus the "same 4.1 toll" gives
12.0. That arithmetic is no longer available, and it is worth being explicit about why.

Once the step is modelled, the toll is not one number. It is **2.00** for f16 and **2.42**
for q8_0 — a 21% spread between two precisions measured on the same machine, the same day,
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

### How it measures

For each `num_ctx`: unload the model, record idle VRAM, load at that context length, record
loaded VRAM, take the delta.

Always compare **deltas**, never absolute VRAM. Desktop usage drifted by 419 MiB between the
two runs in this repo; deltas absorbed it completely.

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
docs/kv-cache-report-zh.pdf    15-page Chinese write-up with charts and derivations
docs/PUBLISHING-zh.md          how this repo was published, step by step (Chinese)
.github/ISSUE_TEMPLATE/        "report your own number" issue form
CITATION.cff                   GitHub's "Cite this repository" metadata
```

## What this repo is not

- **Not a benchmark.** No tokens/s, no latency, no quality evaluation. Memory only.
- **No repeated measurements.** Every `num_ctx` was loaded **once**. All three degrees of
  freedom in the confidence intervals come from model residuals, not from pure error, so a
  lack-of-fit test is impossible and the error bars are optimistic. This is the single biggest
  weakness in the dataset and it is the first item on the roadmap.
- **One machine, one model.** Five points per configuration. The method generalizes; the
  numbers do not. Two runs here disagree on the idle baseline by ~480 MiB when the desktop is
  busy — enough to move a two-point slope by 9 KiB/token.
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
