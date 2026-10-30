# kv-cache-lab

> **Scope note (2026.10.30)**: this repository measures **memory only**, on **one GPU** with
> **one model**. It is not a throughput benchmark and it does not evaluate output quality.
> The method generalizes; the numbers do not. Read `## What this repo is not` before citing
> anything here.

**English** | [中文](README_zh.md)

## Introduction

`kv-cache-lab` is a one-script lab that answers a single question with arithmetic instead of
marketing: **how much VRAM does KV cache quantization actually save on a consumer GPU?**

Vendors advertise "50% off". On an 8 GB laptop GPU the measured answer is **40.6% off the KV
cache** and **19.6% off the total VRAM bill at 32K context**. The difference is a fixed
per-token overhead — roughly **4.1 KiB/token** — that quantization cannot discount. This repo
contains the script that produces the numbers, the raw CSVs it produced, and the derivation
of the fit, so that every claim can be recomputed rather than trusted.

No frameworks, no dependencies. One script, two CSVs, and the arithmetic.

本仓库用一次控制变量的实测回答一个问题：**在消费级显卡上，KV cache 量化到底省多少显存？**
厂商说"省一半"。在 8 GB 笔记本显卡上，实测是 **KV cache 省 40.6%、32K 上下文的总显存省 19.6%**。
差额来自一笔量化打折不了的固定开销——约 **4.1 KiB/token**。仓库里有产出这些数字的脚本、
脚本跑出的原始 CSV、以及拟合的推导过程，每个结论都可以自己复算，不必相信。

## News

- **2026.10.30 v0.1.0 — first public release.** Two configurations measured (f16, q8_0) across
  five context lengths each; least-squares fit with 95% confidence intervals; 15-page Chinese
  write-up with charts and derivations (`docs/kv-cache-report-zh.pdf`).
- **Roadmap** — q4_0 measured (currently predicted only), output-quality comparison at q8_0,
  two extra points at 6144 / 24576 to tighten the confidence interval, and the mechanism behind
  the 4096→8192 step. Contributions welcome; see `## Contributing`.

## Performance

Measured on an RTX 4060 Laptop (8188 MiB), Qwen2.5-1.5B (Q4_K_M), Ollama 0.40.6,
`OLLAMA_NUM_PARALLEL=1`. Each point is one model load; the reported value is the VRAM delta
between a fully unloaded GPU and a load at that `num_ctx`.

### Measured slopes

| KV type | slope (measured) | slope (theory) | measured/theory | intercept | R² |
|---|---|---|---|---|---|
| **f16** (baseline) | **32.11 ± 1.24 KiB/token** | 28.00 | 1.15× | 1119 ± 20 MiB | 0.9956 |
| **q8_0** | **19.08 ± 1.05 KiB/token** | 14.88 | 1.28× | 1118 ± 17 MiB | 0.9910 |

The **slope is the real per-token cost on this machine**. The **intercept is the model's fixed
footprint** — weights plus CUDA overhead — and it never discounts.

Two independent runs, an hour apart, different desktop usage, produced intercepts that differ
by **1 MiB**. That is the evidence that the method is sound.

### The toll over theory

| | pure KV theory | measured slope | difference |
|---|---|---|---|
| f16 | 28.00 | 32.11 | **+4.11** |
| q8_0 | 14.88 | 19.08 | **+4.20** |

Two nearly identical gaps across two precisions. This is not a quantization artifact — it is a
fixed cost:

- **~1.0 KiB/token** — the attention mask buffer (allocated whether FlashAttention is on or off)
- **~1.0 KiB/token** — a temporary f16 copy of quantized K/V, required by the FA kernels
- **~2.0 KiB/token** — allocator fragmentation and alignment

That toll is why quantization never reaches its theoretical ratio. **The saving scales with
context length; the toll does not.**

### What it means for the VRAM bill

| Question | Answer |
|---|---|
| "How much does q8_0 save on the KV cache itself?" | **40.6%** |
| "How much does it save on total VRAM at 32K context?" | **19.6%** |
| "How much at 128K context?" | **31.9%** |

Both "half" and "a fifth" are true. They answer different questions. If you are quoting a
customer, you want the second number.

Model check at 32K — the fit against the raw measurement at `num_ctx=32768`:

| | from the fit | raw @32768 | Δ |
|---|---|---|---|
| f16 | 1119 + 32.11 × 32 = **2146.6 MiB** | 2132 MiB | 14.6 |
| q8_0 | 1118 + 19.08 × 32 = **1728.4 MiB** | 1714 MiB | 14.4 |
| saved | **418.2 MiB** | **418.0 MiB** | — |

418 / 2132 = **19.6%**. Both routes give the same saving, which is the point: the line is a
model of the machine, not a redrawing of the data.

### How sure is that number

Five points, three degrees of freedom, Student-t at 95%:

| KV type | slope ± 1 SE | 95% CI | theory | theory vs CI |
|---|---|---|---|---|
| f16 | 32.11 ± 1.24 | [28.18, 36.05] | 28.00 | outside, by 0.18 |
| q8_0 | 19.08 ± 1.05 | [15.74, 22.43] | 14.88 | outside, by 0.86 |

Both theoretical values sit outside the interval, so the toll is nominally significant.
**Now look at the margin on f16: 0.18 KiB/token — 0.6%.** One more point could pull the theory
back inside the interval and turn "there is a toll" into "no detectable difference".

That is why the toll section above does not hang its claim on this CI. It hangs on
**replication**: two independent runs, two precisions, the same +4.11 / +4.20 gap, intercepts
1 MiB apart. A systematic offset that survives all of that is not a regression artifact.

Want a tighter interval? Add points at 6144 and 24576 — the 4096→8192 step is what widens it.

### Raw measurements

```
num_ctx   f16 (Δ MiB)   q8_0 (Δ MiB)
  2048        1166          1139
  4096        1224          1177
  8192        1412          1297
 16384        1652          1445
 32768        2132          1714
```

The 4096→8192 interval is the one place both curves bend upward (30 vs 19 KiB/token for q8_0,
47 vs 29 for f16). It appeared in two independent runs at the same position, so it is
systematic, not noise. Most likely cause: a FlashAttention kernel switching on at
`n_kv ≥ 4096` with `gqa_ratio > 4` — Qwen2.5-1.5B sits exactly at ratio 6. *(Mechanism
inferred, not confirmed in source.)*

### The third rung — predicted, not measured

| KV type | slope | 32K VRAM on Qwen2.5-1.5B |
|---|---|---|
| f16 (measured) | 32.11 KiB/token | 2146.6 MiB |
| q8_0 (measured) | 19.08 | 1728.4 MiB |
| **q4_0 (predicted)** | **≈ 12.0** | **≈ 1503 MiB** |

q4_0 theory is 7.88 KiB/token; add the same ~4.1 toll and you land at 12.0 — 30% below f16 at
32K. **This row is arithmetic, not a measurement.** It is in the table so you can check the
model instead of trusting it: run `--kv-bytes 0.5625` and see whether it holds.

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
- **One machine, one model.** Five points per configuration. The method generalizes; the
  numbers do not.
- **Not proof that q8_0 is lossless.** Practitioner consensus says it is near-lossless; this
  repo did not test it. Do not cite it as evidence that it is.
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
