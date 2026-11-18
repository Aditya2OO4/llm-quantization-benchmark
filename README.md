# TinyLlama quantization benchmark

Quantization stores a model's weights with fewer bits. That makes it smaller and
faster, but it can also make it worse. This project measures that trade-off for
TinyLlama 1.1B at seven GGUF quantization levels (Q8_0 to Q2_K) against the 16-bit
original, on a laptop CPU with no GPU, and serves the model behind a small
REST API.

## Highlights

- **Full quantization benchmark on a laptop CPU.** Seven GGUF levels (Q8_0 to Q2_K) vs F16 for size, RAM, speed and perplexity. Q4_K_M is 70% smaller and decodes 2.2x faster for +3.5% perplexity.
- **My own int4/int8 quantizer, written from scratch.** The int8 matches llama.cpp's Q8_0 (45.18 vs 45.17 dB weight error). Three ideas, each tested on its own, took my int4 from +5.2% to **+2.6%** perplexity vs fp16, better than llama.cpp's Q4_0 (+3.6%, each measured against its own F16).
- **A real exported model.** My quantized TinyLlama is saved as a 622 MB file (3.4x smaller than F16), and reloading it gives bit-identical outputs.
- **Leak-free importance-matrix study.** Calibration from the train split, with an automatic check against the test set: the imatrix closes 38 to 72% of the quality gap at Q2_K, Q3_K_M and Q4_K_M at no extra file size.
- **Serving experiment with a control.** llama.cpp's batching raises throughput by about 38% at 4 to 8 concurrent clients on 4 cores, and a 1-slot control run confirms the gain comes from batching.
- **Reproducible.** Every number comes from a script run saved in `results/`; 122 tests; one command per experiment.

## What gets measured

| Question | Metric | Script |
|---|---|---|
| How much smaller is it? | file size | `step3_quantize_sweep.py` |
| How much RAM does it need? | peak resident memory (fresh process per model) | `step4_benchmark.py` |
| How fast is it? | decode tok/s, prompt tok/s, time to first token (mean ± std) | `step4_benchmark.py` |
| Does it scale with CPU cores? | tok/s vs threads | `step4_benchmark.py --thread-sweep` |
| How much worse are its answers? | **perplexity** on WikiText-2 | `step4b_perplexity.py` |
| How does the API cope with many users? | latency & throughput vs concurrency | `step7_load_test.py` |
| What does my own quantizer lose, layer by layer, and vs Q8_0 / Q4_0? | weight MSE / SQNR, perplexity delta | `step9_custom_quant.py` |

`step6_report.py` merges everything into `results/RESULTS.md` plus charts.

## What I built vs. what I used

In the main sweep (`step3` to `step8`) the quantization itself is done by llama.cpp: `llama-quantize` makes
every GGUF file, `llama-perplexity` scores them, and `llama-cpp-python` runs them. My part there is the measurement
harness: the benchmark orchestration, the RAM and speed methodology, the report, the API and the load test. My own quantizer
(below) is a separate piece of work, compared against llama.cpp on equal terms.

| Used (not mine) | Built by me |
|---|---|
| llama.cpp quantization formats, `llama-quantize`, `llama-perplexity`, GGUF conversion | `custom_quant.py`: block-wise int8 / int4 quantizer (block 32, fp16 scale per block) in NumPy with three scale rules (symmetric absmax, signed anchoring, per-block scale search with optional importance weights), and round-trip, zero-block and odd-size handling |
| `llama-cpp-python` inference, HuggingFace transformers and tokenizer, PyTorch | `custom_quant_eval.py`: runs the HF TinyLlama on my quantized weights (stored compressed, dequantized layer by layer), collects the importance statistics with forward hooks, and scores it with the same window protocol as `llama-perplexity`. `custom_quant_export.py`: writes and re-reads my quantized model as a `.safetensors` file |
| llama.cpp's `llama-server` (parallel slots, continuous batching) | `step11_serving_compare.py` + the parametrized `step7_load_test.py`: starts one server at a time, same model, threads, prompt and load generator, plus a 1-slot control |
| The importance-matrix algorithm itself (`llama-imatrix`, `llama-quantize --imatrix`) | `step10_imatrix.py`: the experiment design (calibration from the train split, a leakage check that refuses to run if any test paragraph is in the calibration text, the same test windows as the main sweep, the "gap closed" metric), run reproducibly |
| The `gguf` package (only to read llama.cpp's tensors back) | `step9_custom_quant.py`, `gguf_compare.py`: per-tensor MSE and SQNR for every layer, a direct weight-error comparison with llama.cpp's Q8_0 / Q4_0, and the perplexity comparison |

**My quantizer.** I wrote block-wise int8 / int4 quantization from scratch (block of 32 weights,
one fp16 scale each) and applied it to every weight matrix of TinyLlama: all attention and MLP projections, the embedding table
and the output head. It has three ways of choosing the scales, the last using importance statistics that I collect myself.
I saved the result as a real 622 MB file in my own format and checked that reloading it gives bit-identical outputs
(`custom_quant_export.py`). I measured it against llama.cpp's Q8_0 / Q4_0 on identical tensors and on perplexity (Experiments 1 and 1b).

**Scope.** The quantizer is evaluated for quality and size; it uses its own file format loaded by my PyTorch code, so speed
and llama.cpp integration are out of scope here. The techniques are established ones (signed anchoring is what Q4_0 does, the scale search is
standard, importance weighting is the idea behind llama.cpp's imatrix); the work is implementing them independently and measuring which of them
helps on this model.

## Setup (Windows, no compiler needed)

```powershell
step1_setup.bat                 # venv + dependencies (uses a prebuilt llama-cpp-python wheel)
env\Scripts\activate
python step2_download_model.py  # ~2.2 GB raw model
```

Then get llama.cpp, whose tools do the actual conversion and quantization. See
[step3_quantize.md](step3_quantize.md). The quickest route is a prebuilt release, so no CMake is needed.

## Run the whole pipeline

```powershell
python step3_quantize_sweep.py                      # F16 + 6 quantized files
python step4_benchmark.py --thread-sweep 1,2,4      # speed + RAM (~15-30 min)
python step4b_perplexity.py                         # quality (~10-20 min)
python step6_report.py                              # -> results/RESULTS.md + PNG charts
```

My own quantizer (needs `pip install -r requirements-research.txt`; about 25 min total, one model at a time):

```powershell
python step9_custom_quant.py weights   # per-tensor MSE / SQNR + plot (~2 min)
python step9_custom_quant.py gguf      # same error for llama.cpp's Q8_0 / Q4_0 tensors
python step9_custom_quant.py ppl       # fp16, int8, int4, int4_head8 perplexity (~8-12 min each)
python step6_report.py
```

The improved int4 (Experiment 1b, about 1 h in total on this machine) and the export:

```powershell
python step9_custom_quant.py schemes       # weight error of the signed / scale-search rules (~5 min)
python step9_custom_quant.py importance    # activation statistics from WikiText-2 train text (~4 min)
python step9_custom_quant.py ppl --variants int4s_head8,int4opt_head8,int4imp_head8   # ~11-13 min each
python step9_custom_quant.py export --variant int4imp_head8    # writes models\tinyllama-custom-int4imp_head8.safetensors
python step6_report.py
```

Importance-matrix experiment (about 1 h for the 50-chunk run: ~19 min for the imatrix, ~10 min per level):

```powershell
python step10_imatrix.py                                       # Q2_K, Q3_K_M, Q4_K_M, 50 calibration chunks
python step10_imatrix.py --levels Q2_K --chunks-list 10,200    # calibration-size ablation (not run yet; 200 chunks takes ~75 min)
```

Serving comparison (about 10 minutes; one server at a time):

```powershell
python step11_serving_compare.py                                   # FastAPI vs llama-server, 24 requests per level
python step11_serving_compare.py --server llamaserver --slots 1 --out load_test_llamaserver_np1.json   # control
python step6_report.py
```

### Side-by-side demo

Ask one question and watch every quantization level answer token by token, with live speed and an automatic correctness check on the preset questions:

```powershell
python step8_compare.py        # opens http://127.0.0.1:8001
```

Models run one at a time: they don't all fit in 8 GB, and running them together would distort the timings. Preset facts and maths are checked by keyword. Preset "write a Python function" questions are checked by running the generated code against test cases, in a separate `python -I` process with a timeout that refuses imports and file access. That denylist is not a real sandbox, so the check is only enabled for the presets. Stop `step5_api.py` first, since both load a model.

### API server and load test

```powershell
python step5_api.py                 # http://127.0.0.1:8000/docs
python step7_load_test.py           # in a second terminal
python step6_report.py              # re-run to add load-test table
```

## Results

Measured on an AMD Ryzen 5 7520U (4 cores), 7.2 GB RAM, no GPU, Windows 11.
4 threads, 3 runs per prompt (mean ± std), perplexity on WikiText-2 (40 × 512-token windows).

| Level | Size MB | Peak RAM MB | Decode tok/s | Speedup | Perplexity | vs F16 |
|---|---|---|---|---|---|---|
| F16 | 2099 | 2066 | 12.5 ± 0.7 | 1.00x | 8.836 | - |
| Q8_0 | 1116 | 1179 | 22.0 ± 1.4 | 1.76x | 8.840 | +0.1% |
| Q6_K | 862 | 940 | 22.1 ± 3.8 | 1.77x | 8.870 | +0.4% |
| Q5_K_M | 746 | 833 | 23.0 ± 1.2 | 1.84x | 8.929 | +1.1% |
| Q4_K_M | 637 | 1187 | 27.6 ± 1.2 | 2.21x | 9.142 | +3.5% |
| Q4_0 | 607 | 1195 | 18.7 ± 1.9 | 1.49x | 9.155 | +3.6% |
| Q3_K_M | 523 | 811 | 26.3 ± 2.6 | 2.11x | 9.571 | +8.3% |
| Q2_K | 412 | 528 | 32.4 ± 1.3 | 2.59x | 12.853 | +45.5% |

Takeaways:
- Q4_K_M is the sweet spot: 70% smaller on disk, 2.2x faster decoding, +3.5% perplexity. Below it quality drops quickly (Q2_K: +45%).
- Decoding is memory-bound: going from 2 to 4 threads adds only about 15% (22.5 to 26.0 tok/s), so the CPU is waiting on RAM.
- Q4_K_M uses more RAM than Q5_K_M because llama.cpp repacks its weights into a faster AVX2 layout (+455 MiB, seen in the verbose load log). Its RAM saving over F16 is 43%, not 70%.
- The speed differences between Q8_0, Q6_K and Q5_K_M are within run-to-run noise.
- Q4_0 (added later, see below) is 30 MB smaller than Q4_K_M with the same perplexity (+3.6% vs +3.5%) but decodes much slower here (18.7 vs 27.6 tok/s, measured twice with the same result). Q4_K_M is better on both quality and speed; only the file size favours Q4_0. Q4_0 was benchmarked in a separate session from the other levels, with other programs open on this 7.2 GB machine, so treat its speed as less comparable than the rest.

**Reading the ±.** The perplexity ± is the standard error over the 40 text windows, and it is almost the same for every level (about ±0.23 to ±0.33) because every level is scored on the same text. The *differences* between levels are paired (same windows for every level), so they are more precise than the error bars suggest. Overlapping error bars do not mean "no difference": Q4_K_M is +3.5% over F16 even though the bars overlap. I measured this for my own evaluator (step 9): for my int4 the change vs F16 is +5.9% with a paired standard error of 0.42% (window level), versus 7.9% if the two runs were treated as independent samples.

**The 5% budget is arbitrary.** The recommendation (smallest level within 5% of F16 perplexity) uses a budget I picked, not a standard. It is configurable (`python step6_report.py --budget 2` or `MAX_PPL_INCREASE_PCT=2`), and `RESULTS.md` shows the pick at 2%, 5% and 10%: Q5_K_M, Q4_0 and Q3_K_M. The rule only looks at size, so at 5% it now picks Q4_0 (30 MB smaller than Q4_K_M) even though Q4_K_M is no worse on quality and about 1.5x faster; I call Q4_K_M the sweet spot because it wins on all three axes, not because the rule says so. "Sweet spot" is a judgement about a budget, not a universal fact.

Full report with charts and sample answers: [results/RESULTS.md](results/RESULTS.md).

![size vs quality](results/size_vs_quality.png)

**API load test** (Q4_K_M served via FastAPI, 32 new tokens per request, 8 requests per level):

| Concurrent clients | Req/s | Tokens/s | p50 latency | p95 latency |
|---|---|---|---|---|
| 1 | 0.87 | 27.8 | 1.06 s | 1.77 s |
| 2 | 0.92 | 29.5 | 2.07 s | 2.40 s |
| 4 | 0.95 | 30.5 | 4.14 s | 4.23 s |
| 8 | 0.83 | 26.5 | 5.38 s | 9.63 s |

Throughput stays flat (~27-30 tok/s) while latency grows roughly linearly with the number of clients: one CPU model serves one request at a time, so extra clients just queue. My hypothesis was that fixing this needs request batching (as in vLLM). Experiment 3 below tests that. p95 over only 8 requests is essentially the maximum, which is why that experiment uses 24 requests per level.

## Findings

Each experiment: what I expected, what was measured, what was surprising. Every number comes from a script run
(`results/*.json`).

### Experiment 1: my own int8 / int4 quantizer vs llama.cpp's Q8_0 / Q4_0

Same 40 x 512-token WikiText-2 test windows as the main sweep, F16 scored by the same evaluator as my quantized models.
**Only the deltas vs each tool's own F16 are comparable.** My evaluator uses the HuggingFace tokenizer and PyTorch,
`llama-perplexity` uses its own tokenizer and code, so absolute perplexities are different measurements even though
the two F16 values came out almost equal (8.836 vs 8.836 here; I do not rely on that).

| Tool | Level | Perplexity (own evaluator) | Δ vs own F16 |
|---|---|---|---|
| mine | fp16 | 8.836 | - |
| mine | int8 | 8.839 | +0.03% (paired SE 0.03%) |
| mine | int4 | 9.359 | +5.92% (paired SE 0.42%) |
| mine | int4, lm_head at int8 | 9.291 | +5.15% (paired SE 0.41%) |
| llama.cpp | Q8_0 | 8.840 | +0.05% |
| llama.cpp | Q4_0 (output.weight is Q6_K) | 9.155 | +3.62% |

Direct weight error on identical tensors (reference = the F16 GGUF), pooled over all compared tensors:

| Format | llama.cpp SQNR | mine SQNR |
|---|---|---|
| Q8_0 vs my int8 (156 tensors) | 45.17 dB | 45.18 dB |
| Q4_0 vs my int4 (155 tensors; `output.weight` is Q6_K and skipped) | 21.13 dB | 20.06 dB |

![SQNR by layer and tensor type](results/custom_quant_layers.png)

- **Expected:** int8 should match Q8_0 (same scheme). int4 should be a bit worse than Q4_0, because Q4_0 uses 16 levels (-8..7)
  and a scale from the signed max, while mine uses the symmetric 15 levels (-7..7), a step about 12% coarser.
- **Measured:** int8 and Q8_0 are indistinguishable (0.01 dB apart in weight error; +0.03% vs +0.05% perplexity). My int4 is 1.07 dB worse in
  weight error and +5.9% vs +3.6% in perplexity. The 2.3-point gap is much bigger than the paired SE of my own number (0.42%).
- **Surprising:** (1) a 1 dB weight-error difference shows up as a large perplexity gap. (2) Part of the gap is a design choice llama.cpp makes that I did not:
  it keeps the output matrix at 6 bits even in a "4-bit" file. Quantizing my lm_head at int8 recovers 0.77 of the 2.3 points. I did not test what explains the
  remaining ~1.5 points; the 15-vs-16-level difference is my hypothesis, not a result. (3) Quantization error is not uniform: layer 0's `q_proj` / `k_proj` are the hardest tensors
  (about 41 dB at int8, 17 dB at int4, vs about 45 and 20 dB elsewhere) and `v_proj` dips in layers 15 and 18. I did not test whether protecting those tensors would help perplexity.
- Caveat: the paired SE is over 40 windows of one text; the stored ± for llama.cpp's numbers is a different (token-level, unpaired) quantity, so I do not give a paired SE for them.

### Experiment 1b: making my int4 better, one idea at a time

Experiment 1 showed my int4 losing to Q4_0 (+5.9% vs +3.6%). I tested two causes and one extra idea, adding one per rung, all in the same evaluator
with the output head at int8 (so the rungs are comparable with each other and with Q4_0's Q6_K head). Each variant ran once on the same 40 windows;
the grid and the weighting have no tunable settings that I fitted to the test windows (the scale grid was chosen from weight error on four tensors).

| Rung | Idea added | Weight SQNR, all tensors at int4 | Perplexity | Δ vs fp16 (paired SE) | Δ vs previous rung (paired SE) |
|---|---|---|---|---|---|
| 0 | symmetric absmax, 15 levels | 20.02 dB | 9.291 | +5.15% (0.41%) | - |
| 1 | signed anchoring: the biggest weight of each block sits on level -8, all 16 levels used (what Q4_0 does) | 21.08 dB | 9.145 | +3.50% (0.38%) | -1.57% (0.40%) |
| 2 | per-block scale search: try 13 scale multipliers (0.9 to 1.2), keep the lowest squared error | 21.56 dB | 9.168 | +3.76% (0.37%) | +0.25% (0.36%) |
| 3 | importance weighting: the search minimizes error weighted by how large each input column's activations are (collected on WikiText-2 **train**, 20 windows) | n/a (optimizes a weighted error) | 9.063 | +2.57% (0.31%) | -1.15% (0.24%) |
| ref | llama.cpp Q4_0 (its own evaluator and tokenizer) | 21.13 dB | 9.155 | +3.62% | - |

- **Expected:** signed anchoring buys about 1 dB (16 levels instead of 15). The scale search should help further. Importance weighting should help most, by analogy with the imatrix result in Experiment 2.
- **Measured:** signed anchoring: +1.06 dB and -1.57% perplexity, which lands my int4 on top of Q4_0 (21.08 vs 21.13 dB, +3.50% vs +3.62%). So my explanation of the gap in Experiment 1 was right. The importance weighting gives the biggest remaining gain, -1.15% ± 0.24% against rung 2 (same encoder, only the weights differ), -2.45% ± 0.40% against rung 0.
- **Surprising:** the scale search lowers weight error by another 0.5 dB but does **not** measurably help perplexity (+0.25% ± 0.36%, within noise). Lower weight error did not mean a better model here. My untested guess is that the lowest-error scale clips the largest weights, and those matter more than their share of the squared error. The importance statistics also span an enormous range: in layer 10's `down_proj` the largest column weight is 124x the median, and in layer 0's `q_proj` it is about 340,000x (a few channels carry huge activations), which is why weighting by activation changes the result while plain weight error does not.
- **Comparison with llama.cpp:** my best int4 (+2.57%, 622 MB) is better than plain Q4_0 (+3.62%, 607 MB). Context for reading that: the two are scored by different evaluators (each against its own F16), my file has an int8 head and is 15 MB larger, and calibration and test text both come from WikiText-2. llama.cpp's K-quants with imatrix are the stronger option overall (Q4_K_M with imatrix: +1.0% at 637 MB, Experiment 2); Q4_0 with an imatrix is the like-for-like next comparison.
- **Caveats:** one run per rung; 40 windows; 20 calibration windows (10,240 tokens); only perplexity, no task accuracy.

**Exported file.** `python step9_custom_quant.py export --variant int4imp_head8` writes `models/tinyllama-custom-int4imp_head8.safetensors`
(packed int4 weights, fp16 block scales, int8 output head, fp16 norms): **622 MB on disk, 4.74 bits per parameter, 3.4x smaller than the 2099 MB F16 GGUF**.
Reloading the file into a fresh model and scoring 4 test windows gave results identical to the in-memory model (checked bit for bit, not within a tolerance).
The file only loads through `custom_quant_export.py`; llama.cpp cannot read it.

### Experiment 2: does an importance matrix recover the quality lost at low bit widths?

Hypothesis: llama.cpp's imatrix recovers a large share of the perplexity lost at Q2_K, Q3_K_M and maybe Q4_K_M.
Setup: calibration text is the first 1,000,000 characters of the WikiText-2 **train** split (the script checks that no test
paragraph appears in it: 0 found), 50 chunks of 512 tokens, imatrix computed on the F16 model; quantized with
`llama-quantize --imatrix`; scored on the same 40 x 512-token **test** windows as the main sweep. "Plain" numbers are the
ones already in `results/perplexity.json`. Gap closed = (plain - imatrix) / (plain - F16).

| Level | Size MB (same with and without) | Plain | With imatrix | Gap closed |
|---|---|---|---|---|
| Q2_K | 412 | 12.853 (+45.5%) | 11.056 ± 0.284 (+25.1%) | 45% |
| Q3_K_M | 523 | 9.571 (+8.3%) | 9.292 ± 0.239 (+5.2%) | 38% |
| Q4_K_M | 637 | 9.142 (+3.5%) | 8.921 ± 0.229 (+1.0%) | 72% |

- **Expected:** a large share recovered, biggest at the lowest bit width, where plain quantization hurts most.
- **Measured:** the hypothesis holds only partly. The imatrix closes 38% to 72% of the gap, never all of it, and it is free at inference time (the files are the same size). Q2_K is still +25% worse than F16 even with it.
- **Surprising:** (1) the share is not monotonic in bit width: Q3_K_M gains the least (38%), Q4_K_M the most (72%). In absolute terms Q2_K gains the most (1.8 perplexity points). I did not investigate why. (2) Q4_K_M with imatrix (+1.0%, 637 MB) is about as good as plain Q5_K_M (+1.1%, 746 MB), and Q3_K_M with imatrix (+5.2%, 523 MB) is not far from plain Q4_K_M (+3.5%, 637 MB).
- **Caveats:** calibration and test text both come from WikiText-2 (disjoint articles, but the same style), so this probably overstates the gain on text from other domains; I did not test that. One calibration size (50 chunks) and one calibration text per level, so no error bars on the gain itself. The plain files were quantized earlier than the imatrix ones, most likely with the same llama.cpp build (I did not verify that). The calibration-size ablation (10 / 50 / 200 chunks) is supported by the script but I have **not** run it: building the 200-chunk imatrix alone takes over an hour on this machine (the 50-chunk one took 19 minutes).

### Experiment 3: is batching the fix for the serialized server?

The README above says the FastAPI server answers one request at a time and that batching would fix it. Test: run llama.cpp's
`llama-server` with 8 parallel slots and continuous batching on the same Q4_K_M model with the same 4 threads, and load-test both with the
same generator (24 requests per level, 32 new tokens each, same chat-formatted prompt, `cache_prompt` off so no request gets free prompt reuse). As a
control, the same `llama-server` with **1 slot** (no parallel decoding), to separate "batching" from "a different server program".

| Clients | FastAPI tok/s (p50 / p95 s) | llama-server 8 slots tok/s (p50 / p95 s) | llama-server 1 slot, control tok/s (p50 / p95 s) |
|---|---|---|---|
| 1 | 30.0 (1.02 / 1.28) | 30.0 (1.06 / 1.15) | 26.6 (1.13 / 1.42) |
| 2 | 30.4 (2.07 / 2.31) | 34.9 (1.82 / 1.95) | 28.2 (2.23 / 2.47) |
| 4 | 32.7 (3.89 / 4.00) | 44.7 (2.85 / 2.92) | 26.7 (4.65 / 5.52) |
| 8 | 32.3 (7.90 / 8.04) | 45.0 (5.58 / 5.91) | 27.7 (8.99 / 9.64) |

(Req/s for every row is in `results/RESULTS.md`; raw data in `results/load_test_fastapi_n24.json`, `load_test_llamaserver.json`, `load_test_llamaserver_np1.json`.)

- **Expected:** batching removes the queueing, so throughput should rise with clients and latency should stay roughly flat.
- **Measured:** batching helps, but only partly. Throughput at 4 and 8 clients is about 45 tok/s vs about 32 for FastAPI (+38%), and p50 at 8 clients falls from 7.9 s to 5.6 s. With one slot, `llama-server` behaves like FastAPI (flat at about 27 tok/s), so the gain comes from batching, not from the server program.
- **Surprising / honest:** (1) latency still grows with the number of clients (p50 1.06 s at 1 client to 5.58 s at 8); batching raised the ceiling, it did not remove the queue. (2) Throughput stops growing after 4 clients (44.7 to 45.0 at 8): on 4 CPU cores the batched decode is limited by compute and memory bandwidth, not by scheduling. So the claim in my original README, "batching would fix it", was too strong: it gives about 1.4x on this CPU, not a flat-latency server. (3) A single slot of `llama-server` was a little slower than my FastAPI server (about 27 vs 30 to 33 tok/s); I did not investigate why.
- **Caveats:** one run per configuration, 24 requests per level, so p95 is the 23rd-slowest of 24. The first run of the 1-slot control had one request stall for about two minutes at 1 client (throughput 5.1 tok/s while p50 and p95 were normal), which I treated as a machine hiccup and re-ran; the table shows the re-run. Other programs were open on this 7.2 GB machine, so absolute numbers are noisy, the ratios are more trustworthy.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Prompt in, completion plus timing out |
| `GET` | `/health` | Model status and memory use |
| `GET` | `/docs` | Swagger UI |

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
     -d '{"prompt": "What is model quantization?", "max_tokens": 150, "chat": true}'
```

`"chat": true` wraps the prompt in TinyLlama-Chat's template (`format_prompt` in `config.py`). The model is a chat
model, so without it a question is treated as raw text to continue and the answers are much worse. It defaults to
`false` (raw completion), which keeps the old behaviour.

Response fields: `response`, `tokens_generated`, `tokens_per_second`,
`inference_ms`, `queue_wait_ms` (time spent waiting for earlier requests),
`process_rss_mb`, `model`.

Configuration is by environment variable: `MODEL_PATH`, `N_THREADS`, `N_CTX`,
`HOST`, `PORT`, `MODELS_DIR`, `RESULTS_DIR`, `LLAMA_BIN_DIR`, `LLAMA_CPP_DIR`.

## Project structure

```
config.py                  shared paths, prompts, helpers (env-var overridable)
step1_setup.bat            venv + dependencies
step2_download_model.py    download raw model from Hugging Face
step3_quantize_sweep.py    convert to GGUF + quantize at every level
step4_benchmark.py         speed / RAM / thread-scaling orchestrator
bench_worker.py            child process that measures one model
step4b_perplexity.py       quality via llama-perplexity on WikiText-2
step5_api.py               FastAPI server
step6_report.py            tables + RESULTS.md      (report_plots.py = charts)
step11_serving_compare.py  FastAPI vs llama-server load test (step7_load_test.py takes --format/--url/--out)
step10_imatrix.py          importance-matrix experiment (imatrix_report.py = its RESULTS.md section)
step9_custom_quant.py      my quantizer: weight error + perplexity (custom_quant.py = the quantizer,
                           custom_quant_eval.py = HF evaluator + importance statistics,
                           custom_quant_export.py = save/load my quantized model as .safetensors,
                           gguf_compare.py, custom_quant_report.py)
step7_load_test.py         API concurrency test
step8_compare.py           live side-by-side demo server (compare_ui.html = the page,
                           compare_checks.py = automatic answer checks)
tests/                     pytest suite (no model needed)
results/                   generated: speed.json, perplexity.json, RESULTS.md, *.png
```

## Tests

```powershell
pytest          # no model or GPU needed (the HF tests use a tiny random model)
```

## Scope and next steps

- Results are for TinyLlama 1.1B on one 4-core CPU; the scripts take other models and hardware through environment variables, so the same experiments can be repeated elsewhere.
- Quality is measured with perplexity on WikiText-2. Task accuracy (lm-eval-harness: HellaSwag, ARC-Easy) is the natural next metric.
- Next steps: a calibration-size ablation for the imatrix (`step10_imatrix.py --chunks-list 10,50,200`), a second model of 1.5B to 3B parameters, and exporting my quantized format to GGUF so llama.cpp can run it.
