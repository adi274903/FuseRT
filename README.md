# FuseRT

**A graph compiler and performance-analysis project for deep learning inference.**

FuseRT rewrites ONNX models with hand-written fusion passes, verifies that outputs remain correct, and compares execution in ONNX Runtime and TensorRT 11 on an NVIDIA Laptop GPU.

FuseRT 2.0 extends the project from standard ONNX rewrites toward backend-aware optimization: it profiles the TensorRT engine, identifies BERT's repeated attention path, validates a reference implementation, and prepares the path toward a custom CUDA/TensorRT attention plugin.

## Headline results

**BERT-base, FP32, RTX 3060 Laptop GPU**

- Reduced the exported ONNX graph from **1,231 to 599 nodes** through constant inlining, LayerNorm fusion, GELU fusion, and QKV merging.
- Verified optimized-model outputs against the original with maximum absolute error of **2.19e-5**.
- Achieved **2.1x lower batch-1 latency** in ONNX Runtime CUDA with ORT graph optimization disabled: **15.68 ms -> 7.39 ms**.
- Compared original and FuseRT-rewritten ONNX models in TensorRT 11.4. Both compiled to **197 TensorRT layers**, showing that TensorRT already performs many of the same standard fusions.
- Profiled the original TensorRT engine and found that the repeated BERT attention path, `Q x K^T -> masked Softmax -> scores x V`, accounts for **1.586 ms**, or **23.0% of profiled layer time**, at batch 1 and sequence length 128.
- Built and tested an ONNX attention-pattern matcher that finds all **12 BERT attention blocks** with zero skipped Softmax nodes.

## What this project demonstrates

| Technical focus | Implementation details |
|---|---|
| Graph compiler algorithms | Subgraph pattern matching with safety checks and rewrites that insert fused nodes in [`fusert/optimize.py`](fusert/optimize.py) |
| Frontend operators | Rewrites to `LayerNormalization` and `Gelu`, plus a `MatMul` + `Add` + `Split` representation for QKV |
| Correctness | Original-versus-optimized output comparison using ONNX Runtime with graph optimization disabled |
| Performance methodology | GPU warmup, synchronization, p50/p95/p99 latency, throughput, and ORT optimizer-on/off controls |
| TensorRT analysis | Engine construction, engine-layer comparison, layer-by-layer profiling, generated-kernel breakdown, and attention-path analysis |
| Custom-op groundwork | Residual + LayerNorm matcher, fused-attention reference implementation, attention-pattern matcher, unit tests, CUDA/CMake toolchain setup |
| Frameworks and tools | PyTorch, ONNX, ONNX Runtime CUDA, TensorRT, CUDA, CMake, Python |

## How FuseRT v1.0 works

FuseRT loads an ONNX graph, runs a list of passes, and saves a smaller graph. Each pass finds a pattern, checks that it is safe to rewrite, and replaces the pattern with a smaller equivalent representation.

| Pass | Pattern found | Rewritten to | Nodes removed on BERT-base |
|---|---|---|---:|
| `inline_constants` | `Constant` nodes | plain weights | 348 |
| `fuse_layernorm` | ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul, Add | one `LayerNormalization` node | 200 |
| `fuse_gelu` | Div, Erf, Add, Mul, Mul | one `Gelu` node | 48 |
| `merge_qkv` | three MatMul + bias Add operations reading the same tensor | one wide MatMul + Add + `Split` | 36 |

The graph changes from:

```text
1,231 nodes
-> 883 after constant inlining
-> 683 after LayerNorm fusion
-> 635 after GELU fusion
-> 599 after QKV merging
```

The three fusion passes account for 284 of the 632 removed nodes. Constant inlining removes graph housekeeping nodes.

## v1.0 results

All numbers use an RTX 3060 Laptop GPU, FP32, sequence length 128, and p50 latency in milliseconds. Small percentage differences on a laptop GPU should be treated as noise.

### ONNX Runtime CUDA

| Batch | Original, ORT optimizer off | FuseRT, ORT optimizer off | Original, ORT default | FuseRT, ORT default |
|---:|---:|---:|---:|---:|
| 1 | 15.68 | **7.39** | 7.15 | **6.12** |
| 8 | 36.68 | **29.38** | 27.01 | 27.19 |
| 32 | 130.32 | **106.10** | 96.78 | 99.52 |

### TensorRT 11.4

| Batch | Original ONNX engine | FuseRT ONNX engine | ONNX Runtime default, for reference |
|---:|---:|---:|---:|
| 1 | **3.83** | 3.93 | 7.15 |
| 8 | **19.98** | 20.33 | 27.01 |
| 32 | 78.17 | **77.96** | 96.78 |

## v1.0 findings

1. **FuseRT helps most when another optimizer is not already rewriting the graph.** With ONNX Runtime optimization disabled, batch-1 latency drops from 15.68 ms to 7.39 ms.
2. **Production runtimes already discover many standard fusions.** With ORT's default optimizer enabled, FuseRT adds a measurable benefit at batch 1 and is within noise at larger batches.
3. **TensorRT already performs equivalent standard fusions.** Engines built from the original and FuseRT-rewritten ONNX graphs both contain 197 TensorRT layers with matching layer-type counts.
4. **TensorRT is the fastest backend in this baseline.** At batch 1, the original TensorRT engine runs in 3.83 ms, compared with 7.15 ms for ONNX Runtime CUDA with its default optimizer.

## Correctness

[`fusert/verify.py`](fusert/verify.py) runs the original and optimized models with ONNX Runtime graph optimization disabled, so both graphs execute as written.

The verification covers four batch and sequence-length configurations, including padded attention masks.

```text
Maximum absolute error: 2.19e-5
Tolerance:               1e-4
Result:                  PASS
```

## FuseRT 2.0: custom-op groundwork

FuseRT 2.0 keeps the v1.0 baseline frozen and adds custom-operation research on top of it. The goal is not to claim that a custom plugin already beats TensorRT. The goal is to use profiling and correctness checks to identify a target that is large enough to justify custom CUDA work.

### Frozen baseline

The v1.0 results are saved in [`results/baseline/`](results/baseline) and tagged as `v1.0-baseline`.

This includes graph statistics, correctness results, TensorRT build logs, TensorRT layer dumps, ONNX Runtime benchmarks, TensorRT benchmarks, and environment information.

Summary: [`results/baseline/baseline.md`](results/baseline/baseline.md).

## Residual + LayerNorm exploration

FuseRT 2.0 first explored a custom operation named `FusedResidualLayerNorm`.

It would compute:

```text
LayerNorm(hidden_state + residual) * gamma + beta
```

without writing the intermediate `hidden_state + residual` tensor to GPU memory.

### Matcher results

[`fusert/fuse_residual_layernorm.py`](fusert/fuse_residual_layernorm.py) finds every `Add -> LayerNormalization` pair and checks whether it is safe to fuse.

A site is considered safe when:

- both `Add` inputs are float32 tensors
- both inputs have provably equal shapes
- the `Add` does not contain a constant bias input
- the summed tensor has only one consumer
- LayerNorm normalizes the final dimension
- gamma and beta are constants

On BERT-base:

| Item | Count |
|---|---:|
| `LayerNormalization` nodes | 25 |
| Safe residual + LayerNorm matches | 24 |
| Skipped matches | 1 |

The skipped case is the embedding LayerNorm, where position embeddings use broadcasting and therefore do not satisfy the same-shape safety requirement.

### Residual + LayerNorm reference and tests

[`fusert/reference_residual_ln.py`](fusert/reference_residual_ln.py) provides a trusted Python reference implementation.

Tests cover a valid residual add, a bias add that must be skipped, a broadcast add that must be skipped, an intermediate sum with another consumer, and numerical agreement with ONNX Runtime LayerNormalization.

The future CUDA/TensorRT plugin for this operation has not yet been implemented.

## TensorRT engine profiling

Before writing custom CUDA, FuseRT profiles the original TensorRT engine to determine where runtime is spent.

At batch 1 and sequence length 128:

```text
TensorRT engine layers: 197
Profiled layer-time sum: 6.901 ms
End-to-end TensorRT p50: 3.83 ms
```

The sum of per-layer profiler measurements is larger than end-to-end latency because profiling adds overhead. Therefore, these measurements are used for **relative shares and target selection**, not as direct latency measurements.

### Layer-type breakdown

| TensorRT layer type | Count | Total profiled time | Share |
|---|---:|---:|---:|
| GEMM | 73 | 4.925 ms | 71.4% |
| TensorRT-generated kernel (`kgen`) | 75 | 1.589 ms | 23.0% |
| Cast | 24 | 0.220 ms | 3.2% |
| Reshape | 24 | 0.149 ms | 2.2% |
| Shape call | 1 | 0.019 ms | 0.3% |

The 15 slowest individual layers are GEMMs, but no single GEMM dominates total runtime. A useful optimization therefore needs to target a repeated group of operations rather than one isolated layer.

### Generated-kernel breakdown

Among the 75 TensorRT-generated kernels:

| Category | Count | Total profiled time | Share of profiled time |
|---|---:|---:|---:|
| LayerNorm and residual-related kernels | 38 | 0.545 ms | 7.9% |
| Transpose and other kernels | 13 | 0.492 ms | 7.1% |
| Softmax | 12 | 0.417 ms | 6.0% |
| GELU | 12 | 0.134 ms | 1.9% |

This showed that Residual + LayerNorm alone is too small to be the only target for a large TensorRT speedup.

## Attention as the next optimization target

BERT repeats the following attention computation in each of its 12 encoder layers:

```text
Q x K^T
-> attention scaling and mask
-> Softmax
-> attention probabilities x V
```

FuseRT profiles and validates this complete path before attempting a plugin.

### Attention timing breakdown

[`bench/attention_breakdown.py`](bench/attention_breakdown.py) separates TensorRT attention-related operations into individual categories.

| Attention-related category | Count | Total profiled time | Share of profiled time |
|---|---:|---:|---:|
| QKV projection GEMMs | 12 | 0.995 ms | 14.4% |
| Q x K^T GEMMs | 12 | 0.658 ms | 9.5% |
| Softmax | 12 | 0.417 ms | 6.0% |
| Attention scores x V GEMMs | 12 | 0.511 ms | 7.4% |
| Attention output projection GEMMs | 12 | 0.993 ms | 14.4% |
| Transpose candidates | 24 | 0.677 ms | 9.8% |

The clean central attention target is:

```text
Q x K^T
+ Softmax
+ attention scores x V
= 1.586 ms
= 23.0% of profiled TensorRT layer time
```

This does **not** mean a future plugin will remove 23% of end-to-end latency. It still performs the same mathematical work. The potential improvement comes from reducing intermediate memory traffic and combining separate GPU launches.

The transpose candidates add up to a larger possible group, but they require further graph-level validation before being included in a custom plugin.

### Attention tensor-shape inspection

[`bench/inspect_attention_layers.py`](bench/inspect_attention_layers.py) verifies the inputs and outputs of the selected TensorRT layers.

The inspection confirms:

```text
Q x K^T:
[batch, 128, 64] x [batch, 64, 128]
-> [batch, 128, 128]

Softmax:
attention scores + attention_mask
-> attention probabilities

scores x V:
[batch, 128, 128] x [batch, 128, 64]
-> [batch, 128, 64]
```

BERT-base uses 12 encoder layers, 12 attention heads, sequence length 128, and head size 64.

## Fused-attention reference implementation

[`fusert/reference_fused_attention.py`](fusert/reference_fused_attention.py) implements a correct PyTorch reference for masked scaled dot-product attention:

```text
Q x K^T
-> divide by sqrt(head_size)
-> apply padding mask
-> Softmax
-> probabilities x V
```

The reference is intentionally not an optimization. It is the trusted answer that a future CUDA/TensorRT plugin must match.

[`tests/test_reference_fused_attention.py`](tests/test_reference_fused_attention.py) compares the FuseRT reference against PyTorch `scaled_dot_product_attention`.

```text
Result: 1 passed
```

## Attention-pattern matcher

[`fusert/fuse_attention.py`](fusert/fuse_attention.py) scans the optimized ONNX graph and finds this pattern:

```text
MatMul
-> Softmax
-> MatMul
```

It traces backward from each Softmax to find `Q x K^T`, then forward to find `scores x V`.

On `bert_opt.onnx`:

```text
attention matches: 12
skipped Softmax nodes: 0
```

One complete pattern is found in every BERT encoder layer, from `layer.0` through `layer.11`.

[`tests/test_attention_matcher.py`](tests/test_attention_matcher.py) verifies that a complete attention pattern is found and an incomplete pattern is skipped.

```text
Result: 2 passed
```

## CUDA and TensorRT C++ development setup

The custom plugin requires native CUDA and TensorRT C++ tooling, not only the TensorRT Python package.

The development environment has been prepared with:

```text
WSL2 Ubuntu 24.04.3
CMake 3.28.3
CUDA Toolkit 13.4.3
nvcc 13.4.92
g++ 13.3.0
NVIDIA RTX 3060 Laptop GPU
```

The next implementation stage is a TensorRT C++ plugin and CUDA kernel for the validated attention path. No custom CUDA attention kernel has been benchmarked yet.

## Reproduce

### v1.0 graph optimization and benchmarking

```bash
python models/export_bert.py
python -m fusert.stats models/bert.onnx
python -m fusert.optimize models/bert.onnx models/bert_opt.onnx
python -m fusert.verify models/bert.onnx models/bert_opt.onnx

python bench/build_trt.py models/bert.onnx models/bert_fp32.plan results/trt_original_layers.json
python bench/build_trt.py models/bert_opt.onnx models/bert_opt_fp32.plan results/trt_optimized_layers.json
python bench/compare_layers.py results/trt_original_layers.json results/trt_optimized_layers.json
python bench/bench_ort.py
python bench/bench_trt.py
```

### FuseRT 2.0 analysis

```bash
python -m fusert.fuse_residual_layernorm models/bert_opt.onnx

python bench/profile_trt_layers.py \
  models/bert_fp32.plan \
  results/trt_original_layers.json \
  1 \
  results/profile_original_b1.json

python bench/kgen_breakdown.py \
  results/profile_original_b1.json \
  results/trt_original_layers.json \
  results/kgen_breakdown_b1.json

python bench/attention_breakdown.py \
  results/profile_original_b1.json \
  results/trt_original_layers.json \
  results/attention_breakdown_b1.json

python bench/inspect_attention_layers.py \
  results/attention_breakdown_b1.json \
  results/trt_original_layers.json \
  results/attention_layer_shapes_b1.json

python fusert/fuse_attention.py \
  models/bert_opt.onnx \
  results/attention_pattern_matches.json

PYTHONPATH=. python -m pytest -q tests
```

Raw output from every run is saved in [`results/logs/`](results/logs). Structured profiling and inspection results are saved in [`results/`](results).

## Repository layout

```text
fusert/
  optimize.py                      v1.0 ONNX graph fusion passes
  verify.py                        original-versus-optimized correctness check
  stats.py                         graph-node statistics
  inspect_graph.py                 graph inspection helpers
  fuse_residual_layernorm.py       residual + LayerNorm matcher
  reference_residual_ln.py         residual + LayerNorm reference
  fuse_attention.py                ONNX attention-pattern matcher
  reference_fused_attention.py     masked scaled-attention reference

bench/
  build_trt.py                     TensorRT engine builder
  compare_layers.py                original-versus-optimized TensorRT layer comparison
  bench_ort.py                     ONNX Runtime benchmark
  bench_trt.py                     TensorRT benchmark
  profile_trt_layers.py            TensorRT per-layer profiler
  kgen_breakdown.py                generated-kernel timing breakdown
  attention_candidates.py          initial broad attention candidate scan
  attention_breakdown.py           exact attention timing breakdown
  inspect_attention_layers.py      attention input/output shape inspection

plugins/
  fused_residual_layernorm.h       residual + LayerNorm kernel interface

docs/
  plugin_interface.md              residual + LayerNorm TensorRT plugin contract

tests/
  test_residual_ln_matcher.py
  test_reference_residual_ln.py
  test_attention_matcher.py
  test_reference_fused_attention.py

results/
  baseline/                        frozen v1.0 baseline
  logs/                            saved command output
  *.json                           saved profiling and inspection results
```
