# FuseRT

**A graph compiler for deep learning inference, validated against TensorRT.**
FuseRT rewrites ONNX models with hand-written fusion passes, proves the outputs still match, and benchmarks the result against ONNX Runtime and TensorRT 11 on an RTX 3060.

**Headline results (BERT-base, FP32, RTX 3060 Laptop GPU)**

- **1231 to 599 graph nodes** from four passes: constant inlining, LayerNorm fusion, GELU fusion and QKV merging. Outputs match the original within 2.2e-5.
- **2.1x lower latency at batch 1** (15.68 to 7.39 ms) on ONNX Runtime CUDA with its optimizer off, and still **1.17x faster on top of ORT's own default optimizer**.
- **TensorRT 11.4 comparison:** engines built from the original and the rewritten graph both come out at 197 layers with matching layer types, so the passes reproduce fusions TensorRT already makes. TensorRT is **1.9x faster than ONNX Runtime at batch 1**.

## What this project demonstrates

| Skill | Evidence in this repo |
|---|---|
| Graph compiler algorithms | Subgraph pattern matching with safety checks (single-consumer, axis and constant validation) and rewrites that insert fused nodes, in [`fusert/optimize.py`](fusert/optimize.py) |
| Frontend operators | Rewrites to `LayerNormalization` (opset 17) and `Gelu` (opset 20), and a `MatMul` + `Add` + `Split` form for QKV |
| Performance analysis methodology | Warmup, GPU synchronization, p50/p95/p99, throughput, and a control run with ORT's optimizer on and off to separate what the passes add ([`bench/`](bench)) |
| Correctness | Output comparison against the original on four batch and sequence configurations, with ORT's optimizer disabled so the fused graph is really executed ([`fusert/verify.py`](fusert/verify.py)) |
| TensorRT and inference stack | Engines built through the TensorRT 11 Python API, with layer-by-layer analysis of what the builder produced ([`bench/build_trt.py`](bench/build_trt.py), [`bench/compare_layers.py`](bench/compare_layers.py)) |
| Deep learning frameworks | PyTorch to ONNX export, ONNX Runtime (CUDA), TensorRT |
| Python | End-to-end implementation |

## How it works

FuseRT loads an ONNX graph, runs a list of passes, and saves a smaller graph. Each pass finds a pattern, checks that it is safe to rewrite, and replaces it with one node. Every skipped match is logged with its reason.

| Pass | Pattern found | Rewritten to | Nodes removed (BERT-base) |
|---|---|---|---|
| `inline_constants` | `Constant` nodes | plain weights | 348 |
| `fuse_layernorm` | ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul, Add (9 nodes) | 1 `LayerNormalization` (25 groups) | 200 |
| `fuse_gelu` | Div, Erf, Add, Mul, Mul (5 nodes) | 1 `Gelu` (12 groups) | 48 |
| `merge_qkv` | three MatMul + bias Add reading one tensor | one wide MatMul + Add + `Split` (12 groups) | 36 |

Constant inlining is housekeeping. The three fusion passes account for 284 of the 632 removed nodes.

## Results

All numbers: RTX 3060 Laptop GPU (6 GB), FP32, sequence length 128, p50 latency in ms, single run on a laptop GPU, so differences of a few percent are noise.

**ONNX Runtime (CUDA)**

| batch | original, ORT optimizer off | FuseRT, ORT optimizer off | original, ORT default | FuseRT, ORT default |
|---|---|---|---|---|
| 1 | 15.68 | **7.39** | 7.15 | **6.12** |
| 8 | 36.68 | **29.38** | 27.01 | 27.19 |
| 32 | 130.32 | **106.10** | 96.78 | 99.52 |

**TensorRT 11.4 (engines built from each ONNX file)**

| batch | original | FuseRT output | ONNX Runtime default (for reference) |
|---|---|---|---|
| 1 | 3.83 | 3.93 | 7.15 |
| 8 | 19.98 | 20.33 | 27.01 |
| 32 | 78.17 | 77.96 | 96.78 |

## Findings

1. **The passes help most when nothing else is optimizing.** With ONNX Runtime's optimizer off, latency drops 2.1x at batch 1 and about 1.2x at batch 8 and 32. Batch 1 is most likely limited by kernel launch overhead, so halving the node count matters most there.
2. **Production optimizers already find most of these fusions.** With ORT's default optimizer on, FuseRT still adds 1.17x at batch 1 and is within noise at batch 8 and 32.
3. **TensorRT performs the same fusions itself.** Engines built from the original and the rewritten graph have 197 layers each, with matching type counts (75 generated kernels, 73 GEMMs, 24 casts, 24 reshapes, 1 shape call). The 73 GEMMs match what BERT-base needs with Q, K and V computed together (12 layers x 6 + 1). Timings of the two engines agree to within noise.
4. **TensorRT is the fastest backend here,** 1.9x faster than ONNX Runtime at batch 1 and 1.2x to 1.35x faster at larger batches.

## Correctness

[`fusert/verify.py`](fusert/verify.py) runs the original and optimized models with ONNX Runtime's graph optimizer disabled, so both graphs run as written. Four configurations: batch 1 to 8, sequence length 64 to 256, one with padded attention masks. Maximum absolute error is 2.19e-05 against a tolerance of 1e-4.

## Reproduce

```
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

Raw output of every run is saved in [`results/logs/`](results/logs).

**Environment:** WSL2 Ubuntu, Python 3.12, PyTorch 2.11 (export only), ONNX Runtime GPU, TensorRT 11.4.0.106, NVIDIA RTX 3060 Laptop GPU.

## Layout

```
fusert/    optimize.py (passes), verify.py, stats.py, inspect_graph.py
models/    export_bert.py
bench/     build_trt.py, compare_layers.py, bench_ort.py, bench_trt.py
results/   logs and TensorRT layer dumps
```