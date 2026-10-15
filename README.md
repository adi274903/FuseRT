# FuseRT

A small graph optimizer for ONNX models. It rewrites BERT-base with hand-written fusion passes, checks that the outputs still match, and compares the result against ONNX Runtime and TensorRT 11.

## What it does

FuseRT loads an ONNX graph, runs a list of passes, and saves a smaller graph. Each pass finds a pattern, checks that it is safe to rewrite, and replaces it with one node.

| Pass | Pattern | Result on BERT-base | Nodes removed |
|---|---|---|---|
| inline_constants | `Constant` nodes | turned into plain weights | 348 |
| fuse_layernorm | ReduceMean, Sub, Pow, ReduceMean, Add, Sqrt, Div, Mul, Add | 25 `LayerNormalization` | 200 |
| fuse_gelu | Div, Erf, Add, Mul, Mul | 12 `Gelu` | 48 |
| merge_qkv | 3 x (MatMul + Add) on the same input | 12 x (MatMul + Add + Split) | 36 |

BERT-base goes from 1231 to 599 nodes. Constant inlining is housekeeping, so the fusion passes account for 284 of the 632 removed nodes.

## Correctness

`fusert/verify.py` runs the original and optimized models with ONNX Runtime's own optimizer disabled, on four input configurations (batch 1 to 8, sequence length 64 to 256, one with padding). Maximum absolute error is 2.19e-05, against a tolerance of 1e-4.

## Results

RTX 3060 Laptop GPU (6 GB), FP32, sequence length 128, p50 latency in ms, single run.

**ONNX Runtime (CUDA)**

| batch | original, ORT optimizer off | optimized, ORT optimizer off | original, ORT default | optimized, ORT default |
|---|---|---|---|---|
| 1 | 15.68 | 7.39 | 7.15 | 6.12 |
| 8 | 36.68 | 29.38 | 27.01 | 27.19 |
| 32 | 130.32 | 106.10 | 96.78 | 99.52 |

**TensorRT 11.4 (engines built from each ONNX file)**

| batch | original | optimized |
|---|---|---|
| 1 | 3.83 | 3.93 |
| 8 | 19.98 | 20.33 |
| 32 | 78.17 | 77.96 |

## Findings

- With ONNX Runtime's optimizer off, the passes cut latency 2.1x at batch 1 and about 1.2x at batch 8 and 32. Batch 1 benefits most, probably because it is dominated by kernel launch overhead and the graph has half as many nodes.
- With ONNX Runtime's default optimizer on, the gain shrinks to 1.17x at batch 1 and is within run-to-run noise at batch 8 and 32. ONNX Runtime already applies similar fusions.
- TensorRT builds engines with 197 layers from both the original and the optimized graph, with identical layer type counts. The engine layer names show it already merged the Q, K and V matrix multiplications on the original graph. FuseRT reproduces fusions TensorRT does itself, and does not add new ones for TensorRT.
- TensorRT is about 1.9x faster than ONNX Runtime at batch 1 and 1.2x to 1.35x faster at larger batches.

## Limitations

- BERT-base only, FP32 only, fixed sequence length 128 in the TensorRT engines.
- Timings are single runs on a laptop GPU, so differences of a few percent are noise.
- The `Gelu` op needs ONNX opset 20.
- No FP16 or INT8 yet. TensorRT 11 builds strongly typed networks, so an FP16 engine needs an FP16 ONNX file.

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

Raw output of every run is in `results/logs/`.

## Environment

WSL2 Ubuntu on Windows, Python 3.12, PyTorch 2.11 (export only), ONNX Runtime GPU, TensorRT 11.4.0.106, NVIDIA RTX 3060 Laptop GPU.

## Layout

```
fusert/    optimize.py (passes), verify.py, stats.py, inspect_graph.py
models/    export_bert.py
bench/     build_trt.py, compare_layers.py, bench_ort.py, bench_trt.py
results/   logs and TensorRT layer dumps
```

## Next

- Fused LayerNorm CUDA kernel, benchmarked against the unfused ops
- FP16 engines
- Constant folding for the shape-computation nodes