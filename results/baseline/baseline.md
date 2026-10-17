# FuseRT v1.0 baseline

Frozen before the FuseRT 2.0 work (custom residual + LayerNorm plugin).
Hardware: RTX 3060 Laptop GPU (6 GB). FP32, sequence length 128, BERT-base. p50 latency in ms, single run.

## Graph
- Original ONNX: 1231 nodes
- After FuseRT passes: 599 nodes (25 LayerNorm, 12 GELU, 12 QKV groups fused)
- Max absolute error vs original: 2.19e-05 (4 configurations, ORT optimizer disabled)

## TensorRT 11.4 engines
- Original: 197 layers, 438.8 MB, build 12.0 s
- FuseRT output: 197 layers, 438.8 MB, build 15.1 s
- Layer type counts identical: 75 kgen, 73 gemm, 24 cast, 24 reshape, 1 shape_call

## ONNX Runtime (CUDA)
| batch | original, ORT off | FuseRT, ORT off | original, ORT default | FuseRT, ORT default |
|---|---|---|---|---|
| 1 | 15.68 | 7.39 | 7.15 | 6.12 |
| 8 | 36.68 | 29.38 | 27.01 | 27.19 |
| 32 | 130.32 | 106.10 | 96.78 | 99.52 |

## TensorRT
| batch | original | FuseRT output |
|---|---|---|
| 1 | 3.83 | 3.93 |
| 8 | 19.98 | 20.33 |
| 32 | 78.17 | 77.96 |

Raw logs and TensorRT layer dumps are in this folder.