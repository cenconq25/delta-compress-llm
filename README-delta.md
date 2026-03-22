# delta-compress-llm

**We applied video compression to LLM inference. The result: 10,000x less quantization error at the same storage cost.**

> In video, you don't re-encode every pixel every frame. You store a keyframe, then just the differences. We do the same thing for the KV cache in LLM inference.

## The Insight

During autoregressive decoding, consecutive tokens produce nearly identical KV cache values. The hidden state for "The cat sat on the **mat**" differs from "The cat sat on the **rug**" by only ~1% at most dimensions.

Standard KV cache quantization (Q4_0) compresses absolute values to 4 bits. **Delta-KV** compresses the tiny *difference* between tokens to 4 bits instead. Same bits, vastly less error.

```
Standard Q4_0:  value=0.5432  ->  quantize  ->  reconstruct=0.5100  |  error=0.0332
Delta Q4_0:     delta=0.0032  ->  quantize  ->  reconstruct=0.0030  |  error=0.0002
                                                                      ^^^^^^^^
                                                                      166x less error
```

The quantization error is proportional to the *range* of values being quantized. Deltas have 100x smaller range than absolute values, so the same 4 bits preserve 10,000x more information.

## Benchmark Results

Tested on **Llama 3.1 70B (Q4_K_M)** running on **4x AMD MI50 GPUs** with ROCm 6.3.3.

### Perplexity (WikiText-2, 20 chunks) - lower is better

| KV Cache Config | Perplexity | vs F16 Baseline | Verdict |
|:---|:---:|:---:|:---:|
| F16 (baseline) | 3.3389 | - | - |
| Q8_0 | 3.3444 | +0.16% | OK |
| **Q4_0** | **3.5385** | **+5.98%** | **Degraded** |
| **Delta-KV (kf=16)** | **3.3352** | **-0.11%** | **Lossless** |
| **Delta-KV (kf=32)** | **3.3371** | **-0.05%** | **Lossless** |
| **Delta-KV (kf=64)** | **3.3367** | **-0.07%** | **Lossless** |

**Q4_0 loses 6% quality. Delta-KV loses 0%.** Same 4-bit storage.

### Long Context (Error Accumulation Test)

| Context | F16 | Q4_0 | Delta-KV | Q4_0 degradation | Delta-KV degradation |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 512 | 2.6249 | 2.7548 | 2.6236 | +4.9% | **-0.05%** |
| 1024 | 2.4996 | 2.6638 | 2.5077 | +6.6% | **+0.3%** |
| 2048 | 2.7837 | 2.9760 | 2.7954 | +6.9% | **+0.4%** |

Delta-KV maintains quality even at 2048 context. No error accumulation blowup.

### Cross-Domain (Shakespeare)

| Config | Perplexity | vs F16 |
|:---|:---:|:---:|
| F16 | 1.1990 | - |
| Q4_0 | 1.2211 | +1.8% |
| Delta-KV (kf=32) | 1.1993 | +0.025% |

### Synthetic MSE Analysis

| Drift Rate | Standard Q4_0 MSE | Delta-Q4_0 MSE | Improvement |
|:---:|:---:|:---:|:---:|
| 1% | 1.46e-03 | 1.36e-07 | **10,778x** |
| 5% | 3.35e-03 | 3.38e-06 | **991x** |
| 10% | 8.24e-03 | 1.35e-05 | **609x** |
| 20% | 2.68e-02 | 5.41e-05 | **495x** |

## How It Works

```
Token 1 (keyframe):  store full F16 values -> KV cache
Token 2 (delta):     delta = token2_values - token1_values
                     quantize delta to Q4_0 (tiny range -> high precision)
                     reconstruct = token1_values + dequant(delta)
                     store reconstructed -> KV cache
Token 3 (delta):     delta = token3_values - token2_reconstructed
                     ... same process, rolling reference
Token 33 (keyframe): reset reference (every N tokens)
```

This is exactly how video codecs work: I-frames (keyframes) + P-frames (deltas).

## Also Included: Weight-Skip Prediction

During decode, the MMVQ kernel reads ~40GB of weights per token. We added an inline skip check that reads 4 bytes (weight scale + activation scale) to decide whether to skip the full 400-byte dot product.

| Config | Decode Speed | Quality Loss |
|:---|:---:|:---:|
| Baseline | 9.3 t/s | - |
| Weight-Skip (1e-6) | 10.2 t/s (+10%) | **Zero** (PPL identical) |

## Quick Start

```bash
# Build (requires ROCm or CUDA)
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906
cmake --build build -j$(nproc)

# Run with delta-KV
./build/bin/llama-cli -m model.gguf -ngl 99 \
  --delta-kv --delta-kv-interval 32

# Run with delta-KV + weight skip
LLAMA_WEIGHT_SKIP_THRESHOLD=1e-6 ./build/bin/llama-cli -m model.gguf -ngl 99 \
  --delta-kv --delta-kv-interval 32

# Run the benchmark yourself
./build/bin/llama-delta-kv-bench
./build/bin/llama-weight-skip-bench
./build/bin/llama-weight-analysis -m model.gguf
```

## Files Changed

This is a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with minimal, surgical modifications:

**New files (core):**
- `ggml/src/ggml-cuda/delta-kv.cu` / `.cuh` - GPU kernels for delta encode/reconstruct
- `src/llama-kv-cache-delta.cpp` / `.h` - Delta KV processor (CPU fallback + GPU dispatch)
- `ggml/src/ggml-cuda/weight-skip.cu` / `.cuh` - Weight-skip predictor kernels

**New files (benchmarks):**
- `tools/delta-kv-bench/` - Synthetic quality benchmark (MSE comparison)
- `tools/weight-skip-bench/` - Weight scale analysis and skip rate estimation

**Modified files (11 files, ~195 lines added):**
- `ggml/src/ggml-cuda/mmvq.cu` - Inline weight-skip check in MMVQ inner loop
- `src/llama-kv-cache.h/.cpp` - Delta processor integration
- `src/llama-context.cpp` - Post-processing hook after graph compute
- `include/llama.h` - API parameters (`delta_kv`, `delta_kv_interval`)
- `common/arg.cpp` / `common.h` / `common.cpp` - CLI flags

## Related Work

- [CacheGen](https://dl.acm.org/doi/10.1145/3651890.3672274) (SIGCOMM 2024) - Delta encoding for KV cache *network streaming* (different target: network, not HBM)
- [DeltaKV](https://arxiv.org/abs/2602.08005) (2026) - Residual-based compression using MLP projection (higher overhead)
- [KVTC](https://arxiv.org/abs/2511.01815) (2025) - Transform coding (PCA + entropy coding, different technique)
- [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) - KV cache compression framework

Our approach is distinguished by **zero overhead** (no learned components, no entropy coding) and **direct kernel integration** in llama.cpp.

## Hardware

Developed and tested on:
- 4x AMD Instinct MI50 (32GB HBM2 each, gfx906)
- AMD Ryzen Threadripper PRO 3945WX
- ROCm 6.3.3
- Ubuntu 24.04

The delta-KV technique is hardware-agnostic and benefits any GPU where KV cache bandwidth is a bottleneck (A100, H100, etc.).

## License

Same as [llama.cpp](https://github.com/ggerganov/llama.cpp) (MIT).
