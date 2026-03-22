#pragma once

#include "common.cuh"

// Weight Skip Predictor for Delta Compression
//
// During decode, the MMVQ kernel reads ~40GB of weights per token.
// Most of these reads are wasted when the activation vector is sparse
// or when specific dimensions haven't changed much from the previous token.
//
// The predictor computes a fast approximation of each weight block's
// contribution to the output. Blocks below a threshold are skipped.
//
// Key insight: for Q4_K weights, each super-block has a scale factor (d).
// The dot product contribution is proportional to |d| × |activation_norm|.
// If either is small, the block can be safely skipped.

// Compute skip mask for weight blocks
// For each block, estimates contribution = |weight_scale| × |activation_rms|
// Blocks below threshold are marked for skipping
void ggml_cuda_weight_skip_predict(
    const void  * weight_data,      // Q4_K weight tensor data (GPU)
    const float * activation_data,  // F32 activation vector (GPU), already quantized as Q8_1
    uint8_t     * skip_mask,        // Output: 1 = skip, 0 = compute (GPU) [n_blocks]
    const int64_t n_blocks,         // number of weight super-blocks per row
    const int64_t n_rows,           // number of output rows (weight matrix rows)
    const float   threshold,        // skip threshold (0.0 = skip nothing, 1.0 = skip everything)
    cudaStream_t stream);

// Modified MMVQ that respects skip mask
// Same as mul_mat_vec_q but skips weight blocks marked in skip_mask
void ggml_cuda_mul_mat_vec_q_skip(
    const void  * vx,               // Q4_K weight data
    const void  * vy,               // Q8_1 activation data
    float       * dst,              // output
    const uint8_t * skip_mask,      // skip mask [blocks_per_row]
    const int64_t ncols,            // weight matrix columns
    const int64_t nrows,            // weight matrix rows
    const float   skip_threshold,   // dynamic threshold
    cudaStream_t stream);
