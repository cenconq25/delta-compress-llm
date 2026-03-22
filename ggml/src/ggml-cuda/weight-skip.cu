#include "weight-skip.cuh"
#include "ggml-common.h"

// ============================================================================
// Weight Skip Predictor Kernel
//
// For each weight super-block (Q4_K = 256 elements, 144 bytes):
//   - Read the block's scale factor: dm.x (2 bytes)
//   - Read the corresponding activation's RMS (precomputed)
//   - If |scale| × |act_rms| < threshold → skip
//
// This is MUCH cheaper than the full dot product:
//   - Full dot: read 144 bytes weight + 256 bytes activation = 400 bytes
//   - Predictor: read 2 bytes weight scale + 4 bytes activation RMS = 6 bytes
//   - That's 67x less data read for the prediction
// ============================================================================

#define WEIGHT_SKIP_BLOCK_SIZE 256

// Kernel 1: Compute per-Q4_K-block activation RMS
// Input: Q8_1 quantized activations
// Output: RMS magnitude per super-block (one float per 256-element group)
__global__ void k_compute_activation_block_norms(
    const block_q8_1 * __restrict__ activations,
    float * __restrict__ block_norms,
    const int64_t n_blocks_q8,     // total Q8_1 blocks in activation vector
    const int64_t blocks_per_super) // Q8_1 blocks per Q4_K super-block (256/32 = 8)
{
    const int64_t super_idx = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t n_super_blocks = n_blocks_q8 / blocks_per_super;

    if (super_idx >= n_super_blocks) return;

    // Sum the absolute scale factors of Q8_1 blocks in this super-block
    // This approximates the activation magnitude without full dequantization
    float sum_d = 0.0f;
    for (int i = 0; i < (int)blocks_per_super; i++) {
        const int64_t q8_idx = super_idx * blocks_per_super + i;
        if (q8_idx < n_blocks_q8) {
            sum_d += fabsf(__low2float(activations[q8_idx].ds));
        }
    }

    block_norms[super_idx] = sum_d / blocks_per_super;
}

// Kernel 2: Compute skip mask
// For each weight block, check if |weight_scale| × |activation_norm| < threshold
__global__ void k_compute_skip_mask(
    const void    * __restrict__ weight_data,  // Q4_K blocks
    const float   * __restrict__ act_norms,    // per-super-block activation norms
    uint8_t       * __restrict__ skip_mask,    // output: 1=skip, 0=compute
    const int64_t blocks_per_row,
    const int64_t n_rows,
    const float   threshold)
{
    const int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t total = blocks_per_row * n_rows;

    if (tid >= total) return;

    const int64_t row = tid / blocks_per_row;
    const int64_t col_block = tid % blocks_per_row;

    GGML_UNUSED(row);

    // Read weight super-block scale
    const block_q4_K * w = (const block_q4_K *)weight_data + tid;
    const float w_scale = fabsf(__half2float(__low2half(w->dm)));

    // Read activation norm for this column block
    const float a_norm = act_norms[col_block];

    // Estimate contribution
    const float contribution = w_scale * a_norm;

    // Mark for skipping if below threshold
    skip_mask[tid] = (contribution < threshold) ? 1 : 0;
}

// ============================================================================
// Host dispatch
// ============================================================================

void ggml_cuda_weight_skip_predict(
    const void  * weight_data,
    const float * activation_data,
    uint8_t     * skip_mask,
    const int64_t n_blocks,
    const int64_t n_rows,
    const float   threshold,
    cudaStream_t stream)
{
    // For now, compute activation norms from the raw Q8_1 data
    const int64_t blocks_per_row = n_blocks / n_rows;

    // Allocate temp buffer for activation block norms
    float * d_act_norms;
    CUDA_CHECK(cudaMalloc(&d_act_norms, blocks_per_row * sizeof(float)));

    // Step 1: Compute activation block norms
    // Q4_K super-block = 256 elements = 8 Q8_1 blocks
    const int64_t blocks_per_super = 8; // QK_K / QK8_1 = 256/32
    {
        const int n = (blocks_per_row + WEIGHT_SKIP_BLOCK_SIZE - 1) / WEIGHT_SKIP_BLOCK_SIZE;
        k_compute_activation_block_norms<<<n, WEIGHT_SKIP_BLOCK_SIZE, 0, stream>>>(
            (const block_q8_1 *)activation_data,
            d_act_norms,
            blocks_per_row * blocks_per_super,  // total Q8_1 blocks
            blocks_per_super);
    }

    // Step 2: Compute skip mask
    {
        const int64_t total = n_blocks;
        const int n = (total + WEIGHT_SKIP_BLOCK_SIZE - 1) / WEIGHT_SKIP_BLOCK_SIZE;
        k_compute_skip_mask<<<n, WEIGHT_SKIP_BLOCK_SIZE, 0, stream>>>(
            weight_data, d_act_norms, skip_mask,
            blocks_per_row, n_rows, threshold);
    }

    cudaFree(d_act_norms);
}

// ============================================================================
// Benchmark kernel: measure skip rate for a given threshold
// Useful for tuning the threshold parameter
// ============================================================================

__global__ void k_count_skippable(
    const void    * __restrict__ weight_data,
    const float   * __restrict__ act_norms,
    int           * __restrict__ skip_count,
    const int64_t blocks_per_row,
    const int64_t n_rows,
    const float   threshold)
{
    const int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t total = blocks_per_row * n_rows;

    if (tid >= total) return;

    const int64_t col_block = tid % blocks_per_row;
    const block_q4_K * w = (const block_q4_K *)weight_data + tid;
    const float w_scale = fabsf(__half2float(__low2half(w->dm)));
    const float a_norm = act_norms[col_block];

    if (w_scale * a_norm < threshold) {
        atomicAdd(skip_count, 1);
    }
}
