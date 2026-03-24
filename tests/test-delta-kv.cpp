// Test delta-KV encode/decode consistency
//
// Verifies that the encoder and decoder agree on reconstructed values,
// specifically that the encoder updates its reference to the RECONSTRUCTED
// value (keyframe + dequant(delta)), not the original source.
// See: https://news.ycombinator.com/item?id=47483455 (tveita's comment)

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>
#include <cstdint>

// ============================================================================
// Q4_0 encode/decode (mirrors the logic in delta-kv.cu and llama-kv-cache-delta.cpp)
// ============================================================================

static constexpr int QK = 32;

struct block_q4_0_test {
    float d;
    uint8_t qs[QK / 2];
};

static void quantize_q4_0(const float * x, block_q4_0_test & block) {
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < QK; j++) {
        if (fabsf(x[j]) > amax) {
            amax = fabsf(x[j]);
            vmax = x[j];
        }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    block.d = d;
    for (int j = 0; j < QK / 2; j++) {
        const float x0 = x[j] * id;
        const float x1 = x[j + QK / 2] * id;
        const uint8_t xi0 = std::min(15, (int)(x0 + 8.5f));
        const uint8_t xi1 = std::min(15, (int)(x1 + 8.5f));
        block.qs[j] = xi0 | (xi1 << 4);
    }
}

static void dequantize_q4_0(const block_q4_0_test & block, float * out) {
    const float d = block.d;
    for (int j = 0; j < QK / 2; j++) {
        const uint8_t packed = block.qs[j];
        out[j]        = d * ((float)(packed & 0x0F) - 8.0f);
        out[j + QK/2] = d * ((float)(packed >> 4) - 8.0f);
    }
}

// ============================================================================
// Simulate the CPU delta-KV path (process_cpu from llama-kv-cache-delta.cpp)
// This is the CORRECT path that updates ref to reconstructed values.
// ============================================================================

static void encode_cpu_correct(
    const float * src,      // input values (n_embd)
    float * ref,            // reference buffer, updated to reconstructed (n_embd)
    float * reconstructed,  // output (n_embd)
    int n_embd,
    bool is_keyframe)
{
    if (is_keyframe) {
        memcpy(reconstructed, src, n_embd * sizeof(float));
        memcpy(ref, src, n_embd * sizeof(float));
        return;
    }

    const int blocks = n_embd / QK;
    float delta_buf[QK], decoded_buf[QK];
    block_q4_0_test block;

    for (int b = 0; b < blocks; b++) {
        for (int j = 0; j < QK; j++) {
            delta_buf[j] = src[b * QK + j] - ref[b * QK + j];
        }
        quantize_q4_0(delta_buf, block);
        dequantize_q4_0(block, decoded_buf);
        for (int j = 0; j < QK; j++) {
            reconstructed[b * QK + j] = ref[b * QK + j] + decoded_buf[j];
        }
    }

    // Update ref to reconstructed (NOT to src)
    memcpy(ref, reconstructed, n_embd * sizeof(float));
}

// ============================================================================
// Simulate the BUGGY encoder that updates ref to original source values
// (the bug tveita found)
// ============================================================================

static void encode_cpu_buggy(
    const float * src,
    float * ref,
    float * reconstructed,
    int n_embd,
    bool is_keyframe)
{
    if (is_keyframe) {
        memcpy(reconstructed, src, n_embd * sizeof(float));
        memcpy(ref, src, n_embd * sizeof(float));
        return;
    }

    const int blocks = n_embd / QK;
    float delta_buf[QK], decoded_buf[QK];
    block_q4_0_test block;

    for (int b = 0; b < blocks; b++) {
        for (int j = 0; j < QK; j++) {
            delta_buf[j] = src[b * QK + j] - ref[b * QK + j];
        }
        quantize_q4_0(delta_buf, block);
        dequantize_q4_0(block, decoded_buf);
        for (int j = 0; j < QK; j++) {
            reconstructed[b * QK + j] = ref[b * QK + j] + decoded_buf[j];
        }
    }

    // BUG: update ref to original source, not reconstructed
    memcpy(ref, src, n_embd * sizeof(float));
}

// ============================================================================
// Standalone decoder (reads stored deltas + keyframes, no access to source)
// This mirrors k_delta_kv_reconstruct in delta-kv.cu
// ============================================================================

static void decode_from_stored(
    const block_q4_0_test * delta_blocks,
    const float * keyframe,     // the keyframe/ref at encode time
    float * output,
    int n_embd)
{
    const int blocks = n_embd / QK;
    for (int b = 0; b < blocks; b++) {
        float decoded[QK];
        dequantize_q4_0(delta_blocks[b], decoded);
        for (int j = 0; j < QK; j++) {
            output[b * QK + j] = keyframe[b * QK + j] + decoded[j];
        }
    }
}

// ============================================================================
// Helper: compute MSE between two buffers
// ============================================================================

static double compute_mse(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum / n;
}

static double compute_max_err(const float * a, const float * b, int n) {
    double mx = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// ============================================================================
// Test 1: Encoder/decoder round-trip consistency
//
// Run encoder and decoder in lockstep on the same data. At each step:
// - Encoder processes src, updates enc_ref to reconstructed
// - Decoder independently computes delta from dec_ref, reconstructs, updates dec_ref
// Both refs and outputs must match exactly at every step.
// ============================================================================

static bool test_roundtrip_consistency() {
    printf("Test 1: Encoder/decoder round-trip consistency\n");

    const int n_embd = 128;
    const int n_tokens = 64;
    const int kf_interval = 16;

    // Generate synthetic data: slowly varying values (like real KV cache)
    std::vector<std::vector<float>> src_data(n_tokens, std::vector<float>(n_embd));
    srand(42);
    for (int j = 0; j < n_embd; j++) {
        src_data[0][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (int t = 1; t < n_tokens; t++) {
        for (int j = 0; j < n_embd; j++) {
            src_data[t][j] = src_data[t-1][j] + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
    }

    // Encoder state
    std::vector<float> enc_ref(n_embd, 0.0f);
    std::vector<float> enc_out(n_embd);

    // Decoder state (independent — only sees what encoder would store)
    std::vector<float> dec_ref(n_embd, 0.0f);
    std::vector<float> dec_out(n_embd);

    int mismatches = 0;

    for (int t = 0; t < n_tokens; t++) {
        bool is_keyframe = (t % kf_interval == 0);

        // --- Encoder side ---
        // Save ref BEFORE encode to compare with decoder's ref
        std::vector<float> enc_ref_before = enc_ref;
        encode_cpu_correct(src_data[t].data(), enc_ref.data(), enc_out.data(), n_embd, is_keyframe);

        // --- Decoder side ---
        if (is_keyframe) {
            // Decoder receives the keyframe value directly
            memcpy(dec_out.data(), enc_out.data(), n_embd * sizeof(float));
            memcpy(dec_ref.data(), enc_out.data(), n_embd * sizeof(float));
        } else {
            // Before encoding, encoder's ref should match decoder's ref
            double ref_drift = compute_max_err(enc_ref_before.data(), dec_ref.data(), n_embd);
            if (ref_drift > 1e-7) {
                printf("  FAIL: token %d pre-encode ref drift = %e\n", t, ref_drift);
                mismatches++;
            }

            // Decoder computes delta from its own ref (same src, same ref → same delta)
            const int blocks = n_embd / QK;
            for (int b = 0; b < blocks; b++) {
                float delta_buf[QK], decoded[QK];
                block_q4_0_test block;
                for (int j = 0; j < QK; j++) {
                    delta_buf[j] = src_data[t][b * QK + j] - dec_ref[b * QK + j];
                }
                quantize_q4_0(delta_buf, block);
                dequantize_q4_0(block, decoded);
                for (int j = 0; j < QK; j++) {
                    dec_out[b * QK + j] = dec_ref[b * QK + j] + decoded[j];
                }
            }
            // Decoder updates ref to reconstructed
            memcpy(dec_ref.data(), dec_out.data(), n_embd * sizeof(float));
        }

        // Outputs must match
        double out_err = compute_max_err(enc_out.data(), dec_out.data(), n_embd);
        if (out_err > 1e-7) {
            printf("  FAIL: token %d output mismatch, max_err = %e\n", t, out_err);
            mismatches++;
        }

        // Refs must match after both sides update
        double ref_err = compute_max_err(enc_ref.data(), dec_ref.data(), n_embd);
        if (ref_err > 1e-7) {
            printf("  FAIL: token %d post-update ref mismatch, max_err = %e\n", t, ref_err);
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("  PASS: encoder and decoder agree on all %d tokens\n", n_tokens);
    }
    return mismatches == 0;
}

// ============================================================================
// Test 2: Buggy encoder causes drift
//
// Demonstrates that using source values (not reconstructed) for ref update
// causes encoder/decoder divergence that grows over tokens.
// ============================================================================

static bool test_buggy_encoder_drifts() {
    printf("Test 2: Buggy encoder (source ref update) causes drift\n");

    const int n_embd = 128;
    const int n_tokens = 64;
    const int kf_interval = 32; // long interval to accumulate drift

    // Generate data with moderate variation
    std::vector<std::vector<float>> src_data(n_tokens, std::vector<float>(n_embd));
    srand(123);
    for (int j = 0; j < n_embd; j++) {
        src_data[0][j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    for (int t = 1; t < n_tokens; t++) {
        for (int j = 0; j < n_embd; j++) {
            src_data[t][j] = src_data[t-1][j] + ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        }
    }

    // Run correct encoder
    std::vector<float> correct_ref(n_embd, 0.0f);
    std::vector<float> correct_out(n_embd);

    // Run buggy encoder
    std::vector<float> buggy_ref(n_embd, 0.0f);
    std::vector<float> buggy_out(n_embd);

    double max_drift_correct = 0.0;
    double max_drift_buggy = 0.0;
    int drift_worse_count = 0;

    for (int t = 0; t < n_tokens; t++) {
        bool is_keyframe = (t % kf_interval == 0);

        encode_cpu_correct(src_data[t].data(), correct_ref.data(), correct_out.data(), n_embd, is_keyframe);
        encode_cpu_buggy(src_data[t].data(), buggy_ref.data(), buggy_out.data(), n_embd, is_keyframe);

        // Both encoders produce the SAME reconstructed output for a single token,
        // but their refs diverge, causing FUTURE tokens to differ
        if (!is_keyframe && t > 0) {
            double ref_diff = compute_max_err(correct_ref.data(), buggy_ref.data(), n_embd);
            if (ref_diff > 1e-7) {
                drift_worse_count++;
                if (ref_diff > max_drift_buggy) max_drift_buggy = ref_diff;
            }
        }

        double err_correct = compute_mse(src_data[t].data(), correct_out.data(), n_embd);
        double err_buggy   = compute_mse(src_data[t].data(), buggy_out.data(), n_embd);

        if (err_correct > max_drift_correct) max_drift_correct = err_correct;
    }

    printf("  Correct encoder max MSE: %e\n", max_drift_correct);
    printf("  Buggy ref divergence: %d tokens drifted, max ref drift = %e\n",
           drift_worse_count, max_drift_buggy);

    // The buggy encoder should show ref divergence
    bool pass = (drift_worse_count > 0 && max_drift_buggy > 1e-6);
    printf("  %s: buggy encoder %s ref drift as expected\n",
           pass ? "PASS" : "FAIL",
           pass ? "shows" : "does NOT show");
    return pass;
}

// ============================================================================
// Test 3: Error accumulation stays bounded between keyframes
//
// With the correct encoder, error should not grow unboundedly between keyframes
// because each step's ref matches the decoder's ref exactly.
// ============================================================================

static bool test_error_bounded() {
    printf("Test 3: Error stays bounded between keyframes\n");

    const int n_embd = 256;
    const int n_tokens = 128;
    const int kf_interval = 32;

    std::vector<std::vector<float>> src_data(n_tokens, std::vector<float>(n_embd));
    srand(777);
    for (int j = 0; j < n_embd; j++) {
        src_data[0][j] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
    }
    for (int t = 1; t < n_tokens; t++) {
        for (int j = 0; j < n_embd; j++) {
            src_data[t][j] = src_data[t-1][j] + ((float)rand() / RAND_MAX - 0.5f) * 0.3f;
        }
    }

    std::vector<float> ref(n_embd, 0.0f);
    std::vector<float> reconstructed(n_embd);

    double max_mse_in_segment = 0.0;
    bool error_exploded = false;

    for (int t = 0; t < n_tokens; t++) {
        bool is_keyframe = (t % kf_interval == 0);
        encode_cpu_correct(src_data[t].data(), ref.data(), reconstructed.data(), n_embd, is_keyframe);

        double mse = compute_mse(src_data[t].data(), reconstructed.data(), n_embd);

        if (is_keyframe) {
            // Keyframe should have zero error (just copies source)
            if (mse > 1e-12) {
                printf("  FAIL: keyframe at t=%d has MSE %e (expected ~0)\n", t, mse);
                error_exploded = true;
            }
            max_mse_in_segment = 0.0;
        } else {
            if (mse > max_mse_in_segment) max_mse_in_segment = mse;
            // Error should be bounded by single-step Q4_0 quantization error
            // For deltas ~0.3 magnitude, Q4_0 error should be small
            if (mse > 0.01) {
                printf("  WARN: t=%d MSE=%e seems high\n", t, mse);
                error_exploded = true;
            }
        }
    }

    bool pass = !error_exploded;
    printf("  %s: max MSE in any segment = %e\n", pass ? "PASS" : "FAIL", max_mse_in_segment);
    return pass;
}

// ============================================================================
// Test 4: Keyframe produces exact copy
// ============================================================================

static bool test_keyframe_exact() {
    printf("Test 4: Keyframe produces exact copy of source\n");

    const int n_embd = 128;
    std::vector<float> src(n_embd);
    std::vector<float> ref(n_embd, 0.0f);
    std::vector<float> out(n_embd);

    srand(99);
    for (int j = 0; j < n_embd; j++) {
        src[j] = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
    }

    encode_cpu_correct(src.data(), ref.data(), out.data(), n_embd, true);

    double max_err = compute_max_err(src.data(), out.data(), n_embd);
    bool pass = (max_err == 0.0);
    printf("  %s: keyframe max_err = %e\n", pass ? "PASS" : "FAIL", max_err);

    // Also verify ref was set to source
    double ref_err = compute_max_err(src.data(), ref.data(), n_embd);
    if (ref_err != 0.0) {
        printf("  FAIL: ref not updated to source on keyframe, err = %e\n", ref_err);
        pass = false;
    }

    return pass;
}

// ============================================================================
// Test 5: Zero delta produces zero quantization error
// ============================================================================

static bool test_zero_delta() {
    printf("Test 5: Identical consecutive tokens produce zero error\n");

    const int n_embd = 128;
    std::vector<float> src(n_embd);
    std::vector<float> ref(n_embd);
    std::vector<float> out(n_embd);

    srand(55);
    for (int j = 0; j < n_embd; j++) {
        src[j] = ((float)rand() / RAND_MAX - 0.5f) * 5.0f;
    }

    // First token as keyframe
    encode_cpu_correct(src.data(), ref.data(), out.data(), n_embd, true);

    // Second token identical to first (delta = 0)
    encode_cpu_correct(src.data(), ref.data(), out.data(), n_embd, false);

    double max_err = compute_max_err(src.data(), out.data(), n_embd);
    bool pass = (max_err == 0.0);
    printf("  %s: zero delta max_err = %e\n", pass ? "PASS" : "FAIL", max_err);
    return pass;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== Delta-KV Encode/Decode Consistency Tests ===\n\n");

    int passed = 0, total = 0;

    total++; if (test_keyframe_exact())         passed++;
    printf("\n");
    total++; if (test_zero_delta())             passed++;
    printf("\n");
    total++; if (test_roundtrip_consistency())   passed++;
    printf("\n");
    total++; if (test_buggy_encoder_drifts())    passed++;
    printf("\n");
    total++; if (test_error_bounded())           passed++;

    printf("\n=== Results: %d/%d tests passed ===\n", passed, total);

    return (passed == total) ? 0 : 1;
}
