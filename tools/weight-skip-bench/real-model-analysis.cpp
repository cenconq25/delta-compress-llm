// Real Model Weight Analysis
//
// Reads a GGUF file directly and analyzes Q4_K weight scale distributions.

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstring>

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: %s -m <model.gguf>\n", argv[0]);
        return 1;
    }

    const char * model_path = nullptr;
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
            break;
        }
    }
    if (!model_path) {
        printf("Error: -m <model.gguf> required\n");
        return 1;
    }

    printf("=======================================================\n");
    printf("  Real Model Weight Scale Analysis\n");
    printf("=======================================================\n\n");

    // Open GGUF file
    struct gguf_init_params params = { .no_alloc = false, .ctx = nullptr };
    struct ggml_context * ctx = nullptr;
    params.ctx = &ctx;

    struct gguf_context * gguf = gguf_init_from_file(model_path, params);
    if (!gguf) {
        printf("Error: failed to open %s\n", model_path);
        return 1;
    }

    const int n_tensors = gguf_get_n_tensors(gguf);
    printf("Model: %s\n", model_path);
    printf("Tensors: %d\n\n", n_tensors);

    int n_q4k = 0;
    int64_t total_blocks = 0;
    std::vector<float> all_scales;

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        struct ggml_tensor * t = ggml_get_tensor(ctx, name);

        if (!t || t->type != GGML_TYPE_Q4_K) continue;

        n_q4k++;
        const int64_t n_elements = ggml_nelements(t);
        const int64_t n_blocks = n_elements / 256; // QK_K = 256
        total_blocks += n_blocks;

        // Q4_K block = 144 bytes: first 4 bytes = half2(d, dmin)
        const uint8_t * data = (const uint8_t *)t->data;
        std::vector<float> scales;

        for (int64_t b = 0; b < n_blocks; b++) {
            const uint16_t * dm = (const uint16_t *)(data + b * 144);
            float d = fabsf(ggml_fp16_to_fp32(dm[0]));
            scales.push_back(d);
        }

        std::sort(scales.begin(), scales.end());
        float med = scales[scales.size() / 2];
        float p10 = scales[scales.size() / 10];
        float p90 = scales[9 * scales.size() / 10];

        // Print first 5 + any attention/ffn layers
        if (n_q4k <= 5 || strstr(name, ".0.") || strstr(name, ".39.") || strstr(name, ".79.")) {
            printf("  %-50s | %7lld blocks | p10=%.5f med=%.5f p90=%.5f\n",
                name, (long long)n_blocks, p10, med, p90);
        }

        all_scales.insert(all_scales.end(), scales.begin(), scales.end());
    }

    printf("  ...\n");
    printf("\n  Total Q4_K tensors: %d, total blocks: %lld (%.1f GB)\n\n",
        n_q4k, (long long)total_blocks, (float)total_blocks * 144 / 1e9);

    if (all_scales.empty()) {
        printf("No Q4_K tensors found.\n");
        gguf_free(gguf);
        ggml_free(ctx);
        return 0;
    }

    std::sort(all_scales.begin(), all_scales.end());

    printf("--- Global Weight Scale Distribution ---\n");
    printf("  Min:    %.6f\n", all_scales.front());
    printf("  P1:     %.6f\n", all_scales[all_scales.size() / 100]);
    printf("  P5:     %.6f\n", all_scales[all_scales.size() / 20]);
    printf("  P10:    %.6f\n", all_scales[all_scales.size() / 10]);
    printf("  P25:    %.6f\n", all_scales[all_scales.size() / 4]);
    printf("  Median: %.6f\n", all_scales[all_scales.size() / 2]);
    printf("  P75:    %.6f\n", all_scales[3 * all_scales.size() / 4]);
    printf("  P90:    %.6f\n", all_scales[9 * all_scales.size() / 10]);
    printf("  P99:    %.6f\n", all_scales[99 * all_scales.size() / 100]);
    printf("  Max:    %.6f\n", all_scales.back());

    printf("\n--- Skip Rate by Weight Scale Threshold ---\n");
    printf("  (blocks with |scale| < threshold can potentially be skipped)\n");
    printf("  %-12s | %-10s | %-10s | %-12s\n", "Threshold", "Skip Rate", "BW Save", "Est. Speedup");

    float thresholds[] = {0.0001f, 0.0002f, 0.0005f, 0.001f, 0.002f, 0.005f, 0.01f, 0.02f, 0.05f};
    for (float thresh : thresholds) {
        int64_t below = std::lower_bound(all_scales.begin(), all_scales.end(), thresh) - all_scales.begin();
        float skip_rate = (float)below / all_scales.size();
        float speedup = 1.0f / (1.0f - skip_rate + 0.01f);
        printf("  %-12.4f | %8.1f%% | %8.1f%% | %.2fx\n",
            thresh, skip_rate * 100.0f, skip_rate * 100.0f, speedup);
    }

    // Histogram
    printf("\n--- Scale Histogram ---\n");
    float edges[] = {0, 0.0001f, 0.001f, 0.005f, 0.01f, 0.02f, 0.05f, 0.1f, 0.5f, 1.0f, 1e10f};
    const char * labels[] = {
        "[0, 0.0001)", "[0.0001, 0.001)", "[0.001, 0.005)", "[0.005, 0.01)",
        "[0.01, 0.02)", "[0.02, 0.05)", "[0.05, 0.1)", "[0.1, 0.5)",
        "[0.5, 1.0)", "[1.0, +inf)"
    };
    for (int b = 0; b < 10; b++) {
        int64_t lo = std::lower_bound(all_scales.begin(), all_scales.end(), edges[b]) - all_scales.begin();
        int64_t hi = std::lower_bound(all_scales.begin(), all_scales.end(), edges[b+1]) - all_scales.begin();
        int64_t count = hi - lo;
        float pct = 100.0f * count / all_scales.size();
        printf("  %-20s %8lld (%5.1f%%) ", labels[b], (long long)count, pct);
        int bar = (int)(pct);
        for (int j = 0; j < bar; j++) printf("#");
        printf("\n");
    }

    gguf_free(gguf);
    ggml_free(ctx);

    printf("\n");
    return 0;
}
