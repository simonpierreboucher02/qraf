#pragma once

#include "core/types.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace qraf {

// ─── Quantized KV Cache ───
// Stores K/V in int8 with per-head per-position scales
// Reduces memory by 4x compared to f32 KV cache
// Dequantizes on-the-fly during attention score computation

struct QuantizedKVCache {
    // Quantized storage: int8 values
    std::vector<std::vector<int8_t>> keys_q;    // [layer][pos * kv_dim]
    std::vector<std::vector<int8_t>> values_q;  // [layer][pos * kv_dim]

    // Per-position per-head scales for dequantization
    std::vector<std::vector<f32>> key_scales;   // [layer][pos * num_kv_heads]
    std::vector<std::vector<f32>> value_scales;

    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int current_len;

    void init(int layers, int kv_heads, int hdim, int max_seq) {
        num_layers = layers;
        num_kv_heads = kv_heads;
        head_dim = hdim;
        max_seq_len = max_seq;
        current_len = 0;

        int kv_dim = kv_heads * hdim;
        keys_q.resize(layers);
        values_q.resize(layers);
        key_scales.resize(layers);
        value_scales.resize(layers);

        for (int l = 0; l < layers; l++) {
            keys_q[l].resize(static_cast<size_t>(max_seq) * kv_dim, 0);
            values_q[l].resize(static_cast<size_t>(max_seq) * kv_dim, 0);
            key_scales[l].resize(static_cast<size_t>(max_seq) * kv_heads, 0.0f);
            value_scales[l].resize(static_cast<size_t>(max_seq) * kv_heads, 0.0f);
        }
    }

    void reset() {
        current_len = 0;
        for (auto& k : keys_q) std::fill(k.begin(), k.end(), 0);
        for (auto& v : values_q) std::fill(v.begin(), v.end(), 0);
        for (auto& s : key_scales) std::fill(s.begin(), s.end(), 0.0f);
        for (auto& s : value_scales) std::fill(s.begin(), s.end(), 0.0f);
    }

    // Quantize and store K,V at position
    void store(int layer, int pos, const f32* k, const f32* v) {
        int kv_dim = num_kv_heads * head_dim;
        size_t offset = static_cast<size_t>(pos) * kv_dim;
        size_t scale_offset = static_cast<size_t>(pos) * num_kv_heads;

        // Quantize each head separately for better precision
        for (int h = 0; h < num_kv_heads; h++) {
            const f32* k_head = k + h * head_dim;
            const f32* v_head = v + h * head_dim;
            int8_t* k_out = keys_q[layer].data() + offset + h * head_dim;
            int8_t* v_out = values_q[layer].data() + offset + h * head_dim;

            // Find max absolute value
            f32 k_max = 0.0f, v_max = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                k_max = std::max(k_max, std::fabs(k_head[d]));
                v_max = std::max(v_max, std::fabs(v_head[d]));
            }

            f32 k_scale = k_max / 127.0f;
            f32 v_scale = v_max / 127.0f;
            f32 k_inv = (k_scale > 0) ? 1.0f / k_scale : 0.0f;
            f32 v_inv = (v_scale > 0) ? 1.0f / v_scale : 0.0f;

            key_scales[layer][scale_offset + h] = k_scale;
            value_scales[layer][scale_offset + h] = v_scale;

            for (int d = 0; d < head_dim; d++) {
                k_out[d] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f,
                           std::round(k_head[d] * k_inv))));
                v_out[d] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f,
                           std::round(v_head[d] * v_inv))));
            }
        }
    }

    // Dequantize and read K at position for a specific kv_head
    void dequant_key(int layer, int pos, int kv_head, f32* out) const {
        size_t offset = static_cast<size_t>(pos) * num_kv_heads * head_dim + kv_head * head_dim;
        f32 scale = key_scales[layer][static_cast<size_t>(pos) * num_kv_heads + kv_head];
        const int8_t* q = keys_q[layer].data() + offset;
        for (int d = 0; d < head_dim; d++) {
            out[d] = static_cast<f32>(q[d]) * scale;
        }
    }

    // Dequantize and read V at position for a specific kv_head
    void dequant_value(int layer, int pos, int kv_head, f32* out) const {
        size_t offset = static_cast<size_t>(pos) * num_kv_heads * head_dim + kv_head * head_dim;
        f32 scale = value_scales[layer][static_cast<size_t>(pos) * num_kv_heads + kv_head];
        const int8_t* q = values_q[layer].data() + offset;
        for (int d = 0; d < head_dim; d++) {
            out[d] = static_cast<f32>(q[d]) * scale;
        }
    }

    // Compute dot product directly with quantized key (no full dequant)
    f32 dot_with_key(int layer, int pos, int kv_head, const f32* query) const {
        size_t offset = static_cast<size_t>(pos) * num_kv_heads * head_dim + kv_head * head_dim;
        f32 scale = key_scales[layer][static_cast<size_t>(pos) * num_kv_heads + kv_head];
        const int8_t* q = keys_q[layer].data() + offset;

        f32 sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += static_cast<f32>(q[d]) * query[d];
        }
        return sum * scale;
    }

    // Memory usage in bytes
    size_t memory_usage() const {
        size_t total = 0;
        for (const auto& k : keys_q) total += k.size();
        for (const auto& v : values_q) total += v.size();
        for (const auto& s : key_scales) total += s.size() * sizeof(f32);
        for (const auto& s : value_scales) total += s.size() * sizeof(f32);
        return total;
    }
};

} // namespace qraf
