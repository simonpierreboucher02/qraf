#include "nn/attention.h"
#include "nn/ops.h"
#include "nn/backend.h"
#include "tensor/quantize.h"
#include "core/logging.h"
#include <cmath>
#include <cstring>
#include <vector>

namespace qraf {

// ─── KV Cache Implementation ───

void KVCache::init(int layers, int kv_heads, int hdim, int max_seq) {
    num_layers = layers;
    num_kv_heads = kv_heads;
    head_dim = hdim;
    max_seq_len = max_seq;
    current_len = 0;

    size_t kv_size = static_cast<size_t>(max_seq) * kv_heads * hdim;
    keys.resize(layers);
    values.resize(layers);
    for (int l = 0; l < layers; l++) {
        keys[l].resize(kv_size, 0.0f);
        values[l].resize(kv_size, 0.0f);
    }

    log::info("KV cache initialized: %d layers, %d kv_heads, %d head_dim, max_seq=%d",
              layers, kv_heads, hdim, max_seq);
}

void KVCache::reset() {
    current_len = 0;
    for (auto& k : keys) std::fill(k.begin(), k.end(), 0.0f);
    for (auto& v : values) std::fill(v.begin(), v.end(), 0.0f);
}

void KVCache::store(int layer, int pos, const f32* k, const f32* v) {
    size_t offset = static_cast<size_t>(pos) * num_kv_heads * head_dim;
    size_t kv_dim = static_cast<size_t>(num_kv_heads) * head_dim;
    memcpy(keys[layer].data() + offset, k, kv_dim * sizeof(f32));
    memcpy(values[layer].data() + offset, v, kv_dim * sizeof(f32));
}

const f32* KVCache::key_at(int layer, int pos) const {
    return keys[layer].data() + static_cast<size_t>(pos) * num_kv_heads * head_dim;
}

const f32* KVCache::value_at(int layer, int pos) const {
    return values[layer].data() + static_cast<size_t>(pos) * num_kv_heads * head_dim;
}

// ─── Unified matvec dispatch (f32 or quantized) ───

static void do_matvec(const void* w_data, u64 w_size,
                      DType dtype, const QuantScheme* quant,
                      u32 out_dim, u32 in_dim,
                      const f32* x, f32* y) {
    if (dtype == DType::F32) {
        dispatch_matvec_opt(static_cast<const f32*>(w_data), x, y, out_dim, in_dim);
    } else {
        dispatch_matvec_q(w_data, w_size, dtype, quant, out_dim, in_dim, x, y);
    }
}

// ─── Attention Forward (fixed RoPE) ───

void attention_forward(
    f32* output,
    const f32* x,
    const AttentionWeights& weights,
    KVCache& cache,
    int layer,
    int pos,
    const ModelConfig& config
) {
    int hidden = static_cast<int>(config.hidden_size);
    int num_heads = static_cast<int>(config.num_heads);
    int num_kv_heads = static_cast<int>(config.num_kv_heads);
    int head_dim = static_cast<int>(config.head_dim);
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    int kv_rep = num_heads / num_kv_heads;

    std::vector<f32> q(q_dim);
    std::vector<f32> k(kv_dim);
    std::vector<f32> v(kv_dim);
    std::vector<f32> attn_out(q_dim, 0.0f);

    // Q, K, V projections
    do_matvec(weights.wq_data, weights.wq_size, weights.dtype, weights.quant,
              static_cast<u32>(q_dim), static_cast<u32>(hidden), x, q.data());
    do_matvec(weights.wk_data, weights.wk_size, weights.dtype, weights.quant,
              static_cast<u32>(kv_dim), static_cast<u32>(hidden), x, k.data());
    do_matvec(weights.wv_data, weights.wv_size, weights.dtype, weights.quant,
              static_cast<u32>(kv_dim), static_cast<u32>(hidden), x, v.data());

    // Add bias if present (Qwen-style)
    if (weights.bq) dispatch_add(q.data(), weights.bq, q_dim);
    if (weights.bk) dispatch_add(k.data(), weights.bk, kv_dim);
    if (weights.bv) dispatch_add(v.data(), weights.bv, kv_dim);

    // Apply RoPE if architecture uses it (Llama, Pythia, CodeGen)
    if (config.use_rope) {
        for (int h = 0; h < num_heads; h++) {
            f32* q_head = q.data() + h * head_dim;
            f32 dummy_k[256] = {};
            ops::rope(q_head, dummy_k, head_dim, pos, config.rope_theta);
        }
        for (int kh = 0; kh < num_kv_heads; kh++) {
            f32* k_head = k.data() + kh * head_dim;
            f32 dummy_q[256] = {};
            ops::rope(dummy_q, k_head, head_dim, pos, config.rope_theta);
        }
    }

    // Store K, V in cache
    cache.store(layer, pos, k.data(), v.data());

    // Attention computation per head
    f32 scale = 1.0f / std::sqrt(static_cast<f32>(head_dim));
    int seq_len = pos + 1;
    std::vector<f32> scores(seq_len);

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_rep;
        f32* q_head = q.data() + h * head_dim;
        f32* out_head = attn_out.data() + h * head_dim;

        // Compute attention scores via optimized dot product
        for (int t = 0; t < seq_len; t++) {
            const f32* k_t = cache.key_at(layer, t) + kv_h * head_dim;
            scores[t] = dispatch_dot_f32(q_head, k_t, head_dim) * scale;
        }

        // Softmax
        dispatch_softmax(scores.data(), seq_len);

        // Weighted sum of values
        memset(out_head, 0, head_dim * sizeof(f32));
        for (int t = 0; t < seq_len; t++) {
            const f32* v_t = cache.value_at(layer, t) + kv_h * head_dim;
            f32 s = scores[t];
            // NEON-friendly: accumulate with broadcast scalar
            for (int d = 0; d < head_dim; d++) {
                out_head[d] += s * v_t[d];
            }
        }
    }

    // Output projection
    do_matvec(weights.wo_data, weights.wo_size, weights.dtype, weights.quant,
              static_cast<u32>(hidden), static_cast<u32>(q_dim), attn_out.data(), output);
}

// ─── MLP Forward (SwiGLU) ───

void mlp_forward(
    f32* output,
    const f32* x,
    const MLPWeights& weights,
    const ModelConfig& config
) {
    int hidden = static_cast<int>(config.hidden_size);
    int inter = static_cast<int>(config.intermediate_size);

    std::vector<f32> gate(inter);
    std::vector<f32> up(inter);

    // Gate & Up projections (via optimized dispatch)
    do_matvec(weights.w_gate_data, weights.w_gate_size, weights.dtype, weights.quant,
              static_cast<u32>(inter), static_cast<u32>(hidden), x, gate.data());
    do_matvec(weights.w_up_data, weights.w_up_size, weights.dtype, weights.quant,
              static_cast<u32>(inter), static_cast<u32>(hidden), x, up.data());

    // SiLU(gate) * up
    dispatch_silu(gate.data(), inter);
    dispatch_mul(gate.data(), up.data(), inter);

    // Down projection
    do_matvec(weights.w_down_data, weights.w_down_size, weights.dtype, weights.quant,
              static_cast<u32>(hidden), static_cast<u32>(inter), gate.data(), output);
}

} // namespace qraf
