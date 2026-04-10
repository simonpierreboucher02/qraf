#pragma once

#include "core/types.h"
#include "tensor/tensor.h"
#include "qraf/format.h"
#include <vector>

namespace qraf {

// ─── KV Cache ───
// Stores key and value tensors for each layer across sequence positions

struct KVCache {
    // keys[layer][pos * num_kv_heads * head_dim]
    // values[layer][pos * num_kv_heads * head_dim]
    std::vector<std::vector<f32>> keys;
    std::vector<std::vector<f32>> values;

    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int current_len;

    void init(int layers, int kv_heads, int hdim, int max_seq);
    void reset();

    // Store k,v for a single position at given layer
    void store(int layer, int pos, const f32* k, const f32* v);

    // Get pointer to cached keys/values for a layer
    const f32* key_at(int layer, int pos) const;
    const f32* value_at(int layer, int pos) const;
};

// ─── Attention Module ───
// Implements multi-head attention with GQA support and KV cache

struct AttentionWeights {
    // These point into mmap memory (not owned)
    const void* wq_data = nullptr;   // [num_heads * head_dim, hidden_size]
    const void* wk_data = nullptr;   // [num_kv_heads * head_dim, hidden_size]
    const void* wv_data = nullptr;   // [num_kv_heads * head_dim, hidden_size]
    const void* wo_data = nullptr;   // [hidden_size, num_heads * head_dim]

    u64 wq_size = 0, wk_size = 0, wv_size = 0, wo_size = 0;
    DType dtype = DType::F32;
    const QuantScheme* quant = nullptr;

    // Optional bias (Qwen-style) — f32, always dequantized
    const f32* bq = nullptr;  // [num_heads * head_dim]
    const f32* bk = nullptr;  // [num_kv_heads * head_dim]
    const f32* bv = nullptr;  // [num_kv_heads * head_dim]
};

// Compute attention for a single token position
// x: [hidden_size], output: [hidden_size]
void attention_forward(
    f32* output,
    const f32* x,
    const AttentionWeights& weights,
    KVCache& cache,
    int layer,
    int pos,
    const ModelConfig& config
);

// ─── MLP Module ───
struct MLPWeights {
    const void* w_gate_data = nullptr;  // [intermediate_size, hidden_size]
    const void* w_up_data = nullptr;    // [intermediate_size, hidden_size]
    const void* w_down_data = nullptr;  // [hidden_size, intermediate_size]

    u64 w_gate_size = 0, w_up_size = 0, w_down_size = 0;
    DType dtype = DType::F32;
    const QuantScheme* quant = nullptr;
};

// SwiGLU MLP forward
void mlp_forward(
    f32* output,
    const f32* x,
    const MLPWeights& weights,
    const ModelConfig& config
);

} // namespace qraf
