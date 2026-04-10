#pragma once

#include "core/types.h"
#include "nn/attention.h"
#include "qraf/loader.h"
#include "nn/ops.h"
#include <vector>
#include <string>

namespace qraf {

// ─── Layer weights (pointers into mmap) ───
struct TransformerLayerWeights {
    AttentionWeights attn;
    MLPWeights mlp;

    // Norm weights (always f32, small enough to dequantize up-front)
    Tensor attn_norm;   // [hidden_size]
    Tensor ffn_norm;    // [hidden_size]
};

// ─── Full model weights ───
struct TransformerWeights {
    Tensor token_embedding;  // [vocab_size, hidden_size]
    Tensor output_norm;      // [hidden_size]
    Tensor output_proj;      // [vocab_size, hidden_size]

    // Per-layer weights
    std::vector<TransformerLayerWeights> layers;

    // Raw pointers for quantized output projection
    const void* output_proj_data = nullptr;
    u64 output_proj_size = 0;
    DType output_proj_dtype = DType::F32;
    const QuantScheme* output_proj_quant = nullptr;
};

// ─── Transformer Engine ───
class Transformer {
public:
    Transformer() = default;

    // Initialize from a loaded QRAF model
    bool init(QrafModel& model);

    // Forward pass for a single token at a given position
    // Returns logits [vocab_size]
    Tensor forward(u32 token, int pos);

    // Reset state (clear KV cache)
    void reset();

    const ModelConfig& config() const { return config_; }
    const KVCache& kv_cache() const { return cache_; }

private:
    // Load tensor from model, handling quantized -> f32 conversion for norms
    Tensor load_norm_tensor(QrafModel& model, const std::string& name);

    // Set up attention/MLP weight pointers from TensorViews
    void setup_layer_weights(QrafModel& model, int layer);

    ModelConfig config_;
    TransformerWeights weights_;
    KVCache cache_;
    ops::RopeCache rope_cache_;
};

} // namespace qraf
