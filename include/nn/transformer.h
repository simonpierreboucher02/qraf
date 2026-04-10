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

    // Norm weights (always f32)
    Tensor attn_norm;       // [hidden_size]  (RMSNorm weight or LayerNorm weight)
    Tensor attn_norm_bias;  // [hidden_size]  (LayerNorm bias, empty for RMSNorm)
    Tensor ffn_norm;        // [hidden_size]
    Tensor ffn_norm_bias;   // [hidden_size]

    // GPT-2 style: MLP has single fc_in/fc_out instead of gate/up/down
    const void* mlp_fc_in_data = nullptr;   // [intermediate, hidden]
    const void* mlp_fc_out_data = nullptr;  // [hidden, intermediate]
    u64 mlp_fc_in_size = 0, mlp_fc_out_size = 0;
    // Biases for GPT-2 MLP
    const f32* mlp_fc_in_bias = nullptr;    // [intermediate]
    const f32* mlp_fc_out_bias = nullptr;   // [hidden]
};

// ─── Full model weights ───
struct TransformerWeights {
    Tensor token_embedding;       // [vocab_size, hidden_size]
    Tensor position_embedding;    // [max_seq_len, hidden_size]  (GPT-2/OPT only)
    Tensor output_norm;           // [hidden_size]
    Tensor output_norm_bias;      // [hidden_size]  (LayerNorm bias)
    Tensor output_proj;           // [vocab_size, hidden_size]

    std::vector<TransformerLayerWeights> layers;

    // Raw pointers for quantized output projection
    const void* output_proj_data = nullptr;
    u64 output_proj_size = 0;
    DType output_proj_dtype = DType::F32;
    const QuantScheme* output_proj_quant = nullptr;
    bool tie_word_embeddings = false;

    // OPT: embed_dim != hidden_size projection
    Tensor project_in;   // [hidden, embed_dim]
    Tensor project_out;  // [embed_dim, hidden]
};

// ─── Transformer Engine ───
class Transformer {
public:
    Transformer() = default;

    bool init(QrafModel& model);
    Tensor forward(u32 token, int pos);
    void reset();

    const ModelConfig& config() const { return config_; }
    const KVCache& kv_cache() const { return cache_; }

private:
    Tensor load_tensor_f32(QrafModel& model, const std::string& name);
    bool has_tensor(QrafModel& model, const std::string& name);
    const f32* try_load_bias(QrafModel& model, const std::string& name);

    void setup_layer_llama(QrafModel& model, int layer);
    void setup_layer_gpt2(QrafModel& model, int layer);
    void setup_layer_gpt_neox(QrafModel& model, int layer);
    void setup_layer_opt(QrafModel& model, int layer);

    Tensor forward_llama(u32 token, int pos);
    Tensor forward_gpt2(u32 token, int pos);
    Tensor forward_gpt_neox(u32 token, int pos);

    // Normalization dispatch
    void norm_inplace(f32* x, const Tensor& weight, const Tensor& bias, int size);

    // MLP dispatch
    void mlp_standard(f32* output, const f32* x, const TransformerLayerWeights& lw);

    ModelConfig config_;
    TransformerWeights weights_;
    KVCache cache_;
    ops::RopeCache rope_cache_;
};

} // namespace qraf
