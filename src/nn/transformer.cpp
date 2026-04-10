#include "nn/transformer.h"
#include "nn/backend.h"
#include "tensor/quantize.h"
#include "core/logging.h"
#include "core/error.h"
#include <cstring>

namespace qraf {

static std::string layer_name(int l, const std::string& suffix) {
    return "model.layers." + std::to_string(l) + "." + suffix;
}

Tensor Transformer::load_norm_tensor(QrafModel& model, const std::string& name) {
    TensorView tv = model.get_tensor(name);
    if (tv.dtype == DType::F32) {
        return Tensor::from_data(tv.shape, DType::F32, tv.data, tv.data_size);
    }
    return dequantize_tensor(tv.data, tv.data_size, tv.shape, tv.dtype, tv.quant);
}

void Transformer::setup_layer_weights(QrafModel& model, int layer) {
    auto& lw = weights_.layers[layer];

    auto set_attn_weight = [&](const std::string& name, const void*& data_ptr, u64& size_ref) {
        TensorView tv = model.get_tensor(name);
        data_ptr = tv.data;
        size_ref = tv.data_size;
        lw.attn.dtype = tv.dtype;
        if (tv.quant) lw.attn.quant = tv.quant;
    };

    set_attn_weight(layer_name(layer, "self_attn.q_proj.weight"), lw.attn.wq_data, lw.attn.wq_size);
    set_attn_weight(layer_name(layer, "self_attn.k_proj.weight"), lw.attn.wk_data, lw.attn.wk_size);
    set_attn_weight(layer_name(layer, "self_attn.v_proj.weight"), lw.attn.wv_data, lw.attn.wv_size);
    set_attn_weight(layer_name(layer, "self_attn.o_proj.weight"), lw.attn.wo_data, lw.attn.wo_size);

    // Load optional attention bias (Qwen, Gemma, etc.)
    auto try_load_bias = [&](const std::string& name) -> const f32* {
        if (!model.has_tensor(name)) return nullptr;
        TensorView tv = model.get_tensor(name);
        // Bias is always small, just use the mmap pointer if f32
        if (tv.dtype == DType::F32) return static_cast<const f32*>(tv.data);
        // Otherwise dequantize (shouldn't happen for bias, but safety)
        return nullptr;
    };
    lw.attn.bq = try_load_bias(layer_name(layer, "self_attn.q_proj.bias"));
    lw.attn.bk = try_load_bias(layer_name(layer, "self_attn.k_proj.bias"));
    lw.attn.bv = try_load_bias(layer_name(layer, "self_attn.v_proj.bias"));

    auto set_mlp_weight = [&](const std::string& name, const void*& data_ptr, u64& size_ref) {
        TensorView tv = model.get_tensor(name);
        data_ptr = tv.data;
        size_ref = tv.data_size;
        lw.mlp.dtype = tv.dtype;
        if (tv.quant) lw.mlp.quant = tv.quant;
    };

    set_mlp_weight(layer_name(layer, "mlp.gate_proj.weight"), lw.mlp.w_gate_data, lw.mlp.w_gate_size);
    set_mlp_weight(layer_name(layer, "mlp.up_proj.weight"), lw.mlp.w_up_data, lw.mlp.w_up_size);
    set_mlp_weight(layer_name(layer, "mlp.down_proj.weight"), lw.mlp.w_down_data, lw.mlp.w_down_size);

    lw.attn_norm = load_norm_tensor(model, layer_name(layer, "input_layernorm.weight"));
    lw.ffn_norm = load_norm_tensor(model, layer_name(layer, "post_attention_layernorm.weight"));
}

bool Transformer::init(QrafModel& model) {
    config_ = model.config();
    config_.compute_derived();

    // Initialize optimized backend (Accelerate / NEON / threading)
    backend_init();

    log::info("Initializing transformer: %s, %u layers, hidden=%u, heads=%u, kv_heads=%u",
              config_.architecture.c_str(), config_.num_layers, config_.hidden_size,
              config_.num_heads, config_.num_kv_heads);

    // Load embedding table
    {
        TensorView tv = model.get_tensor("model.embed_tokens.weight");
        if (tv.dtype == DType::F32) {
            weights_.token_embedding = Tensor::from_data(tv.shape, DType::F32, tv.data, tv.data_size);
        } else {
            weights_.token_embedding = dequantize_tensor(tv.data, tv.data_size, tv.shape, tv.dtype, tv.quant);
        }
        log::info("Embedding: [%u, %u]", tv.shape[0], tv.shape[1]);
    }

    weights_.output_norm = load_norm_tensor(model, "model.norm.weight");

    {
        TensorView tv = model.get_tensor("lm_head.weight");
        weights_.output_proj_data = tv.data;
        weights_.output_proj_size = tv.data_size;
        weights_.output_proj_dtype = tv.dtype;
        weights_.output_proj_quant = tv.quant;
        if (tv.dtype == DType::F32) {
            weights_.output_proj = Tensor::from_data(tv.shape, DType::F32, tv.data, tv.data_size);
        }
    }

    weights_.layers.resize(config_.num_layers);
    for (u32 l = 0; l < config_.num_layers; l++) {
        setup_layer_weights(model, static_cast<int>(l));
    }

    cache_.init(
        static_cast<int>(config_.num_layers),
        static_cast<int>(config_.num_kv_heads),
        static_cast<int>(config_.head_dim),
        static_cast<int>(config_.max_seq_len)
    );

    rope_cache_.init(
        static_cast<int>(config_.max_seq_len),
        static_cast<int>(config_.head_dim),
        config_.rope_theta
    );

    log::info("Transformer initialized successfully");
    return true;
}

Tensor Transformer::forward(u32 token, int pos) {
    int hidden = static_cast<int>(config_.hidden_size);
    int vocab = static_cast<int>(config_.vocab_size);

    QRAF_CHECK(token < config_.vocab_size, "Token ID %u >= vocab_size %u", token, config_.vocab_size);
    Tensor x = weights_.token_embedding.row(token);
    f32* x_buf = x.data_f32();

    std::vector<f32> residual(hidden);
    std::vector<f32> normed(hidden);
    std::vector<f32> attn_out(hidden);
    std::vector<f32> mlp_out(hidden);

    for (u32 l = 0; l < config_.num_layers; l++) {
        const auto& lw = weights_.layers[l];

        // Save residual
        memcpy(residual.data(), x_buf, hidden * sizeof(f32));

        // Pre-attention RMSNorm (optimized)
        memcpy(normed.data(), x_buf, hidden * sizeof(f32));
        dispatch_rms_norm(normed.data(), lw.attn_norm.data_f32(), hidden, config_.rms_norm_eps);

        // Attention (uses optimized dispatch internally)
        attention_forward(attn_out.data(), normed.data(), lw.attn, cache_,
                          static_cast<int>(l), pos, config_);

        // Residual connection (optimized)
        dispatch_add(attn_out.data(), residual.data(), hidden);

        // Save residual
        memcpy(residual.data(), attn_out.data(), hidden * sizeof(f32));

        // Pre-MLP RMSNorm (optimized)
        memcpy(normed.data(), attn_out.data(), hidden * sizeof(f32));
        dispatch_rms_norm(normed.data(), lw.ffn_norm.data_f32(), hidden, config_.rms_norm_eps);

        // MLP (uses optimized dispatch internally)
        mlp_forward(mlp_out.data(), normed.data(), lw.mlp, config_);

        // Residual connection (optimized)
        dispatch_add(mlp_out.data(), residual.data(), hidden);

        memcpy(x_buf, mlp_out.data(), hidden * sizeof(f32));
    }

    // Final RMSNorm (optimized)
    dispatch_rms_norm(x_buf, weights_.output_norm.data_f32(), hidden, config_.rms_norm_eps);

    // Output projection (optimized)
    Tensor logits = Tensor::zeros({static_cast<u32>(vocab)}, DType::F32);
    f32* logits_buf = logits.data_f32();

    if (weights_.output_proj_dtype == DType::F32) {
        dispatch_matvec_opt(weights_.output_proj.data_f32(), x_buf, logits_buf,
                            static_cast<u32>(vocab), static_cast<u32>(hidden));
    } else {
        dispatch_matvec_q(
            weights_.output_proj_data, weights_.output_proj_size,
            weights_.output_proj_dtype, weights_.output_proj_quant,
            static_cast<u32>(vocab), static_cast<u32>(hidden),
            x_buf, logits_buf
        );
    }

    return logits;
}

void Transformer::reset() {
    cache_.reset();
}

} // namespace qraf
