#include "nn/transformer.h"
#include "nn/backend.h"
#include "tensor/quantize.h"
#include "core/logging.h"
#include "core/error.h"
#include <cstring>
#include <cmath>

namespace qraf {

// ─── Tensor name helpers ───

static std::string llama_layer(int l, const std::string& s) {
    return "model.layers." + std::to_string(l) + "." + s;
}

// GPT-2: transformer.h.{l}.attn.*, transformer.h.{l}.mlp.*, transformer.h.{l}.ln_*
static std::string gpt2_layer(int l, const std::string& s) {
    return "transformer.h." + std::to_string(l) + "." + s;
}

// GPT-NeoX (Pythia): gpt_neox.layers.{l}.*
static std::string neox_layer(int l, const std::string& s) {
    return "gpt_neox.layers." + std::to_string(l) + "." + s;
}

// OPT: model.decoder.layers.{l}.*
static std::string opt_layer(int l, const std::string& s) {
    return "model.decoder.layers." + std::to_string(l) + "." + s;
}

// ─── Utilities ───

Tensor Transformer::load_tensor_f32(QrafModel& model, const std::string& name) {
    TensorView tv = model.get_tensor(name);
    if (tv.dtype == DType::F32)
        return Tensor::from_data(tv.shape, DType::F32, tv.data, tv.data_size);
    return dequantize_tensor(tv.data, tv.data_size, tv.shape, tv.dtype, tv.quant);
}

bool Transformer::has_tensor(QrafModel& model, const std::string& name) {
    return model.has_tensor(name);
}

const f32* Transformer::try_load_bias(QrafModel& model, const std::string& name) {
    if (!model.has_tensor(name)) return nullptr;
    TensorView tv = model.get_tensor(name);
    if (tv.dtype == DType::F32) return static_cast<const f32*>(tv.data);
    return nullptr;
}

void Transformer::norm_inplace(f32* x, const Tensor& weight, const Tensor& bias, int size) {
    if (config_.use_rms_norm) {
        dispatch_rms_norm(x, weight.data_f32(), size, config_.rms_norm_eps);
    } else {
        const f32* b = bias.empty() ? nullptr : bias.data_f32();
        ops::layer_norm_inplace(x, weight.data_f32(), b, size, config_.rms_norm_eps);
    }
}

// Standard MLP (GPT-2/OPT/Pythia): fc_in -> activation -> fc_out
void Transformer::mlp_standard(f32* output, const f32* x, const TransformerLayerWeights& lw) {
    int hidden = static_cast<int>(config_.hidden_size);
    int inter = static_cast<int>(config_.intermediate_size);

    std::vector<f32> h(inter);

    // fc_in: [inter, hidden] @ x + bias
    if (lw.mlp_fc_in_data) {
        dispatch_matvec_opt(static_cast<const f32*>(lw.mlp_fc_in_data),
                            x, h.data(), static_cast<u32>(inter), static_cast<u32>(hidden));
        if (lw.mlp_fc_in_bias)
            dispatch_add(h.data(), lw.mlp_fc_in_bias, inter);
    }

    // Activation
    if (config_.activation == "gelu") {
        ops::gelu_inplace(h.data(), inter);
    } else if (config_.activation == "relu") {
        for (int i = 0; i < inter; i++) if (h[i] < 0) h[i] = 0;
    } else {
        dispatch_silu(h.data(), inter);
    }

    // fc_out: [hidden, inter] @ h + bias
    if (lw.mlp_fc_out_data) {
        dispatch_matvec_opt(static_cast<const f32*>(lw.mlp_fc_out_data),
                            h.data(), output, static_cast<u32>(hidden), static_cast<u32>(inter));
        if (lw.mlp_fc_out_bias)
            dispatch_add(output, lw.mlp_fc_out_bias, hidden);
    }
}

// ─── Layer Setup: Llama/Qwen ───

void Transformer::setup_layer_llama(QrafModel& model, int l) {
    auto& lw = weights_.layers[l];
    auto set_w = [&](const std::string& n, const void*& d, u64& s) {
        TensorView tv = model.get_tensor(n);
        d = tv.data; s = tv.data_size;
        lw.attn.dtype = tv.dtype;
        if (tv.quant) lw.attn.quant = tv.quant;
    };

    set_w(llama_layer(l, "self_attn.q_proj.weight"), lw.attn.wq_data, lw.attn.wq_size);
    set_w(llama_layer(l, "self_attn.k_proj.weight"), lw.attn.wk_data, lw.attn.wk_size);
    set_w(llama_layer(l, "self_attn.v_proj.weight"), lw.attn.wv_data, lw.attn.wv_size);
    set_w(llama_layer(l, "self_attn.o_proj.weight"), lw.attn.wo_data, lw.attn.wo_size);

    lw.attn.bq = try_load_bias(model, llama_layer(l, "self_attn.q_proj.bias"));
    lw.attn.bk = try_load_bias(model, llama_layer(l, "self_attn.k_proj.bias"));
    lw.attn.bv = try_load_bias(model, llama_layer(l, "self_attn.v_proj.bias"));

    auto set_mlp = [&](const std::string& n, const void*& d, u64& s) {
        TensorView tv = model.get_tensor(n);
        d = tv.data; s = tv.data_size;
        lw.mlp.dtype = tv.dtype;
        if (tv.quant) lw.mlp.quant = tv.quant;
    };
    set_mlp(llama_layer(l, "mlp.gate_proj.weight"), lw.mlp.w_gate_data, lw.mlp.w_gate_size);
    set_mlp(llama_layer(l, "mlp.up_proj.weight"), lw.mlp.w_up_data, lw.mlp.w_up_size);
    set_mlp(llama_layer(l, "mlp.down_proj.weight"), lw.mlp.w_down_data, lw.mlp.w_down_size);

    lw.attn_norm = load_tensor_f32(model, llama_layer(l, "input_layernorm.weight"));
    lw.ffn_norm = load_tensor_f32(model, llama_layer(l, "post_attention_layernorm.weight"));
}

// ─── Layer Setup: GPT-2 ───

void Transformer::setup_layer_gpt2(QrafModel& model, int l) {
    auto& lw = weights_.layers[l];

    std::string prefix = gpt2_layer(l, "");
    int hidden = static_cast<int>(config_.hidden_size);

    // Combined QKV: c_attn or qkv_proj
    for (auto& qkv_name : {std::string("attn.c_attn.weight"), std::string("attn.qkv_proj.weight")}) {
        if (has_tensor(model, prefix + qkv_name)) {
            TensorView tv = model.get_tensor(prefix + qkv_name);
            const f32* data = static_cast<const f32*>(tv.data);
            lw.attn.wq_data = data;
            lw.attn.wq_size = hidden * hidden * sizeof(f32);
            lw.attn.wk_data = data + hidden * hidden;
            lw.attn.wk_size = hidden * hidden * sizeof(f32);
            lw.attn.wv_data = data + 2 * hidden * hidden;
            lw.attn.wv_size = hidden * hidden * sizeof(f32);
            lw.attn.dtype = DType::F32;

            // Combined bias
            std::string bias_name = prefix + qkv_name.substr(0, qkv_name.size()-6) + "bias";
            if (has_tensor(model, bias_name)) {
                const f32* bias = static_cast<const f32*>(model.get_tensor(bias_name).data);
                lw.attn.bq = bias;
                lw.attn.bk = bias + hidden;
                lw.attn.bv = bias + 2 * hidden;
            }
            break;
        }
    }

    // Output projection: c_proj or out_proj
    for (auto& name : {std::string("attn.c_proj.weight"), std::string("attn.out_proj.weight")}) {
        if (has_tensor(model, prefix + name)) {
            TensorView tv = model.get_tensor(prefix + name);
            lw.attn.wo_data = tv.data;
            lw.attn.wo_size = tv.data_size;
            break;
        }
    }

    // MLP: c_fc/fc_in [inter, hidden], c_proj/fc_out [hidden, inter]
    for (auto& name : {std::string("mlp.c_fc.weight"), std::string("mlp.fc_in.weight")}) {
        if (has_tensor(model, prefix + name)) {
            TensorView tv = model.get_tensor(prefix + name);
            lw.mlp_fc_in_data = tv.data;
            lw.mlp_fc_in_size = tv.data_size;
            std::string bn = prefix + name.substr(0, name.size()-6) + "bias";
            lw.mlp_fc_in_bias = try_load_bias(model, bn);
            break;
        }
    }
    for (auto& name : {std::string("mlp.c_proj.weight"), std::string("mlp.fc_out.weight")}) {
        if (has_tensor(model, prefix + name)) {
            TensorView tv = model.get_tensor(prefix + name);
            lw.mlp_fc_out_data = tv.data;
            lw.mlp_fc_out_size = tv.data_size;
            std::string bn = prefix + name.substr(0, name.size()-6) + "bias";
            lw.mlp_fc_out_bias = try_load_bias(model, bn);
            break;
        }
    }

    // Layer norms
    if (has_tensor(model, prefix + "ln_1.weight")) {
        lw.attn_norm = load_tensor_f32(model, prefix + "ln_1.weight");
        if (has_tensor(model, prefix + "ln_1.bias"))
            lw.attn_norm_bias = load_tensor_f32(model, prefix + "ln_1.bias");
    }
    if (has_tensor(model, prefix + "ln_2.weight")) {
        lw.ffn_norm = load_tensor_f32(model, prefix + "ln_2.weight");
        if (has_tensor(model, prefix + "ln_2.bias"))
            lw.ffn_norm_bias = load_tensor_f32(model, prefix + "ln_2.bias");
    }
}

// ─── Layer Setup: GPT-NeoX (Pythia) ───

void Transformer::setup_layer_gpt_neox(QrafModel& model, int l) {
    auto& lw = weights_.layers[l];
    std::string prefix = neox_layer(l, "");

    // Pythia uses separate q/k/v or combined query_key_value
    if (has_tensor(model, prefix + "attention.query_key_value.weight")) {
        TensorView tv = model.get_tensor(prefix + "attention.query_key_value.weight");
        int hidden = static_cast<int>(config_.hidden_size);
        const f32* data = static_cast<const f32*>(tv.data);
        lw.attn.wq_data = data;
        lw.attn.wq_size = hidden * hidden * sizeof(f32);
        lw.attn.wk_data = data + hidden * hidden;
        lw.attn.wk_size = hidden * hidden * sizeof(f32);
        lw.attn.wv_data = data + 2 * hidden * hidden;
        lw.attn.wv_size = hidden * hidden * sizeof(f32);
        lw.attn.dtype = DType::F32;

        if (has_tensor(model, prefix + "attention.query_key_value.bias")) {
            TensorView bv = model.get_tensor(prefix + "attention.query_key_value.bias");
            const f32* bias = static_cast<const f32*>(bv.data);
            lw.attn.bq = bias;
            lw.attn.bk = bias + hidden;
            lw.attn.bv = bias + 2 * hidden;
        }
    }

    if (has_tensor(model, prefix + "attention.dense.weight")) {
        TensorView tv = model.get_tensor(prefix + "attention.dense.weight");
        lw.attn.wo_data = tv.data;
        lw.attn.wo_size = tv.data_size;
    }

    // MLP
    if (has_tensor(model, prefix + "mlp.dense_h_to_4h.weight")) {
        TensorView tv = model.get_tensor(prefix + "mlp.dense_h_to_4h.weight");
        lw.mlp_fc_in_data = tv.data;
        lw.mlp_fc_in_size = tv.data_size;
    }
    lw.mlp_fc_in_bias = try_load_bias(model, prefix + "mlp.dense_h_to_4h.bias");

    if (has_tensor(model, prefix + "mlp.dense_4h_to_h.weight")) {
        TensorView tv = model.get_tensor(prefix + "mlp.dense_4h_to_h.weight");
        lw.mlp_fc_out_data = tv.data;
        lw.mlp_fc_out_size = tv.data_size;
    }
    lw.mlp_fc_out_bias = try_load_bias(model, prefix + "mlp.dense_4h_to_h.bias");

    // Norms
    lw.attn_norm = load_tensor_f32(model, prefix + "input_layernorm.weight");
    if (has_tensor(model, prefix + "input_layernorm.bias"))
        lw.attn_norm_bias = load_tensor_f32(model, prefix + "input_layernorm.bias");
    lw.ffn_norm = load_tensor_f32(model, prefix + "post_attention_layernorm.weight");
    if (has_tensor(model, prefix + "post_attention_layernorm.bias"))
        lw.ffn_norm_bias = load_tensor_f32(model, prefix + "post_attention_layernorm.bias");
}

// ─── Layer Setup: OPT ───

void Transformer::setup_layer_opt(QrafModel& model, int l) {
    auto& lw = weights_.layers[l];
    std::string prefix = opt_layer(l, "");

    auto set_w = [&](const std::string& n, const void*& d, u64& s) {
        if (!has_tensor(model, n)) return;
        TensorView tv = model.get_tensor(n);
        d = tv.data; s = tv.data_size; lw.attn.dtype = tv.dtype;
    };

    set_w(prefix + "self_attn.q_proj.weight", lw.attn.wq_data, lw.attn.wq_size);
    set_w(prefix + "self_attn.k_proj.weight", lw.attn.wk_data, lw.attn.wk_size);
    set_w(prefix + "self_attn.v_proj.weight", lw.attn.wv_data, lw.attn.wv_size);
    set_w(prefix + "self_attn.out_proj.weight", lw.attn.wo_data, lw.attn.wo_size);

    lw.attn.bq = try_load_bias(model, prefix + "self_attn.q_proj.bias");
    lw.attn.bk = try_load_bias(model, prefix + "self_attn.k_proj.bias");
    lw.attn.bv = try_load_bias(model, prefix + "self_attn.v_proj.bias");

    // OPT MLP: fc1 -> relu -> fc2
    if (has_tensor(model, prefix + "fc1.weight")) {
        TensorView tv = model.get_tensor(prefix + "fc1.weight");
        lw.mlp_fc_in_data = tv.data;
        lw.mlp_fc_in_size = tv.data_size;
    }
    lw.mlp_fc_in_bias = try_load_bias(model, prefix + "fc1.bias");

    if (has_tensor(model, prefix + "fc2.weight")) {
        TensorView tv = model.get_tensor(prefix + "fc2.weight");
        lw.mlp_fc_out_data = tv.data;
        lw.mlp_fc_out_size = tv.data_size;
    }
    lw.mlp_fc_out_bias = try_load_bias(model, prefix + "fc2.bias");

    lw.attn_norm = load_tensor_f32(model, prefix + "self_attn_layer_norm.weight");
    if (has_tensor(model, prefix + "self_attn_layer_norm.bias"))
        lw.attn_norm_bias = load_tensor_f32(model, prefix + "self_attn_layer_norm.bias");
    lw.ffn_norm = load_tensor_f32(model, prefix + "final_layer_norm.weight");
    if (has_tensor(model, prefix + "final_layer_norm.bias"))
        lw.ffn_norm_bias = load_tensor_f32(model, prefix + "final_layer_norm.bias");
}

// ─── Init ───

bool Transformer::init(QrafModel& model) {
    config_ = model.config();
    config_.compute_derived();

    backend_init();

    log::info("Init transformer [%s]: %u layers, hidden=%u, heads=%u, kv=%u",
              config_.architecture.c_str(), config_.num_layers,
              config_.hidden_size, config_.num_heads, config_.num_kv_heads);

    // ─── Embedding ───
    std::vector<std::string> embed_names = {
        "model.embed_tokens.weight",
        "transformer.wte.weight",     // GPT-2
        "gpt_neox.embed_in.weight",   // Pythia
        "model.decoder.embed_tokens.weight", // OPT
    };
    for (auto& n : embed_names) {
        if (model.has_tensor(n)) {
            weights_.token_embedding = load_tensor_f32(model, n);
            log::info("Embedding: %s [%u, %u]",
                      n.c_str(), weights_.token_embedding.shape()[0], weights_.token_embedding.shape()[1]);
            break;
        }
    }

    // ─── Position embedding (GPT-2, OPT) ───
    if (!config_.use_rope) {
        std::vector<std::string> pos_names = {
            "transformer.wpe.weight",
            "model.decoder.embed_positions.weight",
        };
        for (auto& n : pos_names) {
            if (model.has_tensor(n)) {
                weights_.position_embedding = load_tensor_f32(model, n);
                log::info("Position embedding: %s [%u, %u]",
                          n.c_str(), weights_.position_embedding.shape()[0], weights_.position_embedding.shape()[1]);
                break;
            }
        }
    }

    // ─── OPT project_in / project_out ───
    if (model.has_tensor("model.decoder.project_in.weight"))
        weights_.project_in = load_tensor_f32(model, "model.decoder.project_in.weight");
    if (model.has_tensor("model.decoder.project_out.weight"))
        weights_.project_out = load_tensor_f32(model, "model.decoder.project_out.weight");

    // ─── Output norm ───
    std::vector<std::string> norm_names = {
        "model.norm.weight", "transformer.ln_f.weight",
        "gpt_neox.final_layer_norm.weight", "model.decoder.final_layer_norm.weight",
    };
    for (auto& n : norm_names) {
        if (model.has_tensor(n)) {
            weights_.output_norm = load_tensor_f32(model, n);
            // Try bias
            std::string bias_name = n.substr(0, n.size() - 6) + "bias";
            if (model.has_tensor(bias_name))
                weights_.output_norm_bias = load_tensor_f32(model, bias_name);
            break;
        }
    }

    // ─── Output projection (lm_head) ───
    std::vector<std::string> head_names = {
        "lm_head.weight", "embed_out.weight",
    };
    bool found_head = false;
    for (auto& n : head_names) {
        if (model.has_tensor(n)) {
            TensorView tv = model.get_tensor(n);
            weights_.output_proj_data = tv.data;
            weights_.output_proj_size = tv.data_size;
            weights_.output_proj_dtype = tv.dtype;
            weights_.output_proj_quant = tv.quant;
            if (tv.dtype == DType::F32)
                weights_.output_proj = Tensor::from_data(tv.shape, DType::F32, tv.data, tv.data_size);
            found_head = true;
            break;
        }
    }
    if (!found_head) {
        // Tie word embeddings
        weights_.output_proj = weights_.token_embedding;
        weights_.output_proj_data = weights_.token_embedding.data();
        weights_.output_proj_size = weights_.token_embedding.nbytes();
        weights_.output_proj_dtype = DType::F32;
        weights_.tie_word_embeddings = true;
        log::info("Tied word embeddings for output projection");
    }

    // ─── Layer weights ───
    weights_.layers.resize(config_.num_layers);
    for (u32 l = 0; l < config_.num_layers; l++) {
        switch (config_.arch_type) {
            case ArchType::LLAMA:    setup_layer_llama(model, static_cast<int>(l)); break;
            case ArchType::GPT2:     setup_layer_gpt2(model, static_cast<int>(l)); break;
            case ArchType::GPT_NEOX:  setup_layer_gpt_neox(model, static_cast<int>(l)); break;
            case ArchType::CODEGEN: {
                setup_layer_gpt2(model, static_cast<int>(l));
                // CodeGen uses parallel attn: only has ln_1, reuse for ffn_norm
                auto& lw2 = weights_.layers[l];
                if (lw2.ffn_norm.empty() && !lw2.attn_norm.empty()) {
                    lw2.ffn_norm = lw2.attn_norm;
                    lw2.ffn_norm_bias = lw2.attn_norm_bias;
                }
                break;
            }
            case ArchType::OPT:      setup_layer_opt(model, static_cast<int>(l)); break;
            case ArchType::STARCODER: setup_layer_gpt2(model, static_cast<int>(l)); break;
        }
    }

    // ─── KV cache ───
    cache_.init(static_cast<int>(config_.num_layers),
                static_cast<int>(config_.num_kv_heads),
                static_cast<int>(config_.head_dim),
                static_cast<int>(std::min(config_.max_seq_len, 8192u)));

    if (config_.use_rope) {
        rope_cache_.init(static_cast<int>(std::min(config_.max_seq_len, 8192u)),
                         static_cast<int>(config_.head_dim), config_.rope_theta);
    }

    log::info("Transformer ready [%s]: rope=%d, rms_norm=%d, swiglu=%d, parallel=%d",
              config_.architecture.c_str(), config_.use_rope,
              config_.use_rms_norm, config_.use_swiglu, config_.use_parallel_attn);
    return true;
}

// ─── Forward: GPT-2 / OPT (LayerNorm, learned pos, standard MLP) ───

Tensor Transformer::forward_gpt2(u32 token, int pos) {
    int hidden = static_cast<int>(config_.hidden_size);
    int vocab = static_cast<int>(config_.vocab_size);
    int embed_dim = static_cast<int>(weights_.token_embedding.shape()[1]);

    // Token embedding
    Tensor x_embed = weights_.token_embedding.row(token);

    // Position embedding (add in embed_dim space)
    if (!weights_.position_embedding.empty()) {
        int pos_offset = (config_.arch_type == ArchType::OPT) ? pos + 2 : pos;
        if (pos_offset < static_cast<int>(weights_.position_embedding.shape()[0])) {
            Tensor pos_emb = weights_.position_embedding.row(pos_offset);
            dispatch_add(x_embed.data_f32(), pos_emb.data_f32(), embed_dim);
        }
    }

    // OPT project_in: embed_dim -> hidden_size
    Tensor x;
    if (!weights_.project_in.empty() && embed_dim != hidden) {
        x = Tensor::zeros({static_cast<u32>(hidden)}, DType::F32);
        dispatch_matvec_opt(weights_.project_in.data_f32(), x_embed.data_f32(),
                            x.data_f32(), static_cast<u32>(hidden), static_cast<u32>(embed_dim));
    } else {
        x = std::move(x_embed);
    }
    f32* x_buf = x.data_f32();

    std::vector<f32> residual(hidden), normed(hidden), attn_out(hidden), mlp_out(hidden);

    for (u32 l = 0; l < config_.num_layers; l++) {
        const auto& lw = weights_.layers[l];

        memcpy(residual.data(), x_buf, hidden * sizeof(f32));

        // Pre-attention LayerNorm
        memcpy(normed.data(), x_buf, hidden * sizeof(f32));
        norm_inplace(normed.data(), lw.attn_norm, lw.attn_norm_bias, hidden);

        // Attention
        attention_forward(attn_out.data(), normed.data(), lw.attn, cache_,
                          static_cast<int>(l), pos, config_);

        // Residual
        dispatch_add(attn_out.data(), residual.data(), hidden);
        memcpy(residual.data(), attn_out.data(), hidden * sizeof(f32));

        // Pre-MLP LayerNorm
        memcpy(normed.data(), attn_out.data(), hidden * sizeof(f32));
        norm_inplace(normed.data(), lw.ffn_norm, lw.ffn_norm_bias, hidden);

        // Standard MLP (fc_in -> activation -> fc_out)
        mlp_standard(mlp_out.data(), normed.data(), lw);

        // Residual
        dispatch_add(mlp_out.data(), residual.data(), hidden);
        memcpy(x_buf, mlp_out.data(), hidden * sizeof(f32));
    }

    // Final norm
    norm_inplace(x_buf, weights_.output_norm, weights_.output_norm_bias, hidden);

    // OPT project_out: hidden -> embed_dim (if needed)
    const f32* proj_input = x_buf;
    int proj_dim = hidden;
    std::vector<f32> proj_buf;
    if (!weights_.project_out.empty() && embed_dim != hidden) {
        proj_buf.resize(embed_dim);
        dispatch_matvec_opt(weights_.project_out.data_f32(), x_buf,
                            proj_buf.data(), static_cast<u32>(embed_dim), static_cast<u32>(hidden));
        proj_input = proj_buf.data();
        proj_dim = embed_dim;
    }

    // Output projection
    Tensor logits = Tensor::zeros({static_cast<u32>(vocab)}, DType::F32);
    dispatch_matvec_opt(static_cast<const f32*>(weights_.output_proj_data ? weights_.output_proj_data : weights_.output_proj.data()),
                        proj_input, logits.data_f32(),
                        static_cast<u32>(vocab), static_cast<u32>(proj_dim));
    return logits;
}

// ─── Forward: GPT-NeoX / Pythia (parallel attn+MLP, RoPE, LayerNorm) ───

Tensor Transformer::forward_gpt_neox(u32 token, int pos) {
    int hidden = static_cast<int>(config_.hidden_size);
    int vocab = static_cast<int>(config_.vocab_size);

    Tensor x = weights_.token_embedding.row(token);
    f32* x_buf = x.data_f32();

    std::vector<f32> residual(hidden), normed_attn(hidden), normed_mlp(hidden);
    std::vector<f32> attn_out(hidden), mlp_out(hidden);

    for (u32 l = 0; l < config_.num_layers; l++) {
        const auto& lw = weights_.layers[l];

        memcpy(residual.data(), x_buf, hidden * sizeof(f32));

        // Parallel: both norms from same input
        memcpy(normed_attn.data(), x_buf, hidden * sizeof(f32));
        norm_inplace(normed_attn.data(), lw.attn_norm, lw.attn_norm_bias, hidden);

        memcpy(normed_mlp.data(), x_buf, hidden * sizeof(f32));
        norm_inplace(normed_mlp.data(), lw.ffn_norm, lw.ffn_norm_bias, hidden);

        // Attention (with RoPE)
        attention_forward(attn_out.data(), normed_attn.data(), lw.attn, cache_,
                          static_cast<int>(l), pos, config_);

        // MLP (standard, from same input)
        mlp_standard(mlp_out.data(), normed_mlp.data(), lw);

        // x = residual + attn_out + mlp_out
        dispatch_add(attn_out.data(), residual.data(), hidden);
        dispatch_add(attn_out.data(), mlp_out.data(), hidden);
        memcpy(x_buf, attn_out.data(), hidden * sizeof(f32));
    }

    norm_inplace(x_buf, weights_.output_norm, weights_.output_norm_bias, hidden);

    Tensor logits = Tensor::zeros({static_cast<u32>(vocab)}, DType::F32);
    dispatch_matvec_opt(static_cast<const f32*>(weights_.output_proj_data ? weights_.output_proj_data : weights_.output_proj.data()),
                        x_buf, logits.data_f32(),
                        static_cast<u32>(vocab), static_cast<u32>(hidden));
    return logits;
}

// ─── Forward: Llama/Qwen (RMSNorm, RoPE, SwiGLU) ───

Tensor Transformer::forward_llama(u32 token, int pos) {
    int hidden = static_cast<int>(config_.hidden_size);
    int vocab = static_cast<int>(config_.vocab_size);

    Tensor x = weights_.token_embedding.row(token);
    f32* x_buf = x.data_f32();

    std::vector<f32> residual(hidden), normed(hidden), attn_out(hidden), mlp_out(hidden);

    for (u32 l = 0; l < config_.num_layers; l++) {
        const auto& lw = weights_.layers[l];

        memcpy(residual.data(), x_buf, hidden * sizeof(f32));

        memcpy(normed.data(), x_buf, hidden * sizeof(f32));
        dispatch_rms_norm(normed.data(), lw.attn_norm.data_f32(), hidden, config_.rms_norm_eps);

        attention_forward(attn_out.data(), normed.data(), lw.attn, cache_,
                          static_cast<int>(l), pos, config_);

        dispatch_add(attn_out.data(), residual.data(), hidden);
        memcpy(residual.data(), attn_out.data(), hidden * sizeof(f32));

        memcpy(normed.data(), attn_out.data(), hidden * sizeof(f32));
        dispatch_rms_norm(normed.data(), lw.ffn_norm.data_f32(), hidden, config_.rms_norm_eps);

        mlp_forward(mlp_out.data(), normed.data(), lw.mlp, config_);

        dispatch_add(mlp_out.data(), residual.data(), hidden);
        memcpy(x_buf, mlp_out.data(), hidden * sizeof(f32));
    }

    dispatch_rms_norm(x_buf, weights_.output_norm.data_f32(), hidden, config_.rms_norm_eps);

    Tensor logits = Tensor::zeros({static_cast<u32>(vocab)}, DType::F32);
    if (weights_.output_proj_dtype == DType::F32) {
        dispatch_matvec_opt(weights_.output_proj.data_f32(), x_buf, logits.data_f32(),
                            static_cast<u32>(vocab), static_cast<u32>(hidden));
    } else {
        dispatch_matvec_q(weights_.output_proj_data, weights_.output_proj_size,
                          weights_.output_proj_dtype, weights_.output_proj_quant,
                          static_cast<u32>(vocab), static_cast<u32>(hidden),
                          x_buf, logits.data_f32());
    }
    return logits;
}

// ─── Main forward dispatch ───

Tensor Transformer::forward(u32 token, int pos) {
    QRAF_CHECK(token < config_.vocab_size, "Token %u >= vocab %u", token, config_.vocab_size);

    switch (config_.arch_type) {
        case ArchType::LLAMA:
            return forward_llama(token, pos);
        case ArchType::GPT2:
        case ArchType::OPT:
        case ArchType::STARCODER:
            return forward_gpt2(token, pos);
        case ArchType::GPT_NEOX:
            return forward_gpt_neox(token, pos);
        case ArchType::CODEGEN:
            return forward_gpt_neox(token, pos);
        default:
            return forward_llama(token, pos);
    }
}

void Transformer::reset() {
    cache_.reset();
}

} // namespace qraf
