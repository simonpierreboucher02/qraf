#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cassert>

namespace qraf {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;
using f16 = uint16_t;  // stored as raw bits; converted on use
using f32 = float;
using f64 = double;

static constexpr u32 QRAF_MAGIC = 0x46415251; // "QRAF" in little-endian
static constexpr u32 QRAF_VERSION = 1;

enum class DType : u32 {
    F32   = 0,
    F16   = 1,
    BF16  = 2,
    Q8_0  = 10,
    Q4_0  = 20,
    Q4_1  = 21,
    Q6_0  = 30,
    I32   = 100,
    I16   = 101,
    I8    = 102,
    U8    = 103,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::Q8_0: return 1;
        case DType::Q4_0: return 1; // packed: 2 values per byte
        case DType::Q4_1: return 1;
        case DType::Q6_0: return 1;
        case DType::I32:  return 4;
        case DType::I16:  return 2;
        case DType::I8:   return 1;
        case DType::U8:   return 1;
        default:          return 0;
    }
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::F32:  return "f32";
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::Q8_0: return "q8_0";
        case DType::Q4_0: return "q4_0";
        case DType::Q4_1: return "q4_1";
        case DType::Q6_0: return "q6_0";
        case DType::I32:  return "i32";
        case DType::I16:  return "i16";
        case DType::I8:   return "i8";
        case DType::U8:   return "u8";
        default:          return "unknown";
    }
}

inline bool dtype_is_quantized(DType dt) {
    switch (dt) {
        case DType::Q8_0:
        case DType::Q4_0:
        case DType::Q4_1:
        case DType::Q6_0:
            return true;
        default:
            return false;
    }
}

// Architecture family
enum class ArchType {
    LLAMA,      // LLaMA, Qwen2, SmolLM, TinyLlama (RMSNorm, RoPE, SwiGLU)
    GPT2,       // GPT-2, DialoGPT, Cerebras-GPT, LaMini (LayerNorm, learned pos, GELU)
    GPT_NEOX,   // Pythia (LayerNorm, RoPE, parallel attn+MLP, GELU)
    OPT,        // OPT (LayerNorm, learned pos, ReLU)
    CODEGEN,    // CodeGen (LayerNorm, RoPE, GELU)
    STARCODER,  // StarCoder/BigCode (LayerNorm, learned pos, MQA, GELU)
};

struct ModelConfig {
    std::string architecture;  // raw string: "llama", "gpt2", etc.
    ArchType arch_type = ArchType::LLAMA;

    u32 vocab_size     = 0;
    u32 hidden_size    = 0;
    u32 num_layers     = 0;
    u32 num_heads      = 0;
    u32 num_kv_heads   = 0;  // for GQA/MQA
    u32 intermediate_size = 0;
    u32 max_seq_len    = 2048;
    f32 rope_theta     = 10000.0f;
    f32 rms_norm_eps   = 1e-5f;  // also used as layer_norm_eps
    u32 head_dim       = 0;

    // Architecture-specific
    bool use_rope           = true;   // false for GPT-2, OPT (learned pos embeds)
    bool use_rms_norm       = true;   // false for GPT-2/OPT/Pythia (use LayerNorm)
    bool use_swiglu         = true;   // false for GPT-2/OPT/Pythia (standard MLP)
    bool use_parallel_attn  = false;  // true for GPT-NeoX/Pythia
    bool has_bias           = false;  // attention bias (Qwen, GPT-2)
    bool has_mlp_bias       = false;  // MLP bias (GPT-2, OPT)
    std::string activation  = "silu"; // "silu", "gelu", "relu"

    void compute_derived() {
        if (num_heads > 0) head_dim = hidden_size / num_heads;
        if (num_kv_heads == 0) num_kv_heads = num_heads;

        // Set architecture-specific defaults
        if (architecture == "llama" || architecture == "qwen2" || architecture == "mistral") {
            arch_type = ArchType::LLAMA;
            use_rope = true; use_rms_norm = true; use_swiglu = true;
            activation = "silu";
            if (architecture == "qwen2") has_bias = true;
        } else if (architecture == "gpt2") {
            arch_type = ArchType::GPT2;
            use_rope = false; use_rms_norm = false; use_swiglu = false;
            has_bias = true; has_mlp_bias = true; activation = "gelu";
            if (intermediate_size == 0) intermediate_size = hidden_size * 4;
        } else if (architecture == "gpt_neox") {
            arch_type = ArchType::GPT_NEOX;
            use_rope = true; use_rms_norm = false; use_swiglu = false;
            use_parallel_attn = true; activation = "gelu";
            if (intermediate_size == 0) intermediate_size = hidden_size * 4;
        } else if (architecture == "opt") {
            arch_type = ArchType::OPT;
            use_rope = false; use_rms_norm = false; use_swiglu = false;
            has_bias = true; has_mlp_bias = true; activation = "relu";
            if (intermediate_size == 0) intermediate_size = hidden_size * 4;
        } else if (architecture == "codegen") {
            arch_type = ArchType::CODEGEN;
            use_rope = true; use_rms_norm = false; use_swiglu = false;
            use_parallel_attn = true; activation = "gelu";
            if (intermediate_size == 0) intermediate_size = hidden_size * 4;
        } else if (architecture == "gpt_bigcode") {
            arch_type = ArchType::STARCODER;
            use_rope = false; use_rms_norm = false; use_swiglu = false;
            has_bias = true; has_mlp_bias = true; activation = "gelu";
            num_kv_heads = 1; // MQA
            if (intermediate_size == 0) intermediate_size = hidden_size * 4;
        }
    }
};

} // namespace qraf
