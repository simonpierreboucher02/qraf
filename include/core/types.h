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

struct ModelConfig {
    std::string architecture;  // "llama", "gpt2", etc.
    u32 vocab_size     = 0;
    u32 hidden_size    = 0;
    u32 num_layers     = 0;
    u32 num_heads      = 0;
    u32 num_kv_heads   = 0;  // for GQA
    u32 intermediate_size = 0;
    u32 max_seq_len    = 2048;
    f32 rope_theta     = 10000.0f;
    f32 rms_norm_eps   = 1e-5f;
    u32 head_dim       = 0;  // computed: hidden_size / num_heads

    void compute_derived() {
        if (num_heads > 0) {
            head_dim = hidden_size / num_heads;
        }
        if (num_kv_heads == 0) {
            num_kv_heads = num_heads;
        }
    }
};

} // namespace qraf
