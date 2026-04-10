#pragma once

#include "core/types.h"
#include <cstring>

namespace qraf {

// ─── QRAF Binary Format Specification ───
//
// File layout:
//   [QrafHeader]              offset 0
//   [ModelConfigBlock]        offset sizeof(QrafHeader)
//   [TokenizerBlock]          variable
//   [TensorDirectory]         variable
//   [QuantDirectory]          variable
//   [StringTable]             variable
//   [Data blocks...]          aligned to 64 bytes
//
// All multi-byte values are little-endian.
// All data blocks are aligned to 64-byte boundaries for mmap/SIMD.

static constexpr u64 DATA_ALIGNMENT = 64;

// ─── File Header ───
#pragma pack(push, 1)
struct QrafHeader {
    u32 magic;           // QRAF_MAGIC = 0x46415251
    u32 version;         // QRAF_VERSION = 1
    u64 file_size;       // total file size
    u64 config_offset;   // offset to ModelConfigBlock
    u64 config_size;
    u64 tokenizer_offset;
    u64 tokenizer_size;
    u64 tensor_dir_offset;
    u64 tensor_dir_size;
    u64 quant_dir_offset;
    u64 quant_dir_size;
    u64 string_table_offset;
    u64 string_table_size;
    u64 data_offset;     // start of tensor data blocks
    u32 num_tensors;
    u32 num_quant_schemes;
    u32 flags;           // reserved
    u32 padding;         // alignment
    u8  reserved[8];     // pad to 128 bytes
};
#pragma pack(pop)

static_assert(sizeof(QrafHeader) == 128, "QrafHeader must be 128 bytes");

// ─── Model Config Block ───
// Stored as key-value pairs in a simple binary format:
//   [u32 num_entries]
//   [ConfigEntry] * num_entries
// where each ConfigEntry is:
//   [u32 key_offset]  (into string table)
//   [u32 value_type]  (0=u32, 1=f32, 2=string_offset)
//   [u32 value]

#pragma pack(push, 1)
struct ConfigEntry {
    u32 key_offset;    // offset into string table
    u32 value_type;    // 0 = u32, 1 = f32, 2 = string (offset into string table)
    u32 value;         // raw bits (interpret based on value_type)
};
#pragma pack(pop)

// ─── Tensor Metadata Entry ───
#pragma pack(push, 1)
struct TensorMeta {
    u64 name_hash;       // FNV-1a hash of tensor name
    u32 name_offset;     // offset into string table
    u32 ndim;            // number of dimensions (max 8)
    u32 shape[8];        // shape array
    u32 dtype;           // DType enum value
    u32 quant_scheme_id; // index into quant directory (0xFFFFFFFF = none)
    u64 data_offset;     // offset from file start to tensor data
    u64 data_size;       // size in bytes of tensor data
    u32 layout_id;       // 0 = row-major
    u32 padding;         // alignment
};
#pragma pack(pop)

static_assert(sizeof(TensorMeta) == 80, "TensorMeta must be 80 bytes");

// ─── Quantization Scheme Entry ───
#pragma pack(push, 1)
struct QuantScheme {
    u32 id;
    u32 type;          // matches DType for quant types
    u32 block_size;    // number of elements per quantization block
    u32 group_size;    // number of blocks per group (for group quantization)
    f32 scale;         // global scale factor (0 = per-block scales stored in data)
    f32 zero_point;    // global zero point
    u32 flags;         // bit 0: has per-block scales, bit 1: has per-block zeros
    u32 padding;
};
#pragma pack(pop)

static_assert(sizeof(QuantScheme) == 32, "QuantScheme must be 32 bytes");

// ─── Tokenizer Block ───
// Layout:
//   [u32 vocab_size]
//   [u32 merges_count]
//   [u32 special_tokens_count]
//   [u32 reserved]
//   [TokenEntry] * vocab_size
//   [MergeEntry] * merges_count
//   [SpecialToken] * special_tokens_count

#pragma pack(push, 1)
struct TokenEntry {
    u32 string_offset;   // offset into string table
    u32 string_length;
    f32 score;           // BPE score / priority
    u32 type;            // 0=normal, 1=byte, 2=special
};

struct MergeEntry {
    u32 token_a;
    u32 token_b;
    u32 result;
    f32 priority;
};

struct SpecialToken {
    u32 string_offset;
    u32 string_length;
    u32 token_id;
    u32 type;  // 0=bos, 1=eos, 2=pad, 3=unk
};
#pragma pack(pop)

// ─── Utility: FNV-1a hash ───
inline u64 fnv1a_hash(const char* str, size_t len) {
    u64 hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= static_cast<u64>(str[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline u64 fnv1a_hash(const std::string& str) {
    return fnv1a_hash(str.data(), str.size());
}

// ─── Align offset to boundary ───
inline u64 align_offset(u64 offset, u64 alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

} // namespace qraf
