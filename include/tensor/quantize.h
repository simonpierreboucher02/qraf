#pragma once

#include "core/types.h"
#include "tensor/tensor.h"
#include "qraf/format.h"

namespace qraf {

// ─── Quantized Block Formats ───

// Q8_0: 8-bit quantization with per-block scale
// Block layout: [f32 scale][i8 x block_size]
struct BlockQ8_0 {
    f32 scale;
    i32 block_size;  // typically 32

    static constexpr size_t overhead_per_block(int /*bs*/) {
        return sizeof(f32);  // just the scale
    }
};

// Q4_0: 4-bit quantization with per-block scale
// Block layout: [f16 scale][u8 x (block_size/2)]  (two 4-bit values per byte)
struct BlockQ4_0 {
    f16 scale;
    i32 block_size;  // typically 32

    static constexpr size_t overhead_per_block(int /*bs*/) {
        return sizeof(f16);
    }
};

// ─── Dequantization functions ───
// These operate on individual blocks for on-the-fly dequant during matmul.

// Dequantize a Q8_0 block into f32 output
void dequantize_block_q8_0(const void* block_data, f32* output, int block_size);

// Dequantize a Q4_0 block into f32 output
void dequantize_block_q4_0(const void* block_data, f32* output, int block_size);

// ─── Full tensor dequantization ───

// Dequantize an entire quantized tensor to f32
Tensor dequantize_tensor(const void* data, u64 data_size,
                         const std::vector<u32>& shape,
                         DType dtype, const QuantScheme* quant);

// ─── Quantization functions ───

// Quantize f32 data to Q8_0 format
std::vector<u8> quantize_q8_0(const f32* data, size_t count, int block_size = 32);

// Quantize f32 data to Q4_0 format
std::vector<u8> quantize_q4_0(const f32* data, size_t count, int block_size = 32);

// ─── Quantized dot product ───
// Compute dot product directly on quantized data without full dequantization

f32 dot_q8_0_f32(const void* q8_data, const f32* f32_data, size_t count, int block_size = 32);
f32 dot_q4_0_f32(const void* q4_data, const f32* f32_data, size_t count, int block_size = 32);

} // namespace qraf
