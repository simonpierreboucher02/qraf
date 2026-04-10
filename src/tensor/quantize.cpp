#include "tensor/quantize.h"
#include "core/logging.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace qraf {

// Forward declarations for fp16 conversion
f32 fp16_to_fp32(f16 h);
f16 fp32_to_fp16(f32 f);

// ─── Q8_0 Dequantization ───
// Block layout: [f32 scale][i8 x block_size]

void dequantize_block_q8_0(const void* block_data, f32* output, int block_size) {
    const u8* ptr = static_cast<const u8*>(block_data);

    f32 scale;
    memcpy(&scale, ptr, sizeof(f32));
    ptr += sizeof(f32);

    const int8_t* quants = reinterpret_cast<const int8_t*>(ptr);
    for (int i = 0; i < block_size; i++) {
        output[i] = static_cast<f32>(quants[i]) * scale;
    }
}

// ─── Q4_0 Dequantization ───
// Block layout: [f16 scale][u8 x (block_size/2)]
// Each byte holds two 4-bit signed values (low nibble first)
// Values are in range [-8, 7], centered at 0

void dequantize_block_q4_0(const void* block_data, f32* output, int block_size) {
    const u8* ptr = static_cast<const u8*>(block_data);

    f16 scale_f16;
    memcpy(&scale_f16, ptr, sizeof(f16));
    f32 scale = fp16_to_fp32(scale_f16);
    ptr += sizeof(f16);

    int half = block_size / 2;
    for (int i = 0; i < half; i++) {
        u8 packed = ptr[i];
        int lo = (packed & 0x0F) - 8;
        int hi = ((packed >> 4) & 0x0F) - 8;
        output[2 * i]     = static_cast<f32>(lo) * scale;
        output[2 * i + 1] = static_cast<f32>(hi) * scale;
    }
}

// ─── Full tensor dequantization ───

Tensor dequantize_tensor(const void* data, u64 data_size,
                         const std::vector<u32>& shape,
                         DType dtype, const QuantScheme* quant) {
    size_t numel = 1;
    for (auto s : shape) numel *= s;

    Tensor out = Tensor::zeros(shape, DType::F32);
    f32* out_ptr = out.data_f32();

    if (dtype == DType::Q8_0) {
        int block_size = quant ? static_cast<int>(quant->block_size) : 32;
        size_t num_blocks = (numel + block_size - 1) / block_size;
        size_t block_bytes = sizeof(f32) + block_size; // scale + quants

        const u8* ptr = static_cast<const u8*>(data);
        for (size_t b = 0; b < num_blocks; b++) {
            int this_block = static_cast<int>(
                std::min(static_cast<size_t>(block_size), numel - b * block_size)
            );
            dequantize_block_q8_0(ptr, out_ptr + b * block_size, this_block);
            ptr += sizeof(f32) + block_size;
        }
    } else if (dtype == DType::Q4_0) {
        int block_size = quant ? static_cast<int>(quant->block_size) : 32;
        size_t num_blocks = (numel + block_size - 1) / block_size;

        const u8* ptr = static_cast<const u8*>(data);
        for (size_t b = 0; b < num_blocks; b++) {
            int this_block = static_cast<int>(
                std::min(static_cast<size_t>(block_size), numel - b * block_size)
            );
            dequantize_block_q4_0(ptr, out_ptr + b * block_size, this_block);
            ptr += sizeof(f16) + block_size / 2;
        }
    } else if (dtype == DType::F16) {
        const f16* src = static_cast<const f16*>(data);
        for (size_t i = 0; i < numel; i++) {
            out_ptr[i] = fp16_to_fp32(src[i]);
        }
    } else if (dtype == DType::F32) {
        memcpy(out_ptr, data, numel * sizeof(f32));
    } else {
        log::error("Unsupported dtype for dequantization: %s", dtype_name(dtype));
    }

    return out;
}

// ─── Q8_0 Quantization ───

std::vector<u8> quantize_q8_0(const f32* data, size_t count, int block_size) {
    size_t num_blocks = (count + block_size - 1) / block_size;
    size_t block_bytes = sizeof(f32) + block_size;
    std::vector<u8> result(num_blocks * block_bytes, 0);

    u8* out = result.data();
    for (size_t b = 0; b < num_blocks; b++) {
        size_t offset = b * block_size;
        int this_block = static_cast<int>(std::min(static_cast<size_t>(block_size), count - offset));

        // Find max absolute value in block
        f32 max_abs = 0.0f;
        for (int i = 0; i < this_block; i++) {
            f32 abs_val = std::fabs(data[offset + i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        f32 scale = max_abs / 127.0f;
        f32 inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

        // Write scale
        memcpy(out, &scale, sizeof(f32));
        out += sizeof(f32);

        // Quantize values
        int8_t* quants = reinterpret_cast<int8_t*>(out);
        for (int i = 0; i < this_block; i++) {
            f32 v = data[offset + i] * inv_scale;
            v = std::max(-127.0f, std::min(127.0f, v));
            quants[i] = static_cast<int8_t>(std::round(v));
        }
        out += block_size;
    }

    return result;
}

// ─── Q4_0 Quantization ───

std::vector<u8> quantize_q4_0(const f32* data, size_t count, int block_size) {
    size_t num_blocks = (count + block_size - 1) / block_size;
    size_t block_bytes = sizeof(f16) + block_size / 2;
    std::vector<u8> result(num_blocks * block_bytes, 0);

    u8* out = result.data();
    for (size_t b = 0; b < num_blocks; b++) {
        size_t offset = b * block_size;
        int this_block = static_cast<int>(std::min(static_cast<size_t>(block_size), count - offset));

        // Find max absolute value
        f32 max_abs = 0.0f;
        for (int i = 0; i < this_block; i++) {
            f32 abs_val = std::fabs(data[offset + i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }

        f32 scale = max_abs / 7.0f;
        f32 inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        f16 scale_f16 = fp32_to_fp16(scale);

        // Write scale
        memcpy(out, &scale_f16, sizeof(f16));
        out += sizeof(f16);

        // Pack 4-bit values, two per byte
        for (int i = 0; i < this_block / 2; i++) {
            f32 v0 = data[offset + 2 * i] * inv_scale;
            f32 v1 = data[offset + 2 * i + 1] * inv_scale;
            v0 = std::max(-8.0f, std::min(7.0f, std::round(v0)));
            v1 = std::max(-8.0f, std::min(7.0f, std::round(v1)));
            u8 lo = static_cast<u8>(static_cast<int>(v0) + 8) & 0x0F;
            u8 hi = static_cast<u8>(static_cast<int>(v1) + 8) & 0x0F;
            out[i] = lo | (hi << 4);
        }
        out += block_size / 2;
    }

    return result;
}

// ─── Quantized Dot Products ───

f32 dot_q8_0_f32(const void* q8_data, const f32* f32_data, size_t count, int block_size) {
    const u8* ptr = static_cast<const u8*>(q8_data);
    f32 total = 0.0f;
    size_t num_blocks = (count + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        f32 scale;
        memcpy(&scale, ptr, sizeof(f32));
        ptr += sizeof(f32);

        const int8_t* quants = reinterpret_cast<const int8_t*>(ptr);
        size_t offset = b * block_size;
        int this_block = static_cast<int>(std::min(static_cast<size_t>(block_size), count - offset));

        f32 block_sum = 0.0f;
        for (int i = 0; i < this_block; i++) {
            block_sum += static_cast<f32>(quants[i]) * f32_data[offset + i];
        }
        total += block_sum * scale;
        ptr += block_size;
    }

    return total;
}

f32 dot_q4_0_f32(const void* q4_data, const f32* f32_data, size_t count, int block_size) {
    const u8* ptr = static_cast<const u8*>(q4_data);
    f32 total = 0.0f;
    size_t num_blocks = (count + block_size - 1) / block_size;

    for (size_t b = 0; b < num_blocks; b++) {
        f16 scale_f16;
        memcpy(&scale_f16, ptr, sizeof(f16));
        f32 scale = fp16_to_fp32(scale_f16);
        ptr += sizeof(f16);

        size_t offset = b * block_size;
        int this_block = static_cast<int>(std::min(static_cast<size_t>(block_size), count - offset));

        f32 block_sum = 0.0f;
        for (int i = 0; i < this_block / 2; i++) {
            u8 packed = ptr[i];
            int lo = (packed & 0x0F) - 8;
            int hi = ((packed >> 4) & 0x0F) - 8;
            block_sum += static_cast<f32>(lo) * f32_data[offset + 2 * i];
            block_sum += static_cast<f32>(hi) * f32_data[offset + 2 * i + 1];
        }
        total += block_sum * scale;
        ptr += block_size / 2;
    }

    return total;
}

} // namespace qraf
