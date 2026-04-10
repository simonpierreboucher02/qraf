#include "nn/simd_neon.h"
#include "core/types.h"
#include <cmath>
#include <cstring>

#if QRAF_HAS_NEON

namespace qraf {
namespace neon {

// ─── F32 Dot Product (4 accumulators, 16-wide unroll) ───
f32 dot_f32(const f32* a, const f32* b, int n) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int i = 0;
    for (; i + 15 < n; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        acc0 = vfmaq_f32(acc0, a0, b0);
        acc1 = vfmaq_f32(acc1, a1, b1);
        acc2 = vfmaq_f32(acc2, a2, b2);
        acc3 = vfmaq_f32(acc3, a3, b3);
    }

    acc0 = vaddq_f32(acc0, acc1);
    acc2 = vaddq_f32(acc2, acc3);
    acc0 = vaddq_f32(acc0, acc2);
    f32 sum = vaddvq_f32(acc0);

    // Remainder
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// ─── F32 MatVec: y = W @ x ───
void matvec_f32(const f32* W, const f32* x, f32* y, u32 out_dim, u32 in_dim) {
    for (u32 i = 0; i < out_dim; i++) {
        y[i] = dot_f32(W + i * in_dim, x, static_cast<int>(in_dim));
    }
}

// ─── RMSNorm ───
void rms_norm(f32* x, const f32* weight, int size, f32 eps) {
    // Pass 1: sum of squares
    float32x4_t ss_vec = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        ss_vec = vfmaq_f32(ss_vec, v, v);
    }
    f32 ss = vaddvq_f32(ss_vec);
    for (; i < size; i++) ss += x[i] * x[i];

    f32 rsqrt = 1.0f / std::sqrt(ss / static_cast<f32>(size) + eps);
    float32x4_t rsqrt_vec = vdupq_n_f32(rsqrt);

    // Pass 2: normalize and scale
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float32x4_t w = vld1q_f32(weight + i);
        v = vmulq_f32(v, rsqrt_vec);
        v = vmulq_f32(v, w);
        vst1q_f32(x + i, v);
    }
    for (; i < size; i++) {
        x[i] = x[i] * rsqrt * weight[i];
    }
}

// ─── Softmax ───
void softmax(f32* x, int size) {
    // Find max
    float32x4_t max_vec;
    int i = 0;
    if (size >= 4) {
        max_vec = vld1q_f32(x);
        i = 4;
        for (; i + 3 < size; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            max_vec = vmaxq_f32(max_vec, v);
        }
    } else {
        max_vec = vdupq_n_f32(-1e30f);
    }
    f32 max_val = vmaxvq_f32(max_vec);
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    // exp(x - max) and sum
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    float32x4_t max_broadcast = vdupq_n_f32(max_val);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v = vsubq_f32(v, max_broadcast);
        // Scalar exp for correctness (NEON has no native exp)
        f32 tmp[4];
        vst1q_f32(tmp, v);
        tmp[0] = std::exp(tmp[0]);
        tmp[1] = std::exp(tmp[1]);
        tmp[2] = std::exp(tmp[2]);
        tmp[3] = std::exp(tmp[3]);
        v = vld1q_f32(tmp);
        vst1q_f32(x + i, v);
        sum_vec = vaddq_f32(sum_vec, v);
    }
    f32 sum = vaddvq_f32(sum_vec);
    for (; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    // Scale by 1/sum
    f32 inv_sum = 1.0f / sum;
    float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);
    i = 0;
    for (; i + 3 < size; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        v = vmulq_f32(v, inv_sum_vec);
        vst1q_f32(x + i, v);
    }
    for (; i < size; i++) {
        x[i] *= inv_sum;
    }
}

// ─── SiLU: x * sigmoid(x) = x / (1 + exp(-x)) ───
void silu(f32* x, int size) {
    int i = 0;
    for (; i + 3 < size; i += 4) {
        f32 tmp[4];
        vst1q_f32(tmp, vld1q_f32(x + i));
        tmp[0] = tmp[0] / (1.0f + std::exp(-tmp[0]));
        tmp[1] = tmp[1] / (1.0f + std::exp(-tmp[1]));
        tmp[2] = tmp[2] / (1.0f + std::exp(-tmp[2]));
        tmp[3] = tmp[3] / (1.0f + std::exp(-tmp[3]));
        vst1q_f32(x + i, vld1q_f32(tmp));
    }
    for (; i < size; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// ─── Element-wise add ───
void add(f32* x, const f32* y, int size) {
    int i = 0;
    for (; i + 15 < size; i += 16) {
        vst1q_f32(x + i,      vaddq_f32(vld1q_f32(x + i),      vld1q_f32(y + i)));
        vst1q_f32(x + i + 4,  vaddq_f32(vld1q_f32(x + i + 4),  vld1q_f32(y + i + 4)));
        vst1q_f32(x + i + 8,  vaddq_f32(vld1q_f32(x + i + 8),  vld1q_f32(y + i + 8)));
        vst1q_f32(x + i + 12, vaddq_f32(vld1q_f32(x + i + 12), vld1q_f32(y + i + 12)));
    }
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(y + i)));
    }
    for (; i < size; i++) x[i] += y[i];
}

// ─── Element-wise mul ───
void mul(f32* x, const f32* y, int size) {
    int i = 0;
    for (; i + 15 < size; i += 16) {
        vst1q_f32(x + i,      vmulq_f32(vld1q_f32(x + i),      vld1q_f32(y + i)));
        vst1q_f32(x + i + 4,  vmulq_f32(vld1q_f32(x + i + 4),  vld1q_f32(y + i + 4)));
        vst1q_f32(x + i + 8,  vmulq_f32(vld1q_f32(x + i + 8),  vld1q_f32(y + i + 8)));
        vst1q_f32(x + i + 12, vmulq_f32(vld1q_f32(x + i + 12), vld1q_f32(y + i + 12)));
    }
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), vld1q_f32(y + i)));
    }
    for (; i < size; i++) x[i] *= y[i];
}

// ─── Scale ───
void scale(f32* x, f32 s, int size) {
    float32x4_t sv = vdupq_n_f32(s);
    int i = 0;
    for (; i + 15 < size; i += 16) {
        vst1q_f32(x + i,      vmulq_f32(vld1q_f32(x + i), sv));
        vst1q_f32(x + i + 4,  vmulq_f32(vld1q_f32(x + i + 4), sv));
        vst1q_f32(x + i + 8,  vmulq_f32(vld1q_f32(x + i + 8), sv));
        vst1q_f32(x + i + 12, vmulq_f32(vld1q_f32(x + i + 12), sv));
    }
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), sv));
    }
    for (; i < size; i++) x[i] *= s;
}

// ─── RoPE with precomputed cos/sin ───
void rope_apply(f32* q, f32* k, const f32* cos_row, const f32* sin_row,
                int head_dim, int /*pos*/) {
    int half = head_dim / 2;
    int i = 0;
    for (; i + 3 < half; i += 4) {
        // Load interleaved pairs using vld2
        float32x4x2_t qp = vld2q_f32(q + 2 * i);
        float32x4x2_t kp = vld2q_f32(k + 2 * i);
        float32x4_t c = vld1q_f32(cos_row + i);
        float32x4_t s = vld1q_f32(sin_row + i);

        // q_new = (q_even * cos - q_odd * sin, q_even * sin + q_odd * cos)
        float32x4_t q_even_new = vmlsq_f32(vmulq_f32(qp.val[0], c), qp.val[1], s);
        float32x4_t q_odd_new  = vmlaq_f32(vmulq_f32(qp.val[0], s), qp.val[1], c);
        float32x4x2_t q_out = {q_even_new, q_odd_new};
        vst2q_f32(q + 2 * i, q_out);

        // Same for k
        float32x4_t k_even_new = vmlsq_f32(vmulq_f32(kp.val[0], c), kp.val[1], s);
        float32x4_t k_odd_new  = vmlaq_f32(vmulq_f32(kp.val[0], s), kp.val[1], c);
        float32x4x2_t k_out = {k_even_new, k_odd_new};
        vst2q_f32(k + 2 * i, k_out);
    }
    // Scalar remainder
    for (; i < half; i++) {
        f32 cos_v = cos_row[i];
        f32 sin_v = sin_row[i];

        f32 q0 = q[2 * i], q1 = q[2 * i + 1];
        q[2 * i]     = q0 * cos_v - q1 * sin_v;
        q[2 * i + 1] = q0 * sin_v + q1 * cos_v;

        f32 k0 = k[2 * i], k1 = k[2 * i + 1];
        k[2 * i]     = k0 * cos_v - k1 * sin_v;
        k[2 * i + 1] = k0 * sin_v + k1 * cos_v;
    }
}

// ─── Quantized Q8_0 dot product (NEON) ───
f32 dot_q8_0_f32(const void* q8_data, const f32* f32_data, int count, int block_size) {
    const u8* ptr = static_cast<const u8*>(q8_data);
    f32 total = 0.0f;
    int num_blocks = (count + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        f32 scale_val;
        memcpy(&scale_val, ptr, sizeof(f32));
        ptr += sizeof(f32);

        const int8_t* quants = reinterpret_cast<const int8_t*>(ptr);
        int offset = b * block_size;
        int this_block = (offset + block_size <= count) ? block_size : (count - offset);

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j + 15 < this_block; j += 16) {
            // Load 16 int8, widen to f32
            int8x16_t q_i8 = vld1q_s8(quants + j);
            int8x8_t lo8 = vget_low_s8(q_i8);
            int8x8_t hi8 = vget_high_s8(q_i8);

            int16x8_t lo16 = vmovl_s8(lo8);
            int16x8_t hi16 = vmovl_s8(hi8);

            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));

            float32x4_t x0 = vld1q_f32(f32_data + offset + j);
            float32x4_t x1 = vld1q_f32(f32_data + offset + j + 4);
            float32x4_t x2 = vld1q_f32(f32_data + offset + j + 8);
            float32x4_t x3 = vld1q_f32(f32_data + offset + j + 12);

            acc0 = vfmaq_f32(acc0, f0, x0);
            acc1 = vfmaq_f32(acc1, f1, x1);
            acc2 = vfmaq_f32(acc2, f2, x2);
            acc3 = vfmaq_f32(acc3, f3, x3);
        }

        acc0 = vaddq_f32(acc0, acc1);
        acc2 = vaddq_f32(acc2, acc3);
        acc0 = vaddq_f32(acc0, acc2);
        f32 block_sum = vaddvq_f32(acc0);

        // Scalar remainder
        for (; j < this_block; j++) {
            block_sum += static_cast<f32>(quants[j]) * f32_data[offset + j];
        }

        total += block_sum * scale_val;
        ptr += block_size;
    }
    return total;
}

// ─── Quantized Q4_0 dot product (NEON) ───

// fp16 conversion (defined in dtype.cpp, qraf namespace)
namespace { // local helper
    f32 fp16_to_f32_local(f16 h) {
        u32 sign = (h >> 15) & 0x1;
        u32 exp  = (h >> 10) & 0x1F;
        u32 mant = h & 0x3FF;
        u32 result;
        if (exp == 0) {
            result = (mant == 0) ? (sign << 31) : ((sign << 31) | ((128 - 15) << 23) | (mant << 13));
        } else if (exp == 31) {
            result = (sign << 31) | 0x7F800000 | (mant << 13);
        } else {
            result = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        f32 f; memcpy(&f, &result, sizeof(f32));
        return f;
    }
}

f32 dot_q4_0_f32(const void* q4_data, const f32* f32_data, int count, int block_size) {
    const u8* ptr = static_cast<const u8*>(q4_data);
    f32 total = 0.0f;
    int num_blocks = (count + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        f16 scale_f16;
        memcpy(&scale_f16, ptr, sizeof(f16));
        f32 scale_val = fp16_to_f32_local(scale_f16);
        ptr += sizeof(f16);

        int offset = b * block_size;
        int this_block = (offset + block_size <= count) ? block_size : (count - offset);
        int half = this_block / 2;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        int8x16_t eight_vec = vdupq_n_s8(8);

        int j = 0;
        for (; j + 15 < half; j += 16) {
            // Load 16 packed bytes = 32 values
            uint8x16_t packed = vld1q_u8(ptr + j);

            // Extract nibbles
            uint8x16_t lo_u = vandq_u8(packed, vdupq_n_u8(0x0F));
            uint8x16_t hi_u = vshrq_n_u8(packed, 4);

            // Center: subtract 8
            int8x16_t lo_s = vsubq_s8(vreinterpretq_s8_u8(lo_u), eight_vec);
            int8x16_t hi_s = vsubq_s8(vreinterpretq_s8_u8(hi_u), eight_vec);

            // Interleave: lo[0],hi[0],lo[1],hi[1],... to match original order
            int8x16x2_t interleaved = vzipq_s8(lo_s, hi_s);

            // Process first 16 values (interleaved.val[0])
            {
                int8x8_t a8 = vget_low_s8(interleaved.val[0]);
                int8x8_t b8 = vget_high_s8(interleaved.val[0]);
                int16x8_t a16 = vmovl_s8(a8);
                int16x8_t b16 = vmovl_s8(b8);

                float32x4_t fa0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a16)));
                float32x4_t fa1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a16)));
                float32x4_t fb0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t fb1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));

                int base = offset + 2 * j;
                acc0 = vfmaq_f32(acc0, fa0, vld1q_f32(f32_data + base));
                acc0 = vfmaq_f32(acc0, fa1, vld1q_f32(f32_data + base + 4));
                acc0 = vfmaq_f32(acc0, fb0, vld1q_f32(f32_data + base + 8));
                acc0 = vfmaq_f32(acc0, fb1, vld1q_f32(f32_data + base + 12));
            }

            // Process second 16 values (interleaved.val[1])
            {
                int8x8_t a8 = vget_low_s8(interleaved.val[1]);
                int8x8_t b8 = vget_high_s8(interleaved.val[1]);
                int16x8_t a16 = vmovl_s8(a8);
                int16x8_t b16 = vmovl_s8(b8);

                float32x4_t fa0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a16)));
                float32x4_t fa1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a16)));
                float32x4_t fb0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t fb1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));

                int base = offset + 2 * j + 16;
                acc1 = vfmaq_f32(acc1, fa0, vld1q_f32(f32_data + base));
                acc1 = vfmaq_f32(acc1, fa1, vld1q_f32(f32_data + base + 4));
                acc1 = vfmaq_f32(acc1, fb0, vld1q_f32(f32_data + base + 8));
                acc1 = vfmaq_f32(acc1, fb1, vld1q_f32(f32_data + base + 12));
            }
        }

        acc0 = vaddq_f32(acc0, acc1);
        f32 block_sum = vaddvq_f32(acc0);

        // Scalar remainder
        for (; j < half; j++) {
            u8 packed = ptr[j];
            int lo = (packed & 0x0F) - 8;
            int hi = ((packed >> 4) & 0x0F) - 8;
            block_sum += static_cast<f32>(lo) * f32_data[offset + 2 * j];
            block_sum += static_cast<f32>(hi) * f32_data[offset + 2 * j + 1];
        }

        total += block_sum * scale_val;
        ptr += half;
    }
    return total;
}

} // namespace neon
} // namespace qraf

#endif // QRAF_HAS_NEON
