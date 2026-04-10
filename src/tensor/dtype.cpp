#include "core/types.h"
#include <cmath>
#include <cstring>

namespace qraf {

// ─── FP16 conversion utilities ───

f32 fp16_to_fp32(f16 h) {
    u32 sign = (h >> 15) & 0x1;
    u32 exp  = (h >> 10) & 0x1F;
    u32 mant = h & 0x3FF;

    u32 result;
    if (exp == 0) {
        if (mant == 0) {
            result = sign << 31;
        } else {
            // Subnormal
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            result = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        result = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        result = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    f32 f;
    memcpy(&f, &result, sizeof(f32));
    return f;
}

f16 fp32_to_fp16(f32 f) {
    u32 bits;
    memcpy(&bits, &f, sizeof(u32));

    u32 sign = (bits >> 31) & 0x1;
    i32 exp  = static_cast<i32>((bits >> 23) & 0xFF) - 127;
    u32 mant = bits & 0x7FFFFF;

    u16 result;
    if (exp > 15) {
        result = static_cast<u16>((sign << 15) | 0x7C00); // inf
    } else if (exp > -15) {
        u32 h_exp = static_cast<u32>(exp + 15);
        u32 h_mant = mant >> 13;
        result = static_cast<u16>((sign << 15) | (h_exp << 10) | h_mant);
    } else {
        result = static_cast<u16>(sign << 15); // zero / subnormal
    }

    return result;
}

// ─── BF16 conversion utilities ───

f32 bf16_to_fp32(u16 b) {
    u32 bits = static_cast<u32>(b) << 16;
    f32 f;
    memcpy(&f, &bits, sizeof(f32));
    return f;
}

u16 fp32_to_bf16(f32 f) {
    u32 bits;
    memcpy(&bits, &f, sizeof(u32));
    return static_cast<u16>(bits >> 16);
}

} // namespace qraf
