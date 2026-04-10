#pragma once

#include "core/types.h"

#if defined(__aarch64__) && defined(QRAF_NEON)
#include <arm_neon.h>
#define QRAF_HAS_NEON 1
#else
#define QRAF_HAS_NEON 0
#endif

namespace qraf {
namespace neon {

#if QRAF_HAS_NEON

// ─── F32 operations ───
f32 dot_f32(const f32* a, const f32* b, int n);
void matvec_f32(const f32* W, const f32* x, f32* y, u32 out_dim, u32 in_dim);
void rms_norm(f32* x, const f32* weight, int size, f32 eps);
void softmax(f32* x, int size);
void silu(f32* x, int size);
void add(f32* x, const f32* y, int size);
void mul(f32* x, const f32* y, int size);
void scale(f32* x, f32 s, int size);
void rope_apply(f32* q, f32* k, const f32* cos_cache, const f32* sin_cache,
                int head_dim, int pos);

// ─── Quantized operations ───
f32 dot_q8_0_f32(const void* q8_data, const f32* f32_data, int count, int block_size);
f32 dot_q4_0_f32(const void* q4_data, const f32* f32_data, int count, int block_size);

#endif // QRAF_HAS_NEON

} // namespace neon
} // namespace qraf
