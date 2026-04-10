#pragma once

#include "core/types.h"

#ifdef __APPLE__
#define QRAF_HAS_METAL 1
#else
#define QRAF_HAS_METAL 0
#endif

namespace qraf {
namespace metal {

#if QRAF_HAS_METAL

bool init();
void shutdown();
bool is_available();

// GPU MatVec: y = W @ x
void matvec_f32(const f32* W, const f32* x, f32* y, u32 out_dim, u32 in_dim);

// GPU Softmax in-place
void softmax(f32* x, int size);

// GPU RMSNorm in-place
void rms_norm(f32* x, const f32* weight, int size, f32 eps);

// GPU SiLU in-place
void silu(f32* x, int size);

// GPU element-wise add
void add(f32* x, const f32* y, int size);

// GPU element-wise multiply
void mul(f32* x, const f32* y, int size);

#endif

} // namespace metal
} // namespace qraf
