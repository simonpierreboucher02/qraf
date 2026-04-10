#pragma once

#include "core/types.h"

namespace qraf {

enum class Backend {
    SCALAR,
    NEON,
    ACCELERATE,
    NEON_THREADED,
};

struct BackendConfig {
    Backend matvec_backend     = Backend::SCALAR;
    Backend elementwise_backend = Backend::SCALAR;
    Backend quantized_backend  = Backend::SCALAR;
    int num_threads            = 0; // 0 = auto-detect
};

// Global backend configuration
void backend_init();
void backend_set(const BackendConfig& config);
const BackendConfig& backend_config();

// Unified dispatch: f32 matvec
void dispatch_matvec_opt(const f32* W, const f32* x, f32* y,
                         u32 out_dim, u32 in_dim);

// Unified dispatch: quantized matvec
void dispatch_matvec_q(const void* w_data, u64 w_size,
                       DType w_dtype, const struct QuantScheme* quant,
                       u32 out_dim, u32 in_dim,
                       const f32* x, f32* y);

// Unified dispatch: f32 dot product
f32 dispatch_dot_f32(const f32* a, const f32* b, int n);

// Unified dispatch: element-wise ops
void dispatch_rms_norm(f32* x, const f32* weight, int size, f32 eps);
void dispatch_softmax(f32* x, int size);
void dispatch_silu(f32* x, int size);
void dispatch_add(f32* x, const f32* y, int size);
void dispatch_mul(f32* x, const f32* y, int size);
void dispatch_scale(f32* x, f32 scale, int size);

} // namespace qraf
