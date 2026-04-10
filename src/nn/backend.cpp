#include "nn/backend.h"
#include "nn/ops.h"
#include "nn/simd_neon.h"
#include "nn/threading.h"
#include "tensor/quantize.h"
#include "core/logging.h"
#include <cstring>

#ifdef QRAF_ACCELERATE
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace qraf {

static BackendConfig g_config;

void backend_init() {
    // Auto-detect best backend
#ifdef QRAF_ACCELERATE
    g_config.matvec_backend = Backend::ACCELERATE;
    g_config.elementwise_backend = Backend::NEON;
    log::info("Backend: Accelerate (matvec) + NEON (element-wise)");
#elif QRAF_HAS_NEON
    g_config.matvec_backend = Backend::NEON;
    g_config.elementwise_backend = Backend::NEON;
    log::info("Backend: NEON");
#else
    g_config.matvec_backend = Backend::SCALAR;
    g_config.elementwise_backend = Backend::SCALAR;
    log::info("Backend: scalar");
#endif

#if QRAF_HAS_NEON
    g_config.quantized_backend = Backend::NEON;
#else
    g_config.quantized_backend = Backend::SCALAR;
#endif

#ifdef QRAF_THREADING
    threading_init(0);
    // Upgrade matvec to threaded if we have threads
    if (threading_num_threads() > 1) {
        if (g_config.matvec_backend == Backend::NEON) {
            g_config.matvec_backend = Backend::NEON_THREADED;
        }
        // Note: Accelerate already does internal threading
    }
#endif
}

void backend_set(const BackendConfig& config) {
    g_config = config;
}

const BackendConfig& backend_config() {
    return g_config;
}

// ─── F32 MatVec Dispatch ───

void dispatch_matvec_opt(const f32* W, const f32* x, f32* y,
                         u32 out_dim, u32 in_dim) {
    switch (g_config.matvec_backend) {
#ifdef QRAF_ACCELERATE
        case Backend::ACCELERATE:
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        static_cast<int>(out_dim), static_cast<int>(in_dim),
                        1.0f, W, static_cast<int>(in_dim),
                        x, 1, 0.0f, y, 1);
            return;
#endif

#if QRAF_HAS_NEON
        case Backend::NEON_THREADED:
#ifdef QRAF_THREADING
            parallel_for(0, static_cast<int>(out_dim), 32,
                [W, x, y, in_dim](int start, int stop) {
                    for (int i = start; i < stop; i++) {
                        y[i] = neon::dot_f32(W + i * in_dim, x, static_cast<int>(in_dim));
                    }
                });
            return;
#endif
            // fallthrough if threading not available

        case Backend::NEON:
            neon::matvec_f32(W, x, y, out_dim, in_dim);
            return;
#endif

        case Backend::SCALAR:
        default: {
            for (u32 i = 0; i < out_dim; i++) {
                f32 sum = 0.0f;
                const f32* row = W + i * in_dim;
                for (u32 j = 0; j < in_dim; j++) {
                    sum += row[j] * x[j];
                }
                y[i] = sum;
            }
            return;
        }
    }
}

// ─── F32 Dot Product Dispatch ───

f32 dispatch_dot_f32(const f32* a, const f32* b, int n) {
#if QRAF_HAS_NEON
    if (g_config.elementwise_backend == Backend::NEON ||
        g_config.matvec_backend == Backend::NEON ||
        g_config.matvec_backend == Backend::NEON_THREADED) {
        return neon::dot_f32(a, b, n);
    }
#endif
#ifdef QRAF_ACCELERATE
    if (g_config.matvec_backend == Backend::ACCELERATE) {
        f32 result = 0.0f;
        vDSP_dotpr(a, 1, b, 1, &result, static_cast<vDSP_Length>(n));
        return result;
    }
#endif
    f32 sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// ─── Quantized MatVec Dispatch ───

void dispatch_matvec_q(const void* w_data, u64 w_size,
                       DType w_dtype, const QuantScheme* quant,
                       u32 out_dim, u32 in_dim,
                       const f32* x, f32* y) {
    int block_size = quant ? static_cast<int>(quant->block_size) : 32;

    if (w_dtype == DType::Q8_0) {
        size_t blocks_per_row = (in_dim + block_size - 1) / block_size;
        size_t row_bytes = blocks_per_row * (sizeof(f32) + block_size);
        const u8* ptr = static_cast<const u8*>(w_data);

#if QRAF_HAS_NEON
        if (g_config.quantized_backend == Backend::NEON) {
#ifdef QRAF_THREADING
            parallel_for(0, static_cast<int>(out_dim), 32,
                [ptr, row_bytes, x, y, in_dim, block_size](int start, int stop) {
                    for (int i = start; i < stop; i++) {
                        y[i] = neon::dot_q8_0_f32(ptr + i * row_bytes, x,
                                                   static_cast<int>(in_dim), block_size);
                    }
                });
            return;
#else
            for (u32 i = 0; i < out_dim; i++) {
                y[i] = neon::dot_q8_0_f32(ptr + i * row_bytes, x,
                                           static_cast<int>(in_dim), block_size);
            }
            return;
#endif
        }
#endif
        // Scalar fallback
        for (u32 i = 0; i < out_dim; i++) {
            y[i] = dot_q8_0_f32(ptr + i * row_bytes, x, in_dim, block_size);
        }
    } else if (w_dtype == DType::Q4_0) {
        size_t blocks_per_row = (in_dim + block_size - 1) / block_size;
        size_t row_bytes = blocks_per_row * (sizeof(f16) + block_size / 2);
        const u8* ptr = static_cast<const u8*>(w_data);

#if QRAF_HAS_NEON
        if (g_config.quantized_backend == Backend::NEON) {
#ifdef QRAF_THREADING
            parallel_for(0, static_cast<int>(out_dim), 32,
                [ptr, row_bytes, x, y, in_dim, block_size](int start, int stop) {
                    for (int i = start; i < stop; i++) {
                        y[i] = neon::dot_q4_0_f32(ptr + i * row_bytes, x,
                                                   static_cast<int>(in_dim), block_size);
                    }
                });
            return;
#else
            for (u32 i = 0; i < out_dim; i++) {
                y[i] = neon::dot_q4_0_f32(ptr + i * row_bytes, x,
                                           static_cast<int>(in_dim), block_size);
            }
            return;
#endif
        }
#endif
        // Scalar fallback
        for (u32 i = 0; i < out_dim; i++) {
            y[i] = qraf::dot_q4_0_f32(ptr + i * row_bytes, x, in_dim, block_size);
        }
    } else {
        // Unknown quantized type — zero out
        memset(y, 0, out_dim * sizeof(f32));
    }
}

// ─── Element-wise Dispatch ───

void dispatch_rms_norm(f32* x, const f32* weight, int size, f32 eps) {
#if QRAF_HAS_NEON
    if (g_config.elementwise_backend == Backend::NEON) {
        neon::rms_norm(x, weight, size, eps);
        return;
    }
#endif
#ifdef QRAF_ACCELERATE
    {
        // vDSP implementation
        f32 ss = 0.0f;
        vDSP_dotpr(x, 1, x, 1, &ss, static_cast<vDSP_Length>(size));
        ss = ss / static_cast<f32>(size) + eps;
        f32 rsqrt = 1.0f / std::sqrt(ss);
        vDSP_vsmul(x, 1, &rsqrt, x, 1, static_cast<vDSP_Length>(size));
        vDSP_vmul(x, 1, weight, 1, x, 1, static_cast<vDSP_Length>(size));
        return;
    }
#endif
    ops::rms_norm_inplace(x, weight, size, eps);
}

void dispatch_softmax(f32* x, int size) {
#if QRAF_HAS_NEON
    if (g_config.elementwise_backend == Backend::NEON && size >= 16) {
        neon::softmax(x, size);
        return;
    }
#endif
#ifdef QRAF_ACCELERATE
    {
        // Find max
        f32 max_val;
        vDSP_maxv(x, 1, &max_val, static_cast<vDSP_Length>(size));
        // x -= max
        f32 neg_max = -max_val;
        vDSP_vsadd(x, 1, &neg_max, x, 1, static_cast<vDSP_Length>(size));
        // exp
        int n = size;
        vvexpf(x, x, &n);
        // sum
        f32 sum;
        vDSP_sve(x, 1, &sum, static_cast<vDSP_Length>(size));
        // scale
        f32 inv_sum = 1.0f / sum;
        vDSP_vsmul(x, 1, &inv_sum, x, 1, static_cast<vDSP_Length>(size));
        return;
    }
#endif
    ops::softmax_inplace(x, size);
}

void dispatch_silu(f32* x, int size) {
#ifdef QRAF_ACCELERATE
    {
        // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        // Use Accelerate: compute exp(-x), then 1+exp(-x), then x/denom
        std::vector<f32> neg_x(size);
        f32 neg = -1.0f;
        vDSP_vsmul(x, 1, &neg, neg_x.data(), 1, static_cast<vDSP_Length>(size));
        int n = size;
        vvexpf(neg_x.data(), neg_x.data(), &n); // neg_x = exp(-x)
        f32 one = 1.0f;
        vDSP_vsadd(neg_x.data(), 1, &one, neg_x.data(), 1, static_cast<vDSP_Length>(size)); // neg_x = 1 + exp(-x)
        vDSP_vdiv(neg_x.data(), 1, x, 1, x, 1, static_cast<vDSP_Length>(size)); // x = x / (1 + exp(-x))
        return;
    }
#endif
#if QRAF_HAS_NEON
    if (g_config.elementwise_backend == Backend::NEON) {
        neon::silu(x, size);
        return;
    }
#endif
    ops::silu_inplace(x, size);
}

void dispatch_add(f32* x, const f32* y, int size) {
#if QRAF_HAS_NEON
    if (g_config.elementwise_backend == Backend::NEON) {
        neon::add(x, y, size);
        return;
    }
#endif
#ifdef QRAF_ACCELERATE
    vDSP_vadd(x, 1, y, 1, x, 1, static_cast<vDSP_Length>(size));
    return;
#endif
    ops::add_inplace(x, y, size);
}

void dispatch_mul(f32* x, const f32* y, int size) {
#if QRAF_HAS_NEON
    if (g_config.elementwise_backend == Backend::NEON) {
        neon::mul(x, y, size);
        return;
    }
#endif
#ifdef QRAF_ACCELERATE
    vDSP_vmul(x, 1, y, 1, x, 1, static_cast<vDSP_Length>(size));
    return;
#endif
    ops::mul_inplace(x, y, size);
}

void dispatch_scale(f32* x, f32 s, int size) {
#if QRAF_HAS_NEON
    if (g_config.elementwise_backend == Backend::NEON) {
        neon::scale(x, s, size);
        return;
    }
#endif
#ifdef QRAF_ACCELERATE
    vDSP_vsmul(x, 1, &s, x, 1, static_cast<vDSP_Length>(size));
    return;
#endif
    ops::scale_inplace(x, s, size);
}

} // namespace qraf
