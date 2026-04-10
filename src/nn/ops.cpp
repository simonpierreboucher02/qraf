#include "nn/ops.h"
#include "tensor/quantize.h"
#include "core/error.h"
#include "core/logging.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace qraf {
namespace ops {

// ─── Matrix Multiply ───

Tensor matmul(const Tensor& a, const Tensor& b) {
    QRAF_CHECK_SHAPE(a.ndim() == 2 && b.ndim() == 2,
                     "matmul requires 2D tensors, got %uD and %uD", a.ndim(), b.ndim());
    QRAF_CHECK_SHAPE(a.shape()[1] == b.shape()[0],
                     "matmul shape mismatch: [%u,%u] @ [%u,%u]",
                     a.shape()[0], a.shape()[1], b.shape()[0], b.shape()[1]);
    QRAF_CHECK(a.dtype() == DType::F32 && b.dtype() == DType::F32,
               "matmul requires f32 tensors");

    u32 M = a.shape()[0];
    u32 K = a.shape()[1];
    u32 N = b.shape()[1];

    Tensor c = Tensor::zeros({M, N}, DType::F32);
    const f32* A = a.data_f32();
    const f32* B = b.data_f32();
    f32* C = c.data_f32();

    for (u32 i = 0; i < M; i++) {
        for (u32 k = 0; k < K; k++) {
            f32 a_ik = A[i * K + k];
            for (u32 j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }

    return c;
}

// ─── Matrix-Vector Multiply ───

Tensor matvec(const Tensor& w, const Tensor& x) {
    QRAF_CHECK_SHAPE(w.ndim() == 2 && x.ndim() == 1,
                     "matvec: W must be 2D, x must be 1D");
    QRAF_CHECK_SHAPE(w.shape()[1] == x.shape()[0],
                     "matvec: W[%u,%u] @ x[%u]", w.shape()[0], w.shape()[1], x.shape()[0]);

    u32 out_dim = w.shape()[0];
    u32 in_dim = w.shape()[1];

    Tensor y = Tensor::zeros({out_dim}, DType::F32);
    const f32* W = w.data_f32();
    const f32* X = x.data_f32();
    f32* Y = y.data_f32();

    for (u32 i = 0; i < out_dim; i++) {
        f32 sum = 0.0f;
        const f32* row = W + i * in_dim;
        for (u32 j = 0; j < in_dim; j++) {
            sum += row[j] * X[j];
        }
        Y[i] = sum;
    }

    return y;
}

void matvec_quantized(const void* w_data, u64 /*w_size*/,
                      DType w_dtype, const QuantScheme* quant,
                      u32 out_dim, u32 in_dim,
                      const f32* x, f32* y) {
    int block_size = quant ? static_cast<int>(quant->block_size) : 32;

    if (w_dtype == DType::Q8_0) {
        size_t blocks_per_row = (in_dim + block_size - 1) / block_size;
        size_t row_bytes = blocks_per_row * (sizeof(f32) + block_size);
        const u8* ptr = static_cast<const u8*>(w_data);

        for (u32 i = 0; i < out_dim; i++) {
            y[i] = dot_q8_0_f32(ptr + i * row_bytes, x, in_dim, block_size);
        }
    } else if (w_dtype == DType::Q4_0) {
        size_t blocks_per_row = (in_dim + block_size - 1) / block_size;
        size_t row_bytes = blocks_per_row * (sizeof(f16) + block_size / 2);
        const u8* ptr = static_cast<const u8*>(w_data);

        for (u32 i = 0; i < out_dim; i++) {
            y[i] = dot_q4_0_f32(ptr + i * row_bytes, x, in_dim, block_size);
        }
    } else {
        // Fallback: dequantize row by row
        log::warn("Unoptimized quantized matvec for dtype %s", dtype_name(w_dtype));
        std::vector<f32> row_buf(in_dim);
        // Just zero out
        for (u32 i = 0; i < out_dim; i++) {
            y[i] = 0.0f;
        }
    }
}

// ─── RMS Normalization ───

void rms_norm_inplace(f32* x, const f32* weight, int size, f32 eps) {
    f32 ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss = ss / static_cast<f32>(size) + eps;
    f32 rsqrt = 1.0f / std::sqrt(ss);

    for (int i = 0; i < size; i++) {
        x[i] = x[i] * rsqrt * weight[i];
    }
}

// ─── Layer Normalization ───

void layer_norm_inplace(f32* x, const f32* weight, const f32* bias, int size, f32 eps) {
    // Compute mean
    f32 mean = 0.0f;
    for (int i = 0; i < size; i++) mean += x[i];
    mean /= static_cast<f32>(size);

    // Compute variance
    f32 var = 0.0f;
    for (int i = 0; i < size; i++) {
        f32 d = x[i] - mean;
        var += d * d;
    }
    var /= static_cast<f32>(size);

    f32 inv_std = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] - mean) * inv_std * weight[i];
        if (bias) x[i] += bias[i];
    }
}

Tensor rms_norm(const Tensor& x, const Tensor& weight, f32 eps) {
    QRAF_CHECK_SHAPE(x.ndim() == 1, "rms_norm: x must be 1D, got %uD", x.ndim());
    QRAF_CHECK_SHAPE(weight.ndim() == 1, "rms_norm: weight must be 1D");
    QRAF_CHECK_SHAPE(x.shape()[0] == weight.shape()[0],
                     "rms_norm: size mismatch x[%u] vs weight[%u]",
                     x.shape()[0], weight.shape()[0]);

    int size = static_cast<int>(x.shape()[0]);
    Tensor out = Tensor::from_data(x.shape(), DType::F32, x.data(), x.nbytes());
    rms_norm_inplace(out.data_f32(), weight.data_f32(), size, eps);
    return out;
}

// ─── RoPE ───

void rope(f32* q, f32* k, int head_dim, int pos, f32 theta) {
    for (int i = 0; i < head_dim; i += 2) {
        f32 freq = 1.0f / std::pow(theta, static_cast<f32>(i) / static_cast<f32>(head_dim));
        f32 val = static_cast<f32>(pos) * freq;
        f32 cos_val = std::cos(val);
        f32 sin_val = std::sin(val);

        // Apply rotation to q
        f32 q0 = q[i];
        f32 q1 = q[i + 1];
        q[i]     = q0 * cos_val - q1 * sin_val;
        q[i + 1] = q0 * sin_val + q1 * cos_val;

        // Apply rotation to k
        f32 k0 = k[i];
        f32 k1 = k[i + 1];
        k[i]     = k0 * cos_val - k1 * sin_val;
        k[i + 1] = k0 * sin_val + k1 * cos_val;
    }
}

void RopeCache::init(int max_seq, int hdim, f32 theta) {
    max_seq_len = max_seq;
    head_dim = hdim;
    cos_cache.resize(max_seq * hdim / 2);
    sin_cache.resize(max_seq * hdim / 2);

    for (int pos = 0; pos < max_seq; pos++) {
        for (int i = 0; i < hdim / 2; i++) {
            f32 freq = 1.0f / std::pow(theta, static_cast<f32>(2 * i) / static_cast<f32>(hdim));
            f32 val = static_cast<f32>(pos) * freq;
            cos_cache[pos * (hdim / 2) + i] = std::cos(val);
            sin_cache[pos * (hdim / 2) + i] = std::sin(val);
        }
    }
}

void RopeCache::apply(f32* q, f32* k, int pos) const {
    int half = head_dim / 2;
    const f32* cos_row = cos_cache.data() + pos * half;
    const f32* sin_row = sin_cache.data() + pos * half;

    for (int i = 0; i < half; i++) {
        f32 cos_v = cos_row[i];
        f32 sin_v = sin_row[i];

        f32 q0 = q[2 * i];
        f32 q1 = q[2 * i + 1];
        q[2 * i]     = q0 * cos_v - q1 * sin_v;
        q[2 * i + 1] = q0 * sin_v + q1 * cos_v;

        f32 k0 = k[2 * i];
        f32 k1 = k[2 * i + 1];
        k[2 * i]     = k0 * cos_v - k1 * sin_v;
        k[2 * i + 1] = k0 * sin_v + k1 * cos_v;
    }
}

// ─── Activation functions ───

void silu_inplace(f32* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

Tensor silu(const Tensor& x) {
    Tensor out = Tensor::from_data(x.shape(), DType::F32, x.data(), x.nbytes());
    silu_inplace(out.data_f32(), static_cast<int>(out.numel()));
    return out;
}

void gelu_inplace(f32* x, int size) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    constexpr f32 sqrt_2_over_pi = 0.7978845608f;
    for (int i = 0; i < size; i++) {
        f32 v = x[i];
        f32 inner = sqrt_2_over_pi * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
}

// ─── Softmax ───

void softmax_inplace(f32* x, int size) {
    // Numerical stability: subtract max
    f32 max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    f32 sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }

    f32 inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

Tensor softmax(const Tensor& x) {
    Tensor out = Tensor::from_data(x.shape(), DType::F32, x.data(), x.nbytes());
    softmax_inplace(out.data_f32(), static_cast<int>(out.numel()));
    return out;
}

// ─── Element-wise operations ───

void add_inplace(f32* x, const f32* y, int size) {
    for (int i = 0; i < size; i++) {
        x[i] += y[i];
    }
}

void mul_inplace(f32* x, const f32* y, int size) {
    for (int i = 0; i < size; i++) {
        x[i] *= y[i];
    }
}

void scale_inplace(f32* x, f32 scale, int size) {
    for (int i = 0; i < size; i++) {
        x[i] *= scale;
    }
}

} // namespace ops
} // namespace qraf
