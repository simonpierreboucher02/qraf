#pragma once

#include "tensor/tensor.h"
#include "core/types.h"
#include "qraf/format.h"
#include <vector>

namespace qraf {
namespace ops {

// ─── Matrix operations ───

// General matrix multiply: C = A @ B
// A: [M, K], B: [K, N] -> C: [M, N]
Tensor matmul(const Tensor& a, const Tensor& b);

// Matrix-vector multiply: y = W @ x
// W: [out_dim, in_dim], x: [in_dim] -> y: [out_dim]
Tensor matvec(const Tensor& w, const Tensor& x);

// Quantized matvec: y = dequant(W_q) @ x
// Operates on raw quantized data without full dequantization
void matvec_quantized(const void* w_data, u64 w_size,
                      DType w_dtype, const QuantScheme* quant,
                      u32 out_dim, u32 in_dim,
                      const f32* x, f32* y);

// ─── Normalization ───

// RMS Normalization: x = x * rsqrt(mean(x^2) + eps) * weight
Tensor rms_norm(const Tensor& x, const Tensor& weight, f32 eps = 1e-5f);

// In-place RMS norm on raw buffer
void rms_norm_inplace(f32* x, const f32* weight, int size, f32 eps = 1e-5f);

// ─── Positional encoding ───

// Rotary Position Embedding (RoPE)
// Applies RoPE to query and key tensors in-place
void rope(f32* q, f32* k, int head_dim, int pos, f32 theta = 10000.0f);

// Precompute RoPE frequency table
struct RopeCache {
    std::vector<f32> cos_cache;
    std::vector<f32> sin_cache;
    int max_seq_len;
    int head_dim;

    void init(int max_seq, int hdim, f32 theta = 10000.0f);
    void apply(f32* q, f32* k, int pos) const;
};

// ─── Activation functions ───

// SiLU (Swish): x * sigmoid(x)
void silu_inplace(f32* x, int size);
Tensor silu(const Tensor& x);

// GELU approximation
void gelu_inplace(f32* x, int size);

// ─── Softmax ───

void softmax_inplace(f32* x, int size);
Tensor softmax(const Tensor& x);

// ─── Element-wise operations ───

// x = x + y
void add_inplace(f32* x, const f32* y, int size);

// x = x * y
void mul_inplace(f32* x, const f32* y, int size);

// x = x * scale
void scale_inplace(f32* x, f32 scale, int size);

} // namespace ops
} // namespace qraf
