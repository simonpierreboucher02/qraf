#include <metal_stdlib>
using namespace metal;

// ─── MatVec: y[i] = dot(W[i,:], x) ───
// Each thread computes one output element
kernel void matvec_f32(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& out_dim [[buffer(3)]],
    constant uint& in_dim  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= out_dim) return;

    float sum = 0.0f;
    device const float* row = W + tid * in_dim;

    // Unrolled accumulation
    uint i = 0;
    for (; i + 3 < in_dim; i += 4) {
        sum += row[i]     * x[i];
        sum += row[i + 1] * x[i + 1];
        sum += row[i + 2] * x[i + 2];
        sum += row[i + 3] * x[i + 3];
    }
    for (; i < in_dim; i++) {
        sum += row[i] * x[i];
    }

    y[tid] = sum;
}

// ─── Softmax (two-pass: max + exp+sum, then normalize) ───
// Pass 1: find max and compute exp(x - max), accumulate sum
// This simple version works for small arrays (< threadgroup size)
kernel void softmax_pass1(
    device float* x          [[buffer(0)]],
    device float* scratch    [[buffer(1)]],   // [max, sum]
    constant uint& size      [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint tcount [[threads_per_grid]]
) {
    // Find max
    float local_max = -INFINITY;
    for (uint i = tid; i < size; i += tcount) {
        local_max = max(local_max, x[i]);
    }

    // Threadgroup reduction for max (simplified: use atomic or single thread)
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // For simplicity, thread 0 does the full computation
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint i = 0; i < size; i++) mx = max(mx, x[i]);

        float s = 0.0f;
        for (uint i = 0; i < size; i++) {
            x[i] = exp(x[i] - mx);
            s += x[i];
        }

        float inv_s = 1.0f / s;
        for (uint i = 0; i < size; i++) {
            x[i] *= inv_s;
        }
    }
}

// ─── RMSNorm ───
kernel void rms_norm_f32(
    device float* x           [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    constant uint& size       [[buffer(2)]],
    constant float& eps       [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Thread 0 computes norm then all threads apply
    if (tid == 0) {
        float ss = 0.0f;
        for (uint i = 0; i < size; i++) ss += x[i] * x[i];
        ss = ss / float(size) + eps;
        float rsqrt_val = rsqrt(ss);
        for (uint i = 0; i < size; i++) {
            x[i] = x[i] * rsqrt_val * weight[i];
        }
    }
}

// ─── SiLU: x * sigmoid(x) ───
kernel void silu_f32(
    device float* x      [[buffer(0)]],
    constant uint& size  [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    float v = x[tid];
    x[tid] = v / (1.0f + exp(-v));
}

// ─── Element-wise add ───
kernel void add_f32(
    device float* x         [[buffer(0)]],
    device const float* y   [[buffer(1)]],
    constant uint& size     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    x[tid] += y[tid];
}

// ─── Element-wise multiply ───
kernel void mul_f32(
    device float* x         [[buffer(0)]],
    device const float* y   [[buffer(1)]],
    constant uint& size     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;
    x[tid] *= y[tid];
}
