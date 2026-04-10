// Tests for neural network operations

#include "nn/ops.h"
#include <cmath>

using namespace qraf;

TEST(ops_softmax) {
    float data[] = {1.0f, 2.0f, 3.0f};
    ops::softmax_inplace(data, 3);

    // Verify sums to 1
    float sum = data[0] + data[1] + data[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    // Verify ordering preserved
    ASSERT_TRUE(data[2] > data[1]);
    ASSERT_TRUE(data[1] > data[0]);

    // Verify approximate values
    ASSERT_NEAR(data[0], 0.0900f, 0.01f);
    ASSERT_NEAR(data[1], 0.2447f, 0.01f);
    ASSERT_NEAR(data[2], 0.6652f, 0.01f);
    return true;
}

TEST(ops_silu) {
    float data[] = {0.0f, 1.0f, -1.0f};
    ops::silu_inplace(data, 3);

    // silu(0) = 0
    ASSERT_NEAR(data[0], 0.0f, 1e-5f);
    // silu(1) = 1/(1+e^-1) ≈ 0.7311
    ASSERT_NEAR(data[1], 0.7311f, 0.01f);
    // silu(-1) = -1/(1+e^1) ≈ -0.2689
    ASSERT_NEAR(data[2], -0.2689f, 0.01f);
    return true;
}

TEST(ops_rms_norm) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    ops::rms_norm_inplace(x, w, 4, 1e-5f);

    // RMS = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
    // Each value should be divided by RMS
    float rms = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + 1e-5f);
    ASSERT_NEAR(x[0], 1.0f / rms, 0.001f);
    ASSERT_NEAR(x[3], 4.0f / rms, 0.001f);
    return true;
}

TEST(ops_matvec) {
    // W = [[1,2],[3,4],[5,6]], x = [1,1]
    // y = [3, 7, 11]
    float w_data[] = {1, 2, 3, 4, 5, 6};
    float x_data[] = {1, 1};
    auto W = Tensor::from_data({3, 2}, DType::F32, w_data, sizeof(w_data));
    auto x = Tensor::from_data({2}, DType::F32, x_data, sizeof(x_data));

    auto y = ops::matvec(W, x);
    ASSERT_EQ(y.shape()[0], 3u);
    ASSERT_NEAR(y.at(0), 3.0f, 1e-5f);
    ASSERT_NEAR(y.at(1), 7.0f, 1e-5f);
    ASSERT_NEAR(y.at(2), 11.0f, 1e-5f);
    return true;
}

TEST(ops_matmul) {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // C = [[19,22],[43,50]]
    float a_data[] = {1, 2, 3, 4};
    float b_data[] = {5, 6, 7, 8};
    auto A = Tensor::from_data({2, 2}, DType::F32, a_data, sizeof(a_data));
    auto B = Tensor::from_data({2, 2}, DType::F32, b_data, sizeof(b_data));

    auto C = ops::matmul(A, B);
    ASSERT_NEAR(C.at(0, 0), 19.0f, 1e-5f);
    ASSERT_NEAR(C.at(0, 1), 22.0f, 1e-5f);
    ASSERT_NEAR(C.at(1, 0), 43.0f, 1e-5f);
    ASSERT_NEAR(C.at(1, 1), 50.0f, 1e-5f);
    return true;
}

TEST(ops_rope) {
    float q[] = {1.0f, 0.0f, 1.0f, 0.0f};
    float k[] = {1.0f, 0.0f, 1.0f, 0.0f};

    // At pos=0, RoPE should be identity (cos(0)=1, sin(0)=0)
    ops::rope(q, k, 4, 0, 10000.0f);
    ASSERT_NEAR(q[0], 1.0f, 1e-5f);
    ASSERT_NEAR(q[1], 0.0f, 1e-5f);
    ASSERT_NEAR(k[0], 1.0f, 1e-5f);
    return true;
}

TEST(ops_add_mul_scale) {
    float x[] = {1, 2, 3};
    float y[] = {4, 5, 6};

    ops::add_inplace(x, y, 3);
    ASSERT_NEAR(x[0], 5.0f, 1e-5f);
    ASSERT_NEAR(x[2], 9.0f, 1e-5f);

    float a[] = {2, 3, 4};
    float b[] = {3, 2, 1};
    ops::mul_inplace(a, b, 3);
    ASSERT_NEAR(a[0], 6.0f, 1e-5f);
    ASSERT_NEAR(a[2], 4.0f, 1e-5f);

    float c[] = {1, 2, 3};
    ops::scale_inplace(c, 2.0f, 3);
    ASSERT_NEAR(c[0], 2.0f, 1e-5f);
    ASSERT_NEAR(c[2], 6.0f, 1e-5f);
    return true;
}
