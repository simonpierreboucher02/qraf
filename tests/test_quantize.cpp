// Tests for quantization/dequantization

#include "tensor/quantize.h"

using namespace qraf;

TEST(quantize_q8_roundtrip) {
    // Create test data
    std::vector<f32> data = {1.0f, -0.5f, 0.3f, -1.0f, 0.7f, -0.2f, 0.1f, 0.9f};

    // Quantize
    auto q8 = quantize_q8_0(data.data(), data.size(), 8);

    // Dequantize
    std::vector<f32> output(data.size());
    dequantize_block_q8_0(q8.data(), output.data(), 8);

    // Should be close (Q8 loses some precision)
    for (size_t i = 0; i < data.size(); i++) {
        ASSERT_NEAR(output[i], data[i], 0.02f);
    }
    return true;
}

TEST(quantize_q4_roundtrip) {
    std::vector<f32> data = {1.0f, -0.5f, 0.3f, -1.0f, 0.7f, -0.2f, 0.1f, 0.9f};

    auto q4 = quantize_q4_0(data.data(), data.size(), 8);

    std::vector<f32> output(data.size());
    dequantize_block_q4_0(q4.data(), output.data(), 8);

    // Q4 has larger error but should be in right ballpark
    for (size_t i = 0; i < data.size(); i++) {
        ASSERT_NEAR(output[i], data[i], 0.3f);
    }
    return true;
}

TEST(quantize_dot_q8_f32) {
    std::vector<f32> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<f32> b = {0.5f, 0.5f, 0.5f, 0.5f};

    // Expected dot product: 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 5.0
    auto q8 = quantize_q8_0(a.data(), a.size(), 4);
    f32 result = dot_q8_0_f32(q8.data(), b.data(), 4, 4);

    ASSERT_NEAR(result, 5.0f, 0.1f);
    return true;
}
