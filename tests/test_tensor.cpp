// Tests for Tensor class

#include "tensor/tensor.h"

using namespace qraf;

TEST(tensor_zeros) {
    auto t = Tensor::zeros({3, 4}, DType::F32);
    ASSERT_EQ(t.ndim(), 2u);
    ASSERT_EQ(t.shape()[0], 3u);
    ASSERT_EQ(t.shape()[1], 4u);
    ASSERT_EQ(t.numel(), 12u);
    ASSERT_EQ(t.nbytes(), 48u);

    for (size_t i = 0; i < t.numel(); i++) {
        ASSERT_EQ(t.at(i), 0.0f);
    }
    return true;
}

TEST(tensor_from_data) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::from_data({2, 3}, DType::F32, data, sizeof(data));

    ASSERT_EQ(t.at(0, 0), 1.0f);
    ASSERT_EQ(t.at(0, 2), 3.0f);
    ASSERT_EQ(t.at(1, 0), 4.0f);
    ASSERT_EQ(t.at(1, 2), 6.0f);
    return true;
}

TEST(tensor_reshape) {
    float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_data({2, 3}, DType::F32, data, sizeof(data));
    auto r = t.reshape({6});

    ASSERT_EQ(r.ndim(), 1u);
    ASSERT_EQ(r.shape()[0], 6u);
    ASSERT_EQ(r.at(0), 1.0f);
    ASSERT_EQ(r.at(5), 6.0f);
    return true;
}

TEST(tensor_row) {
    float data[] = {1, 2, 3, 4, 5, 6};
    auto t = Tensor::from_data({2, 3}, DType::F32, data, sizeof(data));
    auto row0 = t.row(0);
    auto row1 = t.row(1);

    ASSERT_EQ(row0.ndim(), 1u);
    ASSERT_EQ(row0.shape()[0], 3u);
    ASSERT_EQ(row0.at(0), 1.0f);
    ASSERT_EQ(row0.at(2), 3.0f);
    ASSERT_EQ(row1.at(0), 4.0f);
    return true;
}
