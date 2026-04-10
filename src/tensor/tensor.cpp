#include "tensor/tensor.h"
#include "core/logging.h"
#include <cstring>
#include <numeric>

namespace qraf {

Tensor Tensor::zeros(const std::vector<u32>& shape, DType dtype) {
    Tensor t;
    t.shape_ = shape;
    t.dtype_ = dtype;

    size_t n = 1;
    for (auto s : shape) n *= s;

    size_t bytes;
    if (dtype == DType::Q4_0 || dtype == DType::Q4_1) {
        bytes = (n + 1) / 2;
    } else {
        bytes = n * dtype_size(dtype);
    }

    t.data_.resize(bytes, 0);
    return t;
}

Tensor Tensor::from_data(const std::vector<u32>& shape, DType dtype, const void* data, size_t size) {
    Tensor t;
    t.shape_ = shape;
    t.dtype_ = dtype;
    t.data_.resize(size);
    memcpy(t.data_.data(), data, size);
    return t;
}

Tensor Tensor::reshape(const std::vector<u32>& new_shape) const {
    // Verify element count matches
    size_t old_numel = numel();
    size_t new_numel = 1;
    for (auto s : new_shape) new_numel *= s;
    QRAF_CHECK_SHAPE(old_numel == new_numel,
                     "reshape: %zu elements vs %zu elements", old_numel, new_numel);

    Tensor t;
    t.data_ = data_;  // copy data (could optimize with shared_ptr)
    t.shape_ = new_shape;
    t.dtype_ = dtype_;
    return t;
}

Tensor Tensor::row(size_t i) const {
    QRAF_CHECK_SHAPE(ndim() == 2, "row() requires 2D tensor, got %uD", ndim());
    QRAF_CHECK_SHAPE(i < shape_[0], "row index %zu out of bounds [0, %u)", i, shape_[0]);
    assert(dtype_ == DType::F32);

    u32 cols = shape_[1];
    Tensor t;
    t.shape_ = {cols};
    t.dtype_ = dtype_;
    size_t row_bytes = cols * dtype_size(dtype_);
    t.data_.resize(row_bytes);
    memcpy(t.data_.data(), data_.data() + i * row_bytes, row_bytes);
    return t;
}

} // namespace qraf
