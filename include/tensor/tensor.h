#pragma once

#include "core/types.h"
#include "core/error.h"
#include <vector>
#include <cstring>
#include <numeric>
#include <functional>

namespace qraf {

class Tensor {
public:
    Tensor() = default;

    // Create owned tensor with given shape and dtype
    static Tensor zeros(const std::vector<u32>& shape, DType dtype = DType::F32);
    static Tensor from_data(const std::vector<u32>& shape, DType dtype, const void* data, size_t size);

    // Accessors
    const std::vector<u32>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    u32 ndim() const { return static_cast<u32>(shape_.size()); }

    size_t numel() const {
        if (shape_.empty()) return 0;
        size_t n = 1;
        for (auto s : shape_) n *= s;
        return n;
    }

    size_t nbytes() const {
        if (dtype_ == DType::Q4_0 || dtype_ == DType::Q4_1) {
            return (numel() + 1) / 2; // packed 4-bit
        }
        return numel() * dtype_size(dtype_);
    }

    // Data access
    void* data() { return data_.data(); }
    const void* data() const { return data_.data(); }

    f32* data_f32() {
        assert(dtype_ == DType::F32);
        return reinterpret_cast<f32*>(data_.data());
    }
    const f32* data_f32() const {
        assert(dtype_ == DType::F32);
        return reinterpret_cast<const f32*>(data_.data());
    }

    // Element access (f32 only for simplicity)
    f32& at(size_t i) {
        assert(dtype_ == DType::F32 && i < numel());
        return data_f32()[i];
    }
    f32 at(size_t i) const {
        assert(dtype_ == DType::F32 && i < numel());
        return data_f32()[i];
    }

    // 2D access
    f32& at(size_t row, size_t col) {
        assert(ndim() == 2 && dtype_ == DType::F32);
        return data_f32()[row * shape_[1] + col];
    }
    f32 at(size_t row, size_t col) const {
        assert(ndim() == 2 && dtype_ == DType::F32);
        return data_f32()[row * shape_[1] + col];
    }

    // Reshape (returns a new tensor sharing data)
    Tensor reshape(const std::vector<u32>& new_shape) const;

    // Row slice: returns a view into row i of a 2D tensor
    // For embedding lookups, etc.
    Tensor row(size_t i) const;

    bool empty() const { return data_.empty(); }

private:
    std::vector<u8> data_;
    std::vector<u32> shape_;
    DType dtype_ = DType::F32;
};

} // namespace qraf
