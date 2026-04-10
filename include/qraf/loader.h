#pragma once

#include "qraf/format.h"
#include "core/types.h"
#include "core/error.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace qraf {

struct TensorView {
    void*              data;
    std::vector<u32>   shape;
    DType              dtype;
    const QuantScheme* quant;  // nullptr if not quantized
    std::string        name;
    u64                data_size;

    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    u32 ndim() const { return static_cast<u32>(shape.size()); }
};

class QrafModel {
public:
    QrafModel() = default;
    ~QrafModel();

    QrafModel(const QrafModel&) = delete;
    QrafModel& operator=(const QrafModel&) = delete;
    QrafModel(QrafModel&& other) noexcept;
    QrafModel& operator=(QrafModel&& other) noexcept;

    bool load(const std::string& path);
    void unload();

    TensorView get_tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;

    const ModelConfig& config() const { return config_; }
    const QrafHeader& header() const { return *header_; }
    const std::string& path() const { return path_; }
    u64 file_size() const { return file_size_; }
    bool is_loaded() const { return mapped_ != nullptr; }

    // String table access
    const char* get_string(u32 offset) const;

    // Tokenizer data access
    const void* tokenizer_data() const;
    u64 tokenizer_size() const;

private:
    void parse_config();
    void parse_tensor_directory();
    void parse_quant_directory();

    std::string path_;
    void* mapped_ = nullptr;
    u64 file_size_ = 0;
    int fd_ = -1;

    const QrafHeader* header_ = nullptr;
    ModelConfig config_;

    std::vector<QuantScheme> quant_schemes_;
    std::unordered_map<std::string, TensorMeta> tensor_index_;
    std::unordered_map<u64, std::string> hash_to_name_;
};

// Header validation (implemented in header.cpp)
bool validate_header(const QrafHeader& header, u64 actual_file_size);

} // namespace qraf
