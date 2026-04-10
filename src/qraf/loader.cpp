#include "qraf/loader.h"
#include "core/logging.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstring>

namespace qraf {

QrafModel::~QrafModel() {
    unload();
}

QrafModel::QrafModel(QrafModel&& other) noexcept
    : path_(std::move(other.path_)),
      mapped_(other.mapped_),
      file_size_(other.file_size_),
      fd_(other.fd_),
      header_(other.header_),
      config_(std::move(other.config_)),
      quant_schemes_(std::move(other.quant_schemes_)),
      tensor_index_(std::move(other.tensor_index_)),
      hash_to_name_(std::move(other.hash_to_name_))
{
    other.mapped_ = nullptr;
    other.fd_ = -1;
    other.header_ = nullptr;
}

QrafModel& QrafModel::operator=(QrafModel&& other) noexcept {
    if (this != &other) {
        unload();
        path_ = std::move(other.path_);
        mapped_ = other.mapped_;
        file_size_ = other.file_size_;
        fd_ = other.fd_;
        header_ = other.header_;
        config_ = std::move(other.config_);
        quant_schemes_ = std::move(other.quant_schemes_);
        tensor_index_ = std::move(other.tensor_index_);
        hash_to_name_ = std::move(other.hash_to_name_);
        other.mapped_ = nullptr;
        other.fd_ = -1;
        other.header_ = nullptr;
    }
    return *this;
}

bool QrafModel::load(const std::string& path) {
    path_ = path;
    log::info("Loading QRAF model: %s", path.c_str());

    // Open file
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        log::error("Failed to open file: %s", path.c_str());
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) != 0) {
        log::error("Failed to stat file: %s", path.c_str());
        close(fd_);
        fd_ = -1;
        return false;
    }
    file_size_ = static_cast<u64>(st.st_size);

    if (file_size_ < sizeof(QrafHeader)) {
        log::error("File too small for QRAF header: %llu bytes", (unsigned long long)file_size_);
        close(fd_);
        fd_ = -1;
        return false;
    }

    // Memory-map the entire file (read-only)
    mapped_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        log::error("mmap failed for: %s", path.c_str());
        mapped_ = nullptr;
        close(fd_);
        fd_ = -1;
        return false;
    }

    // Advise kernel for sequential access
    madvise(mapped_, file_size_, MADV_SEQUENTIAL);

    // Validate header
    header_ = reinterpret_cast<const QrafHeader*>(mapped_);
    if (!validate_header(*header_, file_size_)) {
        unload();
        return false;
    }

    // Parse sections
    try {
        parse_quant_directory();
        parse_tensor_directory();
        parse_config();
    } catch (const QrafError& e) {
        log::error("Failed to parse QRAF file: %s", e.what());
        unload();
        return false;
    }

    log::info("Model loaded: %s, %zu tensors, %u layers, hidden=%u",
              config_.architecture.c_str(),
              tensor_index_.size(),
              config_.num_layers,
              config_.hidden_size);

    return true;
}

void QrafModel::unload() {
    if (mapped_) {
        munmap(mapped_, file_size_);
        mapped_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
    header_ = nullptr;
    tensor_index_.clear();
    hash_to_name_.clear();
    quant_schemes_.clear();
    file_size_ = 0;
}

void QrafModel::parse_config() {
    if (header_->config_size == 0) {
        log::warn("No model config block found");
        return;
    }

    const u8* base = static_cast<const u8*>(mapped_) + header_->config_offset;
    u32 num_entries;
    memcpy(&num_entries, base, sizeof(u32));

    const ConfigEntry* entries = reinterpret_cast<const ConfigEntry*>(base + sizeof(u32));

    for (u32 i = 0; i < num_entries; i++) {
        const char* key = get_string(entries[i].key_offset);
        u32 val = entries[i].value;

        if (strcmp(key, "architecture") == 0) {
            config_.architecture = get_string(val);
        } else if (strcmp(key, "vocab_size") == 0) {
            config_.vocab_size = val;
        } else if (strcmp(key, "hidden_size") == 0) {
            config_.hidden_size = val;
        } else if (strcmp(key, "num_layers") == 0) {
            config_.num_layers = val;
        } else if (strcmp(key, "num_heads") == 0) {
            config_.num_heads = val;
        } else if (strcmp(key, "num_kv_heads") == 0) {
            config_.num_kv_heads = val;
        } else if (strcmp(key, "intermediate_size") == 0) {
            config_.intermediate_size = val;
        } else if (strcmp(key, "max_seq_len") == 0) {
            config_.max_seq_len = val;
        } else if (strcmp(key, "rope_theta") == 0) {
            f32 fval;
            memcpy(&fval, &val, sizeof(f32));
            config_.rope_theta = fval;
        } else if (strcmp(key, "rms_norm_eps") == 0) {
            f32 fval;
            memcpy(&fval, &val, sizeof(f32));
            config_.rms_norm_eps = fval;
        } else {
            log::debug("Unknown config key: %s", key);
        }
    }

    config_.compute_derived();
}

void QrafModel::parse_tensor_directory() {
    if (header_->num_tensors == 0) return;

    const TensorMeta* metas = reinterpret_cast<const TensorMeta*>(
        static_cast<const u8*>(mapped_) + header_->tensor_dir_offset
    );

    for (u32 i = 0; i < header_->num_tensors; i++) {
        const TensorMeta& meta = metas[i];
        const char* name = get_string(meta.name_offset);
        std::string name_str(name);

        // Verify hash
        u64 computed_hash = fnv1a_hash(name_str);
        if (computed_hash != meta.name_hash) {
            log::warn("Hash mismatch for tensor '%s': stored=%llu computed=%llu",
                      name, (unsigned long long)meta.name_hash,
                      (unsigned long long)computed_hash);
        }

        // Validate data bounds
        if (meta.data_offset + meta.data_size > file_size_) {
            throw FormatError("Tensor '" + name_str + "' data extends past file end");
        }

        tensor_index_[name_str] = meta;
        hash_to_name_[meta.name_hash] = name_str;

        log::debug("Tensor: %s, shape=[%u", name, meta.shape[0]);
        for (u32 d = 1; d < meta.ndim; d++) {
            log::debug(",%u", meta.shape[d]);
        }
        log::debug("], dtype=%s, size=%llu",
                   dtype_name(static_cast<DType>(meta.dtype)),
                   (unsigned long long)meta.data_size);
    }
}

void QrafModel::parse_quant_directory() {
    if (header_->num_quant_schemes == 0) return;

    const QuantScheme* schemes = reinterpret_cast<const QuantScheme*>(
        static_cast<const u8*>(mapped_) + header_->quant_dir_offset
    );

    quant_schemes_.resize(header_->num_quant_schemes);
    for (u32 i = 0; i < header_->num_quant_schemes; i++) {
        quant_schemes_[i] = schemes[i];
        log::debug("QuantScheme[%u]: type=%u, block_size=%u, group_size=%u",
                   i, schemes[i].type, schemes[i].block_size, schemes[i].group_size);
    }
}

TensorView QrafModel::get_tensor(const std::string& name) const {
    auto it = tensor_index_.find(name);
    QRAF_CHECK(it != tensor_index_.end(), "Tensor not found: %s", name.c_str());

    const TensorMeta& meta = it->second;
    TensorView view;
    view.data = static_cast<u8*>(mapped_) + meta.data_offset;
    view.dtype = static_cast<DType>(meta.dtype);
    view.name = name;
    view.data_size = meta.data_size;

    for (u32 d = 0; d < meta.ndim; d++) {
        view.shape.push_back(meta.shape[d]);
    }

    if (meta.quant_scheme_id != 0xFFFFFFFF && meta.quant_scheme_id < quant_schemes_.size()) {
        view.quant = &quant_schemes_[meta.quant_scheme_id];
    } else {
        view.quant = nullptr;
    }

    return view;
}

bool QrafModel::has_tensor(const std::string& name) const {
    return tensor_index_.count(name) > 0;
}

std::vector<std::string> QrafModel::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_index_.size());
    for (const auto& kv : tensor_index_) {
        names.push_back(kv.first);
    }
    return names;
}

const char* QrafModel::get_string(u32 offset) const {
    QRAF_CHECK(header_->string_table_offset + offset < file_size_,
               "String offset %u out of bounds", offset);
    return static_cast<const char*>(mapped_) + header_->string_table_offset + offset;
}

const void* QrafModel::tokenizer_data() const {
    if (header_->tokenizer_size == 0) return nullptr;
    return static_cast<const u8*>(mapped_) + header_->tokenizer_offset;
}

u64 QrafModel::tokenizer_size() const {
    return header_->tokenizer_size;
}

} // namespace qraf
