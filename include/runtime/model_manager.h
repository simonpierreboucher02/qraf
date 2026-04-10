#pragma once

#include "core/types.h"
#include "runtime/inference.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace qraf {

struct ModelInfo {
    std::string name;
    std::string path;
    u64 file_size = 0;
    bool loaded = false;
    ModelConfig config;
};

class ModelManager {
public:
    explicit ModelManager(const std::string& models_dir = "models");

    // Scan models directory
    void scan();

    // List available models
    std::vector<ModelInfo> list() const;

    // Load a model by name or path
    InferenceEngine* load(const std::string& name_or_path);

    // Unload a model
    void unload(const std::string& name);

    // Get a loaded model
    InferenceEngine* get(const std::string& name);

    // Unload all models
    void unload_all();

    // Get total memory usage estimate
    u64 total_memory_usage() const;

private:
    std::string models_dir_;
    std::vector<ModelInfo> registry_;
    std::unordered_map<std::string, std::unique_ptr<InferenceEngine>> loaded_;

    std::string model_name_from_path(const std::string& path) const;
};

} // namespace qraf
