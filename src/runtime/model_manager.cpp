#include "runtime/model_manager.h"
#include "core/logging.h"
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace qraf {

ModelManager::ModelManager(const std::string& models_dir)
    : models_dir_(models_dir) {}

void ModelManager::scan() {
    registry_.clear();

    if (!fs::exists(models_dir_)) {
        log::warn("Models directory does not exist: %s", models_dir_.c_str());
        return;
    }

    for (const auto& entry : fs::directory_iterator(models_dir_)) {
        if (entry.is_regular_file() && entry.path().extension() == ".qraf") {
            ModelInfo info;
            info.name = entry.path().stem().string();
            info.path = entry.path().string();
            info.file_size = static_cast<u64>(entry.file_size());
            info.loaded = loaded_.count(info.name) > 0;
            registry_.push_back(info);
        }
    }

    std::sort(registry_.begin(), registry_.end(),
              [](const ModelInfo& a, const ModelInfo& b) { return a.name < b.name; });

    log::info("Found %zu models in %s", registry_.size(), models_dir_.c_str());
}

std::vector<ModelInfo> ModelManager::list() const {
    return registry_;
}

InferenceEngine* ModelManager::load(const std::string& name_or_path) {
    // Determine actual path and name
    std::string path = name_or_path;
    std::string name;

    if (fs::exists(name_or_path) && fs::is_regular_file(name_or_path)) {
        path = name_or_path;
        name = model_name_from_path(path);
    } else {
        // Search registry
        name = name_or_path;
        bool found = false;
        for (const auto& info : registry_) {
            if (info.name == name_or_path) {
                path = info.path;
                found = true;
                break;
            }
        }
        if (!found) {
            // Try with .qraf extension
            path = models_dir_ + "/" + name_or_path + ".qraf";
            if (!fs::exists(path)) {
                log::error("Model not found: %s", name_or_path.c_str());
                return nullptr;
            }
        }
    }

    // Check if already loaded
    if (loaded_.count(name) > 0) {
        log::info("Model '%s' already loaded", name.c_str());
        return loaded_[name].get();
    }

    // Load the model
    auto engine = std::make_unique<InferenceEngine>();
    if (!engine->load_model(path)) {
        return nullptr;
    }

    InferenceEngine* ptr = engine.get();
    loaded_[name] = std::move(engine);

    // Update registry
    for (auto& info : registry_) {
        if (info.name == name) {
            info.loaded = true;
            info.config = ptr->model_config();
        }
    }

    return ptr;
}

void ModelManager::unload(const std::string& name) {
    auto it = loaded_.find(name);
    if (it != loaded_.end()) {
        it->second->unload_model();
        loaded_.erase(it);

        for (auto& info : registry_) {
            if (info.name == name) info.loaded = false;
        }

        log::info("Unloaded model: %s", name.c_str());
    }
}

InferenceEngine* ModelManager::get(const std::string& name) {
    auto it = loaded_.find(name);
    if (it != loaded_.end()) return it->second.get();
    return nullptr;
}

void ModelManager::unload_all() {
    for (auto& kv : loaded_) {
        kv.second->unload_model();
    }
    loaded_.clear();
    for (auto& info : registry_) {
        info.loaded = false;
    }
}

u64 ModelManager::total_memory_usage() const {
    u64 total = 0;
    for (const auto& info : registry_) {
        if (info.loaded) {
            total += info.file_size;
        }
    }
    return total;
}

std::string ModelManager::model_name_from_path(const std::string& path) const {
    return fs::path(path).stem().string();
}

} // namespace qraf
