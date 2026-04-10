#include "nn/lora.h"
#include "nn/backend.h"
#include "core/logging.h"
#include <cstring>
#include <filesystem>

namespace qraf {

bool LoraManager::load_adapter(const std::string& path, const std::string& name, f32 alpha) {
    LoraAdapter adapter;
    adapter.name = name.empty() ? std::filesystem::path(path).stem().string() : name;

    // Try loading as QRAF file
    QrafModel model;
    if (!model.load(path)) {
        log::error("LoRA: failed to load adapter: %s", path.c_str());
        return false;
    }

    // Read LoRA config from model config
    const auto& cfg = model.config();
    // Default rank/alpha
    adapter.rank = 8;
    adapter.alpha = alpha > 0 ? alpha : 16.0f;

    // Scan for LoRA tensors: look for patterns like *.lora_A.weight, *.lora_B.weight
    auto tensor_names = model.tensor_names();
    for (const auto& tname : tensor_names) {
        // Check for lora_A pattern
        size_t pos = tname.find(".lora_A");
        if (pos == std::string::npos) pos = tname.find(".lora_a");
        if (pos == std::string::npos) continue;

        std::string base_name = tname.substr(0, pos);
        std::string a_name = base_name + ".lora_A.weight";
        std::string b_name = base_name + ".lora_B.weight";

        // Try alternate naming
        if (!model.has_tensor(a_name)) a_name = base_name + ".lora_a.weight";
        if (!model.has_tensor(b_name)) b_name = base_name + ".lora_b.weight";

        if (!model.has_tensor(a_name) || !model.has_tensor(b_name)) continue;

        TensorView tv_a = model.get_tensor(a_name);
        TensorView tv_b = model.get_tensor(b_name);

        LoraAdapter::LoraWeight lw;
        // A: [rank, in_dim], B: [out_dim, rank]
        if (tv_a.dtype == DType::F32) {
            lw.A = Tensor::from_data(tv_a.shape, DType::F32, tv_a.data, tv_a.data_size);
        }
        if (tv_b.dtype == DType::F32) {
            lw.B = Tensor::from_data(tv_b.shape, DType::F32, tv_b.data, tv_b.data_size);
        }

        if (!lw.A.empty() && !lw.B.empty()) {
            adapter.rank = lw.A.shape()[0];
            adapter.weights[base_name] = std::move(lw);
        }
    }

    adapter.scaling = adapter.alpha / static_cast<f32>(adapter.rank);

    log::info("LoRA adapter '%s': rank=%u, alpha=%.1f, scaling=%.4f, %zu weight pairs",
              adapter.name.c_str(), adapter.rank, adapter.alpha,
              adapter.scaling, adapter.weights.size());

    adapters_.push_back(std::move(adapter));
    return true;
}

Tensor LoraManager::merge(const std::string& tensor_name, const Tensor& base_weight) const {
    QRAF_CHECK(base_weight.ndim() == 2, "LoRA merge requires 2D tensor");

    u32 out_dim = base_weight.shape()[0];
    u32 in_dim = base_weight.shape()[1];

    // Copy base weight
    Tensor result = Tensor::from_data(base_weight.shape(), DType::F32,
                                       base_weight.data(), base_weight.nbytes());

    merge_inplace(tensor_name, result.data_f32(), out_dim, in_dim);
    return result;
}

void LoraManager::merge_inplace(const std::string& tensor_name, f32* weight,
                                 u32 out_dim, u32 in_dim) const {
    for (const auto& adapter : adapters_) {
        auto it = adapter.weights.find(tensor_name);
        if (it == adapter.weights.end()) continue;

        const auto& lw = it->second;
        u32 rank = lw.A.shape()[0];

        // W' = W + scaling * B @ A
        // B: [out_dim, rank], A: [rank, in_dim]
        // temp = B @ A: [out_dim, in_dim]
        const f32* A = lw.A.data_f32();
        const f32* B = lw.B.data_f32();
        f32 s = adapter.scaling;

        // Fused: weight[i][j] += scaling * sum_k(B[i][k] * A[k][j])
        for (u32 i = 0; i < out_dim; i++) {
            for (u32 k = 0; k < rank; k++) {
                f32 b_ik = B[i * rank + k] * s;
                for (u32 j = 0; j < in_dim; j++) {
                    weight[i * in_dim + j] += b_ik * A[k * in_dim + j];
                }
            }
        }
    }
}

bool LoraManager::has_weights(const std::string& tensor_name) const {
    for (const auto& adapter : adapters_) {
        if (adapter.weights.count(tensor_name)) return true;
    }
    return false;
}

std::vector<std::string> LoraManager::list_adapters() const {
    std::vector<std::string> names;
    for (const auto& a : adapters_) names.push_back(a.name);
    return names;
}

void LoraManager::remove(const std::string& name) {
    adapters_.erase(
        std::remove_if(adapters_.begin(), adapters_.end(),
                        [&](const LoraAdapter& a) { return a.name == name; }),
        adapters_.end());
}

} // namespace qraf
