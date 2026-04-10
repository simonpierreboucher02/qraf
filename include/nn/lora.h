#pragma once

#include "core/types.h"
#include "tensor/tensor.h"
#include "qraf/loader.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace qraf {

struct LoraAdapter {
    std::string name;
    f32 alpha;
    u32 rank;
    f32 scaling;  // alpha / rank

    // Per-layer A and B matrices: W' = W + scaling * B @ A
    // Key: tensor name (e.g., "model.layers.0.self_attn.q_proj")
    struct LoraWeight {
        Tensor A;  // [rank, in_dim]
        Tensor B;  // [out_dim, rank]
    };
    std::unordered_map<std::string, LoraWeight> weights;
};

class LoraManager {
public:
    // Load a LoRA adapter from a QRAF file or safetensors directory
    bool load_adapter(const std::string& path, const std::string& name = "",
                      f32 alpha = 0.0f);

    // Merge adapter into base weight: returns W + scaling * B @ A
    Tensor merge(const std::string& tensor_name, const Tensor& base_weight) const;

    // Merge adapter directly into f32 buffer (in-place)
    void merge_inplace(const std::string& tensor_name, f32* weight,
                       u32 out_dim, u32 in_dim) const;

    // Check if adapter has weights for a given tensor
    bool has_weights(const std::string& tensor_name) const;

    // List loaded adapters
    std::vector<std::string> list_adapters() const;

    // Remove adapter
    void remove(const std::string& name);

private:
    std::vector<LoraAdapter> adapters_;
};

} // namespace qraf
