#pragma once

#include "core/types.h"
#include "tensor/tensor.h"
#include <vector>
#include <random>

namespace qraf {

struct SamplingConfig {
    f32 temperature       = 0.8f;
    i32 top_k             = 40;
    f32 top_p             = 0.9f;
    f32 repetition_penalty = 1.1f;
    u64 seed              = 0;        // 0 = random seed
    bool deterministic    = false;
};

class Sampler {
public:
    explicit Sampler(const SamplingConfig& config = {});

    // Sample a token from logits
    u32 sample(const Tensor& logits);
    u32 sample(const f32* logits, int vocab_size);

    // Apply repetition penalty to logits (modifies in place)
    void apply_repetition_penalty(f32* logits, int vocab_size,
                                  const std::vector<u32>& recent_tokens);

    // Greedy (argmax)
    static u32 argmax(const f32* logits, int vocab_size);

    // Update config
    void set_config(const SamplingConfig& config);
    const SamplingConfig& config() const { return config_; }

private:
    SamplingConfig config_;
    std::mt19937 rng_;
};

} // namespace qraf
