#pragma once

#include "core/types.h"
#include "nn/transformer.h"
#include "sampling/sampler.h"
#include "runtime/tokenizer.h"
#include <vector>
#include <functional>
#include <random>

namespace qraf {

// ─── Speculative Decoding ───
// Uses a small draft model to generate K candidate tokens,
// then verifies them with the main model in a single forward pass batch.
// Accepted tokens are kept, rejected tokens are resampled.
// Achieves ~2-3x speedup when draft model is ~10x faster.

struct SpeculativeConfig {
    u32 num_speculative = 4;  // K: number of draft tokens per step
    f32 temperature = 0.8f;
};

struct SpeculativeResult {
    std::vector<u32> tokens;
    std::string text;
    u32 draft_tokens = 0;
    u32 accepted_tokens = 0;
    u32 total_tokens = 0;
    f32 acceptance_rate = 0.0f;
    double time_ms = 0.0;
};

class SpeculativeDecoder {
public:
    SpeculativeDecoder(Transformer& main_model, Transformer& draft_model,
                       Sampler& sampler);

    // Generate with speculative decoding
    SpeculativeResult generate(
        const std::vector<u32>& prompt_tokens,
        u32 max_tokens,
        const SpeculativeConfig& config = {},
        std::function<bool(u32, const std::string&)> callback = nullptr,
        const Tokenizer* tokenizer = nullptr
    );

private:
    // Verify draft tokens against main model
    // Returns number of accepted tokens (0 to K)
    int verify_and_accept(const std::vector<u32>& draft_tokens,
                          const std::vector<Tensor>& draft_logits,
                          const std::vector<Tensor>& main_logits,
                          std::vector<u32>& accepted,
                          f32 temperature);

    Transformer& main_;
    Transformer& draft_;
    Sampler& sampler_;
    std::mt19937 rng_;
};

} // namespace qraf
