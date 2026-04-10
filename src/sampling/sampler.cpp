#include "sampling/sampler.h"
#include "nn/ops.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace qraf {

Sampler::Sampler(const SamplingConfig& config) : config_(config) {
    if (config.seed != 0) {
        rng_.seed(static_cast<unsigned>(config.seed));
    } else {
        std::random_device rd;
        rng_.seed(rd());
    }
}

void Sampler::set_config(const SamplingConfig& config) {
    config_ = config;
    if (config.seed != 0) {
        rng_.seed(static_cast<unsigned>(config.seed));
    }
}

u32 Sampler::argmax(const f32* logits, int vocab_size) {
    u32 best = 0;
    f32 best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = static_cast<u32>(i);
        }
    }
    return best;
}

void Sampler::apply_repetition_penalty(f32* logits, int vocab_size,
                                       const std::vector<u32>& recent_tokens) {
    if (config_.repetition_penalty == 1.0f) return;

    for (u32 tok : recent_tokens) {
        if (tok >= static_cast<u32>(vocab_size)) continue;
        if (logits[tok] > 0) {
            logits[tok] /= config_.repetition_penalty;
        } else {
            logits[tok] *= config_.repetition_penalty;
        }
    }
}

u32 Sampler::sample(const Tensor& logits) {
    assert(logits.ndim() == 1 && logits.dtype() == DType::F32);
    return sample(logits.data_f32(), static_cast<int>(logits.shape()[0]));
}

u32 Sampler::sample(const f32* logits, int vocab_size) {
    // Deterministic / greedy mode
    if (config_.deterministic || config_.temperature <= 0.0f) {
        return argmax(logits, vocab_size);
    }

    // Apply temperature
    std::vector<f32> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = logits[i] / config_.temperature;
    }

    // Sort indices by logit value (descending) for top-k/top-p
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&probs](int a, int b) { return probs[a] > probs[b]; });

    // Top-K filtering
    int cutoff = vocab_size;
    if (config_.top_k > 0 && config_.top_k < vocab_size) {
        cutoff = config_.top_k;
    }

    // Softmax over top-k candidates
    f32 max_val = probs[indices[0]];
    f32 sum = 0.0f;
    std::vector<f32> filtered_probs(cutoff);
    for (int i = 0; i < cutoff; i++) {
        filtered_probs[i] = std::exp(probs[indices[i]] - max_val);
        sum += filtered_probs[i];
    }
    for (int i = 0; i < cutoff; i++) {
        filtered_probs[i] /= sum;
    }

    // Top-P (nucleus) filtering
    if (config_.top_p > 0.0f && config_.top_p < 1.0f) {
        f32 cumsum = 0.0f;
        int nucleus_size = cutoff;
        for (int i = 0; i < cutoff; i++) {
            cumsum += filtered_probs[i];
            if (cumsum >= config_.top_p) {
                nucleus_size = i + 1;
                break;
            }
        }

        // Renormalize
        f32 nucleus_sum = 0.0f;
        for (int i = 0; i < nucleus_size; i++) {
            nucleus_sum += filtered_probs[i];
        }
        for (int i = 0; i < nucleus_size; i++) {
            filtered_probs[i] /= nucleus_sum;
        }
        cutoff = nucleus_size;
    }

    // Sample from filtered distribution
    std::uniform_real_distribution<f32> dist(0.0f, 1.0f);
    f32 r = dist(rng_);
    f32 cumsum = 0.0f;
    for (int i = 0; i < cutoff; i++) {
        cumsum += filtered_probs[i];
        if (r <= cumsum) {
            return static_cast<u32>(indices[i]);
        }
    }

    // Fallback to last candidate
    return static_cast<u32>(indices[cutoff - 1]);
}

} // namespace qraf
