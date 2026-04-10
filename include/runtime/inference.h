#pragma once

#include "core/types.h"
#include "nn/transformer.h"
#include "runtime/tokenizer.h"
#include "sampling/sampler.h"
#include "qraf/loader.h"
#include <string>
#include <vector>
#include <functional>

namespace qraf {

// Callback for streaming token output
using TokenCallback = std::function<bool(u32 token_id, const std::string& text)>;

struct GenerateConfig {
    u32 max_tokens     = 256;
    SamplingConfig sampling;
    bool echo_prompt   = false;  // include prompt tokens in output
    std::vector<u32> stop_tokens;  // additional stop tokens
};

struct GenerateResult {
    std::vector<u32> tokens;
    std::string text;
    u32 prompt_tokens = 0;
    u32 generated_tokens = 0;
    double generation_time_ms = 0.0;
    double tokens_per_sec = 0.0;
};

class InferenceEngine {
public:
    InferenceEngine() = default;

    // Load a QRAF model
    bool load_model(const std::string& path);
    void unload_model();
    bool is_loaded() const { return loaded_; }

    // Generate text from prompt
    GenerateResult generate(const std::string& prompt,
                            const GenerateConfig& config = {},
                            TokenCallback callback = nullptr);

    // Generate from pre-tokenized input
    GenerateResult generate_tokens(const std::vector<u32>& prompt_tokens,
                                   const GenerateConfig& config = {},
                                   TokenCallback callback = nullptr);

    // Access components
    const ModelConfig& model_config() const { return transformer_.config(); }
    const Tokenizer& tokenizer() const { return tokenizer_; }

    // Reset state for new conversation
    void reset();

private:
    QrafModel model_;
    Transformer transformer_;
    Tokenizer tokenizer_;
    bool loaded_ = false;
};

} // namespace qraf
