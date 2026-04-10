#include "runtime/inference.h"
#include "core/logging.h"
#include <chrono>
#include <algorithm>

namespace qraf {

bool InferenceEngine::load_model(const std::string& path) {
    log::info("InferenceEngine: loading model %s", path.c_str());

    if (!model_.load(path)) {
        log::error("Failed to load QRAF model: %s", path.c_str());
        return false;
    }

    // Initialize tokenizer
    const void* tok_data = model_.tokenizer_data();
    u64 tok_size = model_.tokenizer_size();
    if (tok_data && tok_size > 0) {
        const char* str_table = model_.get_string(0);
        if (!tokenizer_.load_from_qraf(tok_data, tok_size, str_table)) {
            log::error("Failed to load tokenizer");
            model_.unload();
            return false;
        }
    } else {
        log::warn("No tokenizer in model, generate_tokens() will work but generate() won't");
    }

    // Initialize transformer
    if (!transformer_.init(model_)) {
        log::error("Failed to initialize transformer");
        model_.unload();
        return false;
    }

    loaded_ = true;
    log::info("Model ready: %s", path.c_str());
    return true;
}

void InferenceEngine::unload_model() {
    loaded_ = false;
    model_.unload();
    log::info("Model unloaded");
}

void InferenceEngine::reset() {
    transformer_.reset();
}

GenerateResult InferenceEngine::generate(const std::string& prompt,
                                          const GenerateConfig& config,
                                          TokenCallback callback) {
    QRAF_CHECK(loaded_, "No model loaded");

    // Tokenize the prompt
    std::vector<u32> prompt_tokens = tokenizer_.encode(prompt);

    // Add BOS token at the start
    std::vector<u32> tokens;
    tokens.push_back(tokenizer_.bos_token());
    tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());

    return generate_tokens(tokens, config, callback);
}

GenerateResult InferenceEngine::generate_tokens(const std::vector<u32>& prompt_tokens,
                                                 const GenerateConfig& config,
                                                 TokenCallback callback) {
    QRAF_CHECK(loaded_, "No model loaded");

    GenerateResult result;
    result.prompt_tokens = static_cast<u32>(prompt_tokens.size());

    Sampler sampler(config.sampling);

    // Build stop token set
    std::vector<u32> stop_tokens = config.stop_tokens;
    stop_tokens.push_back(tokenizer_.eos_token());

    // Reset KV cache
    transformer_.reset();

    auto start_time = std::chrono::high_resolution_clock::now();

    // Phase 1: Process prompt tokens (prefill)
    log::info("Prefill: %zu tokens", prompt_tokens.size());
    Tensor logits;
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        logits = transformer_.forward(prompt_tokens[i], static_cast<int>(i));

        if (config.echo_prompt && callback) {
            std::string tok_text = tokenizer_.decode(prompt_tokens[i]);
            if (!callback(prompt_tokens[i], tok_text)) {
                break;  // user cancelled
            }
        }
    }

    // Phase 2: Generate tokens
    log::info("Generating (max %u tokens)...", config.max_tokens);
    int pos = static_cast<int>(prompt_tokens.size());
    std::vector<u32> generated;
    std::vector<u32> recent_for_penalty;
    recent_for_penalty.insert(recent_for_penalty.end(),
                              prompt_tokens.begin(), prompt_tokens.end());

    for (u32 t = 0; t < config.max_tokens; t++) {
        // Apply repetition penalty
        f32* logits_data = logits.data_f32();
        int vocab_size = static_cast<int>(logits.shape()[0]);
        sampler.apply_repetition_penalty(logits_data, vocab_size, recent_for_penalty);

        // Sample next token
        u32 next_token = sampler.sample(logits_data, vocab_size);

        // Check stop condition
        bool should_stop = false;
        for (u32 stop : stop_tokens) {
            if (next_token == stop) {
                should_stop = true;
                break;
            }
        }
        if (should_stop) break;

        generated.push_back(next_token);
        recent_for_penalty.push_back(next_token);

        // Stream callback
        if (callback) {
            std::string tok_text = tokenizer_.decode(next_token);
            if (!callback(next_token, tok_text)) {
                break;  // user cancelled
            }
        }

        // Forward pass for next token
        logits = transformer_.forward(next_token, pos);
        pos++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    result.tokens = generated;
    result.text = tokenizer_.decode(generated);
    result.generated_tokens = static_cast<u32>(generated.size());
    result.generation_time_ms = elapsed_ms;
    if (elapsed_ms > 0) {
        result.tokens_per_sec = static_cast<double>(generated.size()) / (elapsed_ms / 1000.0);
    }

    log::info("Generated %u tokens in %.1f ms (%.1f tok/s)",
              result.generated_tokens, result.generation_time_ms, result.tokens_per_sec);

    return result;
}

} // namespace qraf
