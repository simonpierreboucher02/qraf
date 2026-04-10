#include "runtime/speculative.h"
#include "nn/backend.h"
#include "core/logging.h"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace qraf {

SpeculativeDecoder::SpeculativeDecoder(Transformer& main, Transformer& draft,
                                       Sampler& sampler)
    : main_(main), draft_(draft), sampler_(sampler) {
    std::random_device rd;
    rng_.seed(rd());
}

int SpeculativeDecoder::verify_and_accept(
    const std::vector<u32>& draft_tokens,
    const std::vector<Tensor>& draft_logits,
    const std::vector<Tensor>& main_logits,
    std::vector<u32>& accepted,
    f32 temperature
) {
    int K = static_cast<int>(draft_tokens.size());
    int vocab = static_cast<int>(main_logits[0].shape()[0]);
    std::uniform_real_distribution<f32> dist(0.0f, 1.0f);

    accepted.clear();

    for (int i = 0; i < K; i++) {
        u32 token = draft_tokens[i];

        // Get probabilities from draft and main models
        // Apply temperature and softmax
        std::vector<f32> p_draft(vocab), p_main(vocab);
        const f32* dl = draft_logits[i].data_f32();
        const f32* ml = main_logits[i].data_f32();

        // Temperature scaling
        f32 inv_temp = (temperature > 0) ? 1.0f / temperature : 1.0f;
        f32 max_d = *std::max_element(dl, dl + vocab);
        f32 max_m = *std::max_element(ml, ml + vocab);

        f32 sum_d = 0, sum_m = 0;
        for (int v = 0; v < vocab; v++) {
            p_draft[v] = std::exp((dl[v] - max_d) * inv_temp);
            p_main[v] = std::exp((ml[v] - max_m) * inv_temp);
            sum_d += p_draft[v];
            sum_m += p_main[v];
        }
        for (int v = 0; v < vocab; v++) {
            p_draft[v] /= sum_d;
            p_main[v] /= sum_m;
        }

        // Rejection sampling: accept with probability min(1, p_main(x) / p_draft(x))
        f32 p_d = p_draft[token];
        f32 p_m = p_main[token];
        f32 accept_prob = (p_d > 0) ? std::min(1.0f, p_m / p_d) : 0.0f;

        f32 r = dist(rng_);
        if (r < accept_prob) {
            accepted.push_back(token);
        } else {
            // Reject: resample from adjusted distribution
            // p_adjusted = max(0, p_main - p_draft) / Z
            std::vector<f32> adjusted(vocab);
            f32 adj_sum = 0;
            for (int v = 0; v < vocab; v++) {
                adjusted[v] = std::max(0.0f, p_main[v] - p_draft[v]);
                adj_sum += adjusted[v];
            }
            if (adj_sum > 0) {
                for (int v = 0; v < vocab; v++) adjusted[v] /= adj_sum;
            }

            // Sample from adjusted distribution
            f32 r2 = dist(rng_);
            f32 cumsum = 0;
            u32 resampled = draft_tokens[i]; // fallback
            for (int v = 0; v < vocab; v++) {
                cumsum += adjusted[v];
                if (r2 <= cumsum) {
                    resampled = static_cast<u32>(v);
                    break;
                }
            }
            accepted.push_back(resampled);
            return static_cast<int>(accepted.size()); // stop at first rejection
        }
    }

    return static_cast<int>(accepted.size());
}

SpeculativeResult SpeculativeDecoder::generate(
    const std::vector<u32>& prompt_tokens,
    u32 max_tokens,
    const SpeculativeConfig& config,
    std::function<bool(u32, const std::string&)> callback,
    const Tokenizer* tokenizer
) {
    SpeculativeResult result;
    int K = static_cast<int>(config.num_speculative);

    auto start = std::chrono::high_resolution_clock::now();

    // Reset both models
    main_.reset();
    draft_.reset();

    // Prefill both models with prompt
    int pos = 0;
    Tensor main_logits, draft_logits_last;
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        main_logits = main_.forward(prompt_tokens[i], pos);
        draft_logits_last = draft_.forward(prompt_tokens[i], pos);
        pos++;
    }

    u32 eos = tokenizer ? tokenizer->eos_token() : 2;

    // Generation loop
    u32 generated = 0;
    while (generated < max_tokens) {
        // Step 1: Draft model generates K candidates
        std::vector<u32> draft_tokens;
        std::vector<Tensor> draft_logits_vec;
        int draft_pos = pos;

        // Sample first draft token from draft model's last logits
        u32 draft_tok = sampler_.sample(draft_logits_last);
        draft_tokens.push_back(draft_tok);
        draft_logits_vec.push_back(draft_logits_last);

        for (int k = 1; k < K && (generated + k) < max_tokens; k++) {
            Tensor dl = draft_.forward(draft_tok, draft_pos);
            draft_pos++;
            draft_tok = sampler_.sample(dl);
            draft_tokens.push_back(draft_tok);
            draft_logits_vec.push_back(dl);
        }

        result.draft_tokens += static_cast<u32>(draft_tokens.size());

        // Step 2: Run main model on all draft tokens to get verification logits
        std::vector<Tensor> main_logits_vec;
        int verify_pos = pos;
        for (size_t i = 0; i < draft_tokens.size(); i++) {
            Tensor ml = main_.forward(draft_tokens[i], verify_pos);
            verify_pos++;
            main_logits_vec.push_back(ml);
        }

        // Step 3: Verify and accept
        std::vector<u32> accepted;
        int num_accepted = verify_and_accept(
            draft_tokens, draft_logits_vec, main_logits_vec,
            accepted, config.temperature
        );

        result.accepted_tokens += static_cast<u32>(num_accepted);

        // Step 4: Output accepted tokens
        for (size_t i = 0; i < accepted.size(); i++) {
            u32 tok = accepted[i];
            if (tok == eos) goto done;

            result.tokens.push_back(tok);
            generated++;

            if (callback && tokenizer) {
                std::string text = tokenizer->decode(tok);
                if (!callback(tok, text)) goto done;
            }
        }

        // Advance position
        pos += static_cast<int>(accepted.size());

        // Reset draft model's KV cache to match accepted state
        // (Simplified: just advance. For full correctness, would need cache rollback)
        draft_.reset();
        // Re-prefill draft with all accepted context
        int re_pos = 0;
        for (u32 pt : prompt_tokens) {
            draft_logits_last = draft_.forward(pt, re_pos++);
        }
        for (u32 at : result.tokens) {
            draft_logits_last = draft_.forward(at, re_pos++);
        }

        // Get main model's logits for next draft step
        if (!main_logits_vec.empty()) {
            main_logits = main_logits_vec.back();
            draft_logits_last = main_logits; // use main's logits for next draft
        }
    }

done:
    auto end = std::chrono::high_resolution_clock::now();
    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.total_tokens = static_cast<u32>(result.tokens.size());
    if (result.draft_tokens > 0) {
        result.acceptance_rate = static_cast<f32>(result.accepted_tokens) /
                                 static_cast<f32>(result.draft_tokens);
    }
    if (tokenizer) result.text = tokenizer->decode(result.tokens);

    log::info("Speculative: %u tokens, acceptance=%.1f%%, draft=%u, accepted=%u",
              result.total_tokens, result.acceptance_rate * 100.0f,
              result.draft_tokens, result.accepted_tokens);

    return result;
}

} // namespace qraf
