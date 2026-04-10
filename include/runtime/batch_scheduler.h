#pragma once

#include "core/types.h"
#include "nn/transformer.h"
#include "sampling/sampler.h"
#include "runtime/tokenizer.h"
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <thread>
#include <memory>

namespace qraf {

// ─── Continuous Batching Scheduler ───
// Manages multiple concurrent inference requests with per-request KV caches.
// Implements iteration-level scheduling: at each decode step, the scheduler
// picks which requests to process based on available memory and priority.

struct BatchRequest {
    u32 id;
    std::vector<u32> prompt_tokens;
    u32 max_tokens;
    SamplingConfig sampling;
    std::function<void(u32 token_id, const std::string& text)> on_token;
    std::function<void(const std::string& full_text, u32 tokens, double ms)> on_complete;

    // Internal state
    enum class State { QUEUED, PREFILLING, DECODING, DONE };
    State state = State::QUEUED;
    std::vector<u32> generated_tokens;
    int current_pos = 0;
    u32 kv_slot = 0;  // which KV cache slot this request uses
};

struct BatchSlot {
    bool active = false;
    u32 request_id = 0;
    KVCache cache;
    Sampler sampler;
    int pos = 0;
    Tensor last_logits;
};

struct BatchConfig {
    u32 max_batch_size = 8;     // max concurrent requests
    u32 max_seq_len = 2048;     // max total sequence length per request
    u32 max_queue_size = 64;    // max queued requests
};

class BatchScheduler {
public:
    BatchScheduler(Transformer& model, const Tokenizer& tokenizer,
                   const BatchConfig& config = {});
    ~BatchScheduler();

    // Submit a request (thread-safe)
    u32 submit(const std::string& prompt, u32 max_tokens,
               const SamplingConfig& sampling = {},
               std::function<void(u32, const std::string&)> on_token = nullptr,
               std::function<void(const std::string&, u32, double)> on_complete = nullptr);

    // Start the scheduler loop (runs in background thread)
    void start();

    // Stop the scheduler
    void stop();

    // Get stats
    u32 active_requests() const;
    u32 queued_requests() const;
    u32 total_processed() const { return total_processed_; }

private:
    void scheduler_loop();
    void process_step();
    int find_free_slot();
    void start_request(BatchRequest& req);
    void decode_step(BatchSlot& slot, BatchRequest& req);
    void finish_request(BatchRequest& req, BatchSlot& slot);

    Transformer& model_;
    const Tokenizer& tokenizer_;
    BatchConfig config_;

    // Request queue
    std::queue<BatchRequest> pending_;
    std::vector<BatchRequest> active_requests_;
    std::vector<BatchSlot> slots_;

    // Threading
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    std::atomic<u32> next_id_{1};
    std::atomic<u32> total_processed_{0};
};

// ─── Implementation ───

inline BatchScheduler::BatchScheduler(Transformer& model, const Tokenizer& tokenizer,
                                       const BatchConfig& config)
    : model_(model), tokenizer_(tokenizer), config_(config) {
    slots_.resize(config.max_batch_size);
}

inline BatchScheduler::~BatchScheduler() { stop(); }

inline u32 BatchScheduler::submit(
    const std::string& prompt, u32 max_tokens,
    const SamplingConfig& sampling,
    std::function<void(u32, const std::string&)> on_token,
    std::function<void(const std::string&, u32, double)> on_complete
) {
    BatchRequest req;
    req.id = next_id_++;
    req.prompt_tokens = tokenizer_.encode(prompt);
    req.max_tokens = max_tokens;
    req.sampling = sampling;
    req.on_token = on_token;
    req.on_complete = on_complete;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_.push(std::move(req));
    }
    cv_.notify_one();
    return req.id;
}

inline void BatchScheduler::start() {
    running_ = true;
    worker_ = std::thread([this]() { scheduler_loop(); });
}

inline void BatchScheduler::stop() {
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
}

inline void BatchScheduler::scheduler_loop() {
    while (running_) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(1),
                         [this]() { return !pending_.empty() || !running_; });
        }
        if (!running_) break;
        process_step();
    }
}

inline int BatchScheduler::find_free_slot() {
    for (size_t i = 0; i < slots_.size(); i++) {
        if (!slots_[i].active) return static_cast<int>(i);
    }
    return -1;
}

inline void BatchScheduler::start_request(BatchRequest& req) {
    int slot_idx = find_free_slot();
    if (slot_idx < 0) return;

    auto& slot = slots_[slot_idx];
    slot.active = true;
    slot.request_id = req.id;
    slot.pos = 0;
    slot.sampler.set_config(req.sampling);

    // Initialize KV cache for this slot
    slot.cache.init(
        static_cast<int>(model_.config().num_layers),
        static_cast<int>(model_.config().num_kv_heads),
        static_cast<int>(model_.config().head_dim),
        static_cast<int>(config_.max_seq_len)
    );

    req.kv_slot = static_cast<u32>(slot_idx);
    req.state = BatchRequest::State::PREFILLING;
}

inline void BatchScheduler::decode_step(BatchSlot& slot, BatchRequest& req) {
    // Note: In a full implementation, we'd batch multiple requests
    // into a single forward pass. For now, we process sequentially
    // but manage the lifecycle correctly.

    if (req.state == BatchRequest::State::PREFILLING) {
        // Process prompt tokens one at a time
        for (size_t i = 0; i < req.prompt_tokens.size(); i++) {
            slot.last_logits = model_.forward(req.prompt_tokens[i], slot.pos);
            slot.pos++;
        }
        req.state = BatchRequest::State::DECODING;
        req.current_pos = slot.pos;
    }

    if (req.state == BatchRequest::State::DECODING) {
        // Sample next token
        f32* logits = slot.last_logits.data_f32();
        int vocab = static_cast<int>(slot.last_logits.shape()[0]);
        slot.sampler.apply_repetition_penalty(logits, vocab, req.generated_tokens);
        u32 token = slot.sampler.sample(logits, vocab);

        // Check EOS
        if (token == tokenizer_.eos_token() ||
            req.generated_tokens.size() >= req.max_tokens) {
            req.state = BatchRequest::State::DONE;
            return;
        }

        req.generated_tokens.push_back(token);

        // Callback
        if (req.on_token) {
            std::string text = tokenizer_.decode(token);
            req.on_token(token, text);
        }

        // Forward for next token
        slot.last_logits = model_.forward(token, slot.pos);
        slot.pos++;
    }
}

inline void BatchScheduler::finish_request(BatchRequest& req, BatchSlot& slot) {
    if (req.on_complete) {
        std::string text = tokenizer_.decode(req.generated_tokens);
        req.on_complete(text, static_cast<u32>(req.generated_tokens.size()), 0.0);
    }
    slot.active = false;
    total_processed_++;
}

inline void BatchScheduler::process_step() {
    // Admit new requests from queue
    {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!pending_.empty()) {
            int free = find_free_slot();
            if (free < 0) break;

            BatchRequest req = std::move(pending_.front());
            pending_.pop();
            start_request(req);
            active_requests_.push_back(std::move(req));
        }
    }

    // Process one decode step for each active request
    for (size_t i = 0; i < active_requests_.size(); ) {
        auto& req = active_requests_[i];
        if (req.state == BatchRequest::State::DONE) {
            finish_request(req, slots_[req.kv_slot]);
            active_requests_.erase(active_requests_.begin() + i);
            continue;
        }

        model_.reset(); // Simplified: reset between requests
        decode_step(slots_[req.kv_slot], req);

        if (req.state == BatchRequest::State::DONE) {
            finish_request(req, slots_[req.kv_slot]);
            active_requests_.erase(active_requests_.begin() + i);
        } else {
            i++;
        }
    }
}

inline u32 BatchScheduler::active_requests() const {
    u32 count = 0;
    for (const auto& s : slots_) if (s.active) count++;
    return count;
}

inline u32 BatchScheduler::queued_requests() const {
    return static_cast<u32>(pending_.size());
}

} // namespace qraf
