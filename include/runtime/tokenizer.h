#pragma once

#include "core/types.h"
#include "qraf/format.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace qraf {

class Tokenizer {
public:
    Tokenizer() = default;

    // Load from QRAF tokenizer block (mmap pointer)
    bool load_from_qraf(const void* data, u64 size, const char* string_table);

    // Load a simple vocab from a text file (one token per line)
    bool load_vocab_file(const std::string& path);

    // Encode text to token IDs
    std::vector<u32> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<u32>& tokens) const;
    std::string decode(u32 token) const;

    // Vocabulary access
    u32 vocab_size() const { return static_cast<u32>(id_to_token_.size()); }
    const std::string& id_to_token(u32 id) const;
    u32 token_to_id(const std::string& token) const;
    bool has_token(const std::string& token) const;

    // Special tokens
    u32 bos_token() const { return bos_id_; }
    u32 eos_token() const { return eos_id_; }
    u32 pad_token() const { return pad_id_; }
    u32 unk_token() const { return unk_id_; }

private:
    // BPE merge step
    struct Merge {
        u32 token_a;
        u32 token_b;
        u32 result;
        f32 priority;
    };

    // Apply BPE merges to a sequence of token IDs
    std::vector<u32> apply_bpe(const std::vector<u32>& tokens) const;

    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, u32> token_to_id_;
    std::vector<f32> token_scores_;

    // BPE merges sorted by priority (highest first)
    std::vector<Merge> merges_;
    // Map (token_a, token_b) -> merge index for O(1) lookup
    std::unordered_map<u64, u32> merge_map_;

    u32 bos_id_ = 1;
    u32 eos_id_ = 2;
    u32 pad_id_ = 0;
    u32 unk_id_ = 0;
};

} // namespace qraf
