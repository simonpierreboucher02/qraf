#include "runtime/tokenizer.h"
#include "core/logging.h"
#include "core/error.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstring>
#include <limits>

namespace qraf {

static u64 merge_key(u32 a, u32 b) {
    return (static_cast<u64>(a) << 32) | static_cast<u64>(b);
}

// ─── GPT-2 byte-level BPE unicode-to-byte mapping ───
// GPT-2 tokenizers map bytes 0-255 to specific Unicode code points.
// We need the inverse: unicode code point -> original byte.
static std::unordered_map<u32, u8> build_unicode_to_byte_map() {
    std::unordered_map<u32, u8> m;
    // Printable ASCII range maps to itself
    // '!' (33) to '~' (126), plus some extras
    int n = 0;
    for (int b = 0; b < 256; b++) {
        // These byte values map to themselves as unicode:
        // 33-126 ('!' to '~'), 161-172, 174-255
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            m[static_cast<u32>(b)] = static_cast<u8>(b);
        }
    }
    // The remaining bytes (0-32, 127-160, 173) are mapped to U+0100 onwards
    n = 0;
    for (int b = 0; b < 256; b++) {
        if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))) {
            m[256 + n] = static_cast<u8>(b);
            n++;
        }
    }
    return m;
}

// ─── GPT-2 byte-to-unicode mapping (for encoding) ───
// Converts raw bytes to the unicode representation used in vocab
static std::unordered_map<u8, u32> build_byte_to_unicode_map() {
    std::unordered_map<u8, u32> m;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            m[static_cast<u8>(b)] = static_cast<u32>(b);
        }
    }
    n = 0;
    for (int b = 0; b < 256; b++) {
        if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))) {
            m[static_cast<u8>(b)] = 256 + n;
            n++;
        }
    }
    return m;
}

// Convert raw text to GPT-2 unicode representation for vocab lookup
static std::string encode_bytes_to_unicode(const std::string& text) {
    static auto byte_to_unicode = build_byte_to_unicode_map();
    std::string result;
    for (u8 b : text) {
        u32 cp = byte_to_unicode[b];
        // Encode code point as UTF-8
        if (cp < 0x80) {
            result += static_cast<char>(cp);
        } else if (cp < 0x800) {
            result += static_cast<char>(0xC0 | (cp >> 6));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            result += static_cast<char>(0xE0 | (cp >> 12));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
    }
    return result;
}

// Decode a BPE token string (which may contain GPT-2 byte-level unicode) to raw bytes
static std::string decode_bpe_token(const std::string& token) {
    static auto unicode_to_byte = build_unicode_to_byte_map();

    std::string result;
    size_t i = 0;
    while (i < token.size()) {
        // Decode UTF-8 to get unicode code point
        u32 cp = 0;
        u8 c = static_cast<u8>(token[i]);
        int bytes = 1;

        if (c < 0x80) {
            cp = c;
        } else if (c < 0xE0 && i + 1 < token.size()) {
            cp = (c & 0x1F) << 6;
            cp |= (static_cast<u8>(token[i + 1]) & 0x3F);
            bytes = 2;
        } else if (c < 0xF0 && i + 2 < token.size()) {
            cp = (c & 0x0F) << 12;
            cp |= (static_cast<u8>(token[i + 1]) & 0x3F) << 6;
            cp |= (static_cast<u8>(token[i + 2]) & 0x3F);
            bytes = 3;
        } else if (i + 3 < token.size()) {
            cp = (c & 0x07) << 18;
            cp |= (static_cast<u8>(token[i + 1]) & 0x3F) << 12;
            cp |= (static_cast<u8>(token[i + 2]) & 0x3F) << 6;
            cp |= (static_cast<u8>(token[i + 3]) & 0x3F);
            bytes = 4;
        }

        auto it = unicode_to_byte.find(cp);
        if (it != unicode_to_byte.end()) {
            result += static_cast<char>(it->second);
        } else {
            // Not in mapping, keep original bytes
            for (int j = 0; j < bytes; j++) {
                result += token[i + j];
            }
        }
        i += bytes;
    }
    return result;
}

bool Tokenizer::load_from_qraf(const void* data, u64 size, const char* string_table) {
    if (!data || size == 0) {
        log::error("No tokenizer data");
        return false;
    }

    const u8* ptr = static_cast<const u8*>(data);

    // Read header
    u32 vocab_size, merges_count, special_count, reserved;
    memcpy(&vocab_size, ptr, sizeof(u32)); ptr += sizeof(u32);
    memcpy(&merges_count, ptr, sizeof(u32)); ptr += sizeof(u32);
    memcpy(&special_count, ptr, sizeof(u32)); ptr += sizeof(u32);
    memcpy(&reserved, ptr, sizeof(u32)); ptr += sizeof(u32);

    log::info("Tokenizer: vocab_size=%u, merges=%u, special=%u",
              vocab_size, merges_count, special_count);

    // Read token entries
    id_to_token_.resize(vocab_size);
    token_scores_.resize(vocab_size, 0.0f);

    const TokenEntry* entries = reinterpret_cast<const TokenEntry*>(ptr);
    for (u32 i = 0; i < vocab_size; i++) {
        const char* str = string_table + entries[i].string_offset;
        std::string token(str, entries[i].string_length);
        id_to_token_[i] = token;
        token_to_id_[token] = i;
        token_scores_[i] = entries[i].score;
    }
    ptr += vocab_size * sizeof(TokenEntry);

    // Read BPE merges
    const MergeEntry* merge_entries = reinterpret_cast<const MergeEntry*>(ptr);
    merges_.resize(merges_count);
    for (u32 i = 0; i < merges_count; i++) {
        merges_[i].token_a = merge_entries[i].token_a;
        merges_[i].token_b = merge_entries[i].token_b;
        merges_[i].result = merge_entries[i].result;
        merges_[i].priority = merge_entries[i].priority;
        merge_map_[merge_key(merges_[i].token_a, merges_[i].token_b)] = i;
    }
    ptr += merges_count * sizeof(MergeEntry);

    // Read special tokens
    const SpecialToken* specials = reinterpret_cast<const SpecialToken*>(ptr);
    for (u32 i = 0; i < special_count; i++) {
        switch (specials[i].type) {
            case 0: bos_id_ = specials[i].token_id; break;
            case 1: eos_id_ = specials[i].token_id; break;
            case 2: pad_id_ = specials[i].token_id; break;
            case 3: unk_id_ = specials[i].token_id; break;
        }
    }

    log::info("Special tokens: bos=%u, eos=%u, pad=%u, unk=%u",
              bos_id_, eos_id_, pad_id_, unk_id_);

    return true;
}

bool Tokenizer::load_vocab_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        log::error("Failed to open vocab file: %s", path.c_str());
        return false;
    }

    std::string line;
    u32 id = 0;
    while (std::getline(file, line)) {
        // Format: token [score]
        std::istringstream iss(line);
        std::string token;
        f32 score = 0.0f;
        iss >> token;
        iss >> score;

        id_to_token_.push_back(token);
        token_to_id_[token] = id;
        token_scores_.push_back(score);
        id++;
    }

    log::info("Loaded vocab: %u tokens from %s", id, path.c_str());
    return true;
}

std::vector<u32> Tokenizer::encode(const std::string& text) const {
    if (text.empty()) return {};

    // Convert raw text to GPT-2 byte-level unicode for vocab lookup
    std::string unicode_text = encode_bytes_to_unicode(text);

    if (!merges_.empty()) {
        // ─── BPE path: char-level init, then merge ───
        // This is the correct BPE algorithm:
        // 1. Split into individual unicode characters
        // 2. Look up each character in vocab
        // 3. Apply BPE merges iteratively (most common pairs first)
        std::vector<u32> tokens;
        size_t i = 0;
        while (i < unicode_text.size()) {
            u8 c = static_cast<u8>(unicode_text[i]);
            int char_len = 1;
            if (c >= 0xC0 && c < 0xE0) char_len = 2;
            else if (c >= 0xE0 && c < 0xF0) char_len = 3;
            else if (c >= 0xF0) char_len = 4;

            std::string ch = unicode_text.substr(i, char_len);
            auto it = token_to_id_.find(ch);
            tokens.push_back(it != token_to_id_.end() ? it->second : unk_id_);
            i += char_len;
        }
        return apply_bpe(tokens);
    } else {
        // ─── Legacy greedy path (no merges) ───
        std::vector<u32> tokens;
        for (size_t i = 0; i < unicode_text.size(); ) {
            bool found = false;
            for (size_t len = std::min(unicode_text.size() - i, size_t(128)); len > 0; len--) {
                std::string substr = unicode_text.substr(i, len);
                auto it = token_to_id_.find(substr);
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                    i += len;
                    found = true;
                    break;
                }
            }
            if (!found) {
                u8 c = static_cast<u8>(unicode_text[i]);
                int char_len = 1;
                if (c >= 0xC0 && c < 0xE0) char_len = 2;
                else if (c >= 0xE0 && c < 0xF0) char_len = 3;
                else if (c >= 0xF0) char_len = 4;
                std::string ch = unicode_text.substr(i, char_len);
                auto it = token_to_id_.find(ch);
                tokens.push_back(it != token_to_id_.end() ? it->second : unk_id_);
                i += char_len;
            }
        }
        return tokens;
    }
}

std::vector<u32> Tokenizer::apply_bpe(const std::vector<u32>& input) const {
    if (input.size() <= 1) return input;

    std::vector<u32> tokens = input;

    while (tokens.size() > 1) {
        // Find the highest-priority merge
        f32 best_priority = -std::numeric_limits<f32>::infinity();
        size_t best_pos = SIZE_MAX;
        u32 best_result = 0;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            auto it = merge_map_.find(merge_key(tokens[i], tokens[i + 1]));
            if (it != merge_map_.end()) {
                f32 priority = merges_[it->second].priority;
                if (priority > best_priority) {
                    best_priority = priority;
                    best_pos = i;
                    best_result = merges_[it->second].result;
                }
            }
        }

        if (best_pos == SIZE_MAX) break; // No more merges

        // Apply the merge
        std::vector<u32> new_tokens;
        new_tokens.reserve(tokens.size() - 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i == best_pos) {
                new_tokens.push_back(best_result);
                i++; // Skip next token (it was merged)
            } else {
                new_tokens.push_back(tokens[i]);
            }
        }
        tokens = std::move(new_tokens);
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<u32>& tokens) const {
    std::string result;
    for (u32 t : tokens) {
        result += decode(t);
    }
    return result;
}

std::string Tokenizer::decode(u32 token) const {
    if (token < id_to_token_.size()) {
        return decode_bpe_token(id_to_token_[token]);
    }
    return "<unk>";
}

const std::string& Tokenizer::id_to_token(u32 id) const {
    static const std::string unk = "<unk>";
    if (id < id_to_token_.size()) return id_to_token_[id];
    return unk;
}

u32 Tokenizer::token_to_id(const std::string& token) const {
    auto it = token_to_id_.find(token);
    if (it != token_to_id_.end()) return it->second;
    return unk_id_;
}

bool Tokenizer::has_token(const std::string& token) const {
    return token_to_id_.count(token) > 0;
}

} // namespace qraf
