#include "qraf/writer.h"
#include "core/logging.h"
#include <fstream>
#include <cstring>
#include <algorithm>

namespace qraf {

u32 QrafWriter::add_string(const std::string& str) {
    auto it = string_offsets_.find(str);
    if (it != string_offsets_.end()) return it->second;

    u32 offset = static_cast<u32>(string_table_.size());
    string_table_.insert(string_table_.end(), str.begin(), str.end());
    string_table_.push_back('\0');
    string_offsets_[str] = offset;
    return offset;
}

void QrafWriter::set_config(const std::string& key, u32 value) {
    config_entries_.push_back({key, 0, value});
}

void QrafWriter::set_config(const std::string& key, f32 value) {
    u32 bits;
    memcpy(&bits, &value, sizeof(u32));
    config_entries_.push_back({key, 1, bits});
}

void QrafWriter::set_config_string(const std::string& key, const std::string& value) {
    u32 str_offset = add_string(value);
    config_entries_.push_back({key, 2, str_offset});
}

u32 QrafWriter::add_quant_scheme(DType type, u32 block_size, u32 group_size) {
    u32 id = static_cast<u32>(quant_schemes_.size());
    QuantScheme qs{};
    qs.id = id;
    qs.type = static_cast<u32>(type);
    qs.block_size = block_size;
    qs.group_size = group_size;
    qs.scale = 0.0f;  // per-block scales
    qs.zero_point = 0.0f;
    qs.flags = 1;  // has per-block scales
    quant_schemes_.push_back(qs);
    return id;
}

void QrafWriter::add_tensor(const std::string& name, const std::vector<u32>& shape,
                             DType dtype, const void* data, u64 data_size,
                             u32 quant_scheme_id) {
    TensorEntry entry;
    entry.name = name;
    entry.shape = shape;
    entry.dtype = dtype;
    entry.quant_scheme_id = quant_scheme_id;
    entry.data.resize(data_size);
    memcpy(entry.data.data(), data, data_size);
    tensors_.push_back(std::move(entry));
}

void QrafWriter::set_vocab(const std::vector<std::string>& tokens,
                            const std::vector<f32>& scores) {
    vocab_ = tokens;
    vocab_scores_ = scores;
    if (vocab_scores_.size() < vocab_.size()) {
        vocab_scores_.resize(vocab_.size(), 0.0f);
    }
    has_tokenizer_ = true;
}

void QrafWriter::set_merges(const std::vector<std::tuple<u32, u32, u32, f32>>& merges) {
    merges_ = merges;
}

void QrafWriter::set_special_tokens(u32 bos, u32 eos, u32 pad, u32 unk) {
    bos_id_ = bos;
    eos_id_ = eos;
    pad_id_ = pad;
    unk_id_ = unk;
}

bool QrafWriter::write(const std::string& path) {
    log::info("Writing QRAF file: %s", path.c_str());

    // Pre-populate string table with all names
    for (auto& entry : config_entries_) {
        add_string(entry.key);
    }
    for (auto& t : tensors_) {
        add_string(t.name);
    }
    if (has_tokenizer_) {
        for (auto& tok : vocab_) {
            add_string(tok);
        }
    }

    // ─── Build sections ───

    // Config block
    std::vector<u8> config_block;
    {
        u32 num = static_cast<u32>(config_entries_.size());
        config_block.resize(sizeof(u32) + num * sizeof(ConfigEntry));
        memcpy(config_block.data(), &num, sizeof(u32));

        ConfigEntry* entries = reinterpret_cast<ConfigEntry*>(config_block.data() + sizeof(u32));
        for (u32 i = 0; i < num; i++) {
            entries[i].key_offset = add_string(config_entries_[i].key);
            entries[i].value_type = config_entries_[i].value_type;
            entries[i].value = config_entries_[i].value;
        }
    }

    // Tokenizer block
    std::vector<u8> tokenizer_block;
    if (has_tokenizer_) {
        u32 vocab_size = static_cast<u32>(vocab_.size());
        u32 merges_count = static_cast<u32>(merges_.size());
        u32 special_count = 4; // bos, eos, pad, unk
        u32 reserved = 0;

        size_t tok_size = 4 * sizeof(u32)
                        + vocab_size * sizeof(TokenEntry)
                        + merges_count * sizeof(MergeEntry)
                        + special_count * sizeof(SpecialToken);
        tokenizer_block.resize(tok_size);
        u8* ptr = tokenizer_block.data();

        memcpy(ptr, &vocab_size, sizeof(u32)); ptr += sizeof(u32);
        memcpy(ptr, &merges_count, sizeof(u32)); ptr += sizeof(u32);
        memcpy(ptr, &special_count, sizeof(u32)); ptr += sizeof(u32);
        memcpy(ptr, &reserved, sizeof(u32)); ptr += sizeof(u32);

        // Token entries
        TokenEntry* tok_entries = reinterpret_cast<TokenEntry*>(ptr);
        for (u32 i = 0; i < vocab_size; i++) {
            tok_entries[i].string_offset = add_string(vocab_[i]);
            tok_entries[i].string_length = static_cast<u32>(vocab_[i].size());
            tok_entries[i].score = vocab_scores_[i];
            tok_entries[i].type = 0;
        }
        ptr += vocab_size * sizeof(TokenEntry);

        // Merge entries
        MergeEntry* merge_entries = reinterpret_cast<MergeEntry*>(ptr);
        for (u32 i = 0; i < merges_count; i++) {
            merge_entries[i].token_a = std::get<0>(merges_[i]);
            merge_entries[i].token_b = std::get<1>(merges_[i]);
            merge_entries[i].result  = std::get<2>(merges_[i]);
            merge_entries[i].priority = std::get<3>(merges_[i]);
        }
        ptr += merges_count * sizeof(MergeEntry);

        // Special tokens
        SpecialToken* specials = reinterpret_cast<SpecialToken*>(ptr);
        auto make_special = [&](u32 idx, u32 id, u32 type) {
            specials[idx].string_offset = 0;
            specials[idx].string_length = 0;
            specials[idx].token_id = id;
            specials[idx].type = type;
        };
        make_special(0, bos_id_, 0);
        make_special(1, eos_id_, 1);
        make_special(2, pad_id_, 2);
        make_special(3, unk_id_, 3);
    }

    // Tensor directory
    u32 num_tensors = static_cast<u32>(tensors_.size());
    std::vector<TensorMeta> tensor_metas(num_tensors);

    // Quant directory
    u32 num_quant = static_cast<u32>(quant_schemes_.size());

    // ─── Compute layout ───
    u64 offset = sizeof(QrafHeader); // 128 bytes

    u64 config_offset = offset;
    u64 config_size = config_block.size();
    offset += config_size;
    offset = align_offset(offset, 8);

    u64 tokenizer_offset = offset;
    u64 tokenizer_size_val = tokenizer_block.size();
    offset += tokenizer_size_val;
    offset = align_offset(offset, 8);

    u64 tensor_dir_offset = offset;
    u64 tensor_dir_size = static_cast<u64>(num_tensors) * sizeof(TensorMeta);
    offset += tensor_dir_size;
    offset = align_offset(offset, 8);

    u64 quant_dir_offset = offset;
    u64 quant_dir_size = static_cast<u64>(num_quant) * sizeof(QuantScheme);
    offset += quant_dir_size;
    offset = align_offset(offset, 8);

    u64 string_table_offset = offset;
    u64 string_table_size_val = string_table_.size();
    offset += string_table_size_val;
    offset = align_offset(offset, DATA_ALIGNMENT);

    u64 data_start = offset;

    // Compute tensor data offsets
    for (u32 i = 0; i < num_tensors; i++) {
        auto& t = tensors_[i];
        TensorMeta& meta = tensor_metas[i];

        meta.name_hash = fnv1a_hash(t.name);
        meta.name_offset = add_string(t.name);
        meta.ndim = static_cast<u32>(t.shape.size());
        memset(meta.shape, 0, sizeof(meta.shape));
        for (u32 d = 0; d < meta.ndim && d < 8; d++) {
            meta.shape[d] = t.shape[d];
        }
        meta.dtype = static_cast<u32>(t.dtype);
        meta.quant_scheme_id = t.quant_scheme_id;
        meta.data_offset = offset;
        meta.data_size = t.data.size();
        meta.layout_id = 0;
        meta.padding = 0;

        offset += meta.data_size;
        offset = align_offset(offset, DATA_ALIGNMENT);
    }

    u64 file_size = offset;

    // ─── Build header ───
    QrafHeader header{};
    header.magic = QRAF_MAGIC;
    header.version = QRAF_VERSION;
    header.file_size = file_size;
    header.config_offset = config_offset;
    header.config_size = config_size;
    header.tokenizer_offset = tokenizer_offset;
    header.tokenizer_size = tokenizer_size_val;
    header.tensor_dir_offset = tensor_dir_offset;
    header.tensor_dir_size = tensor_dir_size;
    header.quant_dir_offset = quant_dir_offset;
    header.quant_dir_size = quant_dir_size;
    header.string_table_offset = string_table_offset;
    header.string_table_size = string_table_size_val;
    header.data_offset = data_start;
    header.num_tensors = num_tensors;
    header.num_quant_schemes = num_quant;
    header.flags = 0;
    header.padding = 0;

    // ─── Write file ───
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        log::error("Failed to create file: %s", path.c_str());
        return false;
    }

    auto write_at = [&](u64 off, const void* data, size_t size) {
        out.seekp(static_cast<std::streamoff>(off));
        out.write(static_cast<const char*>(data), size);
    };

    auto write_padding = [&](u64 current, u64 target) {
        if (target > current) {
            std::vector<u8> zeros(target - current, 0);
            out.write(reinterpret_cast<const char*>(zeros.data()), zeros.size());
        }
    };

    // Header
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Config block
    write_at(config_offset, config_block.data(), config_block.size());

    // Tokenizer block
    if (!tokenizer_block.empty()) {
        write_at(tokenizer_offset, tokenizer_block.data(), tokenizer_block.size());
    }

    // Tensor directory
    write_at(tensor_dir_offset, tensor_metas.data(), tensor_dir_size);

    // Quant directory
    if (!quant_schemes_.empty()) {
        write_at(quant_dir_offset, quant_schemes_.data(), quant_dir_size);
    }

    // String table
    write_at(string_table_offset, string_table_.data(), string_table_.size());

    // Tensor data
    for (u32 i = 0; i < num_tensors; i++) {
        write_at(tensor_metas[i].data_offset, tensors_[i].data.data(), tensors_[i].data.size());
    }

    // Pad to file_size
    out.seekp(0, std::ios::end);
    u64 current = static_cast<u64>(out.tellp());
    write_padding(current, file_size);

    out.close();

    log::info("Written %llu bytes to %s", (unsigned long long)file_size, path.c_str());
    return true;
}

} // namespace qraf
