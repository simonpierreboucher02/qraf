#pragma once

#include "qraf/format.h"
#include "core/types.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace qraf {

// ─── QRAF File Writer ───
// Builds a .qraf file from in-memory tensors, config, and tokenizer data.

class QrafWriter {
public:
    QrafWriter() = default;

    // Set model config
    void set_config(const std::string& key, u32 value);
    void set_config(const std::string& key, f32 value);
    void set_config_string(const std::string& key, const std::string& value);

    // Add a quantization scheme
    u32 add_quant_scheme(DType type, u32 block_size, u32 group_size = 1);

    // Add a tensor
    void add_tensor(const std::string& name, const std::vector<u32>& shape,
                    DType dtype, const void* data, u64 data_size,
                    u32 quant_scheme_id = 0xFFFFFFFF);

    // Set tokenizer data
    void set_vocab(const std::vector<std::string>& tokens,
                   const std::vector<f32>& scores = {});
    void set_merges(const std::vector<std::tuple<u32, u32, u32, f32>>& merges);
    void set_special_tokens(u32 bos, u32 eos, u32 pad = 0, u32 unk = 0);

    // Write the complete file
    bool write(const std::string& path);

private:
    u32 add_string(const std::string& str);

    struct TensorEntry {
        std::string name;
        std::vector<u32> shape;
        DType dtype;
        u32 quant_scheme_id;
        std::vector<u8> data;
    };

    struct ConfigKV {
        std::string key;
        u32 value_type;  // 0=u32, 1=f32, 2=string
        u32 value;
    };

    std::vector<ConfigKV> config_entries_;
    std::vector<QuantScheme> quant_schemes_;
    std::vector<TensorEntry> tensors_;

    // String table
    std::vector<u8> string_table_;
    std::unordered_map<std::string, u32> string_offsets_;

    // Tokenizer
    std::vector<std::string> vocab_;
    std::vector<f32> vocab_scores_;
    std::vector<std::tuple<u32, u32, u32, f32>> merges_;
    u32 bos_id_ = 1, eos_id_ = 2, pad_id_ = 0, unk_id_ = 0;
    bool has_tokenizer_ = false;
};

} // namespace qraf
