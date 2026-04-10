// Tool: Generate a small dummy QRAF model for testing
// Creates a tiny transformer model with random weights

#include "qraf/writer.h"
#include "core/logging.h"
#include <random>
#include <iostream>
#include <string>

using namespace qraf;

static std::vector<f32> random_tensor(size_t size, std::mt19937& rng, f32 scale = 0.02f) {
    std::normal_distribution<f32> dist(0.0f, scale);
    std::vector<f32> data(size);
    for (auto& v : data) v = dist(rng);
    return data;
}

static std::vector<f32> ones_tensor(size_t size) {
    return std::vector<f32>(size, 1.0f);
}

int main(int argc, char* argv[]) {
    std::string output = "models/dummy-tiny.qraf";
    if (argc > 1) output = argv[1];

    // Tiny model config
    u32 vocab_size = 256;       // byte-level vocab
    u32 hidden_size = 64;
    u32 num_layers = 2;
    u32 num_heads = 4;
    u32 num_kv_heads = 4;
    u32 head_dim = hidden_size / num_heads;  // 16
    u32 intermediate_size = 128;
    u32 max_seq_len = 512;

    std::mt19937 rng(42);

    QrafWriter writer;

    // ─── Config ───
    writer.set_config_string("architecture", "llama");
    writer.set_config("vocab_size", vocab_size);
    writer.set_config("hidden_size", hidden_size);
    writer.set_config("num_layers", num_layers);
    writer.set_config("num_heads", num_heads);
    writer.set_config("num_kv_heads", num_kv_heads);
    writer.set_config("intermediate_size", intermediate_size);
    writer.set_config("max_seq_len", max_seq_len);
    writer.set_config("rope_theta", 10000.0f);
    writer.set_config("rms_norm_eps", 1e-5f);

    // ─── Tokenizer (byte-level) ───
    std::vector<std::string> tokens;
    tokens.push_back("<pad>");   // 0
    tokens.push_back("<bos>");   // 1
    tokens.push_back("<eos>");   // 2
    tokens.push_back("<unk>");   // 3
    for (int i = 4; i < 256; i++) {
        if (i >= 32 && i < 127) {
            tokens.push_back(std::string(1, static_cast<char>(i)));
        } else {
            tokens.push_back("<0x" + std::to_string(i) + ">");
        }
    }
    writer.set_vocab(tokens);
    writer.set_special_tokens(1, 2, 0, 3);

    // ─── Tensors ───
    auto add_f32_tensor = [&](const std::string& name, const std::vector<u32>& shape,
                               const std::vector<f32>& data) {
        writer.add_tensor(name, shape, DType::F32, data.data(), data.size() * sizeof(f32));
    };

    // Embedding
    auto embed = random_tensor(vocab_size * hidden_size, rng);
    add_f32_tensor("model.embed_tokens.weight", {vocab_size, hidden_size}, embed);

    // Layers
    for (u32 l = 0; l < num_layers; l++) {
        std::string prefix = "model.layers." + std::to_string(l) + ".";

        // Attention
        u32 q_dim = num_heads * head_dim;
        u32 kv_dim = num_kv_heads * head_dim;

        add_f32_tensor(prefix + "self_attn.q_proj.weight",
                       {q_dim, hidden_size},
                       random_tensor(q_dim * hidden_size, rng));

        add_f32_tensor(prefix + "self_attn.k_proj.weight",
                       {kv_dim, hidden_size},
                       random_tensor(kv_dim * hidden_size, rng));

        add_f32_tensor(prefix + "self_attn.v_proj.weight",
                       {kv_dim, hidden_size},
                       random_tensor(kv_dim * hidden_size, rng));

        add_f32_tensor(prefix + "self_attn.o_proj.weight",
                       {hidden_size, q_dim},
                       random_tensor(hidden_size * q_dim, rng));

        // MLP
        add_f32_tensor(prefix + "mlp.gate_proj.weight",
                       {intermediate_size, hidden_size},
                       random_tensor(intermediate_size * hidden_size, rng));

        add_f32_tensor(prefix + "mlp.up_proj.weight",
                       {intermediate_size, hidden_size},
                       random_tensor(intermediate_size * hidden_size, rng));

        add_f32_tensor(prefix + "mlp.down_proj.weight",
                       {hidden_size, intermediate_size},
                       random_tensor(hidden_size * intermediate_size, rng));

        // Norms (initialized to 1.0)
        add_f32_tensor(prefix + "input_layernorm.weight",
                       {hidden_size}, ones_tensor(hidden_size));

        add_f32_tensor(prefix + "post_attention_layernorm.weight",
                       {hidden_size}, ones_tensor(hidden_size));
    }

    // Output norm
    add_f32_tensor("model.norm.weight", {hidden_size}, ones_tensor(hidden_size));

    // LM head
    add_f32_tensor("lm_head.weight", {vocab_size, hidden_size},
                   random_tensor(vocab_size * hidden_size, rng));

    // ─── Write ───
    if (writer.write(output)) {
        std::cout << "Dummy model written to: " << output << "\n";
        std::cout << "Config: " << num_layers << " layers, "
                  << hidden_size << " hidden, "
                  << num_heads << " heads, "
                  << vocab_size << " vocab\n";
        return 0;
    }

    std::cerr << "Failed to write model\n";
    return 1;
}
