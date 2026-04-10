#include "cli/commands.h"
#include "runtime/inference.h"
#include "runtime/model_manager.h"
#include "qraf/loader.h"
#include "core/logging.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <filesystem>

namespace qraf {
namespace cli {

// ─── run: one-shot generation ───
int cmd_run(const std::string& model_path, const std::string& prompt, int max_tokens) {
    InferenceEngine engine;
    if (!engine.load_model(model_path)) {
        std::cerr << "Error: Failed to load model: " << model_path << "\n";
        return 1;
    }

    GenerateConfig config;
    config.max_tokens = static_cast<u32>(max_tokens);

    auto result = engine.generate(prompt, config, [](u32 /*token_id*/, const std::string& text) {
        std::cout << text << std::flush;
        return true;
    });

    std::cout << "\n";
    std::cerr << "\n--- " << result.generated_tokens << " tokens in "
              << std::fixed << std::setprecision(1)
              << result.generation_time_ms << " ms ("
              << result.tokens_per_sec << " tok/s) ---\n";

    return 0;
}

// ─── chat: interactive conversation ───
int cmd_chat(const std::string& model_path) {
    InferenceEngine engine;
    if (!engine.load_model(model_path)) {
        std::cerr << "Error: Failed to load model: " << model_path << "\n";
        return 1;
    }

    std::cout << "QRAF Chat — Model loaded. Type /quit to exit.\n\n";

    GenerateConfig config;
    config.max_tokens = 512;

    while (true) {
        std::cout << ">>> ";
        std::string input;
        if (!std::getline(std::cin, input)) break;

        if (input.empty()) continue;
        if (input == "/quit" || input == "/exit") break;
        if (input == "/reset") {
            engine.reset();
            std::cout << "[context reset]\n";
            continue;
        }

        auto result = engine.generate(input, config, [](u32 /*token_id*/, const std::string& text) {
            std::cout << text << std::flush;
            return true;
        });

        std::cout << "\n\n";
    }

    return 0;
}

// ─── list: show available models ───
int cmd_list(const std::string& models_dir) {
    ModelManager manager(models_dir);
    manager.scan();

    auto models = manager.list();
    if (models.empty()) {
        std::cout << "No models found in " << models_dir << "\n";
        return 0;
    }

    std::cout << "Available models:\n";
    std::cout << std::left
              << std::setw(30) << "NAME"
              << std::setw(15) << "SIZE"
              << std::setw(10) << "STATUS"
              << "\n";
    std::cout << std::string(55, '-') << "\n";

    for (const auto& info : models) {
        std::string size_str;
        if (info.file_size >= 1024ULL * 1024 * 1024) {
            size_str = std::to_string(info.file_size / (1024ULL * 1024 * 1024)) + " GB";
        } else if (info.file_size >= 1024ULL * 1024) {
            size_str = std::to_string(info.file_size / (1024ULL * 1024)) + " MB";
        } else {
            size_str = std::to_string(info.file_size / 1024ULL) + " KB";
        }

        std::cout << std::left
                  << std::setw(30) << info.name
                  << std::setw(15) << size_str
                  << std::setw(10) << (info.loaded ? "loaded" : "")
                  << "\n";
    }

    return 0;
}

// ─── inspect: show model details ───
int cmd_inspect(const std::string& model_path) {
    QrafModel model;
    if (!model.load(model_path)) {
        std::cerr << "Error: Failed to load model: " << model_path << "\n";
        return 1;
    }

    const auto& hdr = model.header();
    const auto& cfg = model.config();

    std::cout << "=== QRAF Model Inspection ===\n\n";
    std::cout << "File:           " << model_path << "\n";
    std::cout << "File size:      " << (model.file_size() / (1024ULL * 1024)) << " MB\n";
    std::cout << "Format version: " << hdr.version << "\n";
    std::cout << "Num tensors:    " << hdr.num_tensors << "\n";
    std::cout << "Quant schemes:  " << hdr.num_quant_schemes << "\n";
    std::cout << "\n";

    std::cout << "--- Model Config ---\n";
    std::cout << "Architecture:     " << cfg.architecture << "\n";
    std::cout << "Vocab size:       " << cfg.vocab_size << "\n";
    std::cout << "Hidden size:      " << cfg.hidden_size << "\n";
    std::cout << "Num layers:       " << cfg.num_layers << "\n";
    std::cout << "Num heads:        " << cfg.num_heads << "\n";
    std::cout << "Num KV heads:     " << cfg.num_kv_heads << "\n";
    std::cout << "Intermediate:     " << cfg.intermediate_size << "\n";
    std::cout << "Max seq length:   " << cfg.max_seq_len << "\n";
    std::cout << "RoPE theta:       " << cfg.rope_theta << "\n";
    std::cout << "RMS norm eps:     " << cfg.rms_norm_eps << "\n";
    std::cout << "\n";

    std::cout << "--- Tensors ---\n";
    auto names = model.tensor_names();
    std::sort(names.begin(), names.end());

    for (const auto& name : names) {
        auto tv = model.get_tensor(name);
        std::cout << "  " << std::left << std::setw(50) << name;
        std::cout << " [";
        for (size_t i = 0; i < tv.shape.size(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << tv.shape[i];
        }
        std::cout << "]";
        std::cout << " " << dtype_name(tv.dtype);
        if (tv.quant) std::cout << " (quantized)";
        std::cout << " " << (tv.data_size / 1024) << " KB";
        std::cout << "\n";
    }

    return 0;
}

// ─── benchmark: measure performance ───
int cmd_benchmark(const std::string& model_path, int num_tokens) {
    InferenceEngine engine;
    if (!engine.load_model(model_path)) {
        std::cerr << "Error: Failed to load model: " << model_path << "\n";
        return 1;
    }

    std::cout << "Benchmarking: " << model_path << "\n";
    std::cout << "Generating " << num_tokens << " tokens...\n\n";

    GenerateConfig config;
    config.max_tokens = static_cast<u32>(num_tokens);
    config.sampling.temperature = 0.0f;  // greedy for reproducibility
    config.sampling.deterministic = true;

    // Warm-up prompt
    std::string prompt = "The quick brown fox";

    auto result = engine.generate(prompt, config);

    std::cout << "Results:\n";
    std::cout << "  Prompt tokens:    " << result.prompt_tokens << "\n";
    std::cout << "  Generated tokens: " << result.generated_tokens << "\n";
    std::cout << "  Total time:       " << std::fixed << std::setprecision(1) << result.generation_time_ms << " ms\n";
    std::cout << "  Tokens/sec:       " << std::setprecision(2) << result.tokens_per_sec << "\n";
    std::cout << "  Latency/token:    " << std::setprecision(2);
    if (result.generated_tokens > 0) {
        std::cout << (result.generation_time_ms / result.generated_tokens) << " ms\n";
    } else {
        std::cout << "N/A\n";
    }

    return 0;
}

// ─── convert: convert models to QRAF ───
int cmd_convert(const std::string& source, const std::string& output, const std::string& format) {
    // Find the Python converter script
    // Try relative paths from executable location
    std::vector<std::string> search_paths = {
        "tools/qraf_convert.py",
        "../tools/qraf_convert.py",
        "../../tools/qraf_convert.py",
    };

    std::string script_path;
    for (const auto& p : search_paths) {
        if (std::ifstream(p).good()) {
            script_path = p;
            break;
        }
    }

    if (script_path.empty()) {
        // Try from executable directory
        std::cerr << "Error: Cannot find qraf_convert.py\n";
        std::cerr << "Make sure tools/qraf_convert.py exists relative to your working directory\n";
        return 1;
    }

    // Build command
    std::string cmd = "python3 " + script_path + " \"" + source + "\" -o \"" + output + "\"";
    if (!format.empty() && format != "auto") {
        cmd += " --format " + format;
    }

    std::cout << "Converting: " << source << " -> " << output << "\n";
    if (!format.empty()) std::cout << "Format: " << format << "\n";
    std::cout << "\n";

    int ret = system(cmd.c_str());
    return WEXITSTATUS(ret);
}

int cmd_version() {
    std::cout << "qraf-runtime v0.1.0\n";
    std::cout << "QRAF format version: " << QRAF_VERSION << "\n";
    return 0;
}

int cmd_help() {
    std::cout << R"(
QRAF Runtime — Local LLM Inference Engine

Usage: qraf <command> [options]

Commands:
  run <model> [--prompt <text>] [--max-tokens <n>]
      Generate text from a prompt

  chat
      Interactive chatbot with model browser and multi-turn conversation
      Use /help inside chat for commands (/model, /system, /reset, /stats...)

  chat <model>
      Quick chat mode with a specific model file

  list [--dir <models_dir>]
      List available models

  inspect <model>
      Show model details (architecture, tensors, etc.)

  benchmark <model> [--tokens <n>]
      Measure inference performance

  convert <source> -o <output.qraf> [--format <fmt>]
      Convert model to QRAF format
      Sources: HuggingFace name, local dir, .gguf, .safetensors, .bin
      Formats: auto, huggingface, safetensors, pytorch, gguf

  version
      Show version information

  help
      Show this help message

Examples:
  qraf chat                       Interactive chatbot (model browser)
  qraf chat --dir ./models        Specify models directory
  qraf run my-model.qraf --prompt "Once upon a time"
  qraf inspect my-model.qraf
  qraf benchmark my-model.qraf --tokens 100

  qraf convert Qwen/Qwen2.5-0.5B -o models/qwen.qraf
  qraf convert ./local-model/ -o models/local.qraf
  qraf convert model.gguf -o models/from-gguf.qraf
  qraf convert weights.safetensors -o models/out.qraf
)";
    return 0;
}

} // namespace cli
} // namespace qraf
