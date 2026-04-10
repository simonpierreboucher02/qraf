#include "cli/commands.h"
#include "runtime/model_manager.h"
#include "core/logging.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdlib>

namespace qraf {
namespace cli {

// ANSI escapes (prefixed to avoid ODR clashes with chat.cpp)
static const char* cR  = "\033[0m";
static const char* cB  = "\033[1m";
static const char* cD  = "\033[2m";
static const char* cRed = "\033[31m";
static const char* cGrn = "\033[32m";
static const char* cYlw = "\033[33m";
static const char* cCyn = "\033[36m";

struct HubModel {
    const char* id;
    const char* name;
    const char* desc;
    const char* arch;
    float params;
    float size_gb;
    const char* cat;
    bool star;
};

static const HubModel CATALOG[] = {
    // Chat
    {"Qwen/Qwen2.5-1.5B-Instruct",       "Qwen 2.5 1.5B Instruct",    "Best small chat model, multilingual",    "qwen2",    1.5f, 7.6f, "chat", true},
    {"HuggingFaceTB/SmolLM2-1.7B-Instruct","SmolLM2 1.7B Instruct",   "Excellent quality for size",             "llama",    1.7f, 8.1f, "chat", true},
    {"Qwen/Qwen2.5-0.5B-Instruct",        "Qwen 2.5 0.5B Instruct",   "Fast lightweight chat",                  "qwen2",    0.5f, 2.4f, "chat", true},
    {"HuggingFaceTB/SmolLM2-360M-Instruct","SmolLM2 360M Instruct",   "Ultra-fast, lower quality",              "llama",    0.4f, 1.5f, "chat", false},
    {"TinyLlama/TinyLlama-1.1B-Chat-v1.0","TinyLlama 1.1B Chat",     "Classic Llama chat",                     "llama",    1.1f, 4.4f, "chat", false},
    {"HuggingFaceTB/SmolLM2-135M-Instruct","SmolLM2 135M Instruct",   "Tiny and fast",                          "llama",    0.1f, 0.6f, "chat", false},
    // Code
    {"Qwen/Qwen2.5-Coder-1.5B-Instruct",  "Qwen Coder 1.5B",         "Best small code model",                  "qwen2",    1.5f, 7.6f, "code", true},
    {"Qwen/Qwen2.5-Coder-0.5B-Instruct",  "Qwen Coder 0.5B",         "Fast code completion",                   "qwen2",    0.5f, 2.4f, "code", true},
    {"bigcode/tiny_starcoder_py",           "Tiny StarCoder Py",        "Python code generation",                 "starcoder",0.2f, 0.8f, "code", false},
    {"Salesforce/codegen-350M-mono",        "CodeGen 350M",             "Code generation",                        "codegen",  0.4f, 1.3f, "code", false},
    // Base
    {"Qwen/Qwen2.5-1.5B",                 "Qwen 2.5 1.5B Base",       "Strong base for fine-tuning",            "qwen2",    1.5f, 7.6f, "base", false},
    {"HuggingFaceTB/SmolLM2-1.7B",        "SmolLM2 1.7B Base",        "1.7B base model",                        "llama",    1.7f, 8.1f, "base", false},
    {"gpt2",                                "GPT-2 124M",               "OpenAI classic",                         "gpt2",     0.1f, 0.6f, "base", false},
    {"gpt2-medium",                         "GPT-2 Medium",             "355M params",                            "gpt2",     0.4f, 1.5f, "base", false},
    {"gpt2-large",                          "GPT-2 Large",              "774M params",                            "gpt2",     0.8f, 3.1f, "base", false},
    {"EleutherAI/pythia-410m",              "Pythia 410M",              "Research model",                         "gpt_neox", 0.4f, 1.5f, "base", false},
    {"EleutherAI/pythia-160m",              "Pythia 160M",              "Tiny research model",                    "gpt_neox", 0.2f, 0.6f, "base", false},
    {"EleutherAI/pythia-70m",               "Pythia 70M",               "Ultra-tiny, 500+ tok/s",                 "gpt_neox", 0.1f, 0.3f, "base", false},
    {"facebook/opt-125m",                   "OPT 125M",                 "Meta open model",                        "opt",      0.1f, 0.6f, "base", false},
    {"distilgpt2",                          "DistilGPT2",               "Distilled, ultra-fast",                  "gpt2",     0.1f, 0.5f, "base", false},
    // Math
    {"Qwen/Qwen2.5-Math-1.5B-Instruct",   "Qwen Math 1.5B",          "Math reasoning",                         "qwen2",    1.5f, 7.6f, "chat", false},
};
static const int CATALOG_SIZE = sizeof(CATALOG) / sizeof(CATALOG[0]);

static std::string make_filename(const char* id) {
    std::string n(id);
    auto p = n.find_last_of('/');
    if (p != std::string::npos) n = n.substr(p + 1);
    std::transform(n.begin(), n.end(), n.begin(), ::tolower);
    for (auto& c : n) if (c == '.') c = '-';
    return n;
}

static bool downloaded(const std::string& dir, const char* id) {
    return std::filesystem::exists(dir + "/" + make_filename(id) + ".qraf");
}

int cmd_browse(const std::string& models_dir) {
    std::string filter = "all";

    while (true) {
        // Header
        std::cout << "\033[2J\033[H\n";
        std::cout << cB << cCyn;
        std::cout << "  ╔═══════════════════════════════════════════════════════════╗\n";
        std::cout << "  ║              QRAF Model Store                            ║\n";
        std::cout << "  ╚═══════════════════════════════════════════════════════════╝\n";
        std::cout << cR << "\n";

        // Tabs
        std::cout << "  ";
        for (auto& f : std::vector<std::pair<std::string,std::string>>{{"All","all"},{"Chat","chat"},{"Code","code"},{"Base","base"}}) {
            if (filter == f.second)
                std::cout << cB << "[" << f.first << "]" << cR << "  ";
            else
                std::cout << cD << " " << f.first << " " << cR << "  ";
        }
        std::cout << "\n\n";

        // Table
        std::cout << cB;
        printf("  %3s  %-30s %6s %6s  %-6s %s\n", "#", "Model", "Params", "QRAF", "Type", "Status");
        std::cout << cD << "  " << std::string(75, '-') << cR << "\n";

        int idx = 0;
        for (int i = 0; i < CATALOG_SIZE; i++) {
            auto& m = CATALOG[i];
            if (filter != "all" && std::string(m.cat) != filter) continue;
            idx++;
            bool dl = downloaded(models_dir, m.id);
            const char* star = m.star ? "*" : " ";
            const char* nc = m.star ? cYlw : cR;
            const char* sc = dl ? cGrn : cD;

            printf("  %s%2d%s  %s%s%-29s%s %4.1fB %5.1fGB  %-6s %s%s%s\n",
                   cB, idx, cR, star, nc, m.name, cR,
                   m.params, m.size_gb, m.cat,
                   sc, dl ? "ready" : "     ", cR);
            if (m.star) printf("       %s%s%s\n", cD, m.desc, cR);
        }

        std::cout << "\n  " << cD << "[number] Download  [a]ll [c]hat [o]de [b]ase  [/query] Search HF  [q]uit" << cR << "\n";
        std::cout << "  " << cCyn << ">" << cR << " ";

        std::string input;
        if (!std::getline(std::cin, input) || input.empty()) continue;

        if (input == "q" || input == "quit") break;
        if (input == "a") { filter = "all"; continue; }
        if (input == "c") { filter = "chat"; continue; }
        if (input == "o") { filter = "code"; continue; }
        if (input == "b") { filter = "base"; continue; }

        // Search HuggingFace
        if (input[0] == '/') {
            std::string q = input.substr(1);
            std::cout << "\n  " << cD << "Searching HuggingFace: " << q << cR << "\n";
            std::string cmd = "curl -s 'https://huggingface.co/api/models?search=" + q +
                              "&filter=text-generation&sort=downloads&limit=8' 2>/dev/null";
            FILE* p = popen(cmd.c_str(), "r");
            if (p) {
                char buf[8192]; std::string res;
                while (fgets(buf, sizeof(buf), p)) res += buf;
                pclose(p);
                std::cout << "\n  Results (use 'qraf convert <name>' to download):\n";
                size_t pos = 0; int c = 0;
                while ((pos = res.find("\"id\":\"", pos)) != std::string::npos && c < 8) {
                    pos += 6;
                    size_t end = res.find("\"", pos);
                    if (end != std::string::npos) {
                        std::cout << "    " << cB << res.substr(pos, end - pos) << cR << "\n";
                        c++;
                    }
                }
                if (c == 0) std::cout << "    No results.\n";
            }
            std::cout << "\n  Press Enter...";
            std::getline(std::cin, input);
            continue;
        }

        // Number selection
        try {
            int choice = std::stoi(input);
            int idx2 = 0;
            for (int i = 0; i < CATALOG_SIZE; i++) {
                auto& m = CATALOG[i];
                if (filter != "all" && std::string(m.cat) != filter) continue;
                idx2++;
                if (idx2 != choice) continue;

                std::string fname = make_filename(m.id);
                std::string outpath = models_dir + "/" + fname + ".qraf";

                if (downloaded(models_dir, m.id)) {
                    std::cout << "\n  " << cGrn << "Already downloaded!" << cR << " Chat? [Y/n] ";
                    std::string ans; std::getline(std::cin, ans);
                    if (ans.empty() || ans[0] == 'Y' || ans[0] == 'y')
                        return cmd_chat(outpath);
                } else {
                    std::cout << "\n  " << cB << "Download " << m.name << cR;
                    std::cout << " (" << std::fixed << std::setprecision(1) << m.size_gb << " GB)? [Y/n] ";
                    std::string ans; std::getline(std::cin, ans);
                    if (!ans.empty() && ans[0] != 'Y' && ans[0] != 'y') break;

                    std::cout << "\n  " << cCyn << "Converting..." << cR << "\n\n";
                    std::filesystem::create_directories(models_dir);

                    std::string script;
                    for (auto& sp : {"tools/qraf_convert.py", "../tools/qraf_convert.py", "../../tools/qraf_convert.py"}) {
                        if (std::ifstream(sp).good()) { script = sp; break; }
                    }
                    if (script.empty()) { std::cerr << cRed << "  qraf_convert.py not found" << cR << "\n"; break; }

                    std::string cmd2 = "python3 " + script + " \"" + std::string(m.id) + "\" -o \"" + outpath + "\"";
                    int ret = system(cmd2.c_str());

                    if (ret == 0) {
                        std::cout << "\n  " << cGrn << "Done!" << cR << " Chat now? [Y/n] ";
                        std::getline(std::cin, ans);
                        if (ans.empty() || ans[0] == 'Y' || ans[0] == 'y')
                            return cmd_chat(outpath);
                    } else {
                        std::cout << "\n  " << cRed << "Failed." << cR << "\n";
                        std::cout << "  Press Enter..."; std::getline(std::cin, ans);
                    }
                }
                break;
            }
        } catch (...) {}
    }
    return 0;
}

} // namespace cli
} // namespace qraf
