#include "cli/commands.h"
#include "runtime/inference.h"
#include "runtime/model_manager.h"
#include "core/logging.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <cmath>

namespace qraf {
namespace cli {

// ─── ANSI Colors ───
namespace col {
    const char* reset   = "\033[0m";
    const char* bold    = "\033[1m";
    const char* dim     = "\033[2m";
    const char* italic  = "\033[3m";
    const char* uline   = "\033[4m";
    const char* red     = "\033[31m";
    const char* green   = "\033[32m";
    const char* yellow  = "\033[33m";
    const char* blue    = "\033[34m";
    const char* magenta = "\033[35m";
    const char* cyan    = "\033[36m";
    const char* white   = "\033[37m";
    const char* bg_dark = "\033[48;5;236m";
    const char* gray    = "\033[38;5;245m";
}

// ─── Chat Message ───
struct ChatMessage {
    std::string role;    // "system", "user", "assistant"
    std::string content;
};

// ─── Chat Session State ───
struct ChatSession {
    std::string model_name;
    std::string model_path;
    std::string system_prompt;
    std::vector<ChatMessage> history;
    InferenceEngine engine;
    bool loaded = false;

    // Stats
    u32 total_tokens_generated = 0;
    u32 total_turns = 0;
    double total_time_ms = 0.0;

    // Config
    int max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
};

// ─── UI Helpers ───

static void clear_screen() {
    std::cout << "\033[2J\033[H";
}

static void draw_separator(const char* color = col::dim, int width = 60) {
    std::cout << color;
    for (int i = 0; i < width; i++) std::cout << "─";
    std::cout << col::reset << "\n";
}

static void draw_box(const std::string& title, const char* color = col::cyan) {
    int width = 60;
    std::cout << color << "╭";
    for (int i = 0; i < width - 2; i++) std::cout << "─";
    std::cout << "╮\n";

    int pad = (width - 2 - static_cast<int>(title.size())) / 2;
    std::cout << "│";
    for (int i = 0; i < pad; i++) std::cout << " ";
    std::cout << col::bold << title << col::reset << color;
    for (int i = 0; i < width - 2 - pad - static_cast<int>(title.size()); i++) std::cout << " ";
    std::cout << "│\n";

    std::cout << "╰";
    for (int i = 0; i < width - 2; i++) std::cout << "─";
    std::cout << "╯" << col::reset << "\n";
}

static void print_welcome() {
    clear_screen();
    std::cout << "\n";
    std::cout << col::bold << col::cyan;
    std::cout << "   ██████╗ ██████╗  █████╗ ███████╗\n";
    std::cout << "  ██╔═══██╗██╔══██╗██╔══██╗██╔════╝\n";
    std::cout << "  ██║   ██║██████╔╝███████║█████╗  \n";
    std::cout << "  ██║▄▄ ██║██╔══██╗██╔══██║██╔══╝  \n";
    std::cout << "  ╚██████╔╝██║  ██║██║  ██║██║     \n";
    std::cout << "   ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ";
    std::cout << col::dim << "v0.1.0" << col::reset << "\n\n";
    std::cout << col::bold << "  QRAF Chat" << col::reset;
    std::cout << col::dim << " — Local LLM Chat Interface" << col::reset << "\n\n";
}

// ─── Model Selection UI ───

static int select_model(const std::string& models_dir, std::string& out_name, std::string& out_path) {
    ModelManager manager(models_dir);
    manager.scan();
    auto models = manager.list();

    if (models.empty()) {
        std::cout << col::red << "  No models found in " << models_dir << col::reset << "\n";
        std::cout << col::dim << "  Convert a model first: qraf convert <model> -o models/<name>.qraf" << col::reset << "\n";
        return -1;
    }

    std::cout << col::bold << "  Select a model:" << col::reset << "\n\n";

    for (size_t i = 0; i < models.size(); i++) {
        std::string size_str;
        auto fs = models[i].file_size;
        if (fs >= 1024ULL * 1024 * 1024) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << (fs / (1024.0 * 1024.0 * 1024.0)) << " GB";
            size_str = oss.str();
        } else {
            size_str = std::to_string(fs / (1024ULL * 1024)) + " MB";
        }

        std::cout << "  " << col::bold << col::green << " [" << (i + 1) << "] " << col::reset;
        std::cout << col::bold << std::left << std::setw(32) << models[i].name << col::reset;
        std::cout << col::dim << size_str << col::reset << "\n";
    }

    std::cout << "\n  " << col::cyan << ">" << col::reset << " ";
    std::string input;
    if (!std::getline(std::cin, input) || input.empty()) return -1;

    int choice;
    try {
        choice = std::stoi(input);
    } catch (...) {
        // Try matching by name
        for (size_t i = 0; i < models.size(); i++) {
            if (models[i].name.find(input) != std::string::npos) {
                out_name = models[i].name;
                out_path = models[i].path;
                return 0;
            }
        }
        return -1;
    }

    if (choice < 1 || choice > static_cast<int>(models.size())) return -1;

    out_name = models[choice - 1].name;
    out_path = models[choice - 1].path;
    return 0;
}

// ─── Build prompt with chat history ───

static std::string build_prompt(const ChatSession& session) {
    std::string prompt;

    // System prompt
    if (!session.system_prompt.empty()) {
        prompt += session.system_prompt + "\n\n";
    }

    // Conversation history (keep last N turns to fit context)
    int start = 0;
    int history_size = static_cast<int>(session.history.size());
    if (history_size > 10) {
        start = history_size - 10; // keep last 5 turns (10 messages)
    }

    for (int i = start; i < history_size; i++) {
        const auto& msg = session.history[i];
        if (msg.role == "user") {
            prompt += "User: " + msg.content + "\n";
        } else if (msg.role == "assistant") {
            prompt += "Assistant: " + msg.content + "\n";
        }
    }

    prompt += "Assistant:";
    return prompt;
}

// ─── Display chat help ───

static void print_chat_help() {
    std::cout << "\n";
    draw_box("Chat Commands");
    std::cout << col::bold << "  /help" << col::reset << "          Show this help\n";
    std::cout << col::bold << "  /quit" << col::reset << "          Exit chat\n";
    std::cout << col::bold << "  /reset" << col::reset << "         Clear conversation history\n";
    std::cout << col::bold << "  /model" << col::reset << "         Switch to a different model\n";
    std::cout << col::bold << "  /system <txt>" << col::reset << "  Set system prompt\n";
    std::cout << col::bold << "  /history" << col::reset << "       Show conversation history\n";
    std::cout << col::bold << "  /stats" << col::reset << "         Show session statistics\n";
    std::cout << col::bold << "  /config" << col::reset << "        Show/change generation settings\n";
    std::cout << col::bold << "  /temp <val>" << col::reset << "    Set temperature (0.0 - 2.0)\n";
    std::cout << col::bold << "  /tokens <n>" << col::reset << "    Set max tokens per response\n";
    std::cout << col::bold << "  /clear" << col::reset << "         Clear screen\n";
    std::cout << "\n";
}

// ─── Load a model into session ───

static bool load_model(ChatSession& session) {
    std::cout << "\n  " << col::dim << "Loading " << session.model_name << "..." << col::reset << std::flush;

    // Suppress log output during load
    auto old_level = log::get_level();
    log::set_level(log::Level::WARN);

    bool ok = session.engine.load_model(session.model_path);

    log::set_level(old_level);

    if (!ok) {
        std::cout << " " << col::red << "FAILED" << col::reset << "\n";
        return false;
    }

    session.loaded = true;
    const auto& cfg = session.engine.model_config();

    std::cout << "\r                                                \r";
    std::cout << col::green << "  ✓ " << col::reset;
    std::cout << col::bold << session.model_name << col::reset;
    std::cout << col::dim << " (" << cfg.num_layers << "L, h=" << cfg.hidden_size
              << ", " << cfg.num_heads << "h)" << col::reset << "\n\n";

    return true;
}

// ─── Main Chat Loop ───

int cmd_chat_interactive(const std::string& models_dir) {
    print_welcome();

    ChatSession session;
    session.system_prompt = "You are a helpful assistant.";

    // ─── Model selection ───
    if (select_model(models_dir, session.model_name, session.model_path) != 0) {
        std::cout << col::red << "\n  No model selected." << col::reset << "\n";
        return 1;
    }

    if (!load_model(session)) {
        return 1;
    }

    draw_separator();
    std::cout << col::dim << "  Type your message or /help for commands" << col::reset << "\n";
    draw_separator();
    std::cout << "\n";

    // ─── Chat loop ───
    while (true) {
        // User input
        std::cout << col::bold << col::blue << "  You" << col::reset << col::dim << " > " << col::reset;
        std::string input;
        if (!std::getline(std::cin, input)) break;

        // Trim
        while (!input.empty() && (input.front() == ' ' || input.front() == '\t')) input.erase(input.begin());
        while (!input.empty() && (input.back() == ' ' || input.back() == '\t')) input.pop_back();

        if (input.empty()) continue;

        // ─── Handle commands ───
        if (input[0] == '/') {
            std::string cmd = input;
            std::string arg;
            auto space_pos = cmd.find(' ');
            if (space_pos != std::string::npos) {
                arg = cmd.substr(space_pos + 1);
                cmd = cmd.substr(0, space_pos);
            }

            if (cmd == "/quit" || cmd == "/exit" || cmd == "/q") {
                std::cout << "\n" << col::dim << "  Goodbye!" << col::reset << "\n\n";
                break;
            }

            if (cmd == "/help" || cmd == "/?") {
                print_chat_help();
                continue;
            }

            if (cmd == "/reset") {
                session.history.clear();
                session.engine.reset();
                std::cout << col::green << "  ✓ Conversation reset" << col::reset << "\n\n";
                continue;
            }

            if (cmd == "/clear") {
                clear_screen();
                draw_box("QRAF Chat — " + session.model_name);
                std::cout << "\n";
                continue;
            }

            if (cmd == "/model") {
                std::cout << "\n";
                std::string new_name, new_path;
                if (select_model(models_dir, new_name, new_path) == 0) {
                    session.engine.unload_model();
                    session.model_name = new_name;
                    session.model_path = new_path;
                    session.loaded = false;
                    session.history.clear();
                    session.total_tokens_generated = 0;
                    session.total_turns = 0;
                    session.total_time_ms = 0.0;

                    if (!load_model(session)) {
                        std::cout << col::red << "  Failed to load model" << col::reset << "\n";
                    } else {
                        draw_separator();
                        std::cout << col::dim << "  Switched to " << session.model_name << col::reset << "\n";
                        draw_separator();
                    }
                }
                std::cout << "\n";
                continue;
            }

            if (cmd == "/system") {
                if (arg.empty()) {
                    std::cout << col::dim << "  Current: " << col::reset << session.system_prompt << "\n\n";
                } else {
                    session.system_prompt = arg;
                    std::cout << col::green << "  ✓ System prompt updated" << col::reset << "\n\n";
                }
                continue;
            }

            if (cmd == "/history") {
                std::cout << "\n";
                if (session.history.empty()) {
                    std::cout << col::dim << "  (empty)" << col::reset << "\n";
                } else {
                    for (size_t i = 0; i < session.history.size(); i++) {
                        const auto& msg = session.history[i];
                        if (msg.role == "user") {
                            std::cout << col::blue << "  [User]" << col::reset << " " << msg.content << "\n";
                        } else {
                            std::cout << col::magenta << "  [Assistant]" << col::reset << " ";
                            // Truncate long messages
                            if (msg.content.size() > 120) {
                                std::cout << msg.content.substr(0, 120) << "..." << "\n";
                            } else {
                                std::cout << msg.content << "\n";
                            }
                        }
                    }
                }
                std::cout << "\n";
                continue;
            }

            if (cmd == "/stats") {
                std::cout << "\n";
                draw_box("Session Stats");
                std::cout << "  Model:           " << col::bold << session.model_name << col::reset << "\n";
                std::cout << "  Turns:           " << session.total_turns << "\n";
                std::cout << "  Tokens generated:" << session.total_tokens_generated << "\n";
                if (session.total_time_ms > 0) {
                    double avg_tps = session.total_tokens_generated / (session.total_time_ms / 1000.0);
                    std::cout << "  Avg speed:       " << std::fixed << std::setprecision(1) << avg_tps << " tok/s\n";
                }
                std::cout << "  History size:    " << session.history.size() << " messages\n";
                std::cout << "  System prompt:   " << (session.system_prompt.empty() ? "(none)" : session.system_prompt.substr(0, 50)) << "\n";
                std::cout << "\n";
                continue;
            }

            if (cmd == "/config") {
                std::cout << "\n";
                draw_box("Generation Config");
                std::cout << "  Max tokens:    " << session.max_tokens << "\n";
                std::cout << "  Temperature:   " << session.temperature << "\n";
                std::cout << "  Top-P:         " << session.top_p << "\n";
                std::cout << "  Top-K:         " << session.top_k << "\n";
                std::cout << "\n";
                continue;
            }

            if (cmd == "/temp") {
                if (!arg.empty()) {
                    try {
                        float val = std::stof(arg);
                        if (val >= 0.0f && val <= 2.0f) {
                            session.temperature = val;
                            std::cout << col::green << "  ✓ Temperature = " << val << col::reset << "\n\n";
                        } else {
                            std::cout << col::red << "  Temperature must be 0.0 - 2.0" << col::reset << "\n\n";
                        }
                    } catch (...) {
                        std::cout << col::red << "  Invalid value" << col::reset << "\n\n";
                    }
                } else {
                    std::cout << col::dim << "  Temperature: " << session.temperature << col::reset << "\n\n";
                }
                continue;
            }

            if (cmd == "/tokens") {
                if (!arg.empty()) {
                    try {
                        int val = std::stoi(arg);
                        if (val >= 1 && val <= 4096) {
                            session.max_tokens = val;
                            std::cout << col::green << "  ✓ Max tokens = " << val << col::reset << "\n\n";
                        }
                    } catch (...) {}
                } else {
                    std::cout << col::dim << "  Max tokens: " << session.max_tokens << col::reset << "\n\n";
                }
                continue;
            }

            std::cout << col::red << "  Unknown command: " << cmd << col::reset << "\n";
            std::cout << col::dim << "  Type /help for available commands" << col::reset << "\n\n";
            continue;
        }

        // ─── Generate response ───
        if (!session.loaded) {
            std::cout << col::red << "  No model loaded. Use /model to select one." << col::reset << "\n\n";
            continue;
        }

        // Add user message to history
        session.history.push_back({"user", input});

        // Build prompt from history
        std::string prompt = build_prompt(session);

        // Configure generation
        GenerateConfig config;
        config.max_tokens = static_cast<u32>(session.max_tokens);
        config.sampling.temperature = session.temperature;
        config.sampling.top_p = session.top_p;
        config.sampling.top_k = session.top_k;

        // Print assistant header
        std::cout << "\n" << col::bold << col::magenta << "  Assistant" << col::reset << col::dim << " > " << col::reset;

        // Stream generation
        std::string full_response;
        auto start = std::chrono::high_resolution_clock::now();

        auto result = session.engine.generate(prompt, config,
            [&full_response](u32 /*token_id*/, const std::string& text) {
                std::cout << text << std::flush;
                full_response += text;
                return true;
            });

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "\n";

        // Stats line
        double tps = (elapsed > 0) ? (result.generated_tokens / (elapsed / 1000.0)) : 0;
        std::cout << col::dim << "  [" << result.generated_tokens << " tokens, "
                  << std::fixed << std::setprecision(0) << elapsed << "ms, "
                  << std::setprecision(1) << tps << " tok/s]" << col::reset << "\n\n";

        // Add assistant response to history
        // Clean up: remove leading/trailing whitespace
        while (!full_response.empty() && full_response.front() == ' ') full_response.erase(full_response.begin());

        session.history.push_back({"assistant", full_response});

        // Update stats
        session.total_turns++;
        session.total_tokens_generated += result.generated_tokens;
        session.total_time_ms += elapsed;

        // Reset engine for next turn (simple approach: re-process entire context)
        session.engine.reset();
    }

    return 0;
}

} // namespace cli
} // namespace qraf
