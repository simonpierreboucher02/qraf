#include "cli/commands.h"
#include "core/logging.h"
#include <string>
#include <vector>
#include <iostream>

using namespace qraf;

static std::string get_arg(const std::vector<std::string>& args, int& i, const std::string& flag) {
    if (i + 1 < static_cast<int>(args.size())) {
        return args[++i];
    }
    std::cerr << "Error: " << flag << " requires a value\n";
    return "";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cli::cmd_help();
        return 1;
    }

    std::vector<std::string> args(argv + 1, argv + argc);
    std::string command = args[0];

    // Parse global flags
    bool verbose = false;
    for (auto& a : args) {
        if (a == "--verbose" || a == "-v") verbose = true;
    }

    if (verbose) {
        log::set_level(log::Level::DEBUG);
    }

    if (command == "help" || command == "--help" || command == "-h") {
        return cli::cmd_help();
    }

    if (command == "version" || command == "--version") {
        return cli::cmd_version();
    }

    if (command == "run") {
        std::string model_path;
        std::string prompt = "Hello";
        int max_tokens = 256;

        for (int i = 1; i < static_cast<int>(args.size()); i++) {
            if (args[i] == "--prompt" || args[i] == "-p") {
                prompt = get_arg(args, i, args[i]);
            } else if (args[i] == "--max-tokens" || args[i] == "-n") {
                std::string val = get_arg(args, i, args[i]);
                if (!val.empty()) max_tokens = std::stoi(val);
            } else if (args[i] == "--verbose" || args[i] == "-v") {
                continue;
            } else if (model_path.empty() && args[i][0] != '-') {
                model_path = args[i];
            }
        }

        if (model_path.empty()) {
            std::cerr << "Error: model path required\n";
            return 1;
        }

        return cli::cmd_run(model_path, prompt, max_tokens);
    }

    if (command == "chat") {
        std::string model_path;
        std::string dir = "models";
        for (int i = 1; i < static_cast<int>(args.size()); i++) {
            if (args[i] == "--dir" || args[i] == "-d") {
                dir = get_arg(args, i, args[i]);
            } else if (args[i] == "--verbose" || args[i] == "-v") {
                continue;
            } else if (args[i][0] != '-') {
                model_path = args[i];
            }
        }
        // If a specific model is given, use old simple chat
        if (!model_path.empty()) {
            return cli::cmd_chat(model_path);
        }
        // Otherwise, launch interactive chat with model browser
        return cli::cmd_chat_interactive(dir);
    }

    if (command == "browse" || command == "store" || command == "download") {
        std::string dir = "models";
        for (int i = 1; i < static_cast<int>(args.size()); i++) {
            if (args[i] == "--dir" || args[i] == "-d") {
                dir = get_arg(args, i, args[i]);
            }
        }
        return cli::cmd_browse(dir);
    }

    if (command == "list") {
        std::string dir = "models";
        for (int i = 1; i < static_cast<int>(args.size()); i++) {
            if (args[i] == "--dir" || args[i] == "-d") {
                dir = get_arg(args, i, args[i]);
            }
        }
        return cli::cmd_list(dir);
    }

    if (command == "inspect") {
        std::string model_path;
        for (int i = 1; i < static_cast<int>(args.size()); i++) {
            if (args[i][0] != '-') { model_path = args[i]; break; }
        }
        if (model_path.empty()) {
            std::cerr << "Error: model path required\n";
            return 1;
        }
        return cli::cmd_inspect(model_path);
    }

    if (command == "benchmark") {
        std::string model_path;
        int num_tokens = 100;

        for (int i = 1; i < static_cast<int>(args.size()); i++) {
            if (args[i] == "--tokens" || args[i] == "-n") {
                std::string val = get_arg(args, i, args[i]);
                if (!val.empty()) num_tokens = std::stoi(val);
            } else if (args[i][0] != '-') {
                model_path = args[i];
            }
        }

        if (model_path.empty()) {
            std::cerr << "Error: model path required\n";
            return 1;
        }
        return cli::cmd_benchmark(model_path, num_tokens);
    }

    if (command == "convert") {
        std::string source;
        std::string output;
        std::string format = "auto";

        for (int i = 1; i < static_cast<int>(args.size()); i++) {
            if (args[i] == "-o" || args[i] == "--output") {
                output = get_arg(args, i, args[i]);
            } else if (args[i] == "--format" || args[i] == "-f") {
                format = get_arg(args, i, args[i]);
            } else if (args[i] == "--verbose" || args[i] == "-v") {
                continue;
            } else if (source.empty() && args[i][0] != '-') {
                source = args[i];
            }
        }

        if (source.empty()) {
            std::cerr << "Error: source model required\n";
            std::cerr << "Usage: qraf convert <source> -o <output.qraf>\n";
            return 1;
        }
        if (output.empty()) {
            // Auto-generate output name
            std::string name = source;
            // Use last path component
            auto pos = name.find_last_of('/');
            if (pos != std::string::npos) name = name.substr(pos + 1);
            // Replace special chars
            for (auto& c : name) {
                if (c == '.' || c == ':') c = '-';
            }
            output = "models/" + name + ".qraf";
            std::cout << "Output: " << output << "\n";
        }

        return cli::cmd_convert(source, output, format);
    }

    std::cerr << "Unknown command: " << command << "\n";
    cli::cmd_help();
    return 1;
}
