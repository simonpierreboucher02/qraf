#pragma once

#include <string>
#include <vector>

namespace qraf {
namespace cli {

int cmd_run(const std::string& model_path, const std::string& prompt, int max_tokens);
int cmd_chat(const std::string& model_path);
int cmd_chat_interactive(const std::string& models_dir);
int cmd_list(const std::string& models_dir);
int cmd_inspect(const std::string& model_path);
int cmd_benchmark(const std::string& model_path, int num_tokens);
int cmd_convert(const std::string& source, const std::string& output, const std::string& format);
int cmd_version();
int cmd_help();

} // namespace cli
} // namespace qraf
