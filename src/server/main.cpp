#include "server/http_server.h"
#include "runtime/model_manager.h"
#include "core/logging.h"
#include <iostream>
#include <string>
#include <csignal>

static qraf::HttpServer* g_server = nullptr;

static void signal_handler(int /*sig*/) {
    if (g_server) {
        g_server->stop();
    }
}

int main(int argc, char* argv[]) {
    int port = 8080;
    std::string models_dir = "models";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
            port = std::stoi(argv[++i]);
        } else if ((arg == "--dir" || arg == "-d") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (arg == "--verbose" || arg == "-v") {
            qraf::log::set_level(qraf::log::Level::DEBUG);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: qraf-server [--port <port>] [--dir <models_dir>] [--verbose]\n";
            return 0;
        }
    }

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    qraf::ModelManager manager(models_dir);
    manager.scan();

    qraf::HttpServer server(manager, port);
    g_server = &server;

    std::cout << "QRAF Server v0.1.0\n";
    std::cout << "Models directory: " << models_dir << "\n";

    server.run();

    return 0;
}
