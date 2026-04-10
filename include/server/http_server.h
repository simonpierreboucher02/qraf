#pragma once

#include "runtime/model_manager.h"
#include <string>
#include <functional>

namespace qraf {

class HttpServer {
public:
    HttpServer(ModelManager& manager, int port = 8080);
    ~HttpServer();

    // Start the server (blocking)
    void run();

    // Stop the server
    void stop();

private:
    // Handle a single client connection
    void handle_client(int client_fd);

    // Parse HTTP request
    struct HttpRequest {
        std::string method;
        std::string path;
        std::string body;
        std::string content_type;
    };

    HttpRequest parse_request(int client_fd);

    // Send HTTP response
    void send_response(int client_fd, int status, const std::string& body,
                       const std::string& content_type = "application/json");
    void send_sse_event(int client_fd, const std::string& data);
    void send_sse_done(int client_fd);

    // Route handlers
    void handle_generate(int client_fd, const HttpRequest& req);
    void handle_chat(int client_fd, const HttpRequest& req);
    void handle_models(int client_fd, const HttpRequest& req);
    void handle_health(int client_fd, const HttpRequest& req);

    // Simple JSON helpers
    static std::string json_string(const std::string& key, const std::string& value);
    static std::string json_error(const std::string& message);

    ModelManager& manager_;
    int port_;
    int server_fd_ = -1;
    bool running_ = false;
};

} // namespace qraf
