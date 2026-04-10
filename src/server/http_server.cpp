#include "server/http_server.h"
#include "core/logging.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <iostream>
#include <thread>

namespace qraf {

HttpServer::HttpServer(ModelManager& manager, int port)
    : manager_(manager), port_(port) {}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::run() {
    // Create socket
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        log::fatal("Failed to create socket");
        return;
    }

    int opt = 1;
    setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(server_fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        log::fatal("Failed to bind to port %d", port_);
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    if (listen(server_fd_, 10) < 0) {
        log::fatal("Failed to listen");
        close(server_fd_);
        server_fd_ = -1;
        return;
    }

    running_ = true;
    log::info("HTTP server listening on port %d", port_);
    std::cout << "QRAF server running at http://localhost:" << port_ << "\n";

    while (running_) {
        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd_, reinterpret_cast<struct sockaddr*>(&client_addr), &client_len);

        if (client_fd < 0) {
            if (running_) log::error("Failed to accept connection");
            continue;
        }

        // Handle in a thread
        std::thread([this, client_fd]() {
            handle_client(client_fd);
            close(client_fd);
        }).detach();
    }
}

void HttpServer::stop() {
    running_ = false;
    if (server_fd_ >= 0) {
        close(server_fd_);
        server_fd_ = -1;
    }
}

HttpServer::HttpRequest HttpServer::parse_request(int client_fd) {
    HttpRequest req;
    char buffer[8192];
    ssize_t n = read(client_fd, buffer, sizeof(buffer) - 1);
    if (n <= 0) return req;
    buffer[n] = '\0';

    std::istringstream stream(buffer);
    stream >> req.method >> req.path;

    // Find Content-Length
    std::string line;
    int content_length = 0;
    while (std::getline(stream, line)) {
        if (line.find("Content-Length:") != std::string::npos) {
            content_length = std::stoi(line.substr(16));
        }
        if (line.find("Content-Type:") != std::string::npos) {
            req.content_type = line.substr(14);
        }
        if (line == "\r" || line.empty()) break;
    }

    // Read body
    if (content_length > 0) {
        std::string full(buffer, n);
        size_t body_start = full.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            req.body = full.substr(body_start + 4);
        }
    }

    return req;
}

void HttpServer::send_response(int client_fd, int status, const std::string& body,
                                const std::string& content_type) {
    std::string status_text;
    switch (status) {
        case 200: status_text = "OK"; break;
        case 400: status_text = "Bad Request"; break;
        case 404: status_text = "Not Found"; break;
        case 500: status_text = "Internal Server Error"; break;
        default:  status_text = "Unknown"; break;
    }

    std::ostringstream response;
    response << "HTTP/1.1 " << status << " " << status_text << "\r\n";
    response << "Content-Type: " << content_type << "\r\n";
    response << "Content-Length: " << body.size() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "\r\n";
    response << body;

    std::string resp = response.str();
    write(client_fd, resp.c_str(), resp.size());
}

void HttpServer::send_sse_event(int client_fd, const std::string& data) {
    std::string event = "data: " + data + "\n\n";
    write(client_fd, event.c_str(), event.size());
}

void HttpServer::send_sse_done(int client_fd) {
    std::string event = "data: [DONE]\n\n";
    write(client_fd, event.c_str(), event.size());
}

void HttpServer::handle_client(int client_fd) {
    HttpRequest req = parse_request(client_fd);

    log::debug("HTTP %s %s", req.method.c_str(), req.path.c_str());

    if (req.path == "/api/generate" && req.method == "POST") {
        handle_generate(client_fd, req);
    } else if (req.path == "/api/chat" && req.method == "POST") {
        handle_chat(client_fd, req);
    } else if (req.path == "/api/models" && req.method == "GET") {
        handle_models(client_fd, req);
    } else if (req.path == "/health" && req.method == "GET") {
        handle_health(client_fd, req);
    } else {
        send_response(client_fd, 404, json_error("Not found"));
    }
}

// Minimal JSON parser — extracts a string value for a key
static std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(":", pos);
    if (pos == std::string::npos) return "";

    pos = json.find("\"", pos + 1);
    if (pos == std::string::npos) return "";

    size_t end = json.find("\"", pos + 1);
    if (end == std::string::npos) return "";

    return json.substr(pos + 1, end - pos - 1);
}

static int json_get_int(const std::string& json, const std::string& key, int default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(":", pos);
    if (pos == std::string::npos) return default_val;

    pos++;
    while (pos < json.size() && json[pos] == ' ') pos++;

    try {
        return std::stoi(json.substr(pos));
    } catch (...) {
        return default_val;
    }
}

void HttpServer::handle_generate(int client_fd, const HttpRequest& req) {
    std::string model_name = json_get_string(req.body, "model");
    std::string prompt = json_get_string(req.body, "prompt");
    int max_tokens = json_get_int(req.body, "max_tokens", 256);
    bool stream = req.body.find("\"stream\":true") != std::string::npos ||
                  req.body.find("\"stream\": true") != std::string::npos;

    if (model_name.empty()) {
        send_response(client_fd, 400, json_error("model field required"));
        return;
    }

    InferenceEngine* engine = manager_.load(model_name);
    if (!engine) {
        send_response(client_fd, 400, json_error("Failed to load model: " + model_name));
        return;
    }

    GenerateConfig config;
    config.max_tokens = static_cast<u32>(max_tokens);

    if (stream) {
        // SSE streaming
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "\r\n";
        write(client_fd, header.c_str(), header.size());

        engine->generate(prompt, config, [this, client_fd](u32 /*token_id*/, const std::string& text) {
            std::string data = "{\"text\":\"" + text + "\"}";
            send_sse_event(client_fd, data);
            return true;
        });

        send_sse_done(client_fd);
    } else {
        auto result = engine->generate(prompt, config);

        std::ostringstream json;
        json << "{";
        json << "\"text\":\"" << result.text << "\",";
        json << "\"tokens\":" << result.generated_tokens << ",";
        json << "\"prompt_tokens\":" << result.prompt_tokens << ",";
        json << "\"time_ms\":" << result.generation_time_ms << ",";
        json << "\"tokens_per_sec\":" << result.tokens_per_sec;
        json << "}";

        send_response(client_fd, 200, json.str());
    }
}

void HttpServer::handle_chat(int client_fd, const HttpRequest& req) {
    // Simplified: treat as generate with conversation format
    handle_generate(client_fd, req);
}

void HttpServer::handle_models(int client_fd, const HttpRequest& /*req*/) {
    manager_.scan();
    auto models = manager_.list();

    std::ostringstream json;
    json << "{\"models\":[";
    for (size_t i = 0; i < models.size(); i++) {
        if (i > 0) json << ",";
        json << "{";
        json << "\"name\":\"" << models[i].name << "\",";
        json << "\"size\":" << models[i].file_size << ",";
        json << "\"loaded\":" << (models[i].loaded ? "true" : "false");
        json << "}";
    }
    json << "]}";

    send_response(client_fd, 200, json.str());
}

void HttpServer::handle_health(int client_fd, const HttpRequest& /*req*/) {
    send_response(client_fd, 200, "{\"status\":\"ok\"}");
}

std::string HttpServer::json_string(const std::string& key, const std::string& value) {
    return "\"" + key + "\":\"" + value + "\"";
}

std::string HttpServer::json_error(const std::string& message) {
    return "{\"error\":\"" + message + "\"}";
}

} // namespace qraf
