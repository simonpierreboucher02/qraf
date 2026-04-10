#include "core/logging.h"
#include <cstdio>
#include <cstdarg>
#include <ctime>

namespace qraf {
namespace log {

static Level g_level = Level::INFO;

void set_level(Level level) { g_level = level; }
Level get_level() { return g_level; }

static void log_msg(Level level, const char* prefix, const char* fmt, va_list args) {
    if (level < g_level) return;

    // timestamp
    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    char time_str[32];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", &tm_buf);

    fprintf(stderr, "[%s] %s: ", time_str, prefix);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    fflush(stderr);
}

void debug(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    log_msg(Level::DEBUG, "DEBUG", fmt, args);
    va_end(args);
}

void info(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    log_msg(Level::INFO, "INFO ", fmt, args);
    va_end(args);
}

void warn(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    log_msg(Level::WARN, "WARN ", fmt, args);
    va_end(args);
}

void error(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    log_msg(Level::ERROR, "ERROR", fmt, args);
    va_end(args);
}

void fatal(const char* fmt, ...) {
    va_list args; va_start(args, fmt);
    log_msg(Level::FATAL, "FATAL", fmt, args);
    va_end(args);
}

} // namespace log
} // namespace qraf
