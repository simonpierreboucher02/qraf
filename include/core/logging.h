#pragma once

#include <string>
#include <cstdio>
#include <cstdarg>

namespace qraf {
namespace log {

enum class Level {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3,
    FATAL = 4,
};

void set_level(Level level);
Level get_level();

void debug(const char* fmt, ...);
void info(const char* fmt, ...);
void warn(const char* fmt, ...);
void error(const char* fmt, ...);
void fatal(const char* fmt, ...);

} // namespace log
} // namespace qraf
