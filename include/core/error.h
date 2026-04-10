#pragma once

#include <stdexcept>
#include <string>

namespace qraf {

class QrafError : public std::runtime_error {
public:
    explicit QrafError(const std::string& msg) : std::runtime_error(msg) {}
};

class FormatError : public QrafError {
public:
    explicit FormatError(const std::string& msg) : QrafError("QRAF format error: " + msg) {}
};

class LoadError : public QrafError {
public:
    explicit LoadError(const std::string& msg) : QrafError("Load error: " + msg) {}
};

class InferenceError : public QrafError {
public:
    explicit InferenceError(const std::string& msg) : QrafError("Inference error: " + msg) {}
};

class ShapeError : public QrafError {
public:
    explicit ShapeError(const std::string& msg) : QrafError("Shape mismatch: " + msg) {}
};

#define QRAF_CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            char buf[512]; \
            snprintf(buf, sizeof(buf), __VA_ARGS__); \
            throw qraf::QrafError(buf); \
        } \
    } while (0)

#define QRAF_CHECK_SHAPE(cond, ...) \
    do { \
        if (!(cond)) { \
            char buf[512]; \
            snprintf(buf, sizeof(buf), __VA_ARGS__); \
            throw qraf::ShapeError(buf); \
        } \
    } while (0)

} // namespace qraf
