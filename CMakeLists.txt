cmake_minimum_required(VERSION 3.16)
project(qraf-runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(QRAF_BUILD_TESTS "Build tests" ON)
option(QRAF_BUILD_BENCHMARKS "Build benchmarks" ON)
option(QRAF_BUILD_TOOLS "Build tools" ON)
option(QRAF_BUILD_SERVER "Build HTTP server" ON)
option(QRAF_USE_NEON "Enable ARM NEON SIMD kernels" ON)
option(QRAF_USE_ACCELERATE "Use Apple Accelerate framework" ON)
option(QRAF_USE_THREADING "Enable multi-threaded ops via GCD" ON)

# Compiler flags
add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

# ─── Apple Silicon / ARM64 detection ───
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(QRAF_IS_ARM64 TRUE)
    message(STATUS "ARM64 detected — enabling Apple Silicon optimizations")
    if(APPLE)
        add_compile_options(-mcpu=apple-m1)
        add_compile_definitions(QRAF_APPLE_SILICON=1)
    endif()
    if(QRAF_USE_NEON)
        add_compile_definitions(QRAF_NEON=1)
        message(STATUS "  NEON SIMD: ON")
    endif()
else()
    set(QRAF_IS_ARM64 FALSE)
    set(QRAF_USE_NEON OFF)
    set(QRAF_USE_ACCELERATE OFF)
    message(STATUS "Non-ARM64 platform — NEON/Accelerate disabled")
endif()

if(QRAF_USE_THREADING)
    add_compile_definitions(QRAF_THREADING=1)
    message(STATUS "  GCD Threading: ON")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# ─── Core library ───
add_library(qraf_core STATIC
    src/core/logging.cpp
    src/core/error.cpp
)

# ─── Tensor library ───
add_library(qraf_tensor STATIC
    src/tensor/tensor.cpp
    src/tensor/dtype.cpp
)
target_link_libraries(qraf_tensor PUBLIC qraf_core)

# ─── Quantization library ───
add_library(qraf_quant STATIC
    src/tensor/quantize.cpp
)
target_link_libraries(qraf_quant PUBLIC qraf_tensor)

# ─── QRAF format library ───
add_library(qraf_format STATIC
    src/qraf/header.cpp
    src/qraf/loader.cpp
    src/qraf/writer.cpp
)
target_link_libraries(qraf_format PUBLIC qraf_tensor qraf_quant)

# ─── Tokenizer ───
add_library(qraf_tokenizer STATIC
    src/runtime/tokenizer.cpp
)
target_link_libraries(qraf_tokenizer PUBLIC qraf_core)

# ─── Neural network ops (with optional SIMD/threading) ───
set(QRAF_NN_SOURCES
    src/nn/ops.cpp
    src/nn/attention.cpp
    src/nn/transformer.cpp
    src/nn/backend.cpp
)

if(QRAF_IS_ARM64 AND QRAF_USE_NEON)
    list(APPEND QRAF_NN_SOURCES src/nn/simd_neon.cpp)
endif()

if(QRAF_USE_THREADING)
    list(APPEND QRAF_NN_SOURCES src/nn/threading.cpp)
endif()

add_library(qraf_nn STATIC ${QRAF_NN_SOURCES})
target_link_libraries(qraf_nn PUBLIC qraf_tensor qraf_quant)

# Apple Accelerate framework
if(APPLE AND QRAF_USE_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(ACCELERATE_FRAMEWORK)
        target_link_libraries(qraf_nn PUBLIC ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(qraf_nn PUBLIC QRAF_ACCELERATE=1)
        message(STATUS "  Apple Accelerate: ON")
    endif()
endif()

# Apple Metal GPU backend
if(APPLE)
    option(QRAF_USE_METAL "Use Metal GPU backend" ON)
    if(QRAF_USE_METAL)
        find_library(METAL_FRAMEWORK Metal)
        find_library(FOUNDATION_FRAMEWORK Foundation)
        if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
            target_sources(qraf_nn PRIVATE src/nn/metal_backend.mm)
            target_link_libraries(qraf_nn PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
            target_compile_definitions(qraf_nn PUBLIC QRAF_METAL=1)
            message(STATUS "  Metal GPU: ON")
        endif()
    endif()
endif()

# ─── Sampling ───
add_library(qraf_sampling STATIC
    src/sampling/sampler.cpp
)
target_link_libraries(qraf_sampling PUBLIC qraf_tensor)

# ─── Runtime ───
add_library(qraf_runtime STATIC
    src/runtime/model_manager.cpp
    src/runtime/inference.cpp
    src/runtime/speculative.cpp
    src/nn/lora.cpp
)
target_link_libraries(qraf_runtime PUBLIC
    qraf_format qraf_nn qraf_sampling qraf_tokenizer
)

# ─── CLI executable ───
add_executable(qraf-cli
    src/cli/main.cpp
    src/cli/commands.cpp
    src/cli/chat.cpp
)
target_link_libraries(qraf-cli PRIVATE qraf_runtime)

# ─── HTTP Server ───
if(QRAF_BUILD_SERVER)
    add_executable(qraf-server
        src/server/http_server.cpp
        src/server/main.cpp
    )
    target_link_libraries(qraf-server PRIVATE qraf_runtime)
endif()

# ─── Tools ───
if(QRAF_BUILD_TOOLS)
    add_executable(qraf-writer
        tools/qraf_writer.cpp
    )
    target_link_libraries(qraf-writer PRIVATE qraf_format qraf_tokenizer)

    add_executable(qraf-dummy-model
        tools/dummy_model.cpp
    )
    target_link_libraries(qraf-dummy-model PRIVATE qraf_format qraf_tokenizer)
endif()

# ─── Tests ───
if(QRAF_BUILD_TESTS)
    enable_testing()

    add_executable(qraf-tests
        tests/test_main.cpp
    )
    target_link_libraries(qraf-tests PRIVATE qraf_runtime)
    add_test(NAME qraf-tests COMMAND qraf-tests)
endif()

# ─── Benchmarks ───
if(QRAF_BUILD_BENCHMARKS)
    add_executable(qraf-bench
        benchmarks/bench_main.cpp
    )
    target_link_libraries(qraf-bench PRIVATE qraf_runtime)
endif()
