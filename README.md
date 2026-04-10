# QRAF — Quantized Runtime Artifact Format

> **A next-gen local LLM inference runtime built from scratch in C++, optimized for Apple Silicon.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20to%20M5-black.svg)]()
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)]()
[![Models](https://img.shields.io/badge/Models-25%20converted-green.svg)]()
[![npm](https://img.shields.io/npm/v/qraf.svg)](https://www.npmjs.com/package/qraf)

---

## What is QRAF?

QRAF is a **complete LLM inference engine** — like Ollama, but lower-level, faster, and built for serious research and control.

It includes:
- A custom **binary model format** (`.qraf`) designed for mmap and zero-copy
- A **C++ inference engine** with Apple Accelerate, ARM NEON SIMD, and GCD threading
- A **universal model converter** (HuggingFace, safetensors, PyTorch, GGUF → QRAF)
- An **interactive CLI chatbot** with model browser
- An **HTTP API server** with streaming

```
                ┌─────────────┐
                │  QRAF v0.1  │
                └──────┬──────┘
       ┌───────────────┼───────────────┐
       │               │               │
  ┌────▼────┐   ┌──────▼──────┐  ┌─────▼─────┐
  │   CLI   │   │  HTTP API   │  │ Converter │
  │  Chat   │   │  /generate  │  │ HF→QRAF   │
  └────┬────┘   └──────┬──────┘  │ GGUF→QRAF │
       │               │         └───────────┘
  ┌────▼───────────────▼────┐
  │    Inference Engine     │
  │  Transformer + KV Cache │
  └────────────┬────────────┘
       ┌───────┼───────┐
       │       │       │
  ┌────▼──┐ ┌──▼───┐ ┌─▼──────┐
  │Accel. │ │ NEON │ │  GCD   │
  │ BLAS  │ │ SIMD │ │Threads │
  └───────┘ └──────┘ └────────┘
```

---

## Performance

Benchmarked on Apple Silicon with Accelerate + NEON + GCD:

### Inference Speed (tok/s)

| Model | Params | QRAF Size | tok/s | ms/token |
|:------|-------:|----------:|------:|---------:|
| SmolLM2-135M | 135M | 622 MB | **214** | 4.7 |
| SmolLM2-135M-Instruct | 135M | 622 MB | **65** | 15.4 |
| SmolLM2-360M | 360M | 1.5 GB | **56** | 17.7 |
| SmolLM2-360M-Instruct | 360M | 1.5 GB | **54** | 18.5 |
| Qwen2.5-0.5B | 494M | 2.4 GB | **31** | 32.1 |
| Qwen2.5-0.5B-Instruct | 494M | 2.4 GB | **31** | 32.4 |
| Qwen2.5-Coder-0.5B | 494M | 2.4 GB | **31** | 32.3 |
| TinyLlama-1.1B-Chat | 1.1B | 4.1 GB | **15** | 67.1 |

### Optimization Speedup (vs scalar baseline)

| Component | Scalar | Optimized | Speedup |
|:----------|-------:|----------:|--------:|
| Full inference (SmolLM2-135M) | 5.5 tok/s | 226 tok/s | **41x** |
| MatVec 128x128 | 0.046 ms | 0.003 ms | **15x** |
| Softmax 32k | 0.161 ms | 0.050 ms | **3.2x** |
| RMSNorm 4096 | 0.015 ms | 0.002 ms | **7.5x** |
| Q8 Dot 4096 | 0.004 ms | 0.001 ms | **4x** |

### Micro-benchmarks

| Operation | Size | Time | Throughput |
|:----------|-----:|-----:|-----------:|
| MatVec | 4096x4096 | 7.7 ms | 130 iter/s |
| Softmax | 32,000 | 0.05 ms | 20,089 iter/s |
| RMSNorm | 4096 | 0.002 ms | 443,114 iter/s |
| SiLU | 4096 | 0.001 ms | 1M+ iter/s |
| Q8 Dot Product | 4096 | 0.001 ms | 1.7M iter/s |
| Q4 Dot Product | 4096 | 0.001 ms | 1.1M iter/s |

---

## Quick Start

### Install from npm (easiest)

```bash
npm install -g qraf
```

### Build from source

```bash
git clone https://github.com/simonpierreboucher02/qraf.git
cd qraf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Convert a model

```bash
# From HuggingFace (auto-downloads)
qraf convert Qwen/Qwen2.5-0.5B-Instruct -o models/qwen.qraf

# From local GGUF
qraf convert model.gguf -o models/model.qraf

# From safetensors
qraf convert weights.safetensors -o models/model.qraf
```

### Chat interactively

```bash
qraf chat
```

This opens the interactive chatbot with model browser:

```
   ██████╗ ██████╗  █████╗ ███████╗
  ██╔═══██╗██╔══██╗██╔══██╗██╔════╝
  ██║   ██║██████╔╝███████║█████╗
  ╚██████╔╝██║  ██║██║  ██║██║
   ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     v0.1.0

  Select a model:
   [1]  qwen2.5-0.5b-instruct    2.4 GB
   [2]  smollm2-360m-instruct    1.5 GB
   [3]  smollm2-135m-instruct    622 MB
   ...

  > 1
  ✓ qwen2.5-0.5b-instruct (24L, h=896, 14h)

  You > Hello! Tell me about quantum computing.

  Assistant > Quantum computing is a type of computation that uses...
  [87 tokens, 2400ms, 36.2 tok/s]

  You > /stats
  ╭──────────────────────────────╮
  │       Session Stats          │
  ╰──────────────────────────────╯
  Model:           qwen2.5-0.5b-instruct
  Turns:           1
  Avg speed:       36.2 tok/s
```

### Chat commands

| Command | Description |
|:--------|:------------|
| `/help` | Show available commands |
| `/quit` | Exit chat |
| `/reset` | Clear conversation history |
| `/model` | Switch to a different model |
| `/system <text>` | Set system prompt |
| `/history` | Show conversation history |
| `/stats` | Show session statistics |
| `/config` | Show generation settings |
| `/temp <val>` | Set temperature (0.0 - 2.0) |
| `/tokens <n>` | Set max tokens per response |
| `/clear` | Clear screen |

---

## CLI Commands

### `qraf chat`
Interactive chatbot with model browser, multi-turn conversation, streaming output.

### `qraf run <model> --prompt <text> [--max-tokens <n>]`
One-shot text generation.

```bash
qraf run models/qwen.qraf --prompt "The meaning of life is" --max-tokens 100
```

### `qraf convert <source> -o <output.qraf>`
Universal model converter.

```bash
# HuggingFace hub
qraf convert Qwen/Qwen2.5-0.5B -o models/qwen.qraf

# Local directory
qraf convert ./my-model/ -o models/local.qraf

# GGUF file
qraf convert model.gguf -o models/from-gguf.qraf

# Safetensors file
qraf convert weights.safetensors -o models/out.qraf

# PyTorch checkpoint
qraf convert model.bin -o models/out.qraf
```

### `qraf list [--dir <path>]`
List available QRAF models.

### `qraf inspect <model>`
Show detailed model information (architecture, tensors, config).

### `qraf benchmark <model> [--tokens <n>]`
Measure inference performance.

### `qraf version`
Show version information.

---

## Model Library

25 pre-tested models compatible with QRAF:

### Llama / Qwen Architecture (full inference support)

| Model | Params | Type | QRAF Size | Speed |
|:------|-------:|:-----|----------:|------:|
| SmolLM2-135M | 135M | Base | 622 MB | 214 tok/s |
| SmolLM2-135M-Instruct | 135M | Instruct | 622 MB | 65 tok/s |
| SmolLM2-360M | 360M | Base | 1.5 GB | 56 tok/s |
| SmolLM2-360M-Instruct | 360M | Instruct | 1.5 GB | 54 tok/s |
| Qwen2.5-0.5B | 494M | Base | 2.4 GB | 31 tok/s |
| Qwen2.5-0.5B-Instruct | 494M | Instruct | 2.4 GB | 31 tok/s |
| Qwen2.5-Coder-0.5B | 494M | Code | 2.4 GB | 31 tok/s |
| Qwen2.5-Coder-0.5B-Instruct | 494M | Code | 2.4 GB | 30 tok/s |
| TinyLlama-1.1B-Chat | 1.1B | Chat | 4.1 GB | 15 tok/s |

### GPT-2 / Pythia / OPT Architecture (QRAF files ready, runtime support coming)

| Model | Params | Type | QRAF Size |
|:------|-------:|:-----|----------:|
| Pythia-70M | 70M | Base | 270 MB |
| DistilGPT2 | 82M | Base | 461 MB |
| Cerebras-GPT-111M | 111M | Base | 572 MB |
| Pythia-160M | 160M | Base | 620 MB |
| GPT-2 | 124M | Base | 623 MB |
| DialoGPT-Small | 124M | Dialogue | 623 MB |
| LaMini-GPT-124M | 124M | Base | 623 MB |
| OPT-125M | 125M | Base | 626 MB |
| Tiny StarCoder Py | 164M | Code | 771 MB |
| Cerebras-GPT-256M | 256M | Base | 1.2 GB |
| CodeGen-350M-Mono | 350M | Code | 1.3 GB |
| OPT-350M | 350M | Base | 1.3 GB |
| Pythia-410M | 410M | Base | 1.5 GB |
| GPT-2 Medium | 355M | Base | 1.5 GB |
| DialoGPT-Medium | 355M | Dialogue | 1.5 GB |
| GPT-2 Large | 774M | Base | 3.1 GB |

---

## QRAF Format Specification

### Binary Layout

```
┌──────────────────────────────────┐  offset 0
│  Header (128 bytes)              │
├──────────────────────────────────┤
│  Model Config Block              │  key-value pairs
├──────────────────────────────────┤
│  Tokenizer Block                 │  vocab + merges + specials
├──────────────────────────────────┤
│  Tensor Directory                │  80 bytes per tensor
├──────────────────────────────────┤
│  Quantization Directory          │  32 bytes per scheme
├──────────────────────────────────┤
│  String Table                    │  null-terminated strings
├──────────────────────────────────┤  aligned to 64 bytes
│  Tensor Data Blocks              │  mmap-friendly, zero-copy
└──────────────────────────────────┘
```

### Header (128 bytes)

```c
struct QrafHeader {
    uint32_t magic;           // "QRAF" = 0x46415251
    uint32_t version;         // 1
    uint64_t file_size;
    uint64_t config_offset, config_size;
    uint64_t tokenizer_offset, tokenizer_size;
    uint64_t tensor_dir_offset, tensor_dir_size;
    uint64_t quant_dir_offset, quant_dir_size;
    uint64_t string_table_offset, string_table_size;
    uint64_t data_offset;
    uint32_t num_tensors, num_quant_schemes;
    uint32_t flags, padding;
    uint8_t  reserved[8];
};
```

### Key Features

- **mmap-first**: Models are memory-mapped, not copied. Load time is near-instant.
- **64-byte alignment**: All tensor data aligned for SIMD/DMA access.
- **Zero-copy tensor views**: Direct pointers into mapped memory.
- **FNV-1a hashing**: Fast tensor lookup by name hash.
- **Quantization-aware**: Native Q4, Q8, FP16 support in the format.

---

## Architecture

### Engine Stack

```
┌─────────────────────────────────────────────┐
│  CLI / HTTP Server                          │
├─────────────────────────────────────────────┤
│  Inference Engine (generate, stream)        │
├─────────────────────────────────────────────┤
│  Transformer (forward pass, KV cache)       │
├──────────┬──────────┬───────────────────────┤
│ Attention│   MLP    │  RMSNorm / RoPE       │
├──────────┴──────────┴───────────────────────┤
│  Backend Dispatch (Accelerate/NEON/Scalar)  │
├──────────┬──────────┬───────────────────────┤
│ cblas    │ NEON     │  GCD Threading        │
│ sgemv   │ FMA/dot  │  dispatch_apply        │
├──────────┴──────────┴───────────────────────┤
│  QRAF Loader (mmap, tensor index)           │
├─────────────────────────────────────────────┤
│  Tensor System (dtype, shape, quantize)     │
└─────────────────────────────────────────────┘
```

### Apple Silicon Optimizations

| Layer | Technology | What it does |
|:------|:-----------|:-------------|
| MatVec (f32) | Apple Accelerate `cblas_sgemv` | Internally NEON + multi-threaded by Apple |
| Dot products | ARM NEON `vfmaq_f32` | 4 accumulators, 16-wide unroll |
| RMSNorm | ARM NEON `vfmaq_f32` + `vaddvq_f32` | Vectorized sum-of-squares + scale |
| Softmax | Accelerate `vvexpf` + `vDSP_maxv` | Hardware-optimized exp() |
| SiLU | Accelerate `vvexpf` + `vDSP_vdiv` | Vectorized sigmoid |
| RoPE | NEON `vld2q_f32` / `vst2q_f32` | Interleaved load for rotation pairs |
| Q8 dot | NEON int8→f32 widening + FMA | 16 values per iteration |
| Q4 dot | NEON nibble unpack `vzipq_s8` | 32 values per iteration |
| Threading | GCD `dispatch_apply` | `QOS_CLASS_USER_INTERACTIVE` → P-cores |
| Quantized matvec | NEON + GCD parallel rows | Threaded quantized dot products |
| Element-wise | NEON 16-wide unroll | add, mul, scale all vectorized |

### Supported Model Architectures

| Architecture | Models | Status |
|:-------------|:-------|:-------|
| **LLaMA** | SmolLM2, TinyLlama, LLaMA-2/3 | Full support |
| **Qwen2** | Qwen2.5, Qwen2.5-Coder | Full support (with attention bias) |
| GPT-2 | GPT-2, DistilGPT2, DialoGPT, Cerebras-GPT | QRAF conversion ready |
| GPT-NeoX | Pythia | QRAF conversion ready |
| OPT | OPT-125M, OPT-350M | QRAF conversion ready |
| CodeGen | CodeGen-350M | QRAF conversion ready |
| StarCoder | Tiny StarCoder | QRAF conversion ready |

---

## Project Structure

```
qraf/
├── include/
│   ├── core/          types.h, logging.h, error.h
│   ├── tensor/        tensor.h, quantize.h
│   ├── qraf/          format.h, loader.h, writer.h
│   ├── nn/            ops.h, attention.h, transformer.h,
│   │                  backend.h, simd_neon.h, threading.h
│   ├── runtime/       tokenizer.h, inference.h, model_manager.h
│   ├── sampling/      sampler.h
│   ├── cli/           commands.h
│   └── server/        http_server.h
├── src/               All implementations
├── tools/
│   ├── qraf_convert.py      Universal converter
│   ├── dummy_model.cpp      Test model generator
│   └── convert_hf_to_qraf.py  HF-specific converter
├── tests/             28 unit tests
├── benchmarks/        Performance measurements
├── models/            Model storage (.qraf files)
└── CMakeLists.txt
```

---

## HTTP API

Start the server:

```bash
qraf-server --port 8080 --dir models
```

### `POST /api/generate`

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-0.5b", "prompt": "Hello!", "max_tokens": 100}'
```

### `POST /api/generate` (streaming)

```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-0.5b", "prompt": "Hello!", "stream": true}'
```

### `GET /api/models`

```bash
curl http://localhost:8080/api/models
```

### `GET /health`

```bash
curl http://localhost:8080/health
```

---

## Building

### Requirements

- C++17 compiler (Clang 14+ or GCC 11+)
- CMake 3.16+
- Python 3.8+ (for converter)
- pip packages: `torch transformers safetensors gguf` (for converter)

### macOS (Apple Silicon)

```bash
brew install cmake
pip3 install torch transformers safetensors sentencepiece gguf

git clone https://github.com/simonpierreboucher02/qraf.git
cd qraf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Build Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `QRAF_USE_NEON` | ON | ARM NEON SIMD kernels |
| `QRAF_USE_ACCELERATE` | ON | Apple Accelerate framework |
| `QRAF_USE_THREADING` | ON | GCD multi-threading |
| `QRAF_BUILD_TESTS` | ON | Build test suite |
| `QRAF_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `QRAF_BUILD_SERVER` | ON | Build HTTP server |
| `QRAF_BUILD_TOOLS` | ON | Build tools |

### Run Tests

```bash
./qraf-tests    # 28 tests
./qraf-bench    # Performance benchmarks
```

---

## Quantization

QRAF supports quantized storage and on-the-fly dequantization:

| Format | Bits | Block Size | Description |
|:-------|-----:|-----------:|:------------|
| F32 | 32 | - | Full precision |
| F16 | 16 | - | Half precision |
| BF16 | 16 | - | Brain float |
| Q8_0 | 8 | 32 | Per-block scale |
| Q4_0 | 4 | 32 | Per-block scale, packed nibbles |

Quantized dot products run directly on compressed data via NEON — no full dequantization needed.

---

## Roadmap

- [x] QRAF binary format specification
- [x] mmap loader with tensor index
- [x] Decoder-only transformer with KV cache
- [x] GQA (Grouped Query Attention)
- [x] BPE tokenizer with GPT-2 byte-level encoding
- [x] Top-k, top-p, temperature sampling
- [x] Apple Accelerate backend
- [x] ARM NEON SIMD kernels
- [x] GCD multi-threading
- [x] Attention bias (Qwen-style)
- [x] Universal converter (HF, safetensors, PyTorch, GGUF)
- [x] Interactive CLI chatbot
- [x] HTTP API with SSE streaming
- [x] 25 pre-tested model conversions
- [ ] GPT-2 / OPT / Pythia architecture support
- [ ] Metal GPU backend (Apple Silicon GPU)
- [ ] LoRA adapter merging at runtime
- [ ] KV cache quantization
- [ ] Speculative decoding
- [ ] Continuous batching
- [ ] CUDA backend

---

## License

MIT License. See [LICENSE](LICENSE) for details.
