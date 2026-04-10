# QRAF — Quantized Runtime Artifact Format

> **A next-gen local LLM inference runtime built from scratch in C++, optimized for Apple Silicon.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux-lightgrey.svg)]()
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20to%20M5-black.svg)]()
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)]()
[![Architectures](https://img.shields.io/badge/Architectures-7%20supported-blue.svg)]()
[![Models](https://img.shields.io/badge/Models-26%20tested-green.svg)]()
[![npm](https://img.shields.io/npm/v/qraf-runtime.svg)](https://www.npmjs.com/package/qraf-runtime)
[![Tests](https://img.shields.io/badge/Tests-28%20passing-brightgreen.svg)]()
[![41x Speedup](https://img.shields.io/badge/Speedup-41x%20vs%20scalar-orange.svg)]()

---

## What is QRAF?

QRAF is a **complete LLM inference engine** — like Ollama, but lower-level, faster, and built for serious research and control.

It includes:
- A custom **binary model format** (`.qraf`) designed for mmap and zero-copy
- A **C++ inference engine** with Apple Accelerate, ARM NEON SIMD, Metal GPU, and GCD threading
- A **universal model converter** (HuggingFace, safetensors, PyTorch, GGUF -> QRAF)
- An **interactive CLI chatbot** with model browser and model store
- An **HTTP API server** with SSE streaming
- **LoRA adapter merging**, **KV cache quantization**, **speculative decoding**, **continuous batching**

```
                ┌─────────────────┐
                │  QRAF v0.3      │
                └────────┬────────┘
       ┌─────────────────┼─────────────────┐
       │                 │                 │
  ┌────▼────┐   ┌────────▼────────┐  ┌─────▼──────┐
  │   CLI   │   │   HTTP API      │  │  Converter │
  │  Chat   │   │  /generate      │  │  HF->QRAF  │
  │ Browse  │   │  /chat          │  │  GGUF->QRAF│
  └────┬────┘   └────────┬────────┘  └────────────┘
  ┌────▼─────────────────▼────┐
  │     Inference Engine      │
  │  Transformer + KV Cache   │
  │  LoRA + Speculative Dec.  │
  └───────────┬───────────────┘
       ┌──────┼──────┐
       │      │      │
  ┌────▼──┐ ┌─▼───┐ ┌▼───────┐
  │Accel. │ │NEON │ │ Metal  │
  │ BLAS  │ │SIMD │ │  GPU   │
  │ vDSP  │ │FMA  │ │Compute │
  └───────┘ └─────┘ └────────┘
```

---

## Text Quality Examples (Qwen2.5-3B-Instruct)

**Quantum Computing:**
> "Quantum computing is a type of technology that uses the principles of quantum mechanics to perform computations at an extremely high speed... leverage the phenomenon called superposition and entanglement."

**French Translation:**
> Input: "Translate to French: The weather is beautiful today, let's go to the park."
> Output: **"La meteorstats est belle aujourd'hui, allons au parc."**

**Creative Writing:**
> "Once upon a time, in the vast expanse of Cyberspace, there was an enigmatic robot named Zephyr. Zephyr lived in the serene landscape known as Pixelhaven..."

**ML for a 10-year-old:**
> "Machine learning is like teaching a computer to recognize pictures by showing it lots of pictures and what they are, and then the computer can do that on its own."

---

## Performance

Benchmarked on Apple Silicon with Accelerate + NEON + GCD:

### Inference Speed — All Tested Models

| Model | Params | Architecture | QRAF Size | tok/s | ms/token |
|:------|-------:|:-------------|----------:|------:|---------:|
| Pythia-70M | 70M | GPT-NeoX | 270 MB | **512** | 2.0 |
| DistilGPT2 | 82M | GPT-2 | 461 MB | **281** | 3.6 |
| Cerebras-GPT-111M | 111M | GPT-2 | 572 MB | **199** | 5.0 |
| SmolLM2-135M | 135M | LLaMA | 622 MB | **173** | 5.8 |
| GPT-2 | 124M | GPT-2 | 623 MB | **179** | 5.6 |
| Pythia-160M | 160M | GPT-NeoX | 620 MB | **179** | 5.6 |
| OPT-125M | 125M | OPT | 626 MB | **188** | 5.3 |
| Tiny StarCoder Py | 164M | StarCoder | 771 MB | **129** | 7.8 |
| Cerebras-GPT-256M | 256M | GPT-2 | 1.2 GB | **90** | 11.1 |
| SmolLM2-360M | 360M | LLaMA | 1.5 GB | **66** | 15.2 |
| GPT-2 Medium | 355M | GPT-2 | 1.5 GB | **59** | 16.9 |
| Pythia-410M | 410M | GPT-NeoX | 1.5 GB | **61** | 16.4 |
| CodeGen-350M | 350M | CodeGen | 1.3 GB | **69** | 14.5 |
| Qwen2.5-0.5B-Instruct | 494M | Qwen2 | 2.4 GB | **23** | 43.5 |
| GPT-2 Large | 774M | GPT-2 | 3.1 GB | **11** | 90.9 |
| TinyLlama-1.1B-Chat | 1.1B | LLaMA | 4.1 GB | **5** | 200 |
| Qwen2.5-1.5B-Instruct | 1.5B | Qwen2 | 6.6 GB | **3.5** | 286 |
| SmolLM2-1.7B-Instruct | 1.7B | LLaMA | 6.8 GB | **9.4** | 106 |
| **Qwen2.5-3B-Instruct** | **3B** | **Qwen2** | **13 GB** | **8.5** | **112** |

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

### Install from npm

```bash
npm install -g qraf-runtime
```

### Or build from source

```bash
git clone https://github.com/simonpierreboucher02/qraf.git
cd qraf
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

### Browse and download models

```bash
qraf browse
```

This opens the interactive **Model Store**:

```
  ╔═══════════════════════════════════════════════════════════╗
  ║              QRAF Model Store                            ║
  ╚═══════════════════════════════════════════════════════════╝

  [All]  Chat  Code  Base

   #  Model                          Params   QRAF  Type   Status
  ---------------------------------------------------------------------------
   1  *Qwen 2.5 1.5B Instruct         1.5B   7.6GB  chat
       Best small chat model, multilingual
   2  *SmolLM2 1.7B Instruct          1.7B   8.1GB  chat
       Excellent quality for size
   3  *Qwen 2.5 0.5B Instruct         0.5B   2.4GB  chat   ready
   ...21 curated models...

  [number] Download  [a]ll [c]hat [o]de [b]ase  [/query] Search HF  [q]uit
```

### Convert any model

```bash
# From HuggingFace (auto-downloads and auto-installs Python deps)
qraf convert Qwen/Qwen2.5-3B-Instruct -o models/qwen3b.qraf

# From local GGUF
qraf convert model.gguf -o models/model.qraf

# From safetensors
qraf convert weights.safetensors -o models/model.qraf
```

### Chat interactively

```bash
qraf chat
```

```
   ██████╗ ██████╗  █████╗ ███████╗
  ██╔═══██╗██╔══██╗██╔══██╗██╔════╝
  ██║   ██║██████╔╝███████║█████╗
  ╚██████╔╝██║  ██║██║  ██║██║
   ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     v0.3.0

  Select a model:
   [1]  qwen2.5-3b-instruct     13 GB
   [2]  smollm2-1.7b-instruct   6.8 GB
   [3]  qwen2.5-0.5b-instruct   2.4 GB

  > 1
  ✓ qwen2.5-3b-instruct (36L, h=2048, 16h)

  You > Write a haiku about AI

  Assistant > Artificial brain
  Thoughts beyond the silicon
  Boundless data streams
  [18 tokens, 2100ms, 8.6 tok/s]
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

| Command | Description |
|:--------|:------------|
| `qraf chat` | Interactive chatbot with model browser |
| `qraf browse` | Model Store — browse, download, convert models |
| `qraf run <model> --prompt <text>` | One-shot text generation |
| `qraf convert <source> -o <out>` | Universal model converter |
| `qraf list` | List local QRAF models |
| `qraf inspect <model>` | Show model architecture details |
| `qraf benchmark <model>` | Measure inference performance |
| `qraf serve` | Start HTTP API server |
| `qraf version` | Show version |

### Convert sources

```bash
qraf convert Qwen/Qwen2.5-3B-Instruct -o models/qwen.qraf    # HuggingFace
qraf convert ./local-model/ -o models/local.qraf                # Local dir
qraf convert model.gguf -o models/from-gguf.qraf                # GGUF
qraf convert weights.safetensors -o models/out.qraf             # Safetensors
qraf convert model.bin -o models/out.qraf                       # PyTorch
```

---

## Supported Architectures

| Architecture | Models | Inference | Features |
|:-------------|:-------|:---------:|:---------|
| **LLaMA** | SmolLM2, TinyLlama, LLaMA-2/3 | Full | RMSNorm, RoPE, SwiGLU |
| **Qwen2** | Qwen2.5, Qwen2.5-Coder, Qwen2.5-Math | Full | RMSNorm, RoPE, SwiGLU, attn bias |
| **GPT-2** | GPT-2, DistilGPT2, DialoGPT, Cerebras-GPT, LaMini | Full | LayerNorm, learned pos, GELU |
| **GPT-NeoX** | Pythia (70M to 410M) | Full | LayerNorm, RoPE, parallel attn+MLP |
| **OPT** | OPT-125M | Full | LayerNorm, learned pos, ReLU, project_in/out |
| **CodeGen** | CodeGen-350M-Mono | Full | LayerNorm, RoPE, parallel attn, GELU |
| **StarCoder** | Tiny StarCoder Py | Full | LayerNorm, learned pos, MQA, GELU |

---

## Model Library

### Recommended Models (best quality-to-speed ratio)

| Model | Params | QRAF Size | Speed | Quality | Use Case |
|:------|-------:|----------:|------:|:--------|:---------|
| **Qwen2.5-3B-Instruct** | 3B | 13 GB | 8.5 tok/s | Excellent | Chat, Q&A, translation, code |
| **SmolLM2-1.7B-Instruct** | 1.7B | 6.8 GB | 9.4 tok/s | Good | Chat, creative writing |
| **Qwen2.5-1.5B-Instruct** | 1.5B | 6.6 GB | 3.5 tok/s | Good | Chat, multilingual |
| **Qwen2.5-Coder-1.5B-Instruct** | 1.5B | 7.6 GB | 3.5 tok/s | Good | Code generation |
| **Qwen2.5-0.5B-Instruct** | 494M | 2.4 GB | 23 tok/s | Medium | Fast chat |
| **SmolLM2-360M-Instruct** | 360M | 1.5 GB | 41 tok/s | Basic | Ultra-fast |

### All 26 Tested Models

| # | Model | Params | Arch | QRAF Size | tok/s | Category |
|--:|:------|-------:|:-----|----------:|------:|:---------|
| 1 | Pythia-70M | 70M | GPT-NeoX | 270 MB | 512 | Base |
| 2 | DistilGPT2 | 82M | GPT-2 | 461 MB | 281 | Base |
| 3 | Cerebras-GPT-111M | 111M | GPT-2 | 572 MB | 199 | Base |
| 4 | SmolLM2-135M-Instruct | 135M | LLaMA | 622 MB | 63 | Instruct |
| 5 | GPT-2 | 124M | GPT-2 | 623 MB | 179 | Base |
| 6 | Pythia-160M | 160M | GPT-NeoX | 620 MB | 179 | Base |
| 7 | OPT-125M | 125M | OPT | 626 MB | 188 | Base |
| 8 | DialoGPT-Small | 124M | GPT-2 | 623 MB | 178 | Dialogue |
| 9 | LaMini-GPT-124M | 124M | GPT-2 | 623 MB | 185 | Base |
| 10 | Tiny StarCoder Py | 164M | StarCoder | 771 MB | 129 | Code |
| 11 | Cerebras-GPT-256M | 256M | GPT-2 | 1.2 GB | 90 | Base |
| 12 | SmolLM2-360M-Instruct | 360M | LLaMA | 1.5 GB | 41 | Instruct |
| 13 | GPT-2 Medium | 355M | GPT-2 | 1.5 GB | 59 | Base |
| 14 | DialoGPT-Medium | 355M | GPT-2 | 1.5 GB | 60 | Dialogue |
| 15 | Pythia-410M | 410M | GPT-NeoX | 1.5 GB | 61 | Base |
| 16 | CodeGen-350M-Mono | 350M | CodeGen | 1.3 GB | 69 | Code |
| 17 | Qwen2.5-0.5B-Instruct | 494M | Qwen2 | 2.4 GB | 23 | Instruct |
| 18 | Qwen2.5-Coder-0.5B-Instruct | 494M | Qwen2 | 2.4 GB | 30 | Code |
| 19 | GPT-2 Large | 774M | GPT-2 | 3.1 GB | 11 | Base |
| 20 | TinyLlama-1.1B-Chat | 1.1B | LLaMA | 4.1 GB | 5 | Chat |
| 21 | Qwen2.5-1.5B-Instruct | 1.5B | Qwen2 | 6.6 GB | 3.5 | Instruct |
| 22 | SmolLM2-1.7B-Instruct | 1.7B | LLaMA | 6.8 GB | 9.4 | Instruct |
| 23 | **Qwen2.5-3B-Instruct** | **3B** | **Qwen2** | **13 GB** | **8.5** | **Instruct** |
| 24+ | Base variants of above | ... | ... | ... | ... | Base |

---

## QRAF Format Specification

### Binary Layout

```
┌──────────────────────────────────┐  offset 0
│  Header (128 bytes)              │
├──────────────────────────────────┤
│  Model Config Block              │  key-value pairs
├──────────────────────────────────┤
│  Tokenizer Block                 │  vocab + BPE merges + specials
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

### Key Design Principles

- **mmap-first**: Models are memory-mapped, not copied. Load time is near-instant.
- **64-byte alignment**: All tensor data aligned for SIMD/DMA access.
- **Zero-copy tensor views**: Direct pointers into mapped memory.
- **FNV-1a hashing**: Fast tensor lookup by name hash.
- **BPE merges stored**: Full tokenizer with merge table for correct encoding.
- **Quantization-aware**: Native Q4, Q8, FP16 support in the format.

---

## Architecture

### Engine Stack

```
┌─────────────────────────────────────────────┐
│  CLI (chat, browse) / HTTP Server           │
├─────────────────────────────────────────────┤
│  Inference Engine                           │
│  Speculative Decoding / Continuous Batching │
├─────────────────────────────────────────────┤
│  Transformer (multi-arch forward pass)      │
│  LLaMA | GPT-2 | GPT-NeoX | OPT | CodeGen │
├──────────┬──────────┬───────────────────────┤
│ Attention│   MLP    │  Norms / Pos Embed    │
│ GQA/MQA  │ SwiGLU   │  RMSNorm / LayerNorm │
│ KV Cache │ Standard │  RoPE / Learned       │
├──────────┴──────────┴───────────────────────┤
│  LoRA Adapter Merging (W + alpha*B@A)       │
├─────────────────────────────────────────────┤
│  Backend Dispatch                           │
├──────────┬──────────┬──────────┬────────────┤
│ Accelerate│  NEON   │  Metal   │   GCD      │
│ cblas     │  FMA    │  GPU     │  Threading │
│ vDSP      │  SIMD   │ Compute  │ dispatch   │
├──────────┴──────────┴──────────┴────────────┤
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
| Q8 dot | NEON int8->f32 widening + FMA | 16 values per iteration |
| Q4 dot | NEON nibble unpack `vzipq_s8` | 32 values per iteration |
| GPU Compute | Metal compute shaders | MatVec, softmax, RMSNorm, SiLU on GPU |
| Threading | GCD `dispatch_apply` | `QOS_CLASS_USER_INTERACTIVE` -> P-cores |

---

## Advanced Features

### LoRA Adapter Merging
```cpp
LoraManager lora;
lora.load_adapter("adapter.qraf", "my-lora", /*alpha=*/16.0f);
// Merges W' = W + (alpha/rank) * B @ A at runtime
```

### KV Cache Quantization
Stores K/V in int8 with per-head scales. **4x memory reduction** for long contexts.

### Speculative Decoding
Uses a fast draft model to generate K candidates, main model verifies in parallel. **~2-3x speedup** on compatible model pairs.

### Continuous Batching
Multiple concurrent requests with per-request KV caches, dynamic slot allocation, thread-safe queue.

---

## HTTP API

```bash
qraf serve --port 8080 --dir models
```

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/api/generate` | POST | Generate text (with optional streaming) |
| `/api/chat` | POST | Chat completion |
| `/api/models` | GET | List available models |
| `/health` | GET | Health check |

```bash
# Non-streaming
curl -X POST http://localhost:8080/api/generate \
  -d '{"model": "qwen2.5-3b-instruct", "prompt": "Hello!", "max_tokens": 100}'

# Streaming (SSE)
curl -X POST http://localhost:8080/api/generate \
  -d '{"model": "qwen2.5-3b-instruct", "prompt": "Hello!", "stream": true}'
```

---

## Building

### Requirements

- C++17 compiler (Clang 14+ or GCC 11+)
- CMake 3.16+
- Python 3.8+ with pip (for converter — deps auto-install on first use)

### macOS (Apple Silicon)

```bash
brew install cmake
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
| `QRAF_USE_METAL` | ON | Metal GPU compute shaders |
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

## Project Structure

```
qraf/
├── include/
│   ├── core/          types.h, logging.h, error.h
│   ├── tensor/        tensor.h, quantize.h
│   ├── qraf/          format.h, loader.h, writer.h
│   ├── nn/            ops.h, attention.h, transformer.h,
│   │                  backend.h, simd_neon.h, threading.h,
│   │                  metal_backend.h, lora.h, kv_cache_quant.h
│   ├── runtime/       tokenizer.h, inference.h, model_manager.h,
│   │                  speculative.h, batch_scheduler.h
│   ├── sampling/      sampler.h
│   ├── cli/           commands.h
│   └── server/        http_server.h
├── src/
│   ├── nn/            ops, attention, transformer, backend,
│   │                  simd_neon, threading, metal_backend, lora
│   ├── runtime/       tokenizer, inference, model_manager, speculative
│   ├── cli/           main, commands, chat, model_browser
│   └── server/        http_server
├── tools/
│   ├── qraf_convert.py      Universal converter (HF/GGUF/safetensors/PyTorch)
│   └── dummy_model.cpp      Test model generator
├── npm/                     npm package wrapper (cli.js, postinstall.js)
├── tests/                   28 unit tests
├── benchmarks/              Performance measurements
└── CMakeLists.txt
```

---

## Quantization

| Format | Bits | Block Size | Description |
|:-------|-----:|-----------:|:------------|
| F32 | 32 | - | Full precision |
| F16 | 16 | - | Half precision |
| BF16 | 16 | - | Brain float |
| Q8_0 | 8 | 32 | Per-block scale, NEON dot product |
| Q4_0 | 4 | 32 | Per-block scale, packed nibbles |

Quantized dot products run directly on compressed data via NEON — no full dequantization needed.

---

## Roadmap

- [x] QRAF binary format specification
- [x] mmap loader with tensor index
- [x] Decoder-only transformer with KV cache
- [x] GQA / MQA (Grouped/Multi Query Attention)
- [x] BPE tokenizer with full merge table (50k+ merges)
- [x] Top-k, top-p, temperature, repetition penalty sampling
- [x] Apple Accelerate backend (`cblas_sgemv`, `vDSP`, `vvexpf`)
- [x] ARM NEON SIMD kernels (dot, matvec, RMSNorm, softmax, SiLU, RoPE)
- [x] GCD multi-threading (`dispatch_apply`)
- [x] Metal GPU compute shaders
- [x] 7 model architectures (LLaMA, Qwen2, GPT-2, GPT-NeoX, OPT, CodeGen, StarCoder)
- [x] Attention bias support (Qwen-style)
- [x] LayerNorm + learned position embeddings (GPT-2/OPT)
- [x] Parallel attention+MLP (GPT-NeoX/CodeGen)
- [x] Universal converter (HuggingFace, safetensors, PyTorch, GGUF)
- [x] Interactive CLI chatbot with chat templates (ChatML)
- [x] Model Store browser with curated catalog
- [x] HTTP API with SSE streaming
- [x] LoRA adapter merging at runtime
- [x] KV cache quantization (int8)
- [x] Speculative decoding
- [x] Continuous batching
- [x] 26 pre-tested model conversions (70M to 3B)
- [x] npm package (`qraf-runtime`)
- [ ] Weight quantization in converter (Q4/Q8 QRAF files)
- [ ] Flash Attention
- [ ] CUDA backend
- [ ] Phi-3 architecture support
- [ ] Gemma architecture support

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with precision by **Simon-Pierre Boucher** and **Claude Opus 4.6**.
