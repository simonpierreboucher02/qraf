// QRAF Benchmarks
// Measures core operation performance

#include "tensor/tensor.h"
#include "tensor/quantize.h"
#include "nn/ops.h"
#include "core/logging.h"

#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>

using namespace qraf;

template<typename Fn>
double bench(const std::string& name, int iterations, Fn fn) {
    // Warm-up
    fn();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        fn();
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_iter = total_ms / iterations;

    std::cout << std::left << std::setw(40) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << per_iter << " ms/iter"
              << std::setw(10) << std::setprecision(1)
              << (1000.0 / per_iter) << " iter/s"
              << "\n";

    return per_iter;
}

int main() {
    std::cout << "=== QRAF Benchmarks ===\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // ─── MatVec benchmark ───
    {
        int sizes[] = {128, 512, 2048, 4096};
        std::cout << "--- Matrix-Vector Multiply ---\n";
        for (int n : sizes) {
            auto W = Tensor::zeros({static_cast<u32>(n), static_cast<u32>(n)});
            auto x = Tensor::zeros({static_cast<u32>(n)});
            for (size_t i = 0; i < W.numel(); i++) W.data_f32()[i] = dist(rng);
            for (size_t i = 0; i < x.numel(); i++) x.data_f32()[i] = dist(rng);

            std::string name = "matvec " + std::to_string(n) + "x" + std::to_string(n);
            int iters = (n <= 512) ? 1000 : 100;
            bench(name, iters, [&]() { ops::matvec(W, x); });
        }
        std::cout << "\n";
    }

    // ─── Softmax benchmark ───
    {
        std::cout << "--- Softmax ---\n";
        int sizes[] = {256, 1024, 32000, 128000};
        for (int n : sizes) {
            std::vector<float> data(n);
            for (auto& v : data) v = dist(rng);

            std::string name = "softmax " + std::to_string(n);
            bench(name, 10000, [&]() {
                std::vector<float> copy = data;
                ops::softmax_inplace(copy.data(), n);
            });
        }
        std::cout << "\n";
    }

    // ─── RMSNorm benchmark ───
    {
        std::cout << "--- RMSNorm ---\n";
        int sizes[] = {512, 2048, 4096, 8192};
        for (int n : sizes) {
            std::vector<float> x(n), w(n);
            for (auto& v : x) v = dist(rng);
            for (auto& v : w) v = dist(rng);

            std::string name = "rms_norm " + std::to_string(n);
            bench(name, 50000, [&]() {
                std::vector<float> copy = x;
                ops::rms_norm_inplace(copy.data(), w.data(), n);
            });
        }
        std::cout << "\n";
    }

    // ─── Quantization benchmark ───
    {
        std::cout << "--- Quantization ---\n";
        int n = 4096;
        std::vector<float> data(n);
        for (auto& v : data) v = dist(rng);

        bench("quantize_q8_0 4096", 10000, [&]() {
            quantize_q8_0(data.data(), data.size());
        });

        bench("quantize_q4_0 4096", 10000, [&]() {
            quantize_q4_0(data.data(), data.size());
        });

        auto q8 = quantize_q8_0(data.data(), data.size());
        bench("dequant_q8_0 4096", 10000, [&]() {
            std::vector<float> out(n);
            size_t bs = 32;
            size_t blocks = (n + bs - 1) / bs;
            const u8* ptr = q8.data();
            for (size_t b = 0; b < blocks; b++) {
                dequantize_block_q8_0(ptr, out.data() + b * bs, 32);
                ptr += sizeof(float) + 32;
            }
        });
        std::cout << "\n";
    }

    // ─── Quantized dot product benchmark ───
    {
        std::cout << "--- Quantized Dot Product ---\n";
        int n = 4096;
        std::vector<float> a(n), b(n);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        auto q8 = quantize_q8_0(a.data(), a.size());
        bench("dot_q8_0_f32 4096", 50000, [&]() {
            dot_q8_0_f32(q8.data(), b.data(), n);
        });

        auto q4 = quantize_q4_0(a.data(), a.size());
        bench("dot_q4_0_f32 4096", 50000, [&]() {
            dot_q4_0_f32(q4.data(), b.data(), n);
        });
    }

    std::cout << "\n=== Done ===\n";
    return 0;
}
