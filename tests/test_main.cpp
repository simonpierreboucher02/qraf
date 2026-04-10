// Minimal test framework (no external dependencies)

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

struct TestCase {
    std::string name;
    std::function<bool()> fn;
};

static std::vector<TestCase> g_tests;

#define TEST(name) \
    static bool test_##name(); \
    static bool _reg_##name = (g_tests.push_back({#name, test_##name}), true); \
    static bool test_##name()

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) { \
        std::cerr << "  FAIL: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    } } while(0)

#define ASSERT_EQ(a, b) \
    do { if ((a) != (b)) { \
        std::cerr << "  FAIL: " << #a << " == " << #b << " (" << (a) << " != " << (b) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    } } while(0)

#define ASSERT_NEAR(a, b, eps) \
    do { if (std::fabs((a) - (b)) > (eps)) { \
        std::cerr << "  FAIL: |" << #a << " - " << #b << "| <= " << (eps) << " (" << (a) << " vs " << (b) << ") at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    } } while(0)

// Include all test files
#include "test_tensor.cpp"
#include "test_qraf_format.cpp"
#include "test_quantize.cpp"
#include "test_tokenizer.cpp"
#include "test_nn_ops.cpp"
#include "test_sampling.cpp"
#include "test_inference.cpp"

int main() {
    int passed = 0, failed = 0;

    std::cout << "Running " << g_tests.size() << " tests...\n\n";

    for (const auto& tc : g_tests) {
        std::cout << "  " << tc.name << "... " << std::flush;
        try {
            if (tc.fn()) {
                std::cout << "PASS\n";
                passed++;
            } else {
                std::cout << "FAIL\n";
                failed++;
            }
        } catch (const std::exception& e) {
            std::cout << "EXCEPTION: " << e.what() << "\n";
            failed++;
        }
    }

    std::cout << "\n" << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
