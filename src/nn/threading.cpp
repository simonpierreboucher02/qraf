#include "nn/threading.h"
#include "core/logging.h"
#include <thread>
#include <algorithm>

#ifdef QRAF_THREADING
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#define QRAF_HAS_GCD 1
#else
#define QRAF_HAS_GCD 0
#endif
#endif

namespace qraf {

static int g_num_threads = 1;

void threading_init(int num_threads) {
    if (num_threads <= 0) {
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        // On Apple Silicon, prefer performance cores
        // M1: 4P+4E, M1 Pro: 6-8P+2E, M2: 4P+4E, M3 Pro: 6P+4E, M4: 4P+6E
        // Use all available cores, GCD handles P/E scheduling via QoS
        g_num_threads = std::max(1, std::min(hw, 12));
    } else {
        g_num_threads = num_threads;
    }
    log::info("Threading initialized: %d threads (hw=%d)",
              g_num_threads, (int)std::thread::hardware_concurrency());
}

int threading_num_threads() {
    return g_num_threads;
}

void parallel_for(int begin, int end, int grain_size,
                  const std::function<void(int start, int stop)>& fn) {
    int range = end - begin;
    if (range <= 0) return;

    // Single-threaded for small work
    if (range <= grain_size || g_num_threads <= 1) {
        fn(begin, end);
        return;
    }

#if defined(QRAF_THREADING) && QRAF_HAS_GCD
    int nchunks = std::min(g_num_threads, (range + grain_size - 1) / grain_size);
    int chunk_size = (range + nchunks - 1) / nchunks;

    dispatch_apply(static_cast<size_t>(nchunks),
                   dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                   ^(size_t idx) {
        int start = begin + static_cast<int>(idx) * chunk_size;
        int stop = std::min(start + chunk_size, end);
        if (start < end) fn(start, stop);
    });
#else
    // Fallback: use std::thread
    int nchunks = std::min(g_num_threads, (range + grain_size - 1) / grain_size);
    int chunk_size = (range + nchunks - 1) / nchunks;
    std::vector<std::thread> threads;
    threads.reserve(nchunks - 1);

    for (int t = 0; t < nchunks; t++) {
        int start = begin + t * chunk_size;
        int stop = std::min(start + chunk_size, end);
        if (start >= end) break;

        if (t == nchunks - 1) {
            fn(start, stop);  // Last chunk runs on current thread
        } else {
            threads.emplace_back([&fn, start, stop]() { fn(start, stop); });
        }
    }
    for (auto& t : threads) t.join();
#endif
}

} // namespace qraf
