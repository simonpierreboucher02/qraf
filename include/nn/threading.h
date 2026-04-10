#pragma once

#include <functional>

namespace qraf {

void threading_init(int num_threads = 0);
int threading_num_threads();

// Parallel for: splits [begin, end) across threads
// grain_size: minimum chunk size per thread to avoid overhead for small work
void parallel_for(int begin, int end, int grain_size,
                  const std::function<void(int start, int stop)>& fn);

} // namespace qraf
