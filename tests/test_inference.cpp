// Tests for inference pipeline (integration test)

#include "nn/attention.h"

using namespace qraf;

TEST(kv_cache_init) {
    KVCache cache;
    cache.init(2, 4, 16, 128);

    ASSERT_EQ(cache.num_layers, 2);
    ASSERT_EQ(cache.num_kv_heads, 4);
    ASSERT_EQ(cache.head_dim, 16);
    ASSERT_EQ(cache.max_seq_len, 128);
    ASSERT_EQ(cache.current_len, 0);
    return true;
}

TEST(kv_cache_store_load) {
    KVCache cache;
    cache.init(1, 2, 4, 16);  // 1 layer, 2 kv_heads, head_dim=4

    // Store at position 0
    float k[] = {1, 2, 3, 4, 5, 6, 7, 8};  // 2 heads * 4 dim
    float v[] = {9, 10, 11, 12, 13, 14, 15, 16};
    cache.store(0, 0, k, v);

    // Read back
    const float* k_out = cache.key_at(0, 0);
    const float* v_out = cache.value_at(0, 0);

    ASSERT_NEAR(k_out[0], 1.0f, 1e-5f);
    ASSERT_NEAR(k_out[7], 8.0f, 1e-5f);
    ASSERT_NEAR(v_out[0], 9.0f, 1e-5f);
    ASSERT_NEAR(v_out[7], 16.0f, 1e-5f);
    return true;
}
