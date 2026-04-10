// Tests for QRAF format structures and writer/loader

#include "qraf/format.h"
#include "qraf/writer.h"
#include "qraf/loader.h"
#include <cstring>
#include <filesystem>

using namespace qraf;

TEST(qraf_header_size) {
    ASSERT_EQ(sizeof(QrafHeader), 128u);
    return true;
}

TEST(qraf_tensor_meta_size) {
    ASSERT_EQ(sizeof(TensorMeta), 80u);
    return true;
}

TEST(qraf_quant_scheme_size) {
    ASSERT_EQ(sizeof(QuantScheme), 32u);
    return true;
}

TEST(qraf_fnv1a_hash) {
    u64 h1 = fnv1a_hash("hello");
    u64 h2 = fnv1a_hash("hello");
    u64 h3 = fnv1a_hash("world");

    ASSERT_EQ(h1, h2);
    ASSERT_TRUE(h1 != h3);
    return true;
}

TEST(qraf_align_offset) {
    ASSERT_EQ(align_offset(0, 64), 0u);
    ASSERT_EQ(align_offset(1, 64), 64u);
    ASSERT_EQ(align_offset(63, 64), 64u);
    ASSERT_EQ(align_offset(64, 64), 64u);
    ASSERT_EQ(align_offset(65, 64), 128u);
    return true;
}

TEST(qraf_write_and_load) {
    // Create a minimal QRAF file
    QrafWriter writer;
    writer.set_config("vocab_size", 4u);
    writer.set_config("hidden_size", 8u);
    writer.set_config("num_layers", 1u);
    writer.set_config("num_heads", 2u);

    // Add a small tensor
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
    writer.add_tensor("test.weight", {2, 4}, DType::F32,
                      data.data(), data.size() * sizeof(float));

    std::string path = "/tmp/test_qraf_rw.qraf";
    ASSERT_TRUE(writer.write(path));

    // Load it back
    QrafModel model;
    ASSERT_TRUE(model.load(path));

    // Verify
    ASSERT_TRUE(model.has_tensor("test.weight"));
    auto tv = model.get_tensor("test.weight");
    ASSERT_EQ(tv.shape.size(), 2u);
    ASSERT_EQ(tv.shape[0], 2u);
    ASSERT_EQ(tv.shape[1], 4u);
    ASSERT_TRUE(tv.dtype == DType::F32);

    const float* loaded = static_cast<const float*>(tv.data);
    ASSERT_NEAR(loaded[0], 1.0f, 1e-6f);
    ASSERT_NEAR(loaded[7], 8.0f, 1e-6f);

    model.unload();
    std::filesystem::remove(path);
    return true;
}
