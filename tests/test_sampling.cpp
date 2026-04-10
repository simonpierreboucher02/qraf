// Tests for sampling system

#include "sampling/sampler.h"

using namespace qraf;

TEST(sampler_argmax) {
    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
    ASSERT_EQ(Sampler::argmax(logits, 4), 1u);
    return true;
}

TEST(sampler_deterministic) {
    SamplingConfig config;
    config.deterministic = true;
    Sampler sampler(config);

    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
    auto t = Tensor::from_data({4}, DType::F32, logits, sizeof(logits));

    // Deterministic mode should always return argmax
    u32 token = sampler.sample(t);
    ASSERT_EQ(token, 1u);
    return true;
}

TEST(sampler_temperature_zero) {
    SamplingConfig config;
    config.temperature = 0.0f;
    Sampler sampler(config);

    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
    u32 token = sampler.sample(logits, 4);
    ASSERT_EQ(token, 1u);
    return true;
}

TEST(sampler_repetition_penalty) {
    SamplingConfig config;
    config.repetition_penalty = 2.0f;
    Sampler sampler(config);

    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f};
    std::vector<u32> recent = {1};  // penalize token 1

    sampler.apply_repetition_penalty(logits, 4, recent);

    // Token 1 (positive logit) should be reduced
    ASSERT_TRUE(logits[1] < 5.0f);
    ASSERT_NEAR(logits[1], 2.5f, 1e-5f);
    // Token 0 should be unchanged
    ASSERT_NEAR(logits[0], 1.0f, 1e-5f);
    return true;
}
