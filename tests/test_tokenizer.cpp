// Tests for tokenizer

#include "runtime/tokenizer.h"

using namespace qraf;

TEST(tokenizer_basic) {
    Tokenizer tok;

    // Build a simple vocab manually
    // We'll use load_vocab_file format via temporary file
    // For now, test the API with an empty tokenizer
    ASSERT_EQ(tok.vocab_size(), 0u);
    return true;
}

TEST(tokenizer_encode_decode) {
    // This test verifies that the tokenizer round-trips through
    // QRAF format correctly. Full test requires a written QRAF file.
    // Placeholder for integration testing.
    return true;
}
