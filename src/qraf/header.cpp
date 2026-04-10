#include "qraf/format.h"
#include "core/error.h"
#include "core/logging.h"
#include <cstring>

namespace qraf {

bool validate_header(const QrafHeader& header, u64 actual_file_size) {
    if (header.magic != QRAF_MAGIC) {
        log::error("Invalid magic: 0x%08X (expected 0x%08X)", header.magic, QRAF_MAGIC);
        return false;
    }

    if (header.version != QRAF_VERSION) {
        log::error("Unsupported version: %u (expected %u)", header.version, QRAF_VERSION);
        return false;
    }

    if (header.file_size != actual_file_size) {
        log::error("File size mismatch: header says %llu, actual %llu",
                   (unsigned long long)header.file_size,
                   (unsigned long long)actual_file_size);
        return false;
    }

    // Validate offsets are within file bounds
    if (header.config_offset + header.config_size > actual_file_size) {
        log::error("Config block extends past file end");
        return false;
    }

    if (header.tensor_dir_offset + header.tensor_dir_size > actual_file_size) {
        log::error("Tensor directory extends past file end");
        return false;
    }

    if (header.data_offset > actual_file_size) {
        log::error("Data offset past file end");
        return false;
    }

    // Validate tensor directory size matches num_tensors
    u64 expected_dir_size = static_cast<u64>(header.num_tensors) * sizeof(TensorMeta);
    if (header.tensor_dir_size != expected_dir_size) {
        log::error("Tensor directory size %llu != expected %llu (%u tensors * %zu bytes)",
                   (unsigned long long)header.tensor_dir_size,
                   (unsigned long long)expected_dir_size,
                   header.num_tensors, sizeof(TensorMeta));
        return false;
    }

    log::info("QRAF header valid: v%u, %u tensors, %llu bytes",
              header.version, header.num_tensors,
              (unsigned long long)header.file_size);
    return true;
}

} // namespace qraf
