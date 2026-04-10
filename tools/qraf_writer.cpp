// Tool: QRAF Writer CLI
// Converts raw weight files into QRAF format (placeholder for future use)

#include "qraf/writer.h"
#include "core/logging.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: qraf-writer <config.json> -o <output.qraf>\n";
        std::cout << "\nConverts model weights to QRAF format.\n";
        std::cout << "Currently a placeholder — use qraf-dummy-model for testing.\n";
        return 0;
    }

    std::cerr << "Full conversion pipeline not yet implemented.\n";
    std::cerr << "Use qraf-dummy-model to generate test models.\n";
    return 1;
}
