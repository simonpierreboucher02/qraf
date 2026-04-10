#!/usr/bin/env node
/**
 * QRAF post-install: builds the C++ runtime from source.
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const BUILD_DIR = path.join(ROOT, 'build');

function run(cmd, opts = {}) {
  console.log(`  > ${cmd}`);
  try {
    execSync(cmd, { stdio: 'inherit', cwd: ROOT, ...opts });
    return true;
  } catch (e) {
    return false;
  }
}

function hasCmake() {
  try {
    execSync('cmake --version', { stdio: 'ignore' });
    return true;
  } catch { return false; }
}

function main() {
  console.log('\n  QRAF — Building C++ runtime...\n');

  // Check cmake
  if (!hasCmake()) {
    console.log('  cmake not found. Install it:');
    console.log('    macOS:  brew install cmake');
    console.log('    Ubuntu: sudo apt install cmake');
    console.log('');
    console.log('  Then re-run: npm rebuild qraf');
    process.exit(0); // Don't fail install — converter still works via Python
  }

  // Create build directory
  if (!fs.existsSync(BUILD_DIR)) {
    fs.mkdirSync(BUILD_DIR, { recursive: true });
  }

  // Configure
  console.log('  [1/2] Configuring...');
  if (!run(`cmake .. -DCMAKE_BUILD_TYPE=Release -DQRAF_BUILD_TESTS=OFF -DQRAF_BUILD_BENCHMARKS=OFF`, { cwd: BUILD_DIR })) {
    console.log('\n  CMake configure failed. Skipping C++ build.');
    console.log('  Python converter (qraf convert) will still work.\n');
    process.exit(0);
  }

  // Build
  const cores = require('os').cpus().length;
  console.log(`  [2/2] Building (${cores} cores)...`);
  if (!run(`make -j${cores}`, { cwd: BUILD_DIR })) {
    console.log('\n  Build failed. Python converter will still work.\n');
    process.exit(0);
  }

  console.log('\n  QRAF runtime built successfully!\n');

  // Create models directory
  const modelsDir = path.join(ROOT, 'models');
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
  }

  // Install Python dependencies for converter
  console.log('  Installing Python dependencies for converter...');
  run('pip3 install torch transformers safetensors sentencepiece gguf 2>/dev/null || true');

  console.log('\n  Ready! Run: qraf chat\n');
}

main();
