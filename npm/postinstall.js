#!/usr/bin/env node
/**
 * QRAF post-install: builds the C++ runtime and installs Python deps.
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
  try { execSync('cmake --version', { stdio: 'ignore' }); return true; }
  catch { return false; }
}

function hasPython() {
  try { execSync('python3 --version', { stdio: 'ignore' }); return true; }
  catch { return false; }
}

function main() {
  console.log('\n  ==============================');
  console.log('  QRAF Runtime — Post-install');
  console.log('  ==============================\n');

  // 1. Create models directory
  const modelsDir = path.join(ROOT, 'models');
  if (!fs.existsSync(modelsDir)) {
    fs.mkdirSync(modelsDir, { recursive: true });
  }

  // 2. Install Python dependencies (required for converter)
  if (hasPython()) {
    console.log('  [1/3] Installing Python dependencies...');
    run('python3 -m pip install --quiet --user torch transformers safetensors sentencepiece gguf numpy 2>/dev/null || python3 -m pip install --quiet torch transformers safetensors sentencepiece gguf numpy');
  } else {
    console.log('  [1/3] Python3 not found. Converter requires Python 3.8+');
    console.log('        Install Python: https://python.org/downloads/');
  }

  // 3. Build C++ runtime
  if (!hasCmake()) {
    console.log('  [2/3] cmake not found — skipping C++ build.');
    console.log('        Install cmake: brew install cmake (macOS) / apt install cmake (Linux)');
    console.log('        The converter (qraf convert) works without the C++ build.\n');
  } else {
    if (!fs.existsSync(BUILD_DIR)) {
      fs.mkdirSync(BUILD_DIR, { recursive: true });
    }

    console.log('  [2/3] Configuring C++ build...');
    if (!run('cmake .. -DCMAKE_BUILD_TYPE=Release -DQRAF_BUILD_TESTS=OFF -DQRAF_BUILD_BENCHMARKS=OFF', { cwd: BUILD_DIR })) {
      console.log('        CMake failed. Converter still works.\n');
    } else {
      const cores = require('os').cpus().length;
      console.log(`  [3/3] Building C++ runtime (${cores} cores)...`);
      if (!run(`make -j${cores}`, { cwd: BUILD_DIR })) {
        console.log('        Build failed. Converter still works.\n');
      } else {
        console.log('\n  C++ runtime built successfully!');
      }
    }
  }

  console.log('\n  ==============================');
  console.log('  QRAF installed! Quick start:');
  console.log('  ==============================');
  console.log('');
  console.log('  1. Convert a model:');
  console.log('     qraf convert Qwen/Qwen2.5-0.5B-Instruct -o models/qwen.qraf');
  console.log('');
  console.log('  2. Chat:');
  console.log('     qraf chat');
  console.log('');
}

main();
