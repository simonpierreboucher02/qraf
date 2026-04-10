#!/usr/bin/env node
/**
 * QRAF CLI wrapper — routes commands to C++ binary or Python converter.
 */
const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const BUILD_DIR = path.join(ROOT, 'build');
const TOOLS_DIR = path.join(ROOT, 'tools');
const MODELS_DIR = path.join(ROOT, 'models');

const CLI_BIN = path.join(BUILD_DIR, 'qraf-cli');
const SERVER_BIN = path.join(BUILD_DIR, 'qraf-server');
const CONVERTER = path.join(TOOLS_DIR, 'qraf_convert.py');

const args = process.argv.slice(2);
const command = args[0] || 'help';

// ─── Check if C++ binary exists ───
function hasBinary() {
  return fs.existsSync(CLI_BIN);
}

// ─── Run C++ CLI ───
function runCli(extraArgs = []) {
  if (!hasBinary()) {
    console.error('QRAF runtime not built. Run: npm run build');
    console.error('Or build manually: mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)');
    process.exit(1);
  }

  const allArgs = args.concat(extraArgs);
  const proc = spawn(CLI_BIN, allArgs, { stdio: 'inherit' });
  proc.on('close', (code) => process.exit(code || 0));
}

// ─── Run converter (Python) ───
function runConvert() {
  const convertArgs = args.slice(1); // remove 'convert'
  if (convertArgs.length === 0) {
    console.log('Usage: qraf convert <source> -o <output.qraf>');
    console.log('');
    console.log('Sources:');
    console.log('  HuggingFace model name    qraf convert Qwen/Qwen2.5-0.5B -o model.qraf');
    console.log('  Local directory            qraf convert ./my-model/ -o model.qraf');
    console.log('  GGUF file                  qraf convert model.gguf -o model.qraf');
    console.log('  Safetensors file           qraf convert weights.safetensors -o model.qraf');
    console.log('  PyTorch file               qraf convert model.bin -o model.qraf');
    process.exit(0);
  }

  // Check if -o is provided, add default if not
  if (!convertArgs.includes('-o') && !convertArgs.includes('--output')) {
    // Auto-generate output name
    let source = convertArgs[0];
    let name = source.split('/').pop().replace(/[.:]/g, '-');
    let output = path.join(MODELS_DIR, `${name}.qraf`);

    // Create models dir
    if (!fs.existsSync(MODELS_DIR)) {
      fs.mkdirSync(MODELS_DIR, { recursive: true });
    }

    convertArgs.push('-o', output);
    console.log(`Output: ${output}`);
  }

  // Check Python3 is available
  try {
    execSync('python3 --version', { stdio: 'ignore' });
  } catch {
    console.error('Error: Python 3 is required for model conversion.');
    console.error('Install it: https://python.org/downloads/');
    process.exit(1);
  }

  // Create models dir if -o target needs it
  const oIdx = convertArgs.indexOf('-o');
  if (oIdx >= 0 && oIdx + 1 < convertArgs.length) {
    const outDir = path.dirname(convertArgs[oIdx + 1]);
    if (outDir && !fs.existsSync(outDir)) {
      fs.mkdirSync(outDir, { recursive: true });
    }
  }

  const proc = spawn('python3', [CONVERTER, ...convertArgs], { stdio: 'inherit' });
  proc.on('error', (err) => {
    if (err.code === 'ENOENT') {
      console.error('Error: python3 not found in PATH');
      console.error('Install Python 3: https://python.org/downloads/');
    } else {
      console.error(`Error: ${err.message}`);
    }
    process.exit(1);
  });
  proc.on('close', (code) => {
    if (code === 0) {
      console.log('\nConversion complete! Run: qraf chat');
    }
    process.exit(code || 0);
  });
}

// ─── Run server ───
function runServer() {
  if (!fs.existsSync(SERVER_BIN)) {
    console.error('Server not built. Run: npm run build');
    process.exit(1);
  }
  const proc = spawn(SERVER_BIN, args.slice(1), { stdio: 'inherit' });
  proc.on('close', (code) => process.exit(code || 0));
}

// ─── Show help ───
function showHelp() {
  console.log(`
QRAF — Local LLM Inference Runtime

Usage: qraf <command> [options]

Commands:
  chat                           Interactive chatbot with model browser
  chat <model.qraf>              Quick chat with specific model
  run <model> --prompt <text>    One-shot text generation
  convert <source> -o <out>      Convert model to QRAF format
  list [--dir <path>]            List available models
  inspect <model>                Show model details
  benchmark <model>              Performance test
  serve [--port N]               Start HTTP API server
  version                        Show version

Convert examples:
  qraf convert Qwen/Qwen2.5-0.5B-Instruct -o models/qwen.qraf
  qraf convert model.gguf -o models/model.qraf
  qraf convert ./local-model/ -o models/local.qraf

Quick start:
  1. qraf convert Qwen/Qwen2.5-0.5B-Instruct -o models/qwen.qraf
  2. qraf chat
`);
}

// ─── Route commands ───
switch (command) {
  case 'convert':
    runConvert();
    break;

  case 'serve':
  case 'server':
    runServer();
    break;

  case 'help':
  case '--help':
  case '-h':
    showHelp();
    break;

  case 'version':
  case '--version':
    console.log('qraf v0.1.0');
    break;

  default:
    // Pass everything else to C++ CLI
    runCli();
    break;
}
