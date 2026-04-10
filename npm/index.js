/**
 * QRAF — Local LLM Inference Runtime
 *
 * Node.js API for programmatic access.
 *
 * Usage:
 *   const qraf = require('qraf');
 *   await qraf.convert('Qwen/Qwen2.5-0.5B', 'models/qwen.qraf');
 *   const result = await qraf.generate('models/qwen.qraf', 'Hello!');
 */

const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const BUILD_DIR = path.join(ROOT, 'build');
const CLI_BIN = path.join(BUILD_DIR, 'qraf-cli');
const CONVERTER = path.join(ROOT, 'tools', 'qraf_convert.py');

function isBuilt() {
  return fs.existsSync(CLI_BIN);
}

/**
 * Convert a model to QRAF format.
 * @param {string} source - HuggingFace name, local path, GGUF/safetensors file
 * @param {string} output - Output .qraf path
 * @param {object} opts - { format: 'auto' }
 * @returns {Promise<void>}
 */
function convert(source, output, opts = {}) {
  return new Promise((resolve, reject) => {
    const args = [CONVERTER, source, '-o', output];
    if (opts.format) args.push('--format', opts.format);

    const proc = spawn('python3', args, { stdio: 'inherit' });
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`Conversion failed with code ${code}`));
    });
  });
}

/**
 * Generate text from a model.
 * @param {string} modelPath - Path to .qraf model
 * @param {string} prompt - Input prompt
 * @param {object} opts - { maxTokens: 256 }
 * @returns {Promise<{text: string, tokens: number, tokensPerSec: number}>}
 */
function generate(modelPath, prompt, opts = {}) {
  return new Promise((resolve, reject) => {
    if (!isBuilt()) {
      reject(new Error('QRAF runtime not built. Run: npm run build'));
      return;
    }

    const maxTokens = opts.maxTokens || 256;
    const args = ['run', modelPath, '--prompt', prompt, '--max-tokens', String(maxTokens)];

    let stdout = '';
    const proc = spawn(CLI_BIN, args);
    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', () => {});
    proc.on('close', (code) => {
      if (code === 0) {
        // Parse stats from output
        const statsMatch = stdout.match(/(\d+) tokens in ([\d.]+) ms \(([\d.]+) tok\/s\)/);
        resolve({
          text: stdout.split('\n')[0] || '',
          tokens: statsMatch ? parseInt(statsMatch[1]) : 0,
          tokensPerSec: statsMatch ? parseFloat(statsMatch[3]) : 0,
        });
      } else {
        reject(new Error(`Generation failed with code ${code}`));
      }
    });
  });
}

/**
 * List available models in a directory.
 * @param {string} dir - Models directory (default: 'models')
 * @returns {Array<{name: string, path: string, size: number}>}
 */
function listModels(dir = 'models') {
  const models = [];
  if (!fs.existsSync(dir)) return models;

  for (const file of fs.readdirSync(dir)) {
    if (file.endsWith('.qraf')) {
      const fullPath = path.join(dir, file);
      const stat = fs.statSync(fullPath);
      models.push({
        name: file.replace('.qraf', ''),
        path: fullPath,
        size: stat.size,
      });
    }
  }
  return models.sort((a, b) => a.name.localeCompare(b.name));
}

module.exports = {
  convert,
  generate,
  listModels,
  isBuilt,
  ROOT,
  BUILD_DIR,
};
