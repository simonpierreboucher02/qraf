#!/usr/bin/env python3
"""
QRAF Universal Model Converter

Converts models from any format to QRAF (.qraf):
  - HuggingFace hub models (by name)
  - Local safetensors files
  - Local PyTorch (.bin/.pt) files
  - Local GGUF files

Usage:
  python qraf_convert.py <source> -o <output.qraf>

Examples:
  python qraf_convert.py Qwen/Qwen2.5-0.5B -o models/qwen.qraf
  python qraf_convert.py ./my-model/ -o models/local.qraf
  python qraf_convert.py model.gguf -o models/from-gguf.qraf
  python qraf_convert.py weights.safetensors --config config.json -o out.qraf
"""

import struct
import sys
import os
import json
import argparse
import subprocess

# ─── Dependency checker & auto-installer ───

REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'torch': 'torch',
    'transformers': 'transformers',
    'safetensors': 'safetensors',
    'sentencepiece': 'sentencepiece',
}

def check_and_install_deps():
    """Check for required Python packages and offer to install them."""
    missing = []
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return True

    print(f"\n[QRAF] Missing Python dependencies: {', '.join(missing)}")
    print(f"[QRAF] Installing automatically...\n")

    cmd = [sys.executable, '-m', 'pip', 'install', '--quiet'] + missing
    try:
        subprocess.check_call(cmd)
        print(f"\n[QRAF] Dependencies installed successfully.\n")
        return True
    except subprocess.CalledProcessError:
        print(f"\n[QRAF] Auto-install failed. Please run manually:")
        print(f"  pip3 install {' '.join(missing)}\n")
        return False

if not check_and_install_deps():
    sys.exit(1)

import numpy as np
from pathlib import Path

# ─── QRAF Writer (same as convert_hf_to_qraf.py, self-contained) ───

def fnv1a_hash(s: bytes) -> int:
    h = 14695981039346656037
    for b in s:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def align(offset, alignment=64):
    return (offset + alignment - 1) & ~(alignment - 1)

class QrafWriter:
    MAGIC = 0x46415251
    VERSION = 1

    def __init__(self):
        self.config_entries = []
        self.tensors = []
        self.string_table = bytearray()
        self.string_offsets = {}
        self.vocab_tokens = []
        self.vocab_scores = []
        self.merges = []
        self.special_tokens = {0: 1, 1: 2, 2: 0, 3: 0}

    def add_string(self, s: str) -> int:
        if s in self.string_offsets:
            return self.string_offsets[s]
        offset = len(self.string_table)
        self.string_table.extend(s.encode('utf-8') + b'\x00')
        self.string_offsets[s] = offset
        return offset

    def set_config_u32(self, key, value):
        self.config_entries.append((key, 0, int(value) & 0xFFFFFFFF))

    def set_config_f32(self, key, value):
        bits = struct.unpack('<I', struct.pack('<f', float(value)))[0]
        self.config_entries.append((key, 1, bits))

    def set_config_str(self, key, value):
        off = self.add_string(value)
        self.config_entries.append((key, 2, off))

    def add_tensor(self, name, data):
        data_f32 = data.astype(np.float32) if data.dtype != np.float32 else data
        shape = list(data_f32.shape)
        raw = data_f32.tobytes()
        self.tensors.append((name, shape, 0, raw))

    def set_vocab(self, tokens, scores=None):
        self.vocab_tokens = tokens
        self.vocab_scores = scores if scores else [0.0] * len(tokens)

    def set_special(self, bos, eos, pad=0, unk=0):
        self.special_tokens = {0: bos, 1: eos, 2: pad, 3: unk}

    def write(self, path):
        print(f"[QRAF] Writing to {path}...")
        for key, _, _ in self.config_entries:
            self.add_string(key)
        for name, _, _, _ in self.tensors:
            self.add_string(name)
        for tok in self.vocab_tokens:
            self.add_string(tok)

        # Config block
        config_block = struct.pack('<I', len(self.config_entries))
        for key, vtype, val in self.config_entries:
            key_off = self.add_string(key)
            config_block += struct.pack('<III', key_off, vtype, val)

        # Tokenizer block
        vocab_size = len(self.vocab_tokens)
        merges_count = len(self.merges)
        special_count = 4
        tok_block = struct.pack('<IIII', vocab_size, merges_count, special_count, 0)
        for i, tok in enumerate(self.vocab_tokens):
            off = self.add_string(tok)
            score = self.vocab_scores[i] if i < len(self.vocab_scores) else 0.0
            tok_block += struct.pack('<IIfI', off, len(tok.encode('utf-8')), score, 0)
        for stype in range(4):
            tok_id = self.special_tokens.get(stype, 0)
            tok_block += struct.pack('<IIII', 0, 0, tok_id, stype)

        # Layout
        HEADER_SIZE = 128
        offset = HEADER_SIZE
        config_offset = offset
        config_size = len(config_block)
        offset = align(offset + config_size, 8)
        tokenizer_offset = offset
        tokenizer_size = len(tok_block)
        offset = align(offset + tokenizer_size, 8)
        TMETA = 80
        num_tensors = len(self.tensors)
        tensor_dir_offset = offset
        tensor_dir_size = num_tensors * TMETA
        offset = align(offset + tensor_dir_size, 8)
        quant_dir_offset = offset
        quant_dir_size = 0
        string_table_offset = offset
        string_table_size = len(self.string_table)
        offset = align(offset + string_table_size, 64)
        data_start = offset

        tensor_metas = []
        for name, shape, dtype_id, raw in self.tensors:
            meta = {
                'name_hash': fnv1a_hash(name.encode('utf-8')),
                'name_offset': self.add_string(name),
                'ndim': len(shape),
                'shape': shape + [0] * (8 - len(shape)),
                'dtype': dtype_id,
                'quant_scheme_id': 0xFFFFFFFF,
                'data_offset': offset,
                'data_size': len(raw),
            }
            tensor_metas.append(meta)
            offset = align(offset + len(raw), 64)

        file_size = offset
        header = struct.pack('<II', self.MAGIC, self.VERSION)
        header += struct.pack('<Q', file_size)
        header += struct.pack('<QQ', config_offset, config_size)
        header += struct.pack('<QQ', tokenizer_offset, tokenizer_size)
        header += struct.pack('<QQ', tensor_dir_offset, tensor_dir_size)
        header += struct.pack('<QQ', quant_dir_offset, quant_dir_size)
        header += struct.pack('<QQ', string_table_offset, string_table_size)
        header += struct.pack('<Q', data_start)
        header += struct.pack('<IIII', num_tensors, 0, 0, 0)
        header += b'\x00' * 8
        assert len(header) == 128

        with open(path, 'wb') as f:
            f.write(header)
            f.seek(config_offset); f.write(config_block)
            f.seek(tokenizer_offset); f.write(tok_block)
            f.seek(tensor_dir_offset)
            for meta in tensor_metas:
                entry = struct.pack('<Q', meta['name_hash'])
                entry += struct.pack('<II', meta['name_offset'], meta['ndim'])
                for s in meta['shape']: entry += struct.pack('<I', s)
                entry += struct.pack('<II', meta['dtype'], meta['quant_scheme_id'])
                entry += struct.pack('<QQ', meta['data_offset'], meta['data_size'])
                entry += struct.pack('<II', 0, 0)
                f.write(entry)
            f.seek(string_table_offset); f.write(bytes(self.string_table))
            for i, (name, shape, dtype_id, raw) in enumerate(self.tensors):
                f.seek(tensor_metas[i]['data_offset']); f.write(raw)
            f.seek(file_size - 1); f.write(b'\x00')

        mb = file_size / (1024 * 1024)
        print(f"[QRAF] Done: {file_size:,} bytes ({mb:.1f} MB), {num_tensors} tensors, {vocab_size} vocab")


# ─── Source Format Detectors ───

def detect_format(source: str) -> str:
    """Detect the format of the model source."""
    p = Path(source)

    if p.is_file():
        ext = p.suffix.lower()
        if ext == '.gguf':
            return 'gguf'
        elif ext == '.safetensors':
            return 'safetensors'
        elif ext in ('.bin', '.pt', '.pth'):
            return 'pytorch'
        elif ext == '.qraf':
            print("Error: source is already QRAF format")
            sys.exit(1)

    if p.is_dir():
        # Check what files are in the directory
        files = list(p.iterdir())
        names = [f.name for f in files]
        if any(n.endswith('.safetensors') for n in names):
            return 'hf_local'
        elif any(n.endswith('.bin') for n in names):
            return 'hf_local'
        elif 'config.json' in names:
            return 'hf_local'

    # Assume HuggingFace model name
    return 'huggingface'


# ─── HuggingFace Converter ───

def convert_huggingface(source: str, writer: QrafWriter):
    """Convert a HuggingFace model (remote or local directory)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch

    print(f"[HF] Loading model: {source}")
    config = AutoConfig.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32)
    model.eval()

    arch = getattr(config, 'model_type', 'llama')
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = getattr(config, 'num_key_value_heads', heads)
    inter = getattr(config, 'intermediate_size', hidden * 4)
    max_seq = getattr(config, 'max_position_embeddings', 2048)
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    rms_eps = getattr(config, 'rms_norm_eps', 1e-5)
    vocab = config.vocab_size

    print(f"[HF] {arch}: {layers}L, hidden={hidden}, heads={heads}, kv={kv_heads}, vocab={vocab}")

    writer.set_config_str('architecture', arch)
    writer.set_config_u32('vocab_size', vocab)
    writer.set_config_u32('hidden_size', hidden)
    writer.set_config_u32('num_layers', layers)
    writer.set_config_u32('num_heads', heads)
    writer.set_config_u32('num_kv_heads', kv_heads)
    writer.set_config_u32('intermediate_size', inter)
    writer.set_config_u32('max_seq_len', min(max_seq, 32768))
    writer.set_config_f32('rope_theta', rope_theta)
    writer.set_config_f32('rms_norm_eps', rms_eps)

    # Tokenizer
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    tokens = [tok for tok, _ in sorted_vocab]
    writer.set_vocab(tokens)

    bos = tokenizer.bos_token_id or 0
    eos = tokenizer.eos_token_id or 0
    pad = tokenizer.pad_token_id or 0
    unk = tokenizer.unk_token_id or 0
    writer.set_special(bos, eos, pad, unk)

    # Tensors
    state = model.state_dict()
    for name, tensor in state.items():
        data = tensor.detach().cpu().numpy()
        print(f"  {name}: {data.shape}")
        writer.add_tensor(name, data)


# ─── Safetensors Converter ───

def convert_safetensors(source: str, writer: QrafWriter, config_path: str = None):
    """Convert a standalone .safetensors file."""
    from safetensors import safe_open

    print(f"[ST] Loading: {source}")

    # Try to find config
    cfg = {}
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        # Try same directory
        parent = Path(source).parent
        for name in ['config.json', 'params.json']:
            cp = parent / name
            if cp.exists():
                with open(cp) as f:
                    cfg = json.load(f)
                print(f"[ST] Found config: {cp}")
                break

    # Set config from JSON
    writer.set_config_str('architecture', cfg.get('model_type', 'llama'))
    writer.set_config_u32('vocab_size', cfg.get('vocab_size', 32000))
    writer.set_config_u32('hidden_size', cfg.get('hidden_size', 4096))
    writer.set_config_u32('num_layers', cfg.get('num_hidden_layers', 32))
    writer.set_config_u32('num_heads', cfg.get('num_attention_heads', 32))
    writer.set_config_u32('num_kv_heads', cfg.get('num_key_value_heads', cfg.get('num_attention_heads', 32)))
    writer.set_config_u32('intermediate_size', cfg.get('intermediate_size', 11008))
    writer.set_config_u32('max_seq_len', min(cfg.get('max_position_embeddings', 2048), 32768))
    writer.set_config_f32('rope_theta', cfg.get('rope_theta', 10000.0))
    writer.set_config_f32('rms_norm_eps', cfg.get('rms_norm_eps', 1e-5))

    # Try to load tokenizer from same dir
    _try_load_tokenizer(Path(source).parent, writer)

    # Load tensors
    with safe_open(source, framework="numpy") as f:
        for name in f.keys():
            data = f.get_tensor(name)
            print(f"  {name}: {data.shape} ({data.dtype})")
            writer.add_tensor(name, data)


# ─── PyTorch Converter ───

def convert_pytorch(source: str, writer: QrafWriter, config_path: str = None):
    """Convert a .bin or .pt PyTorch file."""
    import torch

    print(f"[PT] Loading: {source}")
    state = torch.load(source, map_location='cpu', weights_only=True)

    # Handle nested dicts
    if 'model' in state:
        state = state['model']
    elif 'state_dict' in state:
        state = state['state_dict']

    # Try config
    cfg = {}
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        parent = Path(source).parent
        for name in ['config.json', 'params.json']:
            cp = parent / name
            if cp.exists():
                with open(cp) as f:
                    cfg = json.load(f)
                break

    writer.set_config_str('architecture', cfg.get('model_type', 'llama'))
    writer.set_config_u32('vocab_size', cfg.get('vocab_size', 32000))
    writer.set_config_u32('hidden_size', cfg.get('hidden_size', 4096))
    writer.set_config_u32('num_layers', cfg.get('num_hidden_layers', 32))
    writer.set_config_u32('num_heads', cfg.get('num_attention_heads', 32))
    writer.set_config_u32('num_kv_heads', cfg.get('num_key_value_heads', cfg.get('num_attention_heads', 32)))
    writer.set_config_u32('intermediate_size', cfg.get('intermediate_size', 11008))
    writer.set_config_u32('max_seq_len', min(cfg.get('max_position_embeddings', 2048), 32768))
    writer.set_config_f32('rope_theta', cfg.get('rope_theta', 10000.0))
    writer.set_config_f32('rms_norm_eps', cfg.get('rms_norm_eps', 1e-5))

    _try_load_tokenizer(Path(source).parent, writer)

    for name, tensor in state.items():
        data = tensor.detach().cpu().numpy()
        print(f"  {name}: {data.shape} ({data.dtype})")
        writer.add_tensor(name, data)


# ─── GGUF Converter ───

GGUF_DTYPE_MAP = {
    0: ('f32', np.float32),
    1: ('f16', np.float16),
    2: ('q4_0', None),
    3: ('q4_1', None),
    6: ('q5_0', None),
    7: ('q5_1', None),
    8: ('q8_0', None),
    9: ('q8_1', None),
    10: ('q2_k', None),
    11: ('q3_k', None),
    12: ('q4_k', None),
    13: ('q5_k', None),
    14: ('q6_k', None),
    15: ('q8_k', None),
    26: ('bf16', np.float16),
    30: ('f64', np.float64),
    32: ('i32', np.int32),
}

def convert_gguf(source: str, writer: QrafWriter):
    """Convert a GGUF file to QRAF."""
    try:
        from gguf import GGUFReader
    except ImportError:
        print("Error: 'gguf' package required. Install with: pip install gguf")
        sys.exit(1)

    print(f"[GGUF] Loading: {source}")
    reader = GGUFReader(source)

    # Extract metadata
    metadata = {}
    for field in reader.fields.values():
        if len(field.parts) > 0:
            name = field.name
            # Get value based on type
            try:
                if hasattr(field, 'parts') and len(field.parts) > 1:
                    val_part = field.parts[-1]
                    if hasattr(val_part, 'tolist'):
                        val = val_part.tolist()
                        if isinstance(val, list) and len(val) == 1:
                            val = val[0]
                    else:
                        val = val_part
                    metadata[name] = val
            except Exception:
                pass

    def get_meta(key, default=None):
        # Try various naming conventions
        for prefix in ['', 'llama.', 'general.', 'qwen2.', 'gemma.', 'phi.']:
            full_key = prefix + key
            if full_key in metadata:
                return metadata[full_key]
        return default

    arch = 'llama'
    for k, v in metadata.items():
        if 'architecture' in k.lower() or k == 'general.architecture':
            if isinstance(v, bytes):
                arch = v.decode('utf-8', errors='ignore')
            elif isinstance(v, str):
                arch = v
            break

    vocab_size = get_meta('vocab_size', 32000)
    hidden_size = get_meta('embedding_length', get_meta('hidden_size', 4096))
    num_layers = get_meta('block_count', get_meta('num_hidden_layers', 32))
    num_heads = get_meta('attention.head_count', get_meta('num_attention_heads', 32))
    num_kv_heads = get_meta('attention.head_count_kv', get_meta('num_key_value_heads', num_heads))
    inter_size = get_meta('feed_forward_length', get_meta('intermediate_size', hidden_size * 4))
    max_seq = get_meta('context_length', get_meta('max_position_embeddings', 2048))
    rope_theta = get_meta('rope.freq_base', get_meta('rope_theta', 10000.0))
    rms_eps = get_meta('attention.layer_norm_rms_epsilon', get_meta('rms_norm_eps', 1e-5))

    # Handle values that might be bytes
    for var in [vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, inter_size, max_seq]:
        if isinstance(var, bytes):
            var = int.from_bytes(var, 'little')

    print(f"[GGUF] {arch}: {num_layers}L, hidden={hidden_size}, heads={num_heads}, kv={num_kv_heads}, vocab={vocab_size}")

    writer.set_config_str('architecture', str(arch))
    writer.set_config_u32('vocab_size', int(vocab_size))
    writer.set_config_u32('hidden_size', int(hidden_size))
    writer.set_config_u32('num_layers', int(num_layers))
    writer.set_config_u32('num_heads', int(num_heads))
    writer.set_config_u32('num_kv_heads', int(num_kv_heads))
    writer.set_config_u32('intermediate_size', int(inter_size))
    writer.set_config_u32('max_seq_len', min(int(max_seq), 32768))
    writer.set_config_f32('rope_theta', float(rope_theta))
    writer.set_config_f32('rms_norm_eps', float(rms_eps))

    # Extract tokenizer from GGUF metadata
    tokens_list = []
    token_scores = []
    for field_name, field in reader.fields.items():
        if 'tokenizer.ggml.tokens' in field_name:
            try:
                # tokens are stored as array of strings
                for part in field.parts:
                    if hasattr(part, 'tolist'):
                        data = part.tolist()
                        if isinstance(data, list) and all(isinstance(x, (bytes, str)) for x in data):
                            tokens_list = [x.decode('utf-8', errors='replace') if isinstance(x, bytes) else x for x in data]
            except Exception:
                pass
        if 'tokenizer.ggml.scores' in field_name:
            try:
                for part in field.parts:
                    if hasattr(part, 'tolist'):
                        data = part.tolist()
                        if isinstance(data, list) and len(data) > 0:
                            token_scores = [float(x) for x in data]
            except Exception:
                pass

    if tokens_list:
        writer.set_vocab(tokens_list, token_scores if token_scores else None)
        print(f"[GGUF] Tokenizer: {len(tokens_list)} tokens")
    else:
        # Fallback: byte-level vocab
        tokens_list = [f"<{i}>" for i in range(int(vocab_size))]
        writer.set_vocab(tokens_list)
        print(f"[GGUF] Warning: no tokenizer found, using placeholder")

    bos = int(get_meta('bos_token_id', 1) or 1)
    eos = int(get_meta('eos_token_id', 2) or 2)
    writer.set_special(bos, eos, 0, 0)

    # GGUF tensor name mapping to HF-style names
    GGUF_NAME_MAP = {
        'token_embd.weight': 'model.embed_tokens.weight',
        'output_norm.weight': 'model.norm.weight',
        'output.weight': 'lm_head.weight',
    }

    def map_gguf_name(name):
        if name in GGUF_NAME_MAP:
            return GGUF_NAME_MAP[name]
        # blk.N.* -> model.layers.N.*
        if name.startswith('blk.'):
            parts = name.split('.')
            layer_num = parts[1]
            rest = '.'.join(parts[2:])
            rest_map = {
                'attn_q.weight': 'self_attn.q_proj.weight',
                'attn_k.weight': 'self_attn.k_proj.weight',
                'attn_v.weight': 'self_attn.v_proj.weight',
                'attn_output.weight': 'self_attn.o_proj.weight',
                'attn_q.bias': 'self_attn.q_proj.bias',
                'attn_k.bias': 'self_attn.k_proj.bias',
                'attn_v.bias': 'self_attn.v_proj.bias',
                'ffn_gate.weight': 'mlp.gate_proj.weight',
                'ffn_up.weight': 'mlp.up_proj.weight',
                'ffn_down.weight': 'mlp.down_proj.weight',
                'attn_norm.weight': 'input_layernorm.weight',
                'ffn_norm.weight': 'post_attention_layernorm.weight',
            }
            mapped_rest = rest_map.get(rest, rest)
            return f'model.layers.{layer_num}.{mapped_rest}'
        return name

    # Load tensors
    for tensor in reader.tensors:
        name = tensor.name
        qraf_name = map_gguf_name(name)
        shape = list(tensor.shape)
        dtype_id = tensor.tensor_type

        dtype_info = GGUF_DTYPE_MAP.get(dtype_id, ('unknown', None))
        dtype_name, np_dtype = dtype_info

        if np_dtype is not None:
            # F32, F16, BF16, etc. — convert to f32
            data = tensor.data.copy()
            if np_dtype == np.float16 or dtype_name == 'bf16':
                data = data.view(np.float16 if dtype_name == 'f16' else np.uint16)
                if dtype_name == 'bf16':
                    # BF16 to F32
                    f32_data = np.zeros(data.shape, dtype=np.float32)
                    f32_data.view(np.uint32)[:] = data.astype(np.uint32) << 16
                    data = f32_data
                else:
                    data = data.astype(np.float32)
            elif np_dtype == np.float64:
                data = data.astype(np.float32)
            elif np_dtype == np.int32:
                data = data.astype(np.float32)
            else:
                data = data.astype(np.float32)

            data = data.reshape(shape)
            print(f"  {qraf_name}: {shape} ({dtype_name} -> f32)")
            writer.add_tensor(qraf_name, data)
        else:
            # Quantized tensor — dequantize to f32
            print(f"  {qraf_name}: {shape} ({dtype_name} -> f32, dequantizing...)")
            data = _dequantize_gguf_tensor(tensor, dtype_id, shape)
            writer.add_tensor(qraf_name, data)


def _dequantize_gguf_tensor(tensor, dtype_id, shape):
    """Dequantize a GGUF quantized tensor to f32."""
    raw = tensor.data.copy()
    numel = 1
    for s in shape:
        numel *= s

    # Q8_0: block = [f16 scale, int8 x 32]  (34 bytes per 32 elements)
    if dtype_id == 8:  # q8_0
        block_size = 32
        num_blocks = (numel + block_size - 1) // block_size
        result = np.zeros(numel, dtype=np.float32)
        offset = 0
        for b in range(num_blocks):
            scale = np.frombuffer(raw[offset:offset+2], dtype=np.float16)[0].astype(np.float32)
            offset += 2
            count = min(block_size, numel - b * block_size)
            quants = np.frombuffer(raw[offset:offset+count], dtype=np.int8).astype(np.float32)
            result[b*block_size:b*block_size+count] = quants * scale
            offset += block_size
        return result.reshape(shape)

    # Q4_0: block = [f16 scale, uint8 x 16] (18 bytes per 32 elements)
    if dtype_id == 2:  # q4_0
        block_size = 32
        num_blocks = (numel + block_size - 1) // block_size
        result = np.zeros(numel, dtype=np.float32)
        offset = 0
        for b in range(num_blocks):
            scale = np.frombuffer(raw[offset:offset+2], dtype=np.float16)[0].astype(np.float32)
            offset += 2
            half = block_size // 2
            packed = np.frombuffer(raw[offset:offset+half], dtype=np.uint8)
            lo = (packed & 0x0F).astype(np.int8) - 8
            hi = ((packed >> 4) & 0x0F).astype(np.int8) - 8
            interleaved = np.empty(block_size, dtype=np.float32)
            interleaved[0::2] = lo.astype(np.float32)
            interleaved[1::2] = hi.astype(np.float32)
            count = min(block_size, numel - b * block_size)
            result[b*block_size:b*block_size+count] = interleaved[:count] * scale
            offset += half
        return result.reshape(shape)

    # Fallback: try treating as raw f32
    print(f"    Warning: unsupported quant type {dtype_id}, treating as raw f32")
    try:
        return np.frombuffer(raw[:numel*4], dtype=np.float32).reshape(shape)
    except:
        return np.zeros(shape, dtype=np.float32)


# ─── Tokenizer Helper ───

def _try_load_tokenizer(directory: Path, writer: QrafWriter):
    """Try to load tokenizer from a local directory."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(directory))
        vocab = tok.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [t for t, _ in sorted_vocab]
        writer.set_vocab(tokens)
        bos = tok.bos_token_id or 0
        eos = tok.eos_token_id or 0
        pad = tok.pad_token_id or 0
        unk = tok.unk_token_id or 0
        writer.set_special(bos, eos, pad, unk)
        print(f"  Tokenizer: {len(tokens)} tokens")
    except Exception:
        # Fallback
        print("  Warning: no tokenizer found")
        writer.set_vocab([f"<{i}>" for i in range(32000)])


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description='QRAF Universal Model Converter',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s Qwen/Qwen2.5-0.5B -o models/qwen.qraf
  %(prog)s ./local-model-dir/ -o models/local.qraf
  %(prog)s model.gguf -o models/converted.qraf
  %(prog)s weights.safetensors --config config.json -o out.qraf
        """
    )
    parser.add_argument('source', help='Model source: HF name, local dir, .gguf, .safetensors, .bin')
    parser.add_argument('-o', '--output', required=True, help='Output .qraf file path')
    parser.add_argument('--config', help='Path to config.json (for standalone weight files)')
    parser.add_argument('--format', choices=['auto', 'huggingface', 'safetensors', 'pytorch', 'gguf'],
                        default='auto', help='Source format (default: auto-detect)')

    args = parser.parse_args()

    fmt = args.format if args.format != 'auto' else detect_format(args.source)
    print(f"[QRAF Convert] Source: {args.source}")
    print(f"[QRAF Convert] Format: {fmt}")
    print(f"[QRAF Convert] Output: {args.output}")
    print()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    writer = QrafWriter()

    if fmt == 'huggingface' or fmt == 'hf_local':
        convert_huggingface(args.source, writer)
    elif fmt == 'safetensors':
        convert_safetensors(args.source, writer, args.config)
    elif fmt == 'pytorch':
        convert_pytorch(args.source, writer, args.config)
    elif fmt == 'gguf':
        convert_gguf(args.source, writer)
    else:
        print(f"Error: unknown format '{fmt}'")
        sys.exit(1)

    writer.write(args.output)
    print(f"\nConversion complete: {args.output}")


if __name__ == '__main__':
    main()
