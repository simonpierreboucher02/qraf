#!/usr/bin/env python3
"""Convert a HuggingFace Llama-style model to QRAF format."""

import struct
import sys
import os
import numpy as np

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
        self.config_entries = []  # (key_str, value_type, value_u32)
        self.tensors = []         # (name, shape, dtype_id, data_bytes)
        self.string_table = bytearray()
        self.string_offsets = {}
        self.vocab_tokens = []
        self.vocab_scores = []
        self.merges = []
        self.special_tokens = {0: 1, 1: 2, 2: 0, 3: 0}  # bos, eos, pad, unk

    def add_string(self, s: str) -> int:
        if s in self.string_offsets:
            return self.string_offsets[s]
        offset = len(self.string_table)
        self.string_table.extend(s.encode('utf-8') + b'\x00')
        self.string_offsets[s] = offset
        return offset

    def set_config_u32(self, key: str, value: int):
        self.config_entries.append((key, 0, value & 0xFFFFFFFF))

    def set_config_f32(self, key: str, value: float):
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        self.config_entries.append((key, 1, bits))

    def set_config_str(self, key: str, value: str):
        off = self.add_string(value)
        self.config_entries.append((key, 2, off))

    def add_tensor(self, name: str, data: np.ndarray):
        # Always store as f32
        data_f32 = data.astype(np.float32)
        shape = list(data_f32.shape)
        raw = data_f32.tobytes()
        self.tensors.append((name, shape, 0, raw))  # dtype 0 = F32

    def set_vocab(self, tokens, scores=None):
        self.vocab_tokens = tokens
        self.vocab_scores = scores if scores else [0.0] * len(tokens)

    def set_special(self, bos, eos, pad=0, unk=0):
        self.special_tokens = {0: bos, 1: eos, 2: pad, 3: unk}

    def write(self, path: str):
        print(f"Writing QRAF to {path}...")

        # Pre-populate string table
        for key, _, _ in self.config_entries:
            self.add_string(key)
        for name, _, _, _ in self.tensors:
            self.add_string(name)
        for tok in self.vocab_tokens:
            self.add_string(tok)

        # ─── Build config block ───
        config_block = struct.pack('<I', len(self.config_entries))
        for key, vtype, val in self.config_entries:
            key_off = self.add_string(key)
            config_block += struct.pack('<III', key_off, vtype, val)

        # ─── Build tokenizer block ───
        vocab_size = len(self.vocab_tokens)
        merges_count = len(self.merges)
        special_count = 4

        tok_block = struct.pack('<IIII', vocab_size, merges_count, special_count, 0)
        # Token entries: string_offset(u32), string_length(u32), score(f32), type(u32)
        for i, tok in enumerate(self.vocab_tokens):
            off = self.add_string(tok)
            score = self.vocab_scores[i] if i < len(self.vocab_scores) else 0.0
            tok_block += struct.pack('<IIfI', off, len(tok.encode('utf-8')), score, 0)
        # Merge entries (empty for now)
        # Special tokens
        for stype in range(4):
            tok_id = self.special_tokens.get(stype, 0)
            tok_block += struct.pack('<IIII', 0, 0, tok_id, stype)

        # ─── Layout computation ───
        HEADER_SIZE = 128
        offset = HEADER_SIZE

        config_offset = offset
        config_size = len(config_block)
        offset += config_size
        offset = align(offset, 8)

        tokenizer_offset = offset
        tokenizer_size = len(tok_block)
        offset += tokenizer_size
        offset = align(offset, 8)

        num_tensors = len(self.tensors)
        TENSOR_META_SIZE = 80
        tensor_dir_offset = offset
        tensor_dir_size = num_tensors * TENSOR_META_SIZE
        offset += tensor_dir_size
        offset = align(offset, 8)

        quant_dir_offset = offset
        quant_dir_size = 0
        # no quant schemes for f32

        string_table_offset = offset
        string_table_size = len(self.string_table)
        offset += string_table_size
        offset = align(offset, 64)

        data_start = offset

        # Compute tensor data offsets
        tensor_metas = []
        for name, shape, dtype_id, raw in self.tensors:
            name_hash = fnv1a_hash(name.encode('utf-8'))
            name_off = self.add_string(name)
            ndim = len(shape)
            shape_padded = shape + [0] * (8 - ndim)

            meta = {
                'name_hash': name_hash,
                'name_offset': name_off,
                'ndim': ndim,
                'shape': shape_padded,
                'dtype': dtype_id,
                'quant_scheme_id': 0xFFFFFFFF,
                'data_offset': offset,
                'data_size': len(raw),
                'layout_id': 0,
            }
            tensor_metas.append(meta)
            offset += len(raw)
            offset = align(offset, 64)

        file_size = offset

        # ─── Write file ───
        with open(path, 'wb') as f:
            # Header (128 bytes)
            header = struct.pack('<II',  self.MAGIC, self.VERSION)
            header += struct.pack('<Q', file_size)
            header += struct.pack('<QQ', config_offset, config_size)
            header += struct.pack('<QQ', tokenizer_offset, tokenizer_size)
            header += struct.pack('<QQ', tensor_dir_offset, tensor_dir_size)
            header += struct.pack('<QQ', quant_dir_offset, quant_dir_size)
            header += struct.pack('<QQ', string_table_offset, string_table_size)
            header += struct.pack('<Q', data_start)
            header += struct.pack('<II', num_tensors, 0)  # num_quant_schemes = 0
            header += struct.pack('<II', 0, 0)  # flags, padding
            header += b'\x00' * 8  # reserved[8]
            assert len(header) == 128, f"Header is {len(header)} bytes, expected 128"
            f.write(header)

            # Config block
            f.seek(config_offset)
            f.write(config_block)

            # Tokenizer block
            f.seek(tokenizer_offset)
            f.write(tok_block)

            # Tensor directory
            f.seek(tensor_dir_offset)
            for meta in tensor_metas:
                entry = struct.pack('<Q', meta['name_hash'])
                entry += struct.pack('<I', meta['name_offset'])
                entry += struct.pack('<I', meta['ndim'])
                for s in meta['shape']:
                    entry += struct.pack('<I', s)
                entry += struct.pack('<I', meta['dtype'])
                entry += struct.pack('<I', meta['quant_scheme_id'])
                entry += struct.pack('<Q', meta['data_offset'])
                entry += struct.pack('<Q', meta['data_size'])
                entry += struct.pack('<I', meta['layout_id'])
                entry += struct.pack('<I', 0)  # padding
                assert len(entry) == 80
                f.write(entry)

            # String table
            f.seek(string_table_offset)
            f.write(bytes(self.string_table))

            # Tensor data
            for i, (name, shape, dtype_id, raw) in enumerate(self.tensors):
                f.seek(tensor_metas[i]['data_offset'])
                f.write(raw)

            # Pad to file_size
            f.seek(file_size - 1)
            f.write(b'\x00')

        size_mb = file_size / (1024 * 1024)
        print(f"Written {file_size} bytes ({size_mb:.1f} MB) to {path}")
        print(f"  {num_tensors} tensors, {vocab_size} vocab tokens")


def convert_model(model_name: str, output_path: str):
    """Download and convert a HuggingFace model to QRAF."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch

    print(f"Downloading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    print(f"Model config: {config}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num heads: {config.num_attention_heads}")
    num_kv_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    intermediate = getattr(config, 'intermediate_size', config.hidden_size * 4)
    max_seq = getattr(config, 'max_position_embeddings', 2048)
    rope_theta = getattr(config, 'rope_theta', 10000.0)
    rms_eps = getattr(config, 'rms_norm_eps', 1e-5)

    writer = QrafWriter()

    # ─── Config ───
    writer.set_config_str('architecture', 'llama')
    writer.set_config_u32('vocab_size', config.vocab_size)
    writer.set_config_u32('hidden_size', config.hidden_size)
    writer.set_config_u32('num_layers', config.num_hidden_layers)
    writer.set_config_u32('num_heads', config.num_attention_heads)
    writer.set_config_u32('num_kv_heads', num_kv_heads)
    writer.set_config_u32('intermediate_size', intermediate)
    writer.set_config_u32('max_seq_len', max_seq)
    writer.set_config_f32('rope_theta', rope_theta)
    writer.set_config_f32('rms_norm_eps', rms_eps)

    # ─── Tokenizer ───
    vocab = tokenizer.get_vocab()
    # Sort by ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    tokens = [tok for tok, _ in sorted_vocab]
    scores = [0.0] * len(tokens)
    writer.set_vocab(tokens, scores)

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    unk = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
    writer.set_special(bos, eos, pad, unk)
    print(f"Special tokens: bos={bos}, eos={eos}, pad={pad}, unk={unk}")

    # ─── Extract tensors ───
    state = model.state_dict()

    # Map HF names to QRAF names
    name_map = {}
    for key in state:
        qraf_name = key
        # Common patterns
        qraf_name = qraf_name.replace('model.embed_tokens.weight', 'model.embed_tokens.weight')
        qraf_name = qraf_name.replace('model.norm.weight', 'model.norm.weight')
        qraf_name = qraf_name.replace('lm_head.weight', 'lm_head.weight')
        # Layer patterns are usually already correct for Llama-style models
        name_map[key] = qraf_name

    for hf_name, qraf_name in name_map.items():
        tensor = state[hf_name].detach().cpu().numpy()
        print(f"  {qraf_name}: {tensor.shape} ({tensor.dtype})")
        writer.add_tensor(qraf_name, tensor)

    writer.write(output_path)
    print("Done!")


if __name__ == '__main__':
    model_name = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
    output = sys.argv[2] if len(sys.argv) > 2 else "models/smollm2-135m.qraf"

    os.makedirs(os.path.dirname(output), exist_ok=True)
    convert_model(model_name, output)
