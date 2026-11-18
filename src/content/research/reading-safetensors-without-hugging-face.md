---
title: "Reading Safetensors without Hugging Face"
description: "Mapping original Safetensors weights into the C# inference runtime."
order: 2
track: "File formats"
status: "In progress"
---

## The loader boundary

The first stage of the runtime was a C# Safetensors loader. Each shard starts with an eight-byte little-endian header length followed by a JSON header. The header records each tensor's name, data type, shape and byte offsets relative to the data region. The loader memory-maps the file and resolves these offsets into the mapped address space rather than copying all BF16 weights into separate arrays.

For a sharded model, `model.safetensors.index.json` identifies the shard files; otherwise the runtime opens `model.safetensors`. The engine looks up the named weights it needs and checks their expected BF16 type and shape. Qwen2.5's tied embedding matrix is also used as the source for its output head; Qwen3 has a separate head tensor.

## From file mapping to GPU work

The mapping is not the same thing as loading the whole model into GPU VRAM. Windows brings mapped file pages into host memory as needed. For BF16 execution, the loader can supply source bytes from the mapping; for FP8 or INT4, the engine creates quantized host-memory chunks during initialization. Resident matrices are uploaded to GPU buffers; the rest use reusable streaming slots. The [memory note](/research/layer-streaming-for-large-language-models/) describes what is resident and what moves per token.

The reader is a boundary between a model artifact and GPU execution. The runtime checks expected tensor names, shapes and types when accessing them, but this prototype is not a comprehensive validator for arbitrary untrusted Safetensors files. Its correctness and error handling need independent testing beyond the model runs documented here.
