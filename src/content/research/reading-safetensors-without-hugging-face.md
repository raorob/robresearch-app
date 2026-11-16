---
title: "Reading Safetensors without Hugging Face"
description: "An outline for parsing tensor metadata and reading model data directly from Safetensors files."
order: 2
track: "File formats"
status: "Planned"
---

## Research outline

This note will document a direct Safetensors reader for the Windows runtime: how the file header is parsed, how tensor names, shapes, data types and byte offsets are represented, and how the underlying data is accessed.

## Topics to investigate

- Header parsing and validation
- Tensor metadata and byte ranges
- Mapping tensor data to the runtime's expected layouts
- Reading only the data needed for a streamed layer
- Error handling for unsupported or malformed inputs

The reader is a boundary between a model artifact and GPU execution. Its correctness needs to be verified independently of inference results.
