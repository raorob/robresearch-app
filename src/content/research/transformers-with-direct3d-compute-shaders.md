---
title: "Executing Transformers via Direct3D Compute Shaders"
description: "An outline for mapping transformer operations to a Windows-native Direct3D compute runtime."
order: 3
track: "GPU execution"
status: "Planned"
---

## Research outline

This note will describe the Direct3D runtime: GPU buffer allocation and reuse, compute shader dispatch, synchronization, and the tensor operations needed by the model.

## Topics to investigate

- Buffer lifecycle and limits on a 6 GB GPU
- Matrix multiplication and data layout
- Attention and feed-forward operations
- Numerical precision and correctness checks
- Dispatch and transfer costs in a streamed execution path

The goal is to make the mapping between model operations and GPU work explicit. Performance and numerical behavior remain to be measured.
