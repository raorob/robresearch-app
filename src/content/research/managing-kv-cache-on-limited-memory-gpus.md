---
title: "Managing KV Cache on GPUs with Limited Memory"
description: "An outline for observing context growth and evaluating cache policies under a GPU memory budget."
order: 5
track: "Memory management"
status: "Planned"
---

## Context has a memory cost

As tokens are processed, the key/value cache grows with the context. Even if layer streaming reduces resident weight memory, the cache and intermediate activations still compete for GPU capacity.

For a decoder model, a first-order estimate is:

$$
M_{\mathrm{KV}} \propto
2 \times L \times T \times H_{\mathrm{KV}} \times D_{\mathrm{head}} \times B
$$

where $L$ is the number of layers, $T$ is the cached sequence length, $H_{\mathrm{KV}}$ is the number of key/value heads, $D_{\mathrm{head}}$ is the head dimension, and $B$ is the bytes per value. This estimate omits allocator overhead and implementation-specific layouts.

## An 80% target as a hypothesis

One strategy to investigate is growing the KV cache dynamically up to a target of **80% GPU memory utilization**, leaving headroom for other runtime allocations. That threshold is a proposed policy to evaluate, not a universal optimum or a measured result.

When the target is approached, possible strategies include trimming context or moving some cache data to CPU memory. Both change model behavior or performance and need explicit evaluation.

## Questions to answer

- How is GPU memory usage observed and how often is the policy checked?
- What happens to generation latency when the cache grows?
- Which context-trimming policy preserves the most useful information?
- What is the transfer cost of CPU offload?
- How much safety margin is needed for transient allocations?

Measurements are planned; no cache-policy results are claimed here.
