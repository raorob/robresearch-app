---
title: "Managing KV Cache on GPUs with Limited Memory"
description: "How the runtime allocates key/value caches alongside a constrained GPU weight budget."
order: 5
track: "Memory management"
status: "In progress"
---

## Context has a memory cost

As tokens are processed, the number of used key/value entries grows with the context. In this implementation, however, the FP32 GPU buffers are allocated upfront for the **configured maximum context** rather than growing dynamically. Even if layer streaming reduces resident weight memory, the allocated cache and intermediate activations compete for GPU capacity.

For a decoder model, a first-order estimate is:

$$
M_{\mathrm{KV}} \propto
2 \times L \times T \times H_{\mathrm{KV}} \times D_{\mathrm{head}} \times B
$$

where $L$ is the number of layers, $T$ is the cached sequence length, $H_{\mathrm{KV}}$ is the number of key/value heads, $D_{\mathrm{head}}$ is the head dimension, and $B$ is the bytes per value. This estimate omits allocator overhead and implementation-specific layouts.

For the default 2,048-token context and FP32 cache values ($B=4$), the three configurations require:

| Model | Layers | KV heads | Head dimension | Allocated KV cache |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 24 | 2 | 64 | 48 MiB |
| Qwen2.5-1.5B | 28 | 2 | 128 | 112 MiB |
| Qwen3-8B | 36 | 8 | 128 | 576 MiB |

For example, Qwen3-8B uses $2 \times 36 \times 2048 \times 8 \times 128 \times 4 = 603{,}979{,}776$ bytes, or 576 MiB. This excludes other buffers and driver overhead.

## The 80% weight-planning target

The runtime's default 80% fraction applies to a **weight-residency planning estimate**, not dynamic KV-cache growth or a guaranteed global GPU usage cap. It subtracts the already allocated KV buffers, scratch space and streaming slots from the target before choosing resident layers. Small norm/bias buffers, transfer resources, driver overhead and other processes are not all included. See [layer streaming](/research/layer-streaming-for-large-language-models/) for the observed layer counts and the limitations of that estimate.

The implementation does not dynamically trim the conversation or offload the KV cache to CPU memory. When the configured context is full, the user can reset the conversation or restart with a larger context, if memory permits.

## Questions to answer

- How is GPU memory usage observed and how often is the policy checked?
- What happens to generation latency when the cache grows?
- Which context-trimming policy preserves the most useful information?
- What is the transfer cost of CPU offload?
- How much safety margin is needed for transient allocations?

Measurements are planned; no cache-policy results are claimed here.
