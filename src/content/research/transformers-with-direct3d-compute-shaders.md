---
title: "Executing Transformers via Direct3D Compute Shaders"
description: "Qwen architectures, quantization and measured precision tradeoffs in a Direct3D compute runtime."
order: 3
track: "GPU execution"
status: "In progress"
---

## Research outline

This note will describe the Direct3D runtime: GPU buffer allocation and reuse, compute shader dispatch, synchronization, and the tensor operations needed by the model.

## Topics to investigate

- Buffer lifecycle and limits on a 6 GB GPU
- Matrix multiplication and data layout
- Attention and feed-forward operations
- Numerical precision and correctness checks
- Dispatch and transfer costs in a streamed execution path

The goal is to make the mapping between model operations and GPU work explicit. The first numerical experiment below examines one specific precision tradeoff; a broader runtime evaluation remains open.

## The three model architectures

The runtime reads each model's `config.json` and switches on its `model_type`. The following values are from the three local model configurations; the parameter counts count stored tensors once, including a tied embedding only once.

| Property | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct | Qwen3-8B |
| --- | ---: | ---: | ---: |
| Stored parameters | 494,032,768 | 1,543,714,304 | 8,190,735,360 |
| Transformer layers | 24 | 28 | 36 |
| Hidden size | 896 | 1,536 | 4,096 |
| Feed-forward intermediate size | 4,864 | 8,960 | 12,288 |
| Query heads | 14 | 12 | 32 |
| Key/value heads | 2 | 2 | 8 |
| Dimensions per head | 64 | 128 | 128 |
| Key/value vector width | 128 | 256 | 1,024 |
| Query heads per key/value head | 7 | 6 | 4 |
| Projection weights per layer | 14,909,440 | 46,792,704 | 192,937,984 |
| Vocabulary size | 151,936 | 151,936 | 151,936 |
| Configured maximum positions | 32,768 | 32,768 | 40,960 |
| Embedding and output weights tied | Yes | Yes | No |

These are configured maxima, not a claim that the full contexts fit in the GPU. The experiments below use a 2,048-token context. All three use grouped-query attention, RMS normalization, RoPE with $\theta=1{,}000{,}000$, and a SiLU-gated feed-forward block.

For each token, a layer normalizes its hidden state, projects queries, keys and values, applies rotary position encoding, uses the KV cache for causal attention, then applies the output projection, residual connection and gated feed-forward block. In this implementation:

- **Qwen2.5** adds Q, K and V projection biases and does not apply per-head Q/K normalization. The output head uses the embedding matrix as its source. Its chat template begins with a default system message.
- **Qwen3** applies per-head Q/K normalization and has no Q/K/V projection biases. It has a separate output head and supports the runtime's thinking/non-thinking template switch.
- Even when source weights are tied, the runtime creates a separate GPU output-head representation; input embeddings are read from the BF16 CPU mapping.

For these three configurations the query width equals the hidden size $H$. With key/value width $K$ and feed-forward width $I$, the seven projection matrices contain the following weights per layer, excluding biases and normalization:

$$
P_{\mathrm{layer}} = H^2 + HK + HK + H^2 + 3HI
                   = 2H^2 + 2HK + 3HI
$$

For Qwen2.5-0.5B, $H=896$, $K=128$, $I=4{,}864$, giving $14{,}909{,}440$ weights per layer. Qwen2.5-1.5B uses $H=1{,}536$, $K=256$, $I=8{,}960$, giving $46{,}792{,}704$. Qwen3-8B uses $H=4{,}096$, $K=1{,}024$, $I=12{,}288$, giving $192{,}937{,}984$.

## Quantization, dequantization and group size

Quantization replaces a higher-precision weight with a compact approximation. Dequantization reconstructs an approximate numerical value for the calculation. It reduces storage and transfer volume, but rounding can change model answers. It is not lossless compression.

For a group of $g$ BF16 weights $w_i$, this runtime's symmetric INT4 path calculates a scale, rounds it to BF16, and stores each quantized value as a packed four-bit code:

$$
s = \operatorname{BF16}\!\left(\frac{\max_i |w_i|}{7}\right), \quad
q_i = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w_i}{s}\right),-8,7\right), \quad
\hat w_i = s q_i
$$

For an all-zero group the implementation uses $s=1$. The shader unpacks the code and applies the scale during projection; it does not first expand the entire matrix into a persistent BF16 buffer.

For example, consider one group whose maximum absolute weight is $1$. Its stored BF16 scale is approximately $0.142578$. A weight of $0.8$ rounds to $q=6$ and reconstructs to approximately $0.855469$; a weight of $-0.5$ rounds to $q=-4$ and reconstructs to approximately $-0.570313$. The differences are quantization errors. This is a numerical illustration, not a sampled model weight.

Smaller groups use more scales but adapt to local weight ranges. For groups of 32, 64 and 128, the ideal packed INT4 layer-matrix cost is respectively $4+16/32=4.5$, $4+16/64=4.25$, and $4+16/128=4.125$ bits per weight, before alignment. The runtime keeps the LM head in FP8 for INT4 runs, so these numbers do not describe the entire model.

The FP8 option instead uses E4M3 values with one FP32 scale per matrix row; reconstruction multiplies the decoded E4M3 value by that row scale. BF16 is the original source precision. The command-line alias `fp16` selects **BF16**, not IEEE FP16; `fp4` selects **INT4**, not floating-point FP4.

The [streaming and memory page](/research/layer-streaming-for-large-language-models/) gives calculated sizes and measured residency for every model and group size. The following experiment examines answer quality.

## Experiment: INT4 group size and Qwen3-8B answer quality

**Measured on 18 November 2026.** The question is whether reducing the size of the groups used for INT4 weight quantization preserves answers better, and what happens to GPU residency. This matters because Qwen3-8B cannot fit in this GPU's memory budget at its original precision. The experiment is a small diagnostic, not a standardized model benchmark or a measurement of general model quality.

### Runtime changes

The existing C#/Direct3D 11 runtime was extended with an `--precision int4` option and `--group-size 32|64|128` (default: 32). These were existing changes when the measurement was run; no inference code was changed for the experiment itself. `--precision fp4` is an alias for **INT4**, not a floating-point 4-bit format.

For each group in a matrix row, the loader converts BF16 weights to signed 4-bit integers using the group's maximum absolute weight divided by 7 as the scale. It stores one BF16 scale per group and packs two integer values per byte. The Direct3D projection shader unpacks the weights and applies the group scales during the matrix-vector operation. Smaller groups use more scale metadata but can adapt better to local weight ranges. The LM head remains FP8 in all three INT4 variants; embeddings and small normalization weights remain BF16. FP8 uses E4M3 weights with a per-row scale. The BF16 run uses the original weights.

### Setup and method

- Model: Qwen3-8B, the original BF16 Safetensors files; Windows, NVIDIA GeForce GTX 1660 Ti Max-Q, 5.83 GiB reported dedicated VRAM, C#, .NET 10, Direct3D 11.
- Five variants: BF16 baseline, FP8, and INT4 with group sizes 32, 64 and 128. BF16 is the reference for comparison, not a guarantee of correct answers.
- Each variant was initially loaded once per set of questions. All runs used a 2,048-token context and an 80% VRAM budget. Greedy decoding (`--temp 0`) and the Qwen3 non-thinking chat template were used; no `--think` flag.
- The first ten questions had a 12-token output limit; the six additional questions had a 20-token limit. `/reset` was issued between questions so every prompt began with an empty conversation and KV cache. One output per question and variant is scored; no repeated-run variance was measured.
- Answers were checked against the expected result, ignoring terminal punctuation and allowing commas between the requested uppercase words. The score is the number of correct answers, not similarity to BF16. All prompts and outputs are shown below.

**Encoding correction:** The original automated runs sent UTF-8 input to a process expecting Windows console code page 437, corrupting four prompts containing German characters. Those four prompts were rerun with the correct input encoding for every variant; the unchanged ASCII-only prompts retain their original results. For BF16, the whole first set was rerun correctly. This table supersedes the earlier 8B results; notably, the word-order outputs and scores for groups 64 and 128 changed. The corrected prompts were run in separate model processes with the same settings and resets.

**Output-limit review:** The limits of 12 and 20 tokens were too short for some Qwen2.5 answers, which were therefore rerun with a 128-token limit below. The 8B results were not rerun: the original and encoding-corrected logs show that every scored 8B answer stopped before its respective limit, so none was cut off by those limits. The reported decode-step counts were at most 3 for the first set and 10 for the additional set. This review does not establish that a longer limit would improve answer quality.

These short, independently checkable questions cover arithmetic, factual recall, translation, simple logic and instruction following. The harder set was added after the initial ten produced identical results across all variants. This selection can reveal obvious regressions, but it is too small and simple to rank general capabilities or measure subtle probability changes.

### Results

| Variant | Correct, first 10 | Correct, additional 6 | Total | Resident layers | Streamed per token |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 10/10 | 4/6 | 14/16 | 10/36 | 10.50 GiB |
| FP8 | 10/10 | 4/6 | 14/16 | 21/36 | 3.28 GiB |
| INT4, group 32 | 10/10 | 4/6 | 14/16 | 36/36 | 0.58 GiB |
| INT4, group 64 | 10/10 | 3/6 | 13/16 | 36/36 | 0.58 GiB |
| INT4, group 128 | 10/10 | 3/6 | 13/16 | 36/36 + LM head | 0.00 GiB |

Residency and transfer figures are the runtime's reported values, rounded to two decimal places, not independent peak-memory measurements. In this run only group 128 also fit the FP8 LM head in the budget. The other INT4 variants streamed the head. These results depend on the GPU, context and budget.

The identical totals for BF16, FP8 and INT4 group 32 **do not mean identical answers**: group 32 got the leap-day question right but the multiplication question wrong, while BF16 and FP8 did the reverse. All variants incorrectly answered the modulo question. Group 64 reversed letters rather than word order; group 128 answered that prompt correctly but got the discount question wrong. Both gave a wrong leap-day answer.

### Prompts and outputs

All five variants gave the same output for each of the first ten prompts:

| Exact prompt | Output | Correct? |
| --- | --- | --- |
| Was ist 17 mal 23? Antworte nur mit der Zahl. | `391` | Yes |
| Was ist 144 geteilt durch 12? Antworte nur mit der Zahl. | `12` | Yes |
| Was ist 3 hoch 4? Antworte nur mit der Zahl. | `81` | Yes |
| Was ist die Hauptstadt von Australien? Antworte nur mit dem Stadtnamen. | `Canberra` | Yes |
| Wie lautet das chemische Symbol für Sauerstoff? Antworte nur mit dem Symbol. | `O` | Yes |
| Alle Katzen sind Säugetiere. Alle Säugetiere sind Wirbeltiere. Sind alle Katzen Wirbeltiere? Antworte nur mit Ja oder Nein. | `Ja.` | Yes |
| Welche Zahl folgt in dieser Reihe: 2, 4, 8, 16? Antworte nur mit der Zahl. | `32` | Yes |
| Übersetze das englische Wort apple ins Deutsche. Antworte nur mit dem Wort. | `Apfel` | Yes |
| Welches Wort kommt alphabetisch zuerst: Zebra oder Apfel? Antworte nur mit dem Wort. | `Apfel` | Yes |
| Was ergibt Python len([2, 4, 6])? Antworte nur mit der Zahl. | `3` | Yes |

The additional prompts exposed differences. BF16 and FP8 had identical outputs, shown together:

| Exact prompt | Expected | BF16 / FP8 | INT4 32 | INT4 64 | INT4 128 |
| --- | --- | --- | --- | --- | --- |
| Was ist 37 mal 46? Antworte nur mit der Zahl. | `1702` | `1702` | `1692` | `1702` | `1702` |
| Ein Artikel kostet 80 Euro. Nach 25 Prozent Rabatt: Wie viele Euro kostet er? Antworte nur mit der Zahl. | `60` | `60` | `60` | `60` | `50` |
| Was ergibt Python 17 % 5? Antworte nur mit der Zahl. | `2` | `3` | `3` | `3` | `3` |
| Alle A sind B. Einige B sind C. Folgt daraus, dass alle A C sind? Antworte nur mit Ja oder Nein. | `Nein` | `Nein.` | `Nein.` | `Nein.` | `Nein.` |
| Gib die folgenden Wörter in umgekehrter Reihenfolge und in Großbuchstaben aus: blau, rot, grün. Keine weiteren Wörter. | `GRÜN ROT BLAU` | `GRÜN ROT BLAU` | `GRÜN ROT BLAU` | `NIGR, TRO, LAUB` | `GRÜN, ROT, BLAU` |
| Welches Datum folgt unmittelbar auf den 28. Februar 2024? Antworte nur mit dem Datum. | `29. Februar 2024` | `1. März 2024` | `29. Februar 2024` | `29. März 2024` | `29. März 2024` |

### Reproducing the runs

Build the `QwenInference` project in Release mode, then from the inference repository root start each variant below. The project was named `Qwen3Inference` at the time of the measurements and was later renamed because it also supports Qwen2.5. Enter each prompt from the tables as a separate line and enter `/reset` after each answer. Use `--max-tokens 12` for the first table and `--max-tokens 20` for the additional table. The examples below use 20; change that value for the first set.

```powershell
dotnet build QwenInference -c Release
dotnet run --project QwenInference -c Release --no-build -- --model qwen3-8b --precision bf16 --context 2048 --vram-fraction 0.8 --temp 0 --max-tokens 20
dotnet run --project QwenInference -c Release --no-build -- --model qwen3-8b --precision fp8 --context 2048 --vram-fraction 0.8 --temp 0 --max-tokens 20
dotnet run --project QwenInference -c Release --no-build -- --model qwen3-8b --precision int4 --group-size 32 --context 2048 --vram-fraction 0.8 --temp 0 --max-tokens 20
dotnet run --project QwenInference -c Release --no-build -- --model qwen3-8b --precision int4 --group-size 64 --context 2048 --vram-fraction 0.8 --temp 0 --max-tokens 20
dotnet run --project QwenInference -c Release --no-build -- --model qwen3-8b --precision int4 --group-size 128 --context 2048 --vram-fraction 0.8 --temp 0 --max-tokens 20
```

For per-token timing diagnostics, set `QWEN_TRACE=1`. The old name `QWEN3_TRACE=1` remains supported for existing scripts.

The SHA-256 hashes of the five model shards, in filename order, identify the tested weights:

```text
model-00001-of-00005.safetensors  31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f
model-00002-of-00005.safetensors  5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282
model-00003-of-00005.safetensors  c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836
model-00004-of-00005.safetensors  b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a
model-00005-of-00005.safetensors  20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff
```

For the tested runtime and model metadata, the SHA-256 hashes were as follows. The runtime files were subsequently renamed (`Qwen3Engine.cs` to `QwenEngine.cs`, `Qwen3Kernels.hlsl` to `QwenKernels.hlsl`, and `Qwen3Generator.cs` to `QwenGenerator.cs`) and their contents changed for the rename. These historical hashes identify the tested files, not the current files.

```text
Qwen3Engine.cs    80031c59b674d86dfbbd3b3b9021c563c92a06ca47d4e7e1064182a28a6b7422
Qwen3Kernels.hlsl c7eb51bcc8c5f7f95411c47db665392600d8458056344d8346a28d796e60137d
Qwen3Generator.cs 3de5394773031e01d856608e1dc42a8d8cf79124527834a599bca0d5923e4eb7
Program.cs        3e705c2176519c0d068eab84c687fc9ef5608a7821e9810cdc6a87f6e82e65ba
tokenizer.json    aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
config.json       f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30
```

### Interpretation and next steps

On this small set, group 128 traded some answer accuracy for fitting the head in VRAM. Group 32 scored equally with BF16, but individual answers differed and quantization cannot be isolated from the FP8 head by this test. No statistically reliable quality ranking follows from 16 hand-picked questions and one scored output per prompt and variant.

A stronger follow-up would use a larger held-out benchmark and teacher-forced negative log-likelihood or perplexity on a fixed corpus, with repeated runs and a separately measured throughput comparison. Those measurements have **not** been performed here.

## Follow-up: Qwen2.5-0.5B and 1.5B

**Repeated on 18 November 2026.** The same 16 prompts and expected answers above were also tested with Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct. The aim is to see whether the precision tradeoff observed with 8B also holds for smaller models; this is still a small exploratory test, not a standardized quality benchmark.

The variants, hardware, VRAM budget, context, greedy decoding and reset procedure were the same as above. The original 12/20-token limits cut off some Qwen2.5 responses before they were complete. Therefore, all 16 questions were repeated for both models and all five variants with **128 maximum output tokens**; the results below replace the earlier Qwen2.5 results. Each model and variant was loaded once for all 16 prompts, with `/reset` between prompts. The Qwen2.5 models use the runtime's Qwen2 chat template with its default system message, rather than Qwen3's non-thinking template. No inference code was changed for this follow-up.

Earlier Qwen2.5 support added model-family selection from the configuration, Q/K/V projection biases instead of Qwen3's Q/K normalization, rotary attention with 64- or 128-dimensional heads, and support for tied embedding/output weights. The same INT4 group-size and shader path is used for both model families.

For these runs, automated input was encoded using Windows console code page 437, matching the runtime's input encoding on the test machine. An initial UTF-8 input attempt produced corrupted German characters and was discarded. Responses below are trimmed of surrounding whitespace.

The runtime does not print an explicit stop reason, so its per-response decode-step counts were checked against the limit. Each of the 160 answers stopped well below 127 decode steps, and the reset context was far from full; in the current generator this means generation encountered a stop token, not the output limit. The longest response had 49 decode steps. A model can still stop naturally with an incorrect or awkward answer; it is not necessarily a complete grammatical sentence.

Two manually checked scores are shown:

- **Content:** the expected answer is present and not contradicted, even if surrounded by explanatory text. For the word-order task, correct order counts regardless of uppercase.
- **Strict:** the expected answer is in the requested short form; terminal punctuation and commas between the requested uppercase words are allowed. A correct explanatory sentence does not satisfy a request for only a number or city name.

For example, `144 geteilt durch 12 ist 12.` is correct content but not strict format. These are small, hand-scored results, not a general measure of language quality.

### Results

| Model | Variant | Content correct | Strict format and content |
| --- | --- | ---: | ---: |
| 0.5B | BF16 | 3/16 | 2/16 |
| 0.5B | FP8 | 3/16 | 2/16 |
| 0.5B | INT4, group 32 | 3/16 | 3/16 |
| 0.5B | INT4, group 64 | 4/16 | 3/16 |
| 0.5B | INT4, group 128 | 5/16 | 2/16 |
| 1.5B | BF16 | 11/16 | 11/16 |
| 1.5B | FP8 | 10/16 | 10/16 |
| 1.5B | INT4, group 32 | 12/16 | 11/16 |
| 1.5B | INT4, group 64 | 10/16 | 9/16 |
| 1.5B | INT4, group 128 | 7/16 | 6/16 |

The runtime reported all 24 layers and the LM head resident for 0.5B, and all 28 layers and the LM head resident for 1.5B, in every variant; reported streaming was 0.00 GiB per token. This is not an independent peak-memory measurement. In the INT4 variants the LM head is FP8 and the embeddings remain BF16, even though these models tie their output and embedding weights.

The 0.5B BF16 baseline already struggled with this particular short-answer and instruction-following set. For 1.5B, group 128 scored lower than BF16 on this sample, while group 32 had the same strict score but different individual answers. These observations cannot establish a general quality ranking or isolate quantization error from the FP8 head.

### Individual responses

Rows 1–10 correspond in order to the first prompt table above; rows 11–16 correspond in order to the additional prompt table. The cells show the actual responses; the multiline 0.5B response for prompt 15 is reproduced in full below its table.

**Qwen2.5-0.5B-Instruct**

| Prompt | BF16 | FP8 | INT4 32 | INT4 64 | INT4 128 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `34` | `34` | `17 * 23 = 321` | `30` | `30` |
| 2 | `144 geteilt durch 12 ist 12.` | `144 geteilt durch 12 ist 12.` | `144` | `144 geteilt durch 12 ist 12.` | `144 geteilt durch 12 ist 12.` |
| 3 | `7` | `7` | `7` | `9` | `7` |
| 4 | `Sydney` | `Sydney` | `Hoboken` | `Sydney` | `The capital of Australia is Canberra.` |
| 5 | `O2` | `O2` | `SO2` | `O` | `#chem#S#O#` |
| 6 | `Ja` | `Ja` | `Ja` | `Ja` | `Ja` |
| 7 | `16` | `16` | `16` | `16` | `16` |
| 8 | `Apple` | `Apple` | `Apple ist das englische Wort für "Appel".` | `Apple ist das Wörterbuch.` | `apple - Apfel` |
| 9 | `Zebra` | `Zebra` | `Zebra` | `Zebra` | `Zebra` |
| 10 | `3` | `3` | `3` | `3` | `6` |
| 11 | `142` | `142` | `37 × 46 = 1542` | `112` | `37 * 46 = 1612` |
| 12 | `16` | `16` | `60` | `12` | `60` |
| 13 | `4` | `12` | `12` | `12` | `3` |
| 14 | `Ja` | `Ja` | `Ja` | `Ja` | `Ja.` |
| 15 | `rot, blau, grün.` | `rot, blau, grün.` | `rot, blau, grün.` | `rot, grün, blau` | See full response below |
| 16 | `28. Februar 2024` | `28. Februar 2024` | `2024-02-28` | `2024-02-28` | `2024-02-28` |

0.5B, INT4 group 128, prompt 15:

```text
Hier ist die Übersetzung von den gegebenen Wörter in umgekehrter Reihenfolge und in Großbuchstaben:

blau -> außäße
rot -> rot
grün -> grün
```

**Qwen2.5-1.5B-Instruct**

| Prompt | BF16 | FP8 | INT4 32 | INT4 64 | INT4 128 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `391` | `391` | `391` | `391` | `46` |
| 2 | `12` | `12` | `12` | `12` | `12` |
| 3 | `81` | `81` | `81` | `81` | `81` |
| 4 | `Canberra` | `Canberra` | `Canberra` | `Hobart` | `Sydney` |
| 5 | `O` | `O` | `O` | `Oxygen` | `O` |
| 6 | `Ja` | `Ja` | `Ja.` | `Ja.` | `Ja` |
| 7 | `32` | `32` | `32` | `32` | `32` |
| 8 | `Apfel` | `Apfel` | `Apfel` | `Apfel` | `Apfel` |
| 9 | `Zebra` | `Zebra` | `Apfel` | `Apfel` | `Zeitung` |
| 10 | `3` | `3` | `3` | `3` | `10` |
| 11 | `169` | `169` | `1692` | `1694` | `169` |
| 12 | `64` | `64` | `48` | `64` | `64` |
| 13 | `1` | `1` | `1` | `1` | `3` |
| 14 | `Nein` | `Ja` | `Nein` | `Nein` | `Ja` |
| 15 | `GRÜN, ROT, BLAU` | `GRÜN, ROT, BLAU` | `grün, rot, blau` | `Grün, Rot, Blau` | `Die Wörter in umgekehrter Reihenfolge und in Großbuchstaben sind: GRÜN, ROT, BLAU.` |
| 16 | `01. März 2024` | `01. März 2024` | `01. März 2024` | `3. März 2024` | `3. März 2024` |

### Reproduction

Use the commands in the 8B section, replacing `--model qwen3-8b` with `--model qwen2.5-0.5b` or `--model qwen2.5-1.5b` and replacing the token limit with `--max-tokens 128` for all 16 prompts. Use `/reset` between prompts. If automating redirected input on Windows, encode it to match the runtime's `Console.InputEncoding`; otherwise non-ASCII prompts may change before tokenization.

The tested model files have these SHA-256 hashes:

```text
Qwen2.5-0.5B-Instruct/model.safetensors  fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe
Qwen2.5-0.5B-Instruct/config.json         18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45
Qwen2.5-1.5B-Instruct/model.safetensors  dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee
Qwen2.5-1.5B-Instruct/config.json         98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670
Both tokenizer.json files                 c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
```
