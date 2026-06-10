# VRAM Budget — Qwen/Qwen3.5-9B

Per-step VRAM budget breakdown computed directly from an eval log (2026-06-09 run).
Numbers are in **MiB**, using **reserved-side** values (reserved is what triggers OOM
and never releases once carved out).

## Run configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.5-9B` (hybrid attention + Mamba-2) |
| TP / DP | tp=1, dp=1 |
| `mem_fraction_static` | 0.99 |
| `chunked_prefill_size` | 2048 |
| `cuda_graph_max_bs` | 1 |
| `max_loras_per_batch` | 1 (rank 8) |
| `max_mamba_cache_size` | 1 (auto) — **hard caps bs at 1** |
| `SGLANG_VLM_CACHE_SIZE_MB` | 100 |
| GPU | 22 587 MiB total |
| Workload | bs=1, 10 images per request, ~10 k input tokens, 1024 output tokens |

## Per-step budget table

| # | Step | Budget | Reserved drift | Source log | Notes |
|---:|---|---:|---|---|---|
| 0 | **Other GPU processes** | 578 | `total − free` at startup | `VRAM[startup] free=22009MiB total=22587MiB` | Always present; not under your control |
| 1 | **Model weights** | 18 156 | `reserved-delta=18156` | `VRAM[weights] alloc-delta=18048MiB reserved-delta=18156MiB` | Fixed per checkpoint; scales with quant + TP |
| 2 | **LoRA adapter pool** | 38 | `reserved-delta=38` | `VRAM[lora-pool] max_loras_per_batch=1 max_lora_rank=8 reserved-delta=38MiB` | Scales with `max_loras × max_rank × num_layers`. Budget ≈ 38 × N for N adapters of same rank |
| 3 | **KV + Mamba pool** | 480 | `reserved-delta=480` | `VRAM[kv-pool] reserved-delta=480MiB` + `KV Cache 0.46 GB` + `Mamba Cache 0.09 GB` | KV ≈ 460 MiB (14 848 tokens × bf16), mamba ≈ 92 MiB (1 slot). Bundled in one log line |
| 4 | **CUDA graph capture** | 140 | derived from baseline | `VRAM[baseline] graph_capture_gb=0.14` | Scales ~linearly with `cuda_graph_max_bs`. Currently 1 → 140 MiB. For bs=4 expect ~400 MiB |
| 5 | **Other one-time init** | ~395 | gap `baseline.reserved − (kv-pool.end + graph_capture)` = 19 222 − 18 676 − 140 = **406** | unlogged — cublas, attention backend, kernel warmup, flashinfer workspace | Could add a `VRAM[init-overhead]` wrap to break this down |
| 6 | **Multimodal cache** | 100 | currently 0, capacity = 100 | `VRAM[mm-cache] size_mib=100` | Lazy-fills on use. Reserve worst case = configured capacity |
| 7 | **Activation: VLM + prefill peak** | 566 | `peak-reserved-increase=566` (first cold prefill chunk, 2048 tokens, 10 images) | `VRAM[prefill] num_tokens=2048 peak-reserved-increase=566MiB` | Per concurrent VLM-prefill chunk. For bs=N add ~450 MiB × N more for vision encoders |
| 8 | **Activation: decode** | 5 | `peak-alloc-increase=5MiB peak-reserved-increase=0MiB` (every iter) | `VRAM[decode] peak-alloc-increase=5MiB` | Effectively free. Uses cuda-graph buffers |
| 9 | **Activation: heatmap compute** | 560 | `peak-reserved-increase=560` (first call) | `VRAM[heatmap] peak-reserved-increase=560MiB` | Runs once per finished request, post-decode (not concurrent with prefill) |
| 10 | **Persistent: heatmap query buffer** | 64 | `query_buffer_mib=64` (per req, 1024 output tokens × 8 layers) | `VRAM[heatmap] query_buffer_mib=64` | Lives during decode, freed on `req.finished()`. Budget = N × 64 MiB for N concurrent heatmap reqs |
| 11 | **Allocator fragmentation drift** | ~800 | observed reserved drift `20 640 − 19 222 − 566 = 852` | derived: `peak_reserved − baseline_reserved − activation_peak` | Accumulates across the run from varying chunk shapes. ~5% of total reserved is typical |
| 12 | **Safety margin** | 500 | recommended | n/a | For runtime growth of other processes + 1σ of fragmentation variability |
|   | **TOTAL** | **21 813** | (≤ 22 587 = total) | | leaves ~774 MiB headroom |

## How the budget composes

Activation phases don't all overlap, so we take the worst-case envelope:

```
worst_case = max(
    VLM+prefill (566),                            # active forward, no heatmap building yet
    heatmap_compute (560) + query_buffer (64)     # post-decode, query buffer fully built
) = 624 MiB
```

So **activation budget = 624 MiB** per concurrent request stream. Add fragmentation
+ safety for the global figure.

## Observed vs computed

| Quantity | Computed | Observed |
|---|---:|---:|
| Reserved at baseline | 19 222 MiB | 19 222 MiB ✓ |
| Reserved at peak | 19 222 + 624 + ~800 (frag) = 20 646 | **20 640 MiB** ✓ (within 6 MiB) |
| Free headroom at peak | 22 587 − 20 640 − 578 = **1 369 MiB** | confirms ~1 GB headroom |

## What this means for sizing knobs

| Knob | Current | Implied by budget | If you want more |
|---|---:|---:|---|
| `mem_fraction_static` | 0.99 | OK at bs=1 | **Lower to ~0.93 for bs=4** → frees ~1.4 GB to expand KV/mamba pools |
| `max_mamba_cache_size` | 1 | **Hard cap on bs** | Set `--max-mamba-cache-size 4` to enable bs=4 (+135 MiB persistent) |
| `cuda_graph_max_bs` | 1 | matches mamba cap | Raise to match new mamba cap (+45 MiB per slot for graph buffers) |
| `chunked_prefill_size` | 2048 | activation 566 MiB | Doubling → +150 MiB activation (rough) |
| `SGLANG_VLM_CACHE_SIZE_MB` | 100 | OK | Raising helps avoid re-encoding identical images; 100 MiB is reasonable |

## Scaling formula

For a new target configuration on this same model + workload pattern:

```
total_reserved ≈ weights (18 156)
              + lora_pool (38 × num_loras × rank_ratio)
              + kv_pool_size  (depends on num_tokens)
              + mamba_pool (45 × max_mamba_cache_size + ~10)
              + cuda_graph (~140 × cuda_graph_max_bs)
              + init_overhead (~395)
              + mm_cache (configured cap)
              + activation_peak (~620 × concurrent_VLM_prefill_chunks
                                 + ~330 × concurrent_text_only_chunks)
              + query_buffer (64 × concurrent_heatmap_reqs)
              + fragmentation (~5% of above)
              + safety (~500)

constraint: total_reserved + other_process_VRAM ≤ total_GPU_VRAM
```

## Concurrency: mamba is the gate (read this before raising batch size)

For hybrid SSM models like Qwen3.5-9B, **`max_mamba_cache_size` is the hard
concurrency cap**, period. It maps 1:1 to the number of concurrent requests
the scheduler can run:

```
max_concurrent_requests = max_mamba_cache_size
```

Mamba pool memory scales with **#concurrent requests, not #tokens** — the SSM
state is fixed-size per request regardless of sequence length. So:
- Longer sequences → free (mamba-wise)
- More concurrent requests → +~45 MiB persistent each (mamba state)

### Knobs that must move together for target batch size N

| Knob | Set to | Why |
|---|---|---|
| `--max-mamba-cache-size` | **N** | Hard concurrency cap (mamba slots) |
| `--cuda-graph-max-bs` | **N** (or ≥ N) | Otherwise decode falls off cuda-graph fast path at bs > current cap |
| `--max-running-requests` | ≥ N | Soft cap; defaults usually OK but verify |
| `--chunked-prefill-size` | unchanged | Only bump if you want multi-request prefill chunks (throughput) |

### Cost per +1 to `max_mamba_cache_size`

| Bucket | Cost | Source |
|---|---:|---|
| Mamba state (persistent) | **+45 MiB** | `ssm_state` per slot |
| CUDA graph buffer (if bumping `cuda_graph_max_bs` too) | **+~140 MiB** | linear-ish from `graph_capture_gb=0.14` at bs=1 |
| KV pool (reclaimed from total budget) | **−45 MiB** | mamba grows first, KV gets the leftover |
| Activation peak (transient) | **+~450 MiB** | extra VLM-prefill chunk if concurrent VLM reqs |

So roughly **+200 MiB persistent + 450 MiB transient per +1 bs** (with
concurrent VLM prefill). On the 1 369 MiB headroom observed here:
bs=2 is comfortable, bs=3 marginal, bs=4 needs `mem_fraction_static` lowered
to ~0.95.

### Gotcha: raising mamba alone shrinks KV pool

Just raising `--max-mamba-cache-size` *without* lowering `mem_fraction_static`
will **shrink your KV pool** by the same MiB. At bs=4 you'd lose 135 MiB from
KV → fewer cached tokens. Usually fine, but if KV-tight, lower
`mem_fraction_static` by 0.01–0.02 to keep KV constant.

### Recipe for bs=4 on this 22 GB card

```bash
--max-mamba-cache-size 4 \
--cuda-graph-max-bs 4 \
--mem-fraction-static 0.96    # was 0.99, frees ~700 MiB for activation
```

Verify `VRAM[baseline] => activation_budget≈XMiB` shows ≥ 1 500 MiB and the
run survives without OOM.

## Reproduce

```bash
grep -E "VRAM\[(startup|weights|lora-pool|kv-pool|baseline|mm-cache|vlm-embed|prefill|decode|heatmap)\]" run.log
```

The `alloc=…MiB reserved=…MiB` suffix on every line lets you chain-verify: each
event's `alloc-before=` should match the previous event's trailing `alloc=`. A
mismatch means a missing log step.
