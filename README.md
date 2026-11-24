# omega-qwen36-35b

**Qwen 3.6 35B-A3B running on the Pure Seeded-Omega Zero-CUDA GB10 Engine & Full-Chip APU Stack.**

This repository is a standalone, pinned-dependency deployment and benchmark harness (`omega-mv.lock` → `omega-mv@c30af19e9856`) for running **Qwen 3.6 35B-A3B** (`Qwen3_5MoeForConditionalGeneration`: 40 layers, 30 Gated DeltaNet + 10 Full-Attention `hd=256`, 256 experts top-8, `vocab=248,320`, `131,072`-token / `128K` context window) on **NVIDIA DGX Spark (GB10, SM 12.1)** with **zero CUDA userspace** (`ldd: not a dynamic executable` — zero `libcuda.so.1`, zero cuBLAS/cuDNN, zero `libc`, zero C, zero Rust, zero Python on the inference path).

> **Driver & Firmware Disclosure:** "Zero CUDA" means zero NVIDIA userspace libraries or CUDA toolkit. Execution still opens `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm` through Linux kernel open modules (`nvidia` / `nvidia-uvm` `580.173.02`) and NVIDIA's signed GSP firmware.

---

## Full-Chip GB10 APU & Engine Setup

Built directly from pinned `omega-mv` (`c30af19e98561037e8cb1445035fef948951699a`) via the self-hosted seeded compiler (`seed/out/omc`):

- **Full-Chip GB10 APU (`apu-fullchip`, `488 KB` static ELF):** Fires all **20 physical ARM CPU cores** (`10× Cortex-X925 @ 3.900 GHz` + `10× Cortex-A725 @ 2.808 GHz`) and all **48 Blackwell SM121 SMs** (`6,144` GPU cores) simultaneously over coherent NVLink-C2C (`4.478 T ops/s`, `BIT_EXACT=PASS`).
- **Full-W4 + Exact FP8 Dual Precision (`q36-decode`, `q36-serve`):** On-GPU FP8 → block-32 INT4 quantization (`~19.5 GB` weights + `5.37 GB` 128K KV cache + `32 MB` RoPE + `128 MB` prefill workspace) with W4 Two-Pass Exact `lm_head` (`PARITY=EXACT` vs Full BF16).
- **Native Qwen 3.6 MTP (`--mtp`, Slot 40 W4):** Multi-Token Prediction (`M=4`) speculative decode with bit-exact GDN rollback (`75.99 tok/s`, `acc=2.52 tok/step`, `PARITY=EXACT`).
- **Continuous Batched Decode (`batch4` / `batch8`) & Fused APU Kernels:** Single-launch `qW4B32GroupedM4Gu` / `qW4B32GroupedM4Dn` fused-SwiGLU + shared-expert-256 MoE kernels for `B <= 8` (eliminating `160` full-GPU `WFI` barriers per step), `K=4096` `qW4B32GemmM4`, and **Split-K `S=16` 256-CTA HMMA Flash-Attention-2** for flat long-context throughput (`241.47 tok/s` at `N=8`).
- **Live Arboris Online Co-Activation Prefetcher (Always On):** Pure OM-2 cross-layer expert prefetcher overlapped asynchronously across both prefill (`QP_TRACE`) and decode (`QDW_TRACE`), warming the 40-layer co-activation graph (`32–52%` real-prompt hit rate).
- **128K Context Window + 2-Byte Indexed ByteLevel BPE Tokenizer:** Pure OM-2 `65,536`-bucket tokenizer (`248,044` vocab) and OpenAI-compatible HTTP server (`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, default `max_tokens=8192`, cap `32768`).

---

## Live DGX Spark GB10 Silicon Benchmarks

Measured on **NVIDIA DGX Spark (GB10, `sm_121`, driver `580.173.02`, Linux `7.0.0-1019-nvidia` aarch64)** inside exclusive quiet `spark-window` sessions (`nvrm_errors_delta: 0`) on `2026-11-24`, running real weights from `~/models/qwen3.6-35b-a3b-fp8-95a723d0` built from `omega-mv@c30af19e9856`:

- **Window 1 (APU + Decode + MTP + Batch + Prefill):** `20261124T184033Z-accb1d` — [receipt JSON](evidence/gb10-apu-decode-prefill.cc437d29a8eed60762c2ef7c8dfbcff698b3f0ec89091c085dbf45665a799460.json) · [raw log](evidence/gb10-apu-decode-prefill-log.0b4bca0640f18088d9c5721c134c890d51cc93d09c89714eb56af9d0dc1256d0.txt)
- **Window 2 (Live HTTP Prose + Needle + 128K Self-Test):** `20261124T184839Z-8d330b` (`exit: 0`, `nvrm_errors_delta: 0`) — [receipt JSON](evidence/gb10-http-prose-selftest.dbb58d932bd23bf620df194c5c7fc3adbc77decf850206d4949e95ca4caf5957.json) · [raw log](evidence/gb10-http-prose-selftest-log.809a0d65f2018f98d265980a10cde5b27f9d422d4237139f27b13425779ed804.txt)

### 1. Prose Decode, Native MTP & Continuous Batched Decode

| Workload | Precision / Mode | Prompt / Gen | Prefill | Decode Throughput | Latency | Live Arboris | Resident VRAM | Parity / Verification |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Short Factual Prose (Live HTTP)** | Full-W4 (`q36-serve`) | `19` / `8` tok | `65 ms` | **`67.80 tok/s`** | `14.75 ms/tok` | `39%` (`3,126 / 7,992`) | `~25.0 GB` (incl. 128K KV) | Exact: `"The capital of France is **Paris**."` |
| **Long-Form Prose Decode (Live HTTP)** | Full-W4 (`q36-serve`) | `58` / `256` tok | `94 ms` | **`67.47 tok/s`** | `14.82 ms/tok` | `42%` (`44,558 / 105,640`) | `~25.0 GB` (incl. 128K KV) | Coherent 256-tok technical essay (`finish_reason: stop`) |
| **CLI Decode + Parity (`N=1`)** | Full-W4 (`q36-decode w4`) | `17` / `24` tok | — | **`68.16 tok/s`** | `14.67 ms/tok` | `37%` | `~19.5 GB` weights | `PARITY=EXACT` vs Full BF16 (`tok=271`) |
| **Native MTP Speculative (`M=4`)** | Full-W4 + Slot 40 W4 (`mtp`) | `17` / `64` tok | — | **`75.99 tok/s`** | `13.16 ms/tok` | enabled | `~20.0 GB` | `PARITY=EXACT`, `acc=2.52 tok/step` (`1.13×` vs `N=1`) |
| **Continuous Batched Decode (`N=4`)** | Full-W4 (`batch4`, fused MoE) | `4 × 64` = `256` tok | — | **`147.14 tok/s`** (`36.79`/stream) | `27.18 ms/step` | enabled | `~19.5 GB` weights | 4 concurrent streams, `qW4B32GroupedM4` + `GemmM4` |
| **Continuous Batched Decode (`N=8`)** | Full-W4 (`batch8`, Split-K FA2) | `8 × 64` = `512` tok | — | **`241.47 tok/s`** (`30.18`/stream) | `33.13 ms/step` | enabled | `~19.5 GB` weights | 8 concurrent streams, Split-K `S=16` HMMA FA2 |
| **CLI Exact FP8 Decode (`N=1`)** | Exact FP8 (`q36-decode`) | `17` / `24` tok | — | **`48.03 tok/s`** | `20.82 ms/tok` | `45%` | `~34.7 GB` weights | `PARITY=EXACT` vs Full BF16 (`tok=271`) |

### 2. Prefill Sweep, Needle-in-a-Haystack & 128K Context Boundary

| Workload | Mode | Context / Prompt | Prefill Wall | Prefill Throughput | Decode (post-prefill) | Live Arboris | Verification |
|---|---|---:|---:|---:|---:|---:|---|
| **1K Chunked Prefill** | Full-W4 HMMA FA2 (`1024 p`) | `1,024` tok | `556 ms` | **`1,841 tok/s`** | — | `52%` (`96 ms` overlapped) | `nvrm_errors_delta: 0` |
| **4K Chunked Prefill** | Full-W4 HMMA FA2 (`4096 p`) | `4,096` tok | `2,343 ms` | **`1,748 tok/s`** | — | `52%` (`519 ms` overlapped) | `nvrm_errors_delta: 0` |
| **16K Chunked Prefill** | Full-W4 HMMA FA2 (`16384 p`) | `16,384` tok | `14,442 ms` | **`1,134 tok/s`** | — | `51%` (`1,935 ms` overlapped) | `nvrm_errors_delta: 0` |
| **3K Needle-in-a-Haystack (Live HTTP)** | Full-W4 (`q36-serve`) | `2,994` tok / `17` gen | `1,703 ms` | **`1,758.1 tok/s`** | `47.89 tok/s` (`355 ms`) | `32%` (`341,266 / 1,045,072`) | Exact recall: `"Captain Elena Vance authorized deployment code Aurora-742 at station Omega-9."` |
| **128K Context + Batch-8 Self-Test** | Full-W4 (`q36-serve --self-test --batch`) | `131,072` cap (`128K`) | — | — | `241.34 tok/s` (`N=8`) | `41%` (`6,872 / 16,720`) | `PASS` (Prompts 0 & 1 exact + `N=8` batch) |

### 3. Comparison vs. Other Stacks (DGX Spark GB10)

All stacks measured on the **same NVIDIA DGX Spark GB10** (`131,072`-token context, prefix/radix caching disabled, `temperature=0`, `max_tokens=256`):
- **Side-by-side 5.4K prompt harness (`2026-11-23`, Spark receipts `sbs-35b-omega2`, `sbs-35b-vllm`, `sbs-35b-sglang`):** Identical `5,377–5,379`-token prompt (`prompt1.txt`), fresh engine state, cold (`0%` page cache) and warm (`100%` page cache) startups.
- **Latest `omega-qwen36-35b` (`2026-11-24`, Windows 1–2 above):** Adds the full APU + fused MoE (`qW4B32GroupedM4` + `GemmM4` + Split-K `S=16` HMMA FA2) + Live Arboris + Native MTP stack.
- **Legacy C/Rust `aien-serve-http` (`2026-11-21`, Spark receipt `2026-11-21-q36-35b-verified-kit.json`):** Prior C/Rust Omega serving stack before the pure seeded-OM-2 rewrite.

| Stack / Engine | Userspace Runtime | Cold Ready | Warm Ready | 5.4K Cold TTFT | Launch → 1st Tok (Warm) | Decode (`N=1`) | Batched (`N=4` / `N=8`) | Grounding (`5.4K` P1) | NVRM Errors |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| **Omega `q36-serve` Full-W4 (this repo)** | **Zero-CUDA static ELF (`938 KB`)** | **`57.32 s`** | **`22.04 s`** | **`3.55 s`** | **`25.60 s`** | **`67.47 tok/s`** (`75.99` MTP) | **`147.14` / `241.47 tok/s`** | **`5/5` locs, `7=Y`, `53=Y`** | **`0`** |
| **Omega `q36-serve` Exact FP8 (this repo)** | **Zero-CUDA static ELF (`938 KB`)** | `60.11 s` | `28.66 s` | `120.36 s` | `149.19 s` | `48.03 tok/s` (`37.63` @ 5.4K) | — | `1/5` locs, `7=Y`, `53=N` | **`0`** |
| Legacy C/Rust `aien-serve-http` | Zero-CUDA C/Rust (superseded) | — | — | — | — | `50.43 tok/s` | `105.31` / `119.39 tok/s` | — | `0` |
| **vLLM** (CUDA container, FP8) | Python + PyTorch + CUDA | `368.73 s` | `367.59 s` | `17.98 s` | `387.62 s` | `51.19 tok/s` | — | `0/5` locs, `7=Y`, `53=Y` | `1` |
| **SGLang** (CUDA container, FP8) | Python + PyTorch + CUDA | `318.31 s` | `306.54 s` | `19.56 s` | `323.75 s` | `50.47 tok/s` (`50.76` warm) | — | `0/5` locs, `7=Y`, `53=Y` | `3` (exit `120`) |

Key measured deltas on GB10:
- **Cold / Warm Startup:** Omega Full-W4 reaches ready **`6.43×` faster cold** (`57.32 s` vs `368.73 s`) and **`16.68×` faster warm** (`22.04 s` vs `367.59 s`) than vLLM, and **`5.55×` faster cold / `13.91×` faster warm** than SGLang.
- **Time to First Token (5.4K prompt, no prefix cache):** **`5.06×` faster** than vLLM (`3.55 s` vs `17.98 s`) and **`5.51×` faster** than SGLang (`3.55 s` vs `19.56 s`). Warm Launch → 1st Token is **`15.14×` faster** than vLLM (`25.60 s` vs `387.62 s`) and **`12.65×` faster** than SGLang (`25.60 s` vs `323.75 s`).
- **Prose Decode Throughput:** Live HTTP Full-W4 (`67.47 tok/s`) is **`1.32×` faster** than vLLM (`51.19 tok/s`) and **`1.34×` faster** than SGLang (`50.47 tok/s`); with Native MTP (`75.99 tok/s`), **`1.48×` faster** than vLLM and **`1.51×` faster** than SGLang.
- **Batched Decode vs. Legacy C/Rust Omega:** Pure seeded-OM-2 `q36` is **`1.40×` faster at `N=4`** (`147.14` vs `105.31 tok/s`) and **`2.02×` faster at `N=8`** (`241.47` vs `119.39 tok/s`).
- **Stability:** Zero NVRM kernel errors (`nvrm_errors_delta: 0`) across all Omega runs vs. `1` NVRM error on vLLM and `3` NVRM errors + `[Errno 32] Broken pipe` (`exit 120`) on SGLang warm teardown.

### 4. Full-Chip GB10 APU Engine (`apu-fullchip`)

| Slice | Compute Units Active | Steps (64-bit LCG+XOR) | Wall Time | Throughput | Verification |
|---|---|---:|---:|---:|---|
| `GPU_48SM_ONLY_2B_SLICE` | 48 Blackwell SM121 SMs (6,144 cores) | `1,989,672,960` | `1,347 µs` | `4.431 T ops/s` | Reference GPU slice |
| `FULLCHIP_APU_20CPU_48SM_2B` | 20 ARM CPU + 48 Blackwell SM121 SMs | `2,000,000,000` | `1,648 µs` | `3.641 T ops/s` | `BIT_EXACT=PASS` |
| **`FULLCHIP_APU_20CPU_48SM_20B`** | **20 ARM CPU + 48 Blackwell SM121 SMs (6,164 cores)** | **`20,000,000,000`** | **`13,400 µs`** | **`4.478 T ops/s`** | **`BIT_EXACT=PASS`** |

### 5. Zero-CUDA Static Binary Manifest (`omega-mv@c30af19e9856`)

All binaries are compiled purely by `seed/out/omc` (`not a dynamic executable` under `ldd`):

| Binary | Size | SHA-256 | Role |
|---|---:|---|---|
| `q36-serve` | `938 KB` | `2cb0044caabf7c6899cb513bf5c3b09bce935b9e0f6f43e91ea924e169ed7ae7` | OpenAI HTTP server (Full-W4 / FP8, 128K KV, Live Arboris, BPE tokenizer) |
| `q36-decode` | `902 KB` | `1b9746a856b2038d33bc1f441c7476a9a91ece88857913dd92a0a43aafc1ea39` | Real-weight CLI decoder & benchmark runner (`w4`, `fp8`, `mtp`, `batch4/8`, `p`) |
| `q36-native` | `539 KB` | `959db5aefd5f8fd467a32533d5e2d3ba75bbac75984fb2a68af77063c0161107` | 11-kernel SM121 numerical parity & synthetic cold-bank runner |
| `apu-fullchip` | `488 KB` | `e3da4ae889c141e5eb625ec46c2966987144bd84324c644d20c1f795e8484d64` | Full-Chip GB10 APU (20 CPU cores + 48 SM121 SMs over NVLink-C2C) |
| `q36-head` | `166 KB` | `cb0d9e99cd52c178207a44c35157c1c3a411030f5952085d862b4c12f4982424` | Real-weight W4 Two-Pass Exact `lm_head` verifier |
| `q36-ring` | `14 KB` | `f0f361c9f7a21532dcfaeb5e4ddaf6673cad7de601cabaeee855ae821efee493` | Multi-Spark slab-exchange ring transport |
| `q36-plan` | `11 KB` | `b5249123a0fc5901743cec40d1d2c250dcc428ae745ab2d607365a3a4561e8fc` | Hardware-agnostic Slab Plan IR |

---

## Quick Start (DGX Spark AArch64)

### 1. Build from Pinned `omega-mv`
```sh
git clone https://interplane.io/drakestapleton/omega-mv.git ../omega-mv
git -C ../omega-mv checkout c30af19e98561037e8cb1445035fef948951699a

sh scripts/build.sh
```

### 2. Serve Qwen 3.6 35B-A3B (OpenAI-Compatible API)
```sh
# Full-W4 + Live Arboris + 128K context on :18091
sh scripts/serve.sh --model-dir ~/models/qwen3.6-35b-a3b-fp8-95a723d0 --port 18091

# Exact FP8 mode
sh scripts/serve.sh --fp8 --port 18091

# Self-test (128K boundary + Live Arboris)
sh scripts/serve.sh --self-test
```

### 3. Reproduce the Full Benchmark Suite
Run inside an exclusive quiet DGX Spark GPU window:
```sh
WID=$(spark-window open omega-qwen36-35b-bench --lease 15 --gb10 exclusive --quiet exclusive --wait | tail -n 1)
trap 'spark-window close "$WID" >/dev/null 2>&1 || true' EXIT
spark-window run spark-receipt omega-qwen36-35b-bench -- sh scripts/bench.sh
```

---

## License

Apache-2.0. See [LICENSE](LICENSE).
