# Design Notes — teams-whisperer

This document records technology decisions, tradeoffs, and alternatives considered.

## Philosophy

**Pure Rust preferred**, with pragmatic escape hatches:
- C/C++ bindings (whisper.cpp) are acceptable when they're the most mature option
- PyO3 bridges Python ML models (e.g., pyannote) when no Rust equivalent exists
- The goal is a great tool, not Rust purity for its own sake

## Audio Extraction

### Chosen: Symphonia + Rubato (Pure Rust)

**Symphonia** handles demuxing (MP4/ISOBMFF, MKV/Matroska, WebM, OGG, WAV, FLAC, MP3)
and decoding (AAC, FLAC, Vorbis, MP3, Opus, ALAC, ADPCM) — all in pure Rust.
Google/Chromium is experimentally testing Symphonia as an ffmpeg replacement for audio decoding.
Mozilla built mp4parse-rust for Firefox's MP4 demuxing.

**Rubato** resamples decoded PCM to Whisper's required 16kHz mono format.
Pure Rust, real-time safe (no allocations during processing).

### Alternatives Considered

| Option | Why not? |
|---|---|
| **rsmpeg** (FFmpeg bindings) | Requires FFmpeg installed; C dependency |
| **ffmpeg-next** | Same — FFmpeg system dep |
| **Shell out to `ffmpeg`** | External process; but kept as fallback for unsupported formats |
| **ez-ffmpeg** | Newer, less proven; still requires FFmpeg |

### Fallback

If Symphonia can't decode a format (e.g., some exotic codec), we shell out to `ffmpeg`
with: `ffmpeg -i input.mp4 -ar 16000 -ac 1 -f wav pipe:1`

## Transcription

### Chosen: whisper-rs (default) + Voxtral (alternative)

**whisper-rs** (`--transcriber whisper`, feature `whisper-rs-backend`, default):
- whisper.cpp bindings (~97.9% accuracy on LibriSpeech)
- Metal GPU acceleration on Apple Silicon
- Built-in VAD, word-level timestamps
- Downside: C++ dependency (whisper.cpp compiled at build time)

**Voxtral** (`--transcriber voxtral`, feature `voxtral`):
- Mistral AI's 4.4B ASR model (3.4B Mistral LM + 970M audio encoder)
- Runs via `voxtral-mini-realtime-rs` + burn/wgpu backend with Q4 GGUF weights
- Originally attempted via mistral.rs (candle), but hit Metal NaN bugs in attention
- Custom Q4 WGSL shaders for quantized inference on WebGPU
- Natively streaming architecture with causal attention
- Pure Rust inference path (no C++ deps)
- No word-level timestamps (returns full text as single segment)
- Requires `cubecl-wgpu` patch to cap workgroup size (256 limit)
- Apache 2.0 licensed

### Alternatives Considered

| Option | Why not? |
|---|---|
| **Candle whisper** | Metal backend missing kernels (layer-norm); CPU-only fallback too slow vs whisper-rs with Metal |
| **whisper-burn** | Last commit Oct 2023 (2+ years stale), 14 open issues, effectively abandoned |
| **whisper-cpp-plus** | Newer, less proven; whisper-rs already covers whisper.cpp well |

## Summarization

### Chosen: mistral.rs (local) + Claude CLI + passthrough

**mistral.rs** (`--summarizer local`, feature `summarize-local`):
- Built on HuggingFace Candle, but production-grade: automatic model loading,
  chat template detection, built-in quantization, sampling, and KV cache
- Native Metal GPU acceleration on Apple Silicon
- Default model: Ministral 3 8B Instruct 2512 Q4_K_M (~5 GB, ~33 tok/s on M4 Pro)
- Context window set to 16384 tokens (covers ~1 hour of transcript)
- Auto-downloads from HuggingFace Hub (bartowski GGUF quantizations)
- ~50 lines of code vs ~150 for raw candle implementation

**Claude Code CLI** (`--summarizer claude`):
- Best summarization quality
- One-shot: pipe transcript to `claude -p "Summarize..."`
- Requires Claude Code subscription/API key
- Auto-detected when `claude` is in PATH

**None** (`--summarizer none`):
- Just output the raw transcript
- User pastes to whatever LLM they prefer

### Alternatives Considered

| Option | Why not? |
|---|---|
| **Raw Candle** | Works but requires manual model loading, tokenization, generation loop, chat templates — ~150 lines for what mistral.rs does in ~50. CPU-only (Metal missing kernels for layer-norm). |
| **Burn** | Ecosystem too early for LLM inference. `burn` meta-crate has `lzma` link conflict with yt-dlp. Only ndarray (CPU) backend works via sub-crates. |
| **mlx-rs** | Best Apple Silicon perf (~230 tok/s) but macOS-only, requires MLX C++ lib, smaller ecosystem |
| **llama-cpp-2** | C++ dep; mistral.rs already uses candle (pure Rust) with comparable perf |

### Model Comparison (M4 Pro, 48GB)

| Model | Params | Prompt tok/s | Compl tok/s | Quality |
|---|---|---|---|---|
| Phi-3.5-mini Q8_0 | 3.8B | 487 | 34 | Garbage (Metal bug) |
| Qwen2.5-7B Q4_K_M | 7B | 364 | 38 | Good |
| Mistral 7B v0.3 Q4_K_M | 7B | 327 | 37 | OK (no newlines, no system msg) |
| **Ministral 3 8B Q4_K_M** | **8B** | **310** | **33** | **Very good** |
| Mistral Nemo 12B Q4_K_M | 12B | 221 | 24 | Very good |
| Gemma 3 12B | 12B | — | — | Unsupported architecture |
| **Ministral 3 14B Q4_K_M** | **14B** | **203** | **21** | **Excellent** |

Ministral 3 14B selected as default: best output quality (properly identifies content type,
good formatting, detailed analysis), acceptable speed at 21 tok/s completion. Ministral 3 8B
is a strong alternative — nearly as good quality at 1.5x the speed (~33 tok/s). Qwen2.5-7B
is the fastest option at ~38 tok/s with good quality. All models selectable via `--local-model`.

### Honest Assessment

Local 14B models produce good meeting summaries — **working, tested**.
Claude produces noticeably better structured notes with better
action item extraction. The quality gap is real but local-first is the right default
for privacy-sensitive meeting content. Claude CLI remains the best quality option.

## Speaker Diarization

### Chosen: pyannote via PyO3 (optional)

No Rust-native speaker diarization exists. pyannote is best-in-class:
- Community-1 model significantly outperforms v3.1
- Accepts 16kHz mono audio (matches our pipeline output)
- PyO3 embeds CPython in our Rust binary with zero-copy numpy arrays

This is the "escape hatch pattern" — use PyO3 when Python has something Rust doesn't.

## Input

### Chosen: yt-dlp crate + local files

The `yt-dlp` Rust crate auto-downloads yt-dlp and ffmpeg binaries.
Supports 1,800+ extractors (YouTube, Microsoft Stream/Teams recordings, etc.)

Local files are accepted directly — any format Symphonia can demux.

## Performance Notes (M4 Pro, 48GB)

- Whisper small model: ~3-10x realtime for transcription (Metal GPU) — **working, recommended**
- Voxtral Q4 via burn/wgpu: ~100x slower than whisper-rs — working but impractical for now
- Rubato resampling: negligible overhead
- Symphonia demuxing: very fast, pure Rust
- mistral.rs with Metal: Ministral 3 14B Q4_K_M measured ~21 tok/s completion, ~203 tok/s prompt — **working**
- MLX would be faster (~230 tok/s) but it's Python-only

Memory bandwidth is the LLM bottleneck on Apple Silicon, not compute.
M4 Pro has 273 GB/s — sufficient for 7B-14B models.
