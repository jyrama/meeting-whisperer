# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Check compilation (fast, no linking)
cargo check

# Check with specific feature
cargo check --features summarize-local

# Build release (slow — whisper.cpp compiles from source via whisper-rs-sys)
cargo build --release

# Run with a test URL
cargo run --release -- 'https://www.youtube.com/watch?v=i-pROcBN-d8' -vv

# Run without summarization (faster for testing transcription)
cargo run --release -- input.mp4 --summarizer none
```

No tests, no linter, no CI configured yet.

## Architecture

Four-step pipeline in `main.rs`: **Input → Audio → Transcribe → Summarize**

Each stage is a separate module with pluggable backends behind feature flags:

- **`input.rs`** — yt-dlp downloads with video ID caching in `~/.local/share/teams-whisperer/`
- **`audio.rs`** — Symphonia (pure Rust) decodes and Rubato resamples to 16kHz mono f32 PCM. Falls back to ffmpeg for unsupported codecs.
- **`transcribe/`** — Dispatches to whisper-rs (default, Metal GPU) or voxtral (burn/wgpu, experimental). Returns `Vec<Segment>` with timestamps.
- **`summarize/`** — Dispatches to mistral.rs local LLM (default, Metal GPU), Claude CLI subprocess, or passthrough. Prompt templates in `mod.rs`.

Auto-detection in `config.rs`: if `summarize-local` feature is compiled in, local summarizer is the default. Otherwise falls back to Claude CLI if `claude` binary is in PATH.

## Key Patterns

**Feature gates** — Heavy use of `#[cfg(feature = "...")]` for optional backends. All three main features enabled by default: `whisper-rs-backend`, `summarize-local`, `voxtral`.

**Metal GPU on macOS** — Both `whisper-rs` and `mistralrs` dependencies gain Metal features via `[target.'cfg(target_os = "macos")'.dependencies]` in Cargo.toml.

**GGUF models** — All local models are Q4_K_M quantizations from bartowski's HuggingFace repos, auto-downloaded by hf-hub on first run. Model selection is in the `LocalModel` enum in `config.rs` (repo_id + gguf_file for each).

**Tracing levels** — Third-party crate logs (mistral.rs, symphonia, hf-hub) are suppressed until `-vvv`. Our crate's own logs follow `-v` (INFO), `-vv` (DEBUG), `-vvv` (TRACE + third-party INFO), `-vvvv` (full trace including whisper.cpp internals).

**Local patches** — `patches/cubecl-0.9.0` and `patches/cubecl-wgpu-0.9.0` fix a workgroup size limit (256 max) required by the voxtral-mini-realtime burn/wgpu backend.

## Dependencies to Know

- **mistralrs** — Git dependency from `github.com/EricLBuehler/mistral.rs`. Provides `GgufModelBuilder` for loading GGUF models with auto chat template detection. The `.with_logging()` flag controls whether it emits internal tracing events.
- **whisper-rs / whisper-rs-sys** — Builds whisper.cpp from source at compile time (slow first build). Log callback redirects C++ output to Rust tracing at TRACE level.
- **voxtral-mini-realtime** — Git dependency. Uses burn with wgpu backend for Q4 quantized inference. Requires the cubecl patches.

## Design Decisions

See `DESIGN_NOTES.md` for detailed rationale on every technology choice, model benchmarks, and alternatives considered. Key points:

- Pure Rust preferred, C++ bindings (whisper.cpp) and PyO3 (pyannote) accepted as pragmatic escape hatches
- Ministral 3 8B is the default summarizer (best speed/quality tradeoff at ~33 tok/s)
- Ministral 3 14B available via `--local-model ministral-14b` for best local quality
- Voxtral transcription works but is ~100x slower than whisper-rs — experimental only
