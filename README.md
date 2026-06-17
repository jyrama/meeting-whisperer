# teams-whisperer

Transcribe and summarize meeting recordings — Teams Copilot, but local-first.

A Rust CLI that downloads meeting recordings (YouTube, Teams, etc.), transcribes them with Whisper, and summarizes them with a local LLM — all running on your machine with Metal GPU acceleration on Apple Silicon.

## Quick Start

```bash
# Transcribe + summarize a YouTube video (defaults: whisper + ministral-8b)
teams-whisperer 'https://www.youtube.com/watch?v=...'

# Local file
teams-whisperer recording.mp4

# Skip summarization, just get the transcript
teams-whisperer recording.mp4 --summarizer none

# Use Claude for best-quality summaries (requires Claude Code CLI)
teams-whisperer recording.mp4 --summarizer claude

# Prose summary instead of structured notes
teams-whisperer recording.mp4 --format prose

# Save transcript to a directory
teams-whisperer recording.mp4 -o ./output
```

## How It Works

```
Input (URL/file) → Audio Extraction → Transcription → Summarization → Output
                   (Symphonia/ffmpeg)  (Whisper/Voxtral) (Local LLM/Claude)
```

1. **Input** — URLs are downloaded via yt-dlp (1,800+ extractors: YouTube, Teams, etc.). Local audio/video files are accepted directly.
2. **Audio** — Decoded and resampled to 16kHz mono PCM using Symphonia (pure Rust), with ffmpeg as a fallback for exotic codecs.
3. **Transcription** — Whisper.cpp via whisper-rs with Metal GPU acceleration. Produces timestamped segments.
4. **Summarization** — Local LLM (Ministral 3 8B by default) via mistral.rs with Metal GPU, or Claude Code CLI for higher quality.

## Installation

```bash
# Clone and build (requires Rust toolchain)
git clone <repo-url>
cd teams-whisperer
cargo install --path .
```

Models are auto-downloaded from HuggingFace on first run (~250 MB for Whisper small, ~5 GB for Ministral 8B).

### Requirements

- **macOS** with Apple Silicon (Metal GPU) — tested on M4 Pro
- **Rust** 2024 edition toolchain
- **ffmpeg** (optional, auto-bundled by yt-dlp for downloads)

## CLI Reference

```
teams-whisperer [OPTIONS] <INPUT>
```

| Option | Default | Description |
|--------|---------|-------------|
| `<INPUT>` | required | URL or path to audio/video file |
| `--transcriber` | `whisper` | Transcription engine: `whisper`, `voxtral` |
| `--model` | `small` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `--summarizer` | auto | Summarization: `local`, `claude`, `none` (auto-detects local) |
| `--local-model` | `ministral-8b` | Local LLM model (see table below) |
| `--format` | `structured` | Output format: `structured`, `prose` |
| `--language` | `en` | Language hint for Whisper (e.g., `fi`, `de`) |
| `-o, --output` | — | Save transcript to this directory |
| `--diarize` | false | Speaker diarization (requires Python + pyannote) |
| `-v` | — | Verbosity: `-v` info, `-vv` debug, `-vvv` model loading, `-vvvv` full trace |

### Local Models

All models are Q4_K_M GGUF quantizations, auto-downloaded from HuggingFace (bartowski).

| `--local-model` | Model | VRAM | Speed | Quality |
|-----------------|-------|------|-------|---------|
| `ministral-8b` | Ministral 3 8B Instruct 2512 | ~5 GB | ~33 tok/s | Very good |
| `ministral-14b` | Ministral 3 14B Instruct 2512 | ~8 GB | ~21 tok/s | Excellent |
| `mistral-nemo` | Mistral Nemo 12B Instruct | ~7 GB | ~24 tok/s | Very good |
| `qwen-7b` | Qwen2.5-7B Instruct | ~4.5 GB | ~38 tok/s | Good |
| `mistral-7b` | Mistral 7B Instruct v0.3 | ~4 GB | ~37 tok/s | OK |

Speed measured on M4 Pro (48 GB, 273 GB/s memory bandwidth).

### Output Formats

**Structured** (`--format structured`) produces markdown meeting notes:
- Key Discussion Points
- Decisions Made
- Action Items (with checkboxes)
- Summary

**Prose** (`--format prose`) produces a 2-4 paragraph freeform summary.

## Architecture

```
src/
├── main.rs                    # 4-step pipeline with progress spinners
├── config.rs                  # CLI args, enums, auto-detection
├── audio.rs                   # Symphonia + Rubato (16kHz mono PCM)
├── input.rs                   # yt-dlp downloads with caching
├── transcribe/
│   ├── mod.rs                 # Segment struct, formatting
│   ├── whisper_rs_backend.rs  # Whisper.cpp (Metal GPU)
│   └── voxtral_backend.rs    # Voxtral 4.4B ASR (burn/wgpu)
└── summarize/
    ├── mod.rs                 # Prompt templates
    ├── claude_cli.rs          # Claude Code CLI integration
    └── local.rs               # mistral.rs (Metal GPU)
```

### Features (Cargo)

| Feature | Default | What it enables |
|---------|---------|-----------------|
| `whisper-rs-backend` | yes | Whisper.cpp transcription (Metal GPU) |
| `summarize-local` | yes | Local LLM summarization via mistral.rs |
| `voxtral` | yes | Voxtral ASR backend (experimental, ~100x slower) |
| `diarize` | no | Speaker diarization via PyO3 + pyannote |

### Design Philosophy

**Pure Rust preferred**, with pragmatic escape hatches:
- C/C++ bindings (whisper.cpp) when they're the most mature option
- PyO3 bridges Python ML models (pyannote) when no Rust equivalent exists
- The goal is a great tool, not Rust purity for its own sake

See [DESIGN_NOTES.md](DESIGN_NOTES.md) for detailed technology decisions, model benchmarks, and alternatives considered.

## License

MIT
