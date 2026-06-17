use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WhisperModel {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

impl WhisperModel {
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Tiny => "tiny",
            Self::Base => "base",
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large-v3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Transcriber {
    /// Whisper.cpp via whisper-rs (default)
    Whisper,
    /// Voxtral ASR via mistral.rs (4.4B multimodal model)
    Voxtral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Summarizer {
    /// Local LLM via mistral.rs (Metal GPU on macOS)
    Local,
    /// Claude Code CLI (claude -p)
    Claude,
    /// No summarization, just output transcript
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum LocalModel {
    /// Ministral 3 14B Instruct (Dec 2025) — best quality, ~21 tok/s
    #[value(name = "ministral-14b")]
    Ministral14B,
    /// Ministral 3 8B Instruct (Dec 2025) — very good quality, ~33 tok/s
    #[value(name = "ministral-8b")]
    Ministral8B,
    /// Mistral Nemo 12B Instruct — very good quality, ~24 tok/s
    #[value(name = "mistral-nemo")]
    MistralNemo,
    /// Qwen2.5-7B Instruct — good quality, ~38 tok/s
    #[value(name = "qwen-7b")]
    Qwen7B,
    /// Mistral 7B Instruct v0.3 — OK quality, ~37 tok/s
    #[value(name = "mistral-7b")]
    Mistral7B,
}

impl LocalModel {
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::Ministral14B => "bartowski/mistralai_Ministral-3-14B-Instruct-2512-GGUF",
            Self::Ministral8B => "bartowski/mistralai_Ministral-3-8B-Instruct-2512-GGUF",
            Self::MistralNemo => "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
            Self::Qwen7B => "bartowski/Qwen2.5-7B-Instruct-GGUF",
            Self::Mistral7B => "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        }
    }

    pub fn gguf_file(&self) -> &'static str {
        match self {
            Self::Ministral14B => "mistralai_Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
            Self::Ministral8B => "mistralai_Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
            Self::MistralNemo => "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
            Self::Qwen7B => "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            Self::Mistral7B => "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Structured meeting notes (attendees, decisions, action items)
    Structured,
    /// Freeform prose summary
    Prose,
}

#[derive(Parser, Debug)]
#[command(
    name = "teams-whisperer",
    about = "Transcribe and summarize meeting recordings — Teams Copilot, but local-first",
    version
)]
pub struct Args {
    /// URL (YouTube, Teams, etc.) or path to a local video/audio file
    pub input: String,

    /// Transcription engine
    #[arg(long, value_enum, default_value_t = Transcriber::Whisper)]
    pub transcriber: Transcriber,

    /// Whisper model size (only used with --transcriber whisper)
    #[arg(long, value_enum, default_value_t = WhisperModel::Small)]
    pub model: WhisperModel,

    /// Summarization engine
    #[arg(long, value_enum)]
    pub summarizer: Option<Summarizer>,

    /// Output format for summaries
    #[arg(long, value_enum, default_value_t = OutputFormat::Structured)]
    pub format: OutputFormat,

    /// Enable speaker diarization (requires Python + pyannote)
    #[arg(long, default_value_t = false)]
    pub diarize: bool,

    /// Output directory for transcripts and summaries
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Local model for --summarizer local (default: ministral-8b)
    #[arg(long, value_enum, default_value_t = LocalModel::Ministral8B)]
    pub local_model: LocalModel,

    /// Language hint for Whisper (e.g., "en", "fi", "de")
    #[arg(long)]
    pub language: Option<String>,

    /// Verbose logging (-v info, -vv debug, -vvv model loading, -vvvv full trace)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

impl Args {
    /// Determine the summarizer to use, with runtime auto-detection.
    pub fn resolve_summarizer(&self) -> Summarizer {
        if let Some(s) = self.summarizer {
            return s;
        }

        // Auto-detect: prefer local summarizer, fall back to Claude CLI, otherwise none
        if cfg!(feature = "summarize-local") {
            Summarizer::Local
        } else if claude_cli_available() {
            Summarizer::Claude
        } else {
            Summarizer::None
        }
    }
}

/// Check if the Claude Code CLI is available in PATH.
pub fn claude_cli_available() -> bool {
    which::which("claude").is_ok()
}

/// Check if input looks like a URL (vs a local file path).
pub fn is_url(input: &str) -> bool {
    input.starts_with("http://")
        || input.starts_with("https://")
        || input.starts_with("www.")
}
