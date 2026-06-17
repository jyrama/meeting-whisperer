use anyhow::Result;
use crate::config::{Args, Transcriber};

#[cfg(feature = "whisper-rs-backend")]
mod whisper_rs_backend;

mod voxtral_backend;

/// A transcribed segment with timestamps.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Transcribed text
    pub text: String,
}

/// Transcribe 16kHz mono f32 PCM audio into text segments.
#[allow(unused_variables)]
pub async fn transcribe(args: &Args, pcm_16khz: &[f32]) -> Result<Vec<Segment>> {
    match args.transcriber {
        Transcriber::Whisper => {
            #[cfg(feature = "whisper-rs-backend")]
            {
                return whisper_rs_backend::transcribe(args, pcm_16khz).await;
            }
            #[cfg(not(feature = "whisper-rs-backend"))]
            anyhow::bail!(
                "whisper-rs backend not compiled in. \
                 Rebuild with the `whisper-rs-backend` feature (enabled by default)."
            );
        }
        Transcriber::Voxtral => {
            return voxtral_backend::transcribe(args, pcm_16khz).await;
        }
    }
}

/// Format segments into a human-readable transcript.
pub fn format_transcript(segments: &[Segment]) -> String {
    let mut out = String::new();
    for seg in segments {
        let start = format_time(seg.start);
        let end = format_time(seg.end);
        out.push_str(&format!("[{start} -> {end}]{}\n", seg.text));
    }
    out
}

fn format_time(seconds: f64) -> String {
    let total_secs = seconds as u64;
    let hours = total_secs / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    if hours > 0 {
        format!("{hours:02}:{mins:02}:{secs:02}")
    } else {
        format!("{mins:02}:{secs:02}")
    }
}
