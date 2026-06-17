use anyhow::{Context, Result};
use std::process::Command;

use crate::config::Args;
use super::summary_prompt;

/// Summarize a transcript using the Claude Code CLI (`claude -p`).
pub fn summarize(transcript: &str, args: &Args) -> Result<String> {
    let prompt = format!(
        "{}\n\n--- TRANSCRIPT ---\n{}",
        summary_prompt(args.format),
        transcript
    );

    let output = Command::new("claude")
        .arg("-p")
        .arg(&prompt)
        .output()
        .context("Failed to run 'claude -p'. Is Claude Code CLI installed?")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Claude CLI failed: {stderr}");
    }

    let summary = String::from_utf8(output.stdout)
        .context("Claude CLI output was not valid UTF-8")?;

    Ok(summary.trim().to_string())
}
