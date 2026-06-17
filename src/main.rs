mod audio;
mod config;
mod input;
mod summarize;
mod transcribe;

use anyhow::Result;
use clap::Parser;
use config::{Args, Summarizer, is_url};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Our crate gets the user-requested level; third-party crates stay quiet
    // until -vvv (third-party INFO) / -vvvv (full trace including whisper.cpp).
    let filter = match args.verbose {
        0 => "warn,teams_whisperer=warn",
        1 => "warn,teams_whisperer=info",
        2 => "warn,teams_whisperer=debug",
        3 => "info,teams_whisperer=trace",
        _ => "debug,teams_whisperer=trace,whisper_cpp=trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();

    // Step 1: Resolve input to a local file path
    let spinner = progress_spinner("Fetching input...");
    let media_path = if is_url(&args.input) {
        spinner.set_message(format!("Downloading from {}...", &args.input));
        input::download_url(&args.input).await?
    } else {
        let path = PathBuf::from(&args.input);
        anyhow::ensure!(path.exists(), "File not found: {}", args.input);
        path
    };
    spinner.finish_with_message(format!("Input: {}", media_path.display()));

    // Step 2: Extract and resample audio to 16kHz mono f32 PCM
    let spinner = progress_spinner("Extracting audio...");
    let pcm_data = audio::extract_audio(&media_path)?;
    spinner.finish_with_message(format!(
        "Audio extracted: {:.1}s ({} samples)",
        pcm_data.len() as f64 / 16000.0,
        pcm_data.len()
    ));

    // Step 3: Transcribe
    let spinner = progress_spinner("Transcribing...");
    let segments = transcribe::transcribe(&args, &pcm_data).await?;
    spinner.finish_with_message(format!("Transcribed: {} segments", segments.len()));

    // Build transcript text
    let transcript = transcribe::format_transcript(&segments);
    println!("\n--- Transcript ---\n{transcript}");

    // Step 4: Summarize
    let summarizer = args.resolve_summarizer();
    match summarizer {
        Summarizer::None => {
            println!("\n(No summarization requested. Use --summarizer to enable.)");
        }
        Summarizer::Claude => {
            let spinner = progress_spinner("Summarizing with Claude...");
            let summary = summarize::claude_cli::summarize(&transcript, &args)?;
            spinner.finish_with_message("Summary complete");
            println!("\n--- Summary ---\n{summary}");
        }
        Summarizer::Local => {
            let spinner = progress_spinner("Summarizing with local LLM...");
            let summary = summarize::local::summarize(&transcript, &args).await?;
            spinner.finish_with_message("Summary complete");
            println!("\n--- Summary ---\n{summary}");
        }
    }

    // Save output if requested
    if let Some(ref output_dir) = args.output {
        std::fs::create_dir_all(output_dir)?;

        let transcript_path = output_dir.join("transcript.md");
        std::fs::write(&transcript_path, &transcript)?;
        println!("\nTranscript saved to: {}", transcript_path.display());
    }

    Ok(())
}

fn progress_spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .expect("valid template"),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}
