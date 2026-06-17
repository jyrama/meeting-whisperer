use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::config::Args;
use super::Segment;

/// Route whisper.cpp / ggml internal logs through tracing at TRACE level.
unsafe extern "C" fn whisper_log_callback(
    _level: whisper_rs_sys::ggml_log_level,
    text: *const std::ffi::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    // SAFETY: `text` is a valid C string pointer provided by whisper.cpp/ggml.
    if let Ok(msg) = unsafe { std::ffi::CStr::from_ptr(text) }.to_str() {
        let msg = msg.trim_end();
        if !msg.is_empty() {
            tracing::trace!(target: "whisper_cpp", "{msg}");
        }
    }
}

/// Transcribe audio using whisper-rs (whisper.cpp bindings).
pub async fn transcribe(args: &Args, pcm_16khz: &[f32]) -> Result<Vec<Segment>> {
    // Silence whisper.cpp and ggml log output before anything loads.
    unsafe {
        whisper_rs::set_log_callback(Some(whisper_log_callback), std::ptr::null_mut());
        whisper_rs_sys::ggml_log_set(Some(whisper_log_callback), std::ptr::null_mut());
    }

    let model_path = ensure_model(args).await?;

    let ctx = whisper_rs::WhisperContext::new_with_params(
        model_path.to_str().context("Invalid model path")?,
        whisper_rs::WhisperContextParameters::default(),
    )
    .context("Failed to load Whisper model")?;

    let mut state = ctx.create_state().context("Failed to create Whisper state")?;

    let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    if let Some(ref lang) = args.language {
        params.set_language(Some(lang));
    } else {
        params.set_language(Some("en"));
    }

    state
        .full(params, pcm_16khz)
        .context("Whisper transcription failed")?;

    let n_segments = state.full_n_segments();

    let mut segments = Vec::with_capacity(n_segments as usize);
    for i in 0..n_segments {
        let seg = state
            .get_segment(i)
            .with_context(|| format!("Segment {i} out of bounds"))?;
        let text = seg.to_str().context("Segment text is not valid UTF-8")?.to_owned();
        // Timestamps are in centiseconds (10ms units)
        segments.push(Segment {
            start: seg.start_timestamp() as f64 / 100.0,
            end: seg.end_timestamp() as f64 / 100.0,
            text,
        });
    }

    Ok(segments)
}

/// Download the Whisper model from HuggingFace if not already cached.
async fn ensure_model(args: &Args) -> Result<PathBuf> {
    let model_id = args.model.model_id();
    let repo_id = "ggerganov/whisper.cpp";
    let filename = format!("ggml-{model_id}.bin");

    let api = hf_hub::api::tokio::Api::new().context("Failed to initialize HuggingFace API")?;
    let repo = api.model(repo_id.to_string());

    // Check if model is already in the HuggingFace cache before printing anything.
    if let Ok(path) = repo.get(&filename).await {
        return Ok(path);
    }

    tracing::info!("Downloading Whisper model '{model_id}' (first run only)...");
    let model_path = repo
        .get(&filename)
        .await
        .with_context(|| format!("Failed to download model '{filename}' from HuggingFace"))?;

    Ok(model_path)
}
