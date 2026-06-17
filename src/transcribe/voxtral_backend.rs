use anyhow::Result;

use crate::config::Args;
use super::Segment;

/// Transcribe audio using Voxtral ASR (Q4 GGUF via voxtral-mini-realtime + burn/wgpu).
///
/// Uses the voxtral-mini-realtime-rs library which runs Voxtral Mini 4B through
/// burn's wgpu backend with custom Q4 WGSL shaders. This avoids the Metal NaN bug
/// in mistralrs's candle-based Voxtral implementation.
#[allow(unused_variables)]
pub async fn transcribe(args: &Args, pcm_16khz: &[f32]) -> Result<Vec<Segment>> {
    #[cfg(feature = "voxtral")]
    {
        return run_voxtral(args, pcm_16khz).await;
    }

    #[cfg(not(feature = "voxtral"))]
    {
        anyhow::bail!(
            "Voxtral backend not compiled in. \
             Rebuild with `--features voxtral` or use `--transcriber whisper`."
        );
    }
}

#[cfg(feature = "voxtral")]
async fn run_voxtral(_args: &Args, pcm_16khz: &[f32]) -> Result<Vec<Segment>> {
    use anyhow::Context;
    use burn::backend::Wgpu;
    use voxtral_mini_realtime::audio::{
        chunk::{chunk_audio, needs_chunking, ChunkConfig},
        mel::{MelConfig, MelSpectrogram},
        pad::PadConfig,
        AudioBuffer,
    };
    use voxtral_mini_realtime::gguf::loader::Q4ModelLoader;
    use voxtral_mini_realtime::models::time_embedding::TimeEmbedding;
    use voxtral_mini_realtime::tokenizer::VoxtralTokenizer;

    type Backend = Wgpu;
    let device = burn::backend::wgpu::WgpuDevice::default();

    // Download model files (Q4 GGUF + tokenizer).
    let (gguf_path, tokenizer_path) = ensure_voxtral_model().await?;

    // Load tokenizer.
    tracing::info!("Loading Voxtral tokenizer...");
    let tokenizer = VoxtralTokenizer::from_file(&tokenizer_path)
        .context("Failed to load Voxtral tokenizer")?;

    // Load Q4 GGUF model (~2.5 GB).
    tracing::info!("Loading Voxtral Q4 model...");
    let mut loader = Q4ModelLoader::from_file(&gguf_path)
        .context("Failed to open Voxtral GGUF")?;
    let model = loader.load(&device)
        .context("Failed to load Voxtral Q4 model")?;
    drop(loader);

    tracing::info!("Model loaded, transcribing audio...");

    // Audio preprocessing: peak-normalize to 0.95 (critical for Q4 quantization).
    let mut audio = AudioBuffer::new(pcm_16khz.to_vec(), 16000);
    audio.peak_normalize(0.95);

    let mel_extractor = MelSpectrogram::new(MelConfig::voxtral());
    let pad_config = PadConfig::voxtral();
    let chunk_config = ChunkConfig::voxtral().with_max_frames(1200);

    // Time embedding for decoder (6-token delay = 480ms lookahead).
    let time_embed = TimeEmbedding::new(3072);
    let t_embed = time_embed.embed::<Backend>(6.0, &device);

    // Chunk long audio if needed.
    let chunks = if needs_chunking(audio.samples.len(), &chunk_config) {
        let chunks = chunk_audio(&audio.samples, &chunk_config);
        tracing::info!(chunks = chunks.len(), "Splitting long audio into chunks");
        chunks
    } else {
        vec![voxtral_mini_realtime::audio::AudioChunk {
            samples: audio.samples.clone(),
            start_sample: 0,
            end_sample: audio.samples.len(),
            index: 0,
            is_last: true,
        }]
    };

    // Transcribe each chunk.
    let mut texts = Vec::new();
    for chunk in &chunks {
        let chunk_audio = AudioBuffer::new(chunk.samples.clone(), 16000);
        let mel_tensor = mel_tensor_from_audio::<Backend>(
            &chunk_audio, &mel_extractor, &pad_config, &device,
        )?;

        let generated = model.transcribe_streaming(mel_tensor, t_embed.clone());

        // Decode: token IDs >= 1000 are text tokens.
        let text_tokens: Vec<u32> = generated
            .iter()
            .filter(|&&t| t >= 1000)
            .map(|&t| t as u32)
            .collect();
        let text = tokenizer.decode(&text_tokens)
            .context("Failed to decode Voxtral tokens")?;

        if !text.trim().is_empty() {
            texts.push(text.trim().to_string());
        }
    }

    let full_text = texts.join(" ");
    let duration = pcm_16khz.len() as f64 / 16000.0;

    tracing::debug!(
        duration_secs = format!("{:.1}", duration),
        text_len = full_text.len(),
        "Voxtral transcription complete"
    );

    Ok(vec![Segment {
        start: 0.0,
        end: duration,
        text: full_text,
    }])
}

/// Build a mel spectrogram tensor from an audio buffer.
#[cfg(feature = "voxtral")]
fn mel_tensor_from_audio<B: burn::tensor::backend::Backend>(
    audio: &voxtral_mini_realtime::audio::AudioBuffer,
    mel_extractor: &voxtral_mini_realtime::audio::mel::MelSpectrogram,
    pad_config: &voxtral_mini_realtime::audio::pad::PadConfig,
    device: &B::Device,
) -> Result<burn::tensor::Tensor<B, 3>> {
    use voxtral_mini_realtime::audio::pad::pad_audio;

    let padded = pad_audio(audio, pad_config);
    let mel = mel_extractor.compute_log(&padded.samples);
    let n_frames = mel.len();
    let n_mels = if n_frames > 0 { mel[0].len() } else { 0 };

    anyhow::ensure!(n_frames > 0, "Audio too short to produce mel frames");

    // Transpose from [frames, mels] to [mels, frames] for the model.
    let mut mel_flat = vec![0.0f32; n_mels * n_frames];
    for (frame_idx, frame) in mel.iter().enumerate() {
        for (mel_idx, &val) in frame.iter().enumerate() {
            mel_flat[mel_idx * n_frames + frame_idx] = val;
        }
    }

    Ok(burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(mel_flat, [1, n_mels, n_frames]),
        device,
    ))
}

/// Download the Voxtral Q4 GGUF model and tokenizer from HuggingFace.
#[cfg(feature = "voxtral")]
async fn ensure_voxtral_model() -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    use anyhow::Context;

    let api = hf_hub::api::tokio::ApiBuilder::new()
        .with_progress(false)
        .build()
        .context("Failed to build HuggingFace API")?;

    // Q4 GGUF model (2.51 GB).
    let gguf_repo = api.model("TrevorJS/voxtral-mini-realtime-gguf".to_string());
    tracing::info!("Ensuring Voxtral Q4 GGUF model is downloaded...");
    let gguf_path = gguf_repo.get("voxtral-q4.gguf").await
        .context("Failed to download voxtral-q4.gguf")?;

    // Tokenizer (tekken.json from the official model repo).
    let tok_repo = api.model("mistralai/Voxtral-Mini-4B-Realtime-2602".to_string());
    let tokenizer_path = tok_repo.get("tekken.json").await
        .context("Failed to download tekken.json")?;

    Ok((gguf_path, tokenizer_path))
}
