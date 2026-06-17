use anyhow::{Context, Result};
use rubato::{Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType, WindowFunction};
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Extract audio from any supported media file, returning 16kHz mono f32 PCM.
/// Tries Symphonia first; falls back to ffmpeg for codecs Symphonia can't handle (e.g. Opus).
pub fn extract_audio(path: &Path) -> Result<Vec<f32>> {
    match decode_via_symphonia(path) {
        Ok(samples) => Ok(samples),
        Err(symphonia_err) => {
            // Symphonia has no Opus decoder and limited codec coverage.
            // Fall back to ffmpeg if available (bundled by yt-dlp crate or system).
            if let Some(ffmpeg) = find_ffmpeg() {
                tracing::warn!("Symphonia can't decode this file ({symphonia_err:#}), using ffmpeg");
                decode_via_ffmpeg(path, &ffmpeg)
            } else {
                Err(symphonia_err)
            }
        }
    }
}

/// Locate ffmpeg: check the bundled yt-dlp data dir first, then system PATH.
fn find_ffmpeg() -> Option<std::path::PathBuf> {
    let bundled = dirs::data_local_dir()
        .map(|d| d.join("teams-whisperer").join("bin").join("ffmpeg"));
    if let Some(ref p) = bundled {
        if p.exists() {
            return bundled;
        }
    }
    which::which("ffmpeg").ok()
}

/// Decode audio via ffmpeg, outputting 16kHz mono f32le PCM directly.
fn decode_via_ffmpeg(path: &Path, ffmpeg: &Path) -> Result<Vec<f32>> {
    let output = std::process::Command::new(ffmpeg)
        .args([
            "-v", "quiet",
            "-i", path.to_str().context("Non-UTF8 path")?,
            "-ar", &TARGET_SAMPLE_RATE.to_string(),
            "-ac", "1",
            "-f", "f32le",
            "pipe:1",
        ])
        .output()
        .context("Failed to run ffmpeg")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("ffmpeg failed: {stderr}");
    }

    // Raw f32le bytes → Vec<f32>
    let bytes = output.stdout;
    anyhow::ensure!(bytes.len() % 4 == 0, "ffmpeg output length not a multiple of 4");
    let samples = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    Ok(samples)
}

fn decode_via_symphonia(path: &Path) -> Result<Vec<f32>> {
    let (interleaved, sample_rate, channels) =
        decode_to_f32(path).context("Failed to decode audio")?;

    let mono = downmix_to_mono(&interleaved, channels);

    if sample_rate == TARGET_SAMPLE_RATE {
        return Ok(mono);
    }

    resample(&mono, sample_rate).context("Failed to resample audio")
}

/// Decode audio from a media file to interleaved f32 samples.
fn decode_to_f32(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open file: {}", path.display()))?;

    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .context("Unsupported media format")?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("No supported audio track found")?;

    let track_id = track.id;
    let container_rate = track.codec_params.sample_rate;
    let container_channels = track.codec_params.channels.map(|c| c.count());

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("Failed to create audio decoder")?;

    let mut all_samples: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;
    // Read rate and channels from the actual decoded audio, not container metadata.
    let mut actual_rate: Option<u32> = None;
    let mut actual_channels: Option<usize> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(SymphoniaError::DecodeError(msg)) => {
                tracing::warn!("Decode error (skipping packet): {msg}");
                continue;
            }
            Err(e) => return Err(e.into()),
        };

        if sample_buf.is_none() {
            let spec = *decoded.spec();
            actual_rate = Some(spec.rate);
            actual_channels = Some(spec.channels.count());
            let duration = decoded.capacity() as u64;
            sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
        }

        if let Some(ref mut buf) = sample_buf {
            buf.copy_interleaved_ref(decoded);
            all_samples.extend_from_slice(buf.samples());
        }
    }

    let sample_rate = actual_rate
        .or(container_rate)
        .context("Unknown sample rate")?;
    let channels = actual_channels
        .unwrap_or_else(|| container_channels.unwrap_or(1));

    Ok((all_samples, sample_rate, channels))
}

/// Downmix interleaved multi-channel audio to mono.
fn downmix_to_mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Resample mono f32 PCM to 16kHz using Rubato.
fn resample(mono: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    use audioadapter_buffers::direct::InterleavedSlice;

    let chunk_size = 1024;
    let ratio = TARGET_SAMPLE_RATE as f64 / source_rate as f64;

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = Async::<f32>::new_sinc(ratio, 2.0, &params, chunk_size, 1, FixedAsync::Input)?;

    let nbr_input_frames = mono.len();
    let input_adapter = InterleavedSlice::new(mono, 1, nbr_input_frames)
        .map_err(|e| anyhow::anyhow!("Failed to create input adapter: {e}"))?;

    let out_capacity = (nbr_input_frames as f64 * ratio * 1.1) as usize + chunk_size;
    let mut outdata = vec![0.0f32; out_capacity];
    let mut output_adapter = InterleavedSlice::new_mut(&mut outdata, 1, out_capacity)
        .map_err(|e| anyhow::anyhow!("Failed to create output adapter: {e}"))?;

    let (_nbr_in, nbr_out) = resampler
        .process_all_into_buffer(&input_adapter, &mut output_adapter, nbr_input_frames, None)?;

    outdata.truncate(nbr_out);
    Ok(outdata)
}
