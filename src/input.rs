use anyhow::{Context, Result};
use std::path::PathBuf;
use yt_dlp::Downloader;
use yt_dlp::model::selector::{AudioCodecPreference, AudioQuality};

/// Download only the audio track from a URL using yt-dlp.
/// Returns the path to the downloaded audio file.
pub async fn download_url(url: &str) -> Result<PathBuf> {
    let libs_dir = data_dir();
    let bin_dir = libs_dir.join("bin");
    let output_dir = libs_dir.join("downloads");
    std::fs::create_dir_all(&output_dir)?;
    std::fs::create_dir_all(&bin_dir)?;

    // Check cache by video ID before initializing yt-dlp.
    let video_id = extract_video_id(url);
    let cached = output_dir.join(format!("{video_id}.m4a"));
    if cached.exists() {
        tracing::info!("Reusing cached audio: {}", cached.display());
        return Ok(cached);
    }

    let downloader = Downloader::with_new_binaries(&bin_dir, &output_dir)
        .await
        .context("Failed to initialize yt-dlp (downloading binaries)")?
        .build()
        .await
        .context("Failed to build yt-dlp downloader")?;

    let video = downloader
        .fetch_video_infos(url)
        .await
        .context("Failed to fetch video info")?;

    let filename = format!("{video_id}.m4a");

    downloader
        .download_audio_stream_with_quality(&video, &filename, AudioQuality::Best, AudioCodecPreference::AAC)
        .await
        .context("Failed to download audio")?;

    let downloaded = output_dir.join(&filename);
    anyhow::ensure!(
        downloaded.exists(),
        "Download completed but audio file not found at {}",
        downloaded.display()
    );

    Ok(downloaded)
}

/// Extract a stable, filesystem-safe ID from a URL for caching.
/// Recognises YouTube URL patterns; falls back to a simple hash for other sites.
fn extract_video_id(url: &str) -> String {
    // youtube.com/watch?v=ID or youtube.com/shorts/ID
    if url.contains("youtube.com") || url.contains("youtu.be") {
        if let Some(id) = url
            .split(&['?', '&'][..])
            .find_map(|part| part.strip_prefix("v="))
        {
            // Take only the ID portion (stop at & or #)
            return id.split(&['&', '#'][..]).next().unwrap_or(id).to_string();
        }
        // youtu.be/ID
        if let Some(path) = url.split("youtu.be/").nth(1) {
            return path.split(&['?', '&', '#'][..]).next().unwrap_or(path).to_string();
        }
        // youtube.com/shorts/ID or youtube.com/embed/ID
        for prefix in &["/shorts/", "/embed/", "/v/"] {
            if let Some(rest) = url.split(prefix).nth(1) {
                return rest.split(&['?', '&', '#', '/'][..]).next().unwrap_or(rest).to_string();
            }
        }
    }
    // Fallback: hash the URL
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    url.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Data directory for teams-whisperer (libs, models, downloads).
fn data_dir() -> PathBuf {
    let base = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("teams-whisperer");
    std::fs::create_dir_all(&base).ok();
    base
}
