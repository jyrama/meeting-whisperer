use anyhow::Result;

use crate::config::Args;

/// Summarize a transcript using a local LLM via mistral.rs (Metal GPU on macOS).
pub async fn summarize(transcript: &str, args: &Args) -> Result<String> {
    #[cfg(feature = "summarize-local")]
    {
        return run_local(transcript, args).await;
    }

    #[cfg(not(feature = "summarize-local"))]
    {
        let _ = (transcript, args);
        anyhow::bail!(
            "Local summarizer not compiled in. \
             Rebuild with `--features summarize-local` or use `--summarizer claude`."
        );
    }
}

#[cfg(feature = "summarize-local")]
async fn run_local(transcript: &str, args: &Args) -> Result<String> {
    use anyhow::Context;
    use mistralrs::{
        AutoDeviceMapParams, DeviceMapSetting,
        GgufModelBuilder, RequestBuilder, StopTokens, TextMessageRole,
    };
    use super::summary_prompt;

    let local_model = args.local_model;
    let repo_id = local_model.repo_id();
    let gguf_file = local_model.gguf_file();
    tracing::debug!("Loading model: {repo_id} / {gguf_file}");

    let model = GgufModelBuilder::new(
        repo_id.to_string(),
        vec![gguf_file.to_string()],
    )
    .with_device_mapping(DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
        max_seq_len: 16384,
        max_batch_size: 1,
    }))
    .with_logging()
    .build()
    .await
    .context("Failed to load summarization model")?;

    tracing::debug!("Model loaded, generating summary...");

    let system_prompt = summary_prompt(args.format);
    let user_prompt = format!("{system_prompt}\n\n--- TRANSCRIPT ---\n{transcript}");

    let request = RequestBuilder::new()
        .add_message(TextMessageRole::User, &user_prompt)
        .set_sampler_max_len(1024)
        .set_sampler_temperature(0.3)
        .set_sampler_topp(0.9)
        .set_sampler_stop_toks(StopTokens::Seqs(vec![
            "</s>".to_string(),
        ]));

    let response = model
        .send_chat_request(request)
        .await
        .context("Summarization request failed")?;

    let content = response.choices[0]
        .message
        .content
        .as_ref()
        .context("Model returned empty response")?;

    tracing::debug!(
        prompt_tok_per_sec = response.usage.avg_prompt_tok_per_sec,
        compl_tok_per_sec = response.usage.avg_compl_tok_per_sec,
        "Summarization throughput"
    );

    // mistralrs GGUF models may return "<0x0A>" for newlines — decode them.
    let text = content.replace("<0x0A>", "\n");
    Ok(text.trim().to_string())
}
