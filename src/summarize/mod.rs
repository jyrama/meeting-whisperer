pub mod claude_cli;
pub mod local;

use crate::config::OutputFormat;

/// Build the system prompt for meeting summarization.
pub fn summary_prompt(format: OutputFormat) -> &'static str {
    match format {
        OutputFormat::Structured => {
            "You are a meeting notes assistant. Given a meeting transcript, produce structured \
             markdown meeting notes with the following sections:\n\n\
             ## Key Discussion Points\n\
             - Bullet points of main topics discussed\n\n\
             ## Decisions Made\n\
             - Any decisions that were agreed upon\n\n\
             ## Action Items\n\
             - [ ] Action item with owner if identifiable\n\n\
             ## Summary\n\
             A brief 2-3 sentence overall summary.\n\n\
             Be concise. Focus on substance, not filler. \
             If speakers are labeled, note who said/decided what."
        }
        OutputFormat::Prose => {
            "You are a meeting notes assistant. Given a meeting transcript, write a concise \
             prose summary covering the key points discussed, any decisions made, and action \
             items. Keep it to 2-4 paragraphs. Be direct and substantive."
        }
    }
}
