from __future__ import annotations

import html


def render_user_message(content: str, timestamp: str = "") -> str:
    safe_content = html.escape(content).replace("\n", "<br>")
    time_html = f'<span class="msg-time">{html.escape(timestamp)}</span>' if timestamp else ""
    return (
        '<div class="chat-row user-row">'
        f'<div class="msg-bubble user-bubble">{safe_content}{time_html}</div>'
        "</div>"
    )


def open_assistant_row() -> str:
    return '<div class="chat-row assistant-row"><div class="msg-plain">'


def close_assistant_row() -> str:
    return "</div></div>"


def render_message_actions() -> str:
    return (
        """
        <div class="msg-actions">
            <span title="Share">📤</span>
            <span title="Regenerate">🔁</span>
            <span title="Copy">📋</span>
            <span title="More">⋯</span>
        </div>
        """.strip()
    )


def render_wordmark() -> str:
    return '<div class="wordmark"><span class="bolt">✨</span>Divergent <span class="accent">Agent</span></div>'


def render_hero() -> str:
    return (
        """
        <div class="hero">
            <div class="eyebrow">✨ Divergent Agent · LangGraph + Qdrant + Tavily</div>
            <h1>Ask at the <span class="accent">speed</span><br>of thought</h1>
            <p>Rephrases your question, routes it to search, RAG, or a scraper, then answers.</p>
        </div>
        """.strip()
    )


def render_missing_key_note(missing_keys: list[str]) -> str:
    keys = ", ".join(missing_keys)
    return (
        '<div style="text-align:center;">'
        f'<p class="no-key-note">⚠️ Missing {html.escape(keys)} in .env file</p>'
        "</div>"
    )


def render_debug_pill(route: str | None) -> str:
    if not route:
        return ""
    return f'<span class="no-key-note" style="margin-top:0.5rem;">Route: {html.escape(route)}</span>'
