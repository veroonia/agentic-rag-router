import datetime
import os

import streamlit as st
from dotenv import load_dotenv

from graph.graph import app_graph
from ui.components import (
    close_assistant_row,
    open_assistant_row,
    render_hero,
    render_message_actions,
    render_missing_key_note,
    render_user_message,
    render_wordmark,
)

load_dotenv()

st.set_page_config(page_title="Divergent Agent", page_icon="✨", layout="centered")

_css_path = os.path.join(os.path.dirname(__file__), "ui", "styles.css")
with open(_css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "answer_model" not in st.session_state:
    st.session_state.answer_model = "llama-3.3-70b"

top_col1, top_col2 = st.columns([3, 2])

with top_col1:
    st.markdown(render_wordmark(), unsafe_allow_html=True)

with top_col2:
    model_labels = {
        "llama-3.3-70b": (
            "Llama 3.3 70B",
            "Fast and capable general-purpose model",
        ),
        "gpt-oss-120b": (
            "GPT-OSS 120B",
            "Large open-weight model for deeper reasoning",
        ),
    }

    current_model = st.session_state.answer_model
    current_label = model_labels[current_model][0]

    with st.popover(f"✨ {current_label}", use_container_width=True):
        st.markdown('<div class="model-picker-title">Answering model</div>', unsafe_allow_html=True)

        for model_id, (name, description) in model_labels.items():
            is_selected = st.session_state.answer_model == model_id

            if st.button(
                f"{name}{'  ✓' if is_selected else ''}",
                key=f"model_option_{model_id}",
                use_container_width=True,
            ):
                st.session_state.answer_model = model_id
                st.rerun()

            st.markdown(
                f'<div class="model-option-description">{description}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="model-picker-divider"></div>', unsafe_allow_html=True)

missing_keys = [k for k in ["OPENROUTER_API_KEY", "TAVILY_API_KEY"] if not os.environ.get(k)]

if not st.session_state.messages:
    st.markdown(render_hero(), unsafe_allow_html=True)
    if missing_keys:
        st.markdown(render_missing_key_note(missing_keys), unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(render_user_message(msg["content"], msg.get("time", "")), unsafe_allow_html=True)
    else:
        st.markdown(open_assistant_row() + msg["content"] + close_assistant_row(), unsafe_allow_html=True)
        st.markdown(render_message_actions(), unsafe_allow_html=True)

query = st.chat_input("Ask anything...")
if query:
    now = datetime.datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": query, "time": now})
    st.markdown(render_user_message(query, now), unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        result = app_graph.invoke(
            {
                "original_query": query,
                "answer_model": st.session_state.answer_model,
            }
        )

    answer = result.get("final_answer", "Sorry, I couldn't generate an answer.")
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(open_assistant_row() + answer + close_assistant_row(), unsafe_allow_html=True)
    st.markdown(render_message_actions(), unsafe_allow_html=True)

    with st.expander("Debug: pipeline trace"):
        st.json(
            {
                "expanded_query": result.get("expanded_query"),
                "route": result.get("route"),
                "tool_output_preview": (result.get("tool_output") or "")[:500],
                "answer_model": st.session_state.answer_model,
            }
        )