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


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------

load_dotenv()


# -------------------------------------------------------------------
# Streamlit configuration
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Divergent Agent",
    page_icon="✨",
    layout="centered",
)


# -------------------------------------------------------------------
# CSS
# -------------------------------------------------------------------

_css_path = os.path.join(
    os.path.dirname(__file__),
    "ui",
    "styles.css",
)

with open(
    _css_path,
    encoding="utf-8",
) as f:

    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

top_col1, top_col2 = st.columns(
    [3, 2]
)


with top_col1:

    st.markdown(
        render_wordmark(),
        unsafe_allow_html=True,
    )


with top_col2:

    st.markdown(
        """
        <div style="text-align:right;">
            <small>Agentic RAG • LangGraph</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Check API keys
# -------------------------------------------------------------------

missing_keys = []

if not os.environ.get(
    "OPENROUTER_API_KEY"
):
    missing_keys.append(
        "OPENROUTER_API_KEY"
    )

if not os.environ.get(
    "TAVILY_API_KEY"
):
    missing_keys.append(
        "TAVILY_API_KEY"
    )


# -------------------------------------------------------------------
# Hero
# -------------------------------------------------------------------

if not st.session_state.messages:

    st.markdown(
        render_hero(),
        unsafe_allow_html=True,
    )

    if missing_keys:

        st.markdown(
            render_missing_key_note(
                missing_keys
            ),
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
# Existing messages
# -------------------------------------------------------------------

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            render_user_message(
                msg["content"],
                msg.get("time", ""),
            ),
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            open_assistant_row()
            + msg["content"]
            + close_assistant_row(),
            unsafe_allow_html=True,
        )

        st.markdown(
            render_message_actions(),
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------
# Chat input
# -------------------------------------------------------------------

query = st.chat_input(
    "Ask anything..."
)


if query:

    now = datetime.datetime.now().strftime(
        "%H:%M"
    )

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
            "time": now,
        }
    )

    st.markdown(
        render_user_message(
            query,
            now,
        ),
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------
    # Run LangGraph
    # ---------------------------------------------------------------

    with st.spinner(
        "Thinking..."
    ):

        try:

            result = app_graph.invoke(
                {
                    "original_query": query,
                }
            )

            answer = result.get(
                "final_answer",
                "Sorry, I couldn't generate an answer.",
            )

        except Exception as e:

            answer = (
                "An error occurred while running "
                f"the agent:\n\n`{e}`"
            )

            result = {
                "expanded_query": "",
                "route": "",
                "tool_output": "",
            }

    # ---------------------------------------------------------------
    # Display answer
    # ---------------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    st.markdown(
        open_assistant_row()
        + answer
        + close_assistant_row(),
        unsafe_allow_html=True,
    )

    st.markdown(
        render_message_actions(),
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------
    # Debug trace
    # ---------------------------------------------------------------

    with st.expander("Debug: pipeline trace"):

        st.json(
            {
                "original_query": query,
                "expanded_query": result.get(
                    "expanded_query"
                ),
                "route": result.get(
                    "route"
                ),
                "tool_output_preview": (
                    result.get(
                        "tool_output"
                    )
                    or ""
                )[:1000],
                "llm": "openrouter/free",
            }
        )