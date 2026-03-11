"""
Streamlit web UI: chat-style interface, dark/light theme, sources display.
"""

from __future__ import annotations

import sys
from pathlib import Path

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.config_loader import load_config


load_dotenv()


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    sources: Optional[List[Dict[str, Any]]] = None


def _get_api_base_url() -> str:
    return os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _app_css() -> str:
    # Subtle styling that works with Streamlit's light/dark themes.
    return """
    <style>
      .app-title { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
      .app-subtitle { color: rgba(128,128,128,0.95); margin-top: 0; }
      .sources-title { font-weight: 600; margin-top: 0.5rem; }
      .source-meta { font-size: 0.85rem; opacity: 0.85; }
      .small-muted { font-size: 0.85rem; opacity: 0.75; }
      .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
             font-size: 0.85rem; padding: 2px 6px; border-radius: 6px; border: 1px solid rgba(128,128,128,0.35); }
    </style>
    """


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            ChatMessage(
                role="assistant",
                content=(
                    "Ask a question about your documents. I’ll retrieve relevant context from the FAISS index "
                    "and answer with cited sources."
                ),
            )
        ]


def _render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown('<div class="sources-title">Sources</div>', unsafe_allow_html=True)
    for s in sources:
        md = s.get("metadata") or {}
        source = md.get("source", "unknown")
        page = md.get("page")
        section = md.get("section")
        score = s.get("score")
        header = f"{source}"
        if page is not None:
            header += f" • page {page}"
        if section is not None:
            header += f" • {section}"
        if score is not None:
            header += f" • score {score:.4f}"
        with st.expander(header, expanded=False):
            st.markdown(f'<div class="source-meta">{header}</div>', unsafe_allow_html=True)
            st.write(s.get("text", ""))


def _call_api(question: str, *, top_k: Optional[int] = None, timeout_s: float = 600.0) -> Dict[str, Any]:
    url = f"{_get_api_base_url()}/query"
    
    print("Calling API:", url)
    
    payload: Dict[str, Any] = {"question": question}
    if top_k is not None:
        payload["top_k"] = int(top_k)

    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, json=payload)
            print("Response status:", r.status_code)
            print("Response text:", r.text[:200])
            r.raise_for_status()
            return r.json()
    except Exception as e:
        print("ERROR:", e)
        raise

def main() -> None:
    st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="Enterprise RAG", layout="centered")
    st.markdown(_app_css(), unsafe_allow_html=True)

    st.markdown('<div class="app-title">Enterprise RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Chat with your enterprise documents (PDF, Markdown, TXT) via RAG.</div>',
        unsafe_allow_html=True,
    )

    _init_state()

    cfg = load_config()
    default_top_k = int(cfg.get("retrieval", {}).get("top_k", 5))

    with st.sidebar:
        st.subheader("Settings")
        st.caption("UI calls the FastAPI backend. Start the API first.")

        api_url = st.text_input("API base URL", value=_get_api_base_url())
        os.environ["RAG_API_URL"] = api_url.rstrip("/")

        top_k = st.slider("Top‑k sources", min_value=1, max_value=20, value=default_top_k, step=1)
        st.markdown(
            f'<div class="small-muted">Tip: set <span class="kbd">RAG_API_URL</span> in <span class="kbd">.env</span>.</div>',
            unsafe_allow_html=True,
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            _init_state()
            st.rerun()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg.role):
            st.write(msg.content)
            if msg.role == "assistant" and msg.sources:
                _render_sources(msg.sources)

    prompt = st.chat_input("Ask a question about your documents…")
    if not prompt:
        return

    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer…"):
            try:
                resp = _call_api(prompt, top_k=top_k)
                answer = str(resp.get("answer", "")).strip() or "No answer returned."
                sources = resp.get("sources") or []
            except httpx.HTTPStatusError as e:
                answer = f"API error ({e.response.status_code}). Please check the backend logs and try again."
                sources = []
            except httpx.RequestError:
                answer = (
                    "Could not reach the API. Make sure FastAPI is running and the API base URL is correct "
                    f"({api_url})."
                )
                sources = []

            st.write(answer)
            if sources:
                _render_sources(sources)

    st.session_state.messages.append(ChatMessage(role="assistant", content=answer, sources=sources))


if __name__ == "__main__":
    main()
