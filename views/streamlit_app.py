"""
Streamlit App — Legal AI Chat Interface

Premium dark-themed chat interface with:
- Arabic & English support (RTL aware)
- Debug mode toggle showing agent reasoning trace
- Conversation memory via session state
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from typing import Any

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controllers.graph import run_query
from models.vector_store import get_collection_count
from utils.config import LEGAL_AR_COLLECTION, get_embedding_function, get_llm

# ── Constants ─────────────────────────────────────────────────────────────────

PAGE_TITLE = "⚖️ المستشار القانوني | Legal AI"
PAGE_ICON = "⚖️"

DEFAULT_CONV_TITLE = "محادثة جديدة"
CONV_TITLE_MAX_CHARS = 30
SIDEBAR_TITLE_MAX_CHARS = 25

AVATAR_USER = "👤"
AVATAR_ASSISTANT = "⚖️"

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Kufi+Arabic:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary:      #0a0e1a;
    --bg-secondary:    #111827;
    --bg-glass:        rgba(255,255,255,0.03);
    --border-glass:    rgba(255,255,255,0.08);
    --text-primary:    #f0f4f8;
    --text-secondary:  #94a3b8;
    --accent-gold:     #d4a843;
    --accent-blue:     #3b82f6;
    --accent-emerald:  #10b981;
    --gradient-gold:   linear-gradient(135deg,#d4a843 0%,#f0c66b 50%,#d4a843 100%);
    --gradient-dark:   linear-gradient(180deg,#0a0e1a 0%,#111827 100%);
    --shadow-glow:     0 0 30px rgba(212,168,67,0.1);
}

/* ── Global ── */
.stApp { background: var(--gradient-dark) !important; font-family:'Inter','Noto Kufi Arabic',sans-serif !important; }
#MainMenu,footer,header { visibility:hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-glass) !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-gold) !important;
}

/* ── Header Banner ── */
.header-banner {
    background: linear-gradient(135deg,rgba(212,168,67,0.1) 0%,rgba(59,130,246,0.05) 100%);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    box-shadow: var(--shadow-glow);
    text-align: center;
}
.header-banner h1 {
    font-size: 2rem;
    font-weight: 700;
    background: var(--gradient-gold);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.header-banner p { color: var(--text-secondary); font-size: 0.95rem; margin:0; }

/* ── Chat Messages ── */
.chat-message {
    padding: 1.25rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    animation: fadeIn 0.3s ease-in-out;
    line-height: 1.7;
    color: var(--text-primary);
}
@keyframes fadeIn {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.user-message {
    background: linear-gradient(135deg,rgba(59,130,246,0.12) 0%,rgba(59,130,246,0.05) 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-left: 4px solid var(--accent-blue);
}
.assistant-message {
    background: linear-gradient(135deg,rgba(212,168,67,0.08) 0%,rgba(212,168,67,0.02) 100%);
    border: 1px solid rgba(212,168,67,0.15);
    border-left: 4px solid var(--accent-gold);
}

/* ── Debug Trace ── */
.debug-trace {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono','Courier New',monospace;
    font-size: 0.8rem;
    color: var(--accent-emerald);
    line-height: 1.8;
}

/* ── Metric Card ── */
.metric-card {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-card .value { font-size:1.5rem; font-weight:700; color:var(--accent-gold); }
.metric-card .label { font-size:0.8rem; color:var(--text-secondary); margin-top:4px; }

/* ── Active conversation entry ── */
.conv-active {
    background: rgba(212,168,67,0.15);
    border: 1px solid rgba(212,168,67,0.3);
    padding: 0.5rem;
    border-radius: 8px;
    margin-bottom: 4px;
}

/* ── Streamlit Widget Overrides ── */
.stChatMessage { background:transparent !important; }
.stTextInput > div > div > input {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    color: var(--text-primary) !important;
    border-radius: 12px !important;
}
.stChatInputContainer {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 16px !important;
}
div[data-testid="stChatInput"] textarea { font-size:0.95rem !important; }
.stButton > button {
    background: var(--gradient-gold) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(212,168,67,0.3) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFileUploader"] {
    background: var(--bg-glass) !important;
    border: 1px dashed var(--border-glass) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


# ── Model Preloading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ جاري تحميل نماذج الذكاء الاصطناعي... / Loading AI Models ...")
def preload_models() -> bool:
    """Eagerly load the LLM and Embedding models on first run."""
    get_llm()
    get_embedding_function()
    return True


# ── Session State ─────────────────────────────────────────────────────────────

def _new_conversation_entry() -> dict[str, Any]:
    return {
        "title": DEFAULT_CONV_TITLE,
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }


def init_session_state() -> None:
    """Initialise all required session state keys exactly once."""
    if "conversations" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state.conversations = {first_id: _new_conversation_entry()}

    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = next(iter(st.session_state.conversations))

    st.session_state.setdefault("debug_mode", False)
    st.session_state.setdefault("sidebar_collapsed", False)


def current_conversation() -> dict[str, Any]:
    """Return the currently-active conversation dict."""
    return st.session_state.conversations[st.session_state.current_thread_id]


# ── Conversation CRUD ─────────────────────────────────────────────────────────

def create_conversation() -> None:
    new_id = str(uuid.uuid4())
    st.session_state.conversations[new_id] = _new_conversation_entry()
    st.session_state.current_thread_id = new_id
    st.rerun()


def delete_conversation(thread_id: str) -> None:
    del st.session_state.conversations[thread_id]
    st.session_state.current_thread_id = next(iter(st.session_state.conversations))
    st.rerun()


def switch_conversation(thread_id: str) -> None:
    st.session_state.current_thread_id = thread_id
    st.rerun()


def _auto_title_conversation(conv: dict[str, Any]) -> None:
    """Set a conversation title from the first user message, if still default."""
    if conv["title"] != DEFAULT_CONV_TITLE:
        return
    first_user = next(
        (m["content"] for m in conv["messages"] if m["role"] == "user"), None
    )
    if first_user:
        conv["title"] = (
            first_user[:CONV_TITLE_MAX_CHARS] + "..."
            if len(first_user) > CONV_TITLE_MAX_CHARS
            else first_user
        )


# ── Rendering Helpers ─────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    return text[:max_len] + "..." if len(text) > max_len else text


def _html_message(content: str, role: str) -> str:
    css_class = "user-message" if role == "user" else "assistant-message"
    return f'<div class="chat-message {css_class}">{content}</div>'


def _html_debug_trace(trace: list[str]) -> str:
    return f'<div class="debug-trace">{"<br>".join(trace)}</div>'


def _html_metric_card(value: int | str, label: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div>'
        f"</div>"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_conversation_list() -> None:
    st.markdown("---")

    sorted_convs = sorted(
        st.session_state.conversations.items(),
        key=lambda kv: kv[1]["created_at"],
        reverse=True,
    )

    for thread_id, conv_data in sorted_convs:
        is_active = thread_id == st.session_state.current_thread_id
        label = f"📝 {_truncate(conv_data['title'], SIDEBAR_TITLE_MAX_CHARS)}"

        if is_active:
            if len(st.session_state.conversations) > 1:
                col_btn, col_del = st.columns([0.85, 0.15])
                with col_btn:
                    st.markdown(
                        f'<div class="conv-active" style="margin-bottom: 0;"><strong>✓ {label}</strong></div>',
                        unsafe_allow_html=True,
                    )
                with col_del:
                    if st.button("🗑️", key=f"del_{thread_id}"):
                        delete_conversation(thread_id)
            else:
                st.markdown(
                    f'<div class="conv-active"><strong>✓ {label}</strong></div>',
                    unsafe_allow_html=True,
                )
        else:
            col_btn, col_del = st.columns([0.85, 0.15])
            with col_btn:
                if st.button(label, key=f"conv_{thread_id}", use_container_width=True):
                    switch_conversation(thread_id)
            with col_del:
                if st.button("🗑️", key=f"del_{thread_id}"):
                    delete_conversation(thread_id)

    st.markdown("---")
    if st.button("+ محادثة جديدة", use_container_width=True):
        create_conversation()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ⚖️ المستشار القانوني")
        st.markdown("**Legal AI Assistant**")
        st.markdown("---")

        st.session_state.debug_mode = st.toggle(
            "🔍 Debug Mode / وضع التتبع",
            value=st.session_state.debug_mode,
            help="Show the agent's reasoning trace for each response",
        )

        st.markdown("---")
        st.markdown("### 📊 System Status")

        try:
            legal_count = get_collection_count(LEGAL_AR_COLLECTION)
        except Exception:
            legal_count = 0

        st.markdown(_html_metric_card(legal_count, "📜 Legal Chunks"), unsafe_allow_html=True)

        arrow = "▶" if st.session_state.sidebar_collapsed else "▼"
        if st.button(f"💬 المحادثات {arrow}", use_container_width=True):
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
            st.rerun()

        if not st.session_state.sidebar_collapsed:
            _render_conversation_list()

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;color:#64748b;font-size:0.75rem;'>"
            "Powered by Gemma 4 31B via OpenRouter<br>"
            "Egyptian Civil Code AI Assistant<br>"
            f"Chat: {st.session_state.current_thread_id[:8]}..."
            "</p>",
            unsafe_allow_html=True,
        )


# ── Main Content ──────────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown(
        """
        <div class="header-banner">
            <h1>⚖️ المستشار القانوني الذكي</h1>
            <p>AI-Powered Egyptian Civil Code Assistant — مساعد القانون المدني المصري بالذكاء الاصطناعي</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    for msg in current_conversation().get("messages", []):
        avatar = AVATAR_USER if msg["role"] == "user" else AVATAR_ASSISTANT
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(_html_message(msg["content"], msg["role"]), unsafe_allow_html=True)

            if (
                msg["role"] == "assistant"
                and st.session_state.debug_mode
                and msg.get("debug_trace")
            ):
                st.markdown(_html_debug_trace(msg["debug_trace"]), unsafe_allow_html=True)


def _call_agent(prompt: str, history: list[dict]) -> tuple[str, list[str]]:
    """Call the AI agent and return (response_text, debug_trace)."""
    result = run_query(
        query=prompt,
        conversation_history=history,
        thread_id=st.session_state.current_thread_id,
    )
    return result["response"], result.get("debug_trace", [])


def handle_chat_input() -> None:
    if not (prompt := st.chat_input("اسأل سؤالاً قانونياً... / Ask a legal question...")):
        return

    conv = current_conversation()

    # Record and display user message immediately
    conv["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(_html_message(prompt, "user"), unsafe_allow_html=True)

    # Build history excluding the message we just appended
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in conv["messages"][:-1]
    ]

    with st.chat_message("assistant", avatar=AVATAR_ASSISTANT):
        with st.spinner("جاري التحليل... / Analyzing..."):
            try:
                response, debug_trace = _call_agent(prompt, history)
            except Exception as exc:
                response = f"❌ Error: {exc}"
                debug_trace = [f"[Error] {exc}"]
                st.error(response)

        st.markdown(_html_message(response, "assistant"), unsafe_allow_html=True)

        if st.session_state.debug_mode and debug_trace:
            st.markdown(_html_debug_trace(debug_trace), unsafe_allow_html=True)

    conv["messages"].append({"role": "assistant", "content": response, "debug_trace": debug_trace})
    _auto_title_conversation(conv)


def main() -> None:
    """Main execution function for the Streamlit app."""
    inject_css()
    preload_models()
    init_session_state()
    render_sidebar()
    render_header()
    render_chat_history()
    handle_chat_input()

if __name__ == "__main__":
    main()
