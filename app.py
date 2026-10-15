"""Streamlit front end for the Personal Finance AI Agent.

Single-column layout (works on mobile). The OpenAI key is read from
Streamlit secrets or a .env file, never typed by the visitor. Usage limits
protect the key from abuse.
"""

import base64
import os
import threading
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agent import ask, build_agent

load_dotenv()  # reads .env when running locally; harmless if the file is missing

# ---------------------------------------------------------------- limits
MAX_QUESTIONS_PER_SESSION = 15   # per visitor session
MAX_QUESTIONS_PER_DAY = 100      # across all visitors, resets daily / on app restart
MAX_QUESTION_CHARS = 200

st.set_page_config(
    page_title="Personal Finance AI Agent",
    page_icon="₹",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Hide the sidebar toggle completely so the page is one clean column.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"], [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] { display: none; }
    /* Content column (scrolls inside the fixed frame below) */
    .block-container, [data-testid="stMainBlockContainer"] {
        max-width: 760px;
        padding: 2.5rem 1.2rem 8rem;
    }

    /* Neon frame fixed to the whole window: it never moves while scrolling */
    .stApp::after {
        content: "";
        position: fixed;
        inset: 10px;
        border: 1.5px solid rgba(0, 200, 255, 0.65);
        border-radius: 18px;
        box-shadow: 0 0 14px rgba(0, 200, 255, 0.35), inset 0 0 24px rgba(0, 200, 255, 0.07);
        pointer-events: none;
        z-index: 1000000;
    }
    @media (max-width: 640px) {
        .stApp::after { inset: 6px; border-radius: 14px; }
        .block-container, [data-testid="stMainBlockContainer"] { padding: 2rem 0.9rem 8rem; }
    }

    /* Keep the prompt box pinned at the bottom, inside the frame */
    [data-testid="stBottom"] { z-index: 999; }
    [data-testid="stBottom"] > div { padding-bottom: 1.6rem; }

    /* Hide the Deploy bar so it does not cross the frame */
    [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }

    /* Fallback background colour (the image is added below when the file exists) */
    .stApp { background: #0B1220; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stBottom"] > div { background: transparent; }

    /* Title: emerald to gold */
    h1 {
        background: linear-gradient(90deg, #34D399, #F5C451);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Chat bubbles and expanders as soft cards */
    [data-testid="stChatMessage"] {
        background: #131C2E;
        border: 1px solid #1F2C44;
        border-radius: 14px;
        padding: 0.9rem 1rem;
    }
    [data-testid="stExpander"] {
        background: #101A2B;
        border: 1px solid #1F2C44;
        border-radius: 12px;
    }

    /* Buttons: outlined emerald, filled on hover */
    .stButton > button {
        background: #131C2E;
        color: #E6EDF7;
        border: 1px solid #2A3B57;
        border-radius: 12px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #34D399;
        color: #34D399;
        background: #10261F;
    }

    /* Prompt box: neon blue border with glow */
    [data-testid="stChatInput"] {
        border: 1.5px solid #00C8FF;
        border-radius: 14px;
        box-shadow: 0 0 8px rgba(0, 200, 255, 0.65), 0 0 22px rgba(0, 200, 255, 0.28);
    }
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div:focus-within {
        border: none;
        box-shadow: none;
        border-radius: 14px;
    }

    /* Send arrow: neon blue (filled and glowing once there is text to send) */
    [data-testid="stChatInput"] button {
        background: transparent;
        color: #00C8FF;
    }
    [data-testid="stChatInput"] button svg { fill: currentColor; }
    [data-testid="stChatInput"] button:not(:disabled) {
        background: #00C8FF;
        color: #06121F;
        box-shadow: 0 0 10px rgba(0, 200, 255, 0.8);
    }

    /* Example dropdown */
    [data-baseweb="select"] > div {
        background: #131C2E;
        border: 1px solid rgba(0, 200, 255, 0.45);
        border-radius: 12px;
    }

    /* Header: title on the left, neon "!" tools badge on the right */
    .hdr { position: relative; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .hdr h1 { margin: 0 !important; padding: 0 !important; font-size: clamp(1.7rem, 5.5vw, 2.6rem) !important; line-height: 1.15; }
    .tt-wrap { position: relative; flex-shrink: 0; outline: none; }
    .tt-badge {
        display: flex; align-items: center; justify-content: center;
        width: 38px; height: 38px; border-radius: 50%;
        border: 1.5px solid #00C8FF; color: #00C8FF;
        font-weight: 800; font-size: 1.25rem; cursor: pointer; user-select: none;
        background: rgba(0, 200, 255, 0.08); text-shadow: 0 0 8px #00C8FF;
        box-shadow: 0 0 10px rgba(0, 200, 255, 0.55), inset 0 0 8px rgba(0, 200, 255, 0.2);
        transition: all 0.15s ease;
    }
    .tt-wrap:hover .tt-badge, .tt-wrap:focus-within .tt-badge {
        background: #00C8FF; color: #06121F; text-shadow: none;
        box-shadow: 0 0 16px rgba(0, 200, 255, 0.95);
    }
    .tt-badge.pulse { animation: tt-pulse 1.1s ease-in-out 3; }
    @keyframes tt-pulse {
        0%, 100% { box-shadow: 0 0 10px rgba(0, 200, 255, 0.55); }
        50% { box-shadow: 0 0 22px rgba(0, 200, 255, 1), 0 0 36px rgba(0, 200, 255, 0.6); }
    }
    .tt-panel {
        position: absolute; top: 100%; right: 0; padding-top: 10px;
        width: 310px; max-width: calc(100vw - 3rem);
        opacity: 0; visibility: hidden; transform: translateY(-4px);
        transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s;
        z-index: 1000001;
    }
    .tt-wrap:hover .tt-panel, .tt-wrap:focus-within .tt-panel { opacity: 1; visibility: visible; transform: none; }
    .tt-box {
        background: #0F1A2C; border: 1px solid rgba(0, 200, 255, 0.6); border-radius: 14px;
        padding: 12px; box-shadow: 0 0 18px rgba(0, 200, 255, 0.3);
    }
    .tt-title { font-size: 0.78rem; letter-spacing: 0.05em; color: #9FB0C8; margin: 0 0 8px 2px; }
    .tt-row {
        display: flex; justify-content: space-between; align-items: center; gap: 8px;
        padding: 8px 10px; margin-bottom: 6px; border-radius: 10px;
        border: 1px solid #1F2C44; background: #131C2E; transition: all 0.2s ease;
    }
    .tt-row b { display: block; color: #E6EDF7; font-size: 0.92rem; }
    .tt-row .d { display: block; color: #9FB0C8; font-size: 0.78rem; }
    .tt-row.on { border-color: #00C8FF; background: rgba(0, 200, 255, 0.12); box-shadow: 0 0 12px rgba(0, 200, 255, 0.45); }
    .tt-row.on b { color: #00E5FF; }
    .tt-row.off { opacity: 0.3; filter: grayscale(0.6); }
    .tt-pill { font-size: 0.68rem; font-weight: 700; color: #06121F; background: #00C8FF; border-radius: 999px; padding: 2px 8px; white-space: nowrap; }
    .tt-note { margin: 6px 2px 0; font-size: 0.76rem; color: #9FB0C8; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Background image: assets/background.svg, darkened with an overlay so text stays readable.
_BG_FILE = Path(__file__).parent / "assets" / "background.svg"
if _BG_FILE.exists():
    _b64 = base64.b64encode(_BG_FILE.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(11, 18, 32, 0.45), rgba(11, 18, 32, 0.78)),
                url("data:image/svg+xml;base64,{_b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

EXAMPLES = [
    "Calculate EMI for ₹10 lakh loan at 8% interest for 20 years",
    "If I invest ₹7000 monthly for 15 years at 10% return",
    "Create a budget plan for income ₹90,000",
    "What is 20% of ₹85,000",
    "How much will I get if I invest ₹3000 per month for 5 years at 8% return",
    "How much should I save monthly to reach ₹20 lakh in 10 years at 12% return?",
]


# ---------------------------------------------------------------- helpers
def get_api_key() -> str:
    """Streamlit secrets first (deployed app), then environment / .env (local)."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key
    except Exception:  # no secrets file present
        pass
    return os.getenv("OPENAI_API_KEY", "")


@st.cache_resource
def _daily_counter() -> dict:
    return {"day": date.today(), "count": 0, "lock": threading.Lock()}


def daily_limit_reached() -> bool:
    c = _daily_counter()
    with c["lock"]:
        if c["day"] != date.today():
            c["day"], c["count"] = date.today(), 0
        return c["count"] >= MAX_QUESTIONS_PER_DAY


def count_question() -> None:
    c = _daily_counter()
    with c["lock"]:
        c["count"] += 1


@st.cache_resource
def get_executor(api_key: str):
    """Build the agent once and reuse it across questions."""
    return build_agent(api_key=api_key)


def render_calls(calls) -> None:
    for c in calls:
        with st.expander(f"Tool used: {c['tool']}"):
            st.json(c["inputs"])
            st.write(c["output"])


def pick_example() -> None:
    choice = st.session_state.get("example_pick")
    if choice:
        st.session_state["pending"] = choice
    st.session_state["example_pick"] = None  # reset so the same example can be picked again


def clear_chat() -> None:
    st.session_state["messages"] = []
    st.session_state["used"] = 0


# Tools shown in the "!" panel: (internal tool name, label, short description)
TOOLS = [
    ("calculator", "Calculator", "Percentages and quick sums"),
    ("emi_calculator", "EMI Calculator", "Monthly loan instalment and interest"),
    ("sip_calculator", "SIP Calculator", "Future value of monthly investing"),
    ("required_monthly_savings", "Savings Goal", "Monthly amount needed for a target"),
    ("budget_planner", "Budget Planner", "50-30-20 split of your income"),
]


def tools_header_html() -> str:
    """Title plus the neon '!' badge. Tools used in the last answer glow, the rest fade."""
    msgs = st.session_state.get("messages", [])
    last = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
    answered = last is not None
    used = {c["tool"] for c in last.get("tool_calls", [])} if answered else set()

    rows = []
    for name, label, desc in TOOLS:
        state = "" if not answered else ("on" if name in used else "off")
        pill = '<span class="tt-pill">USED</span>' if state == "on" else ""
        rows.append(
            f'<div class="tt-row {state}"><div><b>{label}</b><span class="d">{desc}</span></div>{pill}</div>'
        )

    if not answered:
        note = "After an answer, the tool that was used lights up."
    elif used:
        note = "Highlighted: the tool used for the last answer."
    else:
        note = "The last answer did not need a tool."

    pulse = " pulse" if used else ""
    # Kept on one line per block: markdown must not see blank lines or indented lines.
    return (
        '<div class="hdr"><h1>Personal Finance AI Agent</h1>'
        '<div class="tt-wrap" tabindex="0">'
        f'<div class="tt-badge{pulse}">!</div>'
        '<div class="tt-panel"><div class="tt-box">'
        f'<div class="tt-title">TOOLS AVAILABLE ({len(TOOLS)})</div>'
        f'{"".join(rows)}<div class="tt-note">{note}</div>'
        '</div></div></div></div>'
    )


# ---------------------------------------------------------------- state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("used", 0)

# ---------------------------------------------------------------- page
st.markdown(tools_header_html(), unsafe_allow_html=True)
st.caption(
    "Smart finance answers. "
    "AI that calculates, not guesses."
)

st.selectbox(
    "Try an example",
    EXAMPLES,
    index=None,
    placeholder="Try an example question...",
    key="example_pick",
    on_change=pick_example,
    label_visibility="collapsed",
)

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        render_calls(m.get("tool_calls", []))

if st.session_state["messages"]:
    st.button("Clear chat", on_click=clear_chat)

typed = st.chat_input("e.g. Calculate EMI for ₹10 lakh at 9% for 20 years")
question = typed or st.session_state.pop("pending", None)

if question:
    api_key = get_api_key()
    if not api_key:
        st.error("The app is not configured with an API key yet. Please contact the owner.")
        st.stop()
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
        st.stop()
    if st.session_state["used"] >= MAX_QUESTIONS_PER_SESSION:
        st.warning("You have reached the question limit for this session. Refresh the page to start again.")
        st.stop()
    if daily_limit_reached():
        st.warning("The demo has reached its daily usage limit. Please try again tomorrow.")
        st.stop()

    history = [(m["role"], m["content"]) for m in st.session_state["messages"]]
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = ask(get_executor(api_key), question, history)
            except Exception as exc:  # show a safe message, never the raw error
                st.error(f"Something went wrong ({type(exc).__name__}). Please try again.")
                st.stop()
        st.write(res["answer"])
        render_calls(res["tool_calls"])

    # Save to history only after a successful answer, so failed calls leave no orphan message.
    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["messages"].append(
        {"role": "assistant", "content": res["answer"], "tool_calls": res["tool_calls"]}
    )
    st.session_state["used"] += 1
    count_question()
    st.rerun()