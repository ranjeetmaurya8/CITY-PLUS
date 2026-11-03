"""
Smart City Intelligence System - Streamlit UI
Run:  streamlit run app.py
"""
from dotenv import load_dotenv

load_dotenv()

import os
import json
from datetime import datetime

import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CITY PLUS",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = (
    "You are the assistant for a Smart City Intelligence System. "
    "Use get_weather for current weather and get_news for latest city news. "
    "Answer clearly and concisely. If a tool call is denied, explain what you "
    "could not do and offer an alternative."
)

# ──────────────────────────────────────────────────────────────
# Styling
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #101828;
  --muted: #667085;
  --line: #E4E8EF;
  --surface: #FFFFFF;
  --canvas: #F6F8FB;
  --transit: #1D4ED8;
  --transit-soft: #E8EEFF;
  --amber: #B45309;
  --amber-soft: #FEF3C7;
  --mint: #047857;
  --mint-soft: #D1FAE5;
  --rose: #BE123C;
  --rose-soft: #FFE4E6;
}

html, body, [class*="css"], .stMarkdown, .stChatInput textarea {
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
code, pre, .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.6rem; padding-bottom: 6rem; max-width: 980px; }

/* Masthead */
.masthead {
  background: var(--ink);
  color: #fff;
  border-radius: 14px;
  padding: 26px 30px;
  margin-bottom: 18px;
  display: flex; align-items: center; justify-content: space-between; gap: 20px;
}
.masthead h1 { margin: 0; font-size: 1.65rem; font-weight: 600; letter-spacing: -0.01em; color: #fff; padding: 0; }
.masthead p { margin: 6px 0 0; color: #AAB4C5; font-size: 0.95rem; }
.masthead .model {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
  background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.16);
  padding: 6px 12px; border-radius: 999px; color: #D7DEEA; white-space: nowrap;
}

/* Badges */
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600;
}
.badge.pending  { background: var(--amber-soft); color: var(--amber); }
.badge.approved { background: var(--mint-soft);  color: var(--mint); }
.badge.denied   { background: var(--rose-soft);  color: var(--rose); }
.badge.auto     { background: var(--transit-soft); color: var(--transit); }

/* Tool activity row */
.tool-row {
  background: var(--surface); border: 1px solid var(--line);
  border-left: 4px solid var(--transit);
  border-radius: 10px; padding: 10px 14px; margin: 4px 0 2px;
  display: flex; justify-content: space-between; align-items: center; gap: 12px;
}
.tool-row.denied { border-left-color: var(--rose); }
.tool-row .name { font-weight: 600; }
.tool-row .args { font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: var(--muted); }

/* Approval panel */
.approval-title { font-weight: 600; font-size: 1.02rem; margin-bottom: 2px; }
.approval-sub { color: var(--muted); font-size: 0.88rem; margin-bottom: 10px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
.status-line { display:flex; justify-content:space-between; font-size:0.88rem; padding:4px 0; }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
.dot.ok { background:#10B981; } .dot.bad { background:#E11D48; }

/* Empty state */
.empty { text-align:center; padding: 38px 10px 14px; color: var(--muted); }
.empty h3 { color: var(--ink); font-weight: 600; margin-bottom: 4px; }

/* Buttons */
.stButton > button { border-radius: 8px; font-weight: 500; border: 1px solid var(--line); }
.stButton > button[kind="primary"] { background: var(--transit); border-color: var(--transit); }

/* Chat */
[data-testid="stChatMessage"] { background: transparent; padding: 6px 0; }
[data-testid="stChatInput"] { border-radius: 12px; }

@media (max-width: 640px) {
  .masthead { flex-direction: column; align-items: flex-start; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY is not set"

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={city},IN&appid={api_key}&units=metric"
    )
    response = requests.get(url, timeout=10)
    data = response.json()

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    feels = data["main"].get("feels_like")
    humidity = data["main"].get("humidity")
    desc = data["weather"][0]["description"]
    extra = f", feels like {feels}°C" if feels is not None else ""
    extra += f", humidity {humidity}%" if humidity is not None else ""
    return f"Weather in {city}: {desc}, {temp}°C{extra}"


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set"

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=f"latest news in {city}", search_depth="basic", max_results=3
    )
    results = response.get("results", [])
    if not results:
        return f"No news found for {city}"

    items = []
    for r in results:
        title = r.get("title", "No title")
        url = r.get("url", "")
        snippet = r.get("content", "")[:140]
        items.append(f"**[{title}]({url})**  \n{snippet}...")
    return f"Latest news in {city}:\n\n" + "\n\n".join(items)


TOOLS = {"get_weather": get_weather, "get_news": get_news}
TOOL_META = {
    "get_weather": ("🌦️", "Weather", "Current conditions from OpenWeather"),
    "get_news": ("📰", "News", "Latest headlines from Tavily"),
}

# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────
def init_state():
    ss = st.session_state
    ss.setdefault("lc_messages", [])      # LangChain message history
    ss.setdefault("ui_log", [])           # what the UI renders
    ss.setdefault("pending", [])          # tool calls awaiting approval
    ss.setdefault("decisions", {})        # tool_call_id -> "approved" | "denied"
    ss.setdefault("auto_approve", False)
    ss.setdefault("queued_prompt", None)
    ss.setdefault("stats", {"tools": 0, "approved": 0, "denied": 0, "turns": 0})


init_state()
ss = st.session_state


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model=MODEL_NAME).bind_tools([get_weather, get_news])


def now():
    return datetime.now().strftime("%H:%M")


# ──────────────────────────────────────────────────────────────
# Agent logic
# ──────────────────────────────────────────────────────────────
def execute_tool(tc, mode):
    """Run one tool call, append ToolMessage + UI entry."""
    ss.stats["tools"] += 1
    if mode == "denied":
        ss.stats["denied"] += 1
        result = "Tool execution denied by user."
    else:
        ss.stats["approved"] += 1
        try:
            result = TOOLS[tc["name"]].invoke(tc["args"])
        except Exception as e:  # keep the agent alive on tool failure
            result = f"Error while running {tc['name']}: {e}"

    ss.lc_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    ss.ui_log.append(
        {"role": "tool", "name": tc["name"], "args": tc["args"],
         "mode": mode, "result": str(result), "time": now()}
    )


def run_agent():
    """Call the model until it produces a final answer or needs approval."""
    llm = get_llm()
    for _ in range(8):  # hard cap on tool-call loops
        result = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + ss.lc_messages)
        ss.lc_messages.append(result)

        if result.tool_calls:
            if ss.auto_approve:
                for tc in result.tool_calls:
                    execute_tool(tc, "auto")
                continue
            ss.pending = list(result.tool_calls)
            ss.decisions = {}
            return

        ss.ui_log.append({"role": "assistant", "content": result.content, "time": now()})
        return

    ss.ui_log.append(
        {"role": "error", "content": "Stopped after too many tool-call rounds."}
    )


def safe_run_agent():
    try:
        run_agent()
    except Exception as e:
        ss.ui_log.append({"role": "error", "content": str(e)})


def decide(call_id, value):
    ss.decisions[call_id] = value


def decide_all(value):
    for tc in ss.pending:
        ss.decisions[tc["id"]] = value


def reset_chat():
    ss.lc_messages, ss.ui_log, ss.pending, ss.decisions = [], [], [], {}
    ss.stats = {"tools": 0, "approved": 0, "denied": 0, "turns": 0}


# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏙️ Smart City")
    st.caption("Weather and news for any Indian city, with you approving every tool call.")

    st.markdown("**Approval mode**")
    ss.auto_approve = st.toggle(
        "Auto-approve tool calls",
        value=ss.auto_approve,
        help="Off: the agent asks before every tool call. On: tools run immediately.",
    )
    if ss.auto_approve:
        st.warning("Tools will run without asking.", icon="⚠️")

    st.divider()
    st.markdown("**Connections**")
    for label, key in [("Groq", "GROQ_API_KEY"), ("OpenWeather", "OPENWEATHER_API_KEY"), ("Tavily", "TAVILY_API_KEY")]:
        ok = bool(os.getenv(key))
        st.markdown(
            f"<div class='status-line'><span><span class='dot {'ok' if ok else 'bad'}'></span>{label}</span>"
            f"<span style='color:var(--muted)'>{'Connected' if ok else 'Key missing'}</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**This session**")
    c1, c2 = st.columns(2)
    c1.metric("Questions", ss.stats["turns"])
    c2.metric("Tool calls", ss.stats["tools"])
    c3, c4 = st.columns(2)
    c3.metric("Approved", ss.stats["approved"])
    c4.metric("Denied", ss.stats["denied"])

    st.divider()
    st.markdown("**Available tools**")
    for name, (icon, title, desc) in TOOL_META.items():
        st.markdown(f"{icon} **{title}**  \n<span style='color:var(--muted);font-size:0.85rem'>{desc}</span>", unsafe_allow_html=True)

    st.divider()
    st.button("Clear conversation", on_click=reset_chat, use_container_width=True)
    if ss.ui_log:
        st.download_button(
            "Export chat (JSON)",
            data=json.dumps(ss.ui_log, indent=2, default=str),
            file_name="smart_city_session.json",
            mime="application/json",
            use_container_width=True,
        )

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="masthead">
  <div>
    <h1>Smart City Intelligence</h1>
    <p>Ask about weather or local news. Review each tool call before it runs.</p>
  </div>
  <div class="model">{MODEL_NAME}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Conversation
# ──────────────────────────────────────────────────────────────
def render_entry(e):
    role = e["role"]
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(e["content"])
    elif role == "assistant":
        with st.chat_message("assistant", avatar="🏙️"):
            st.markdown(e["content"])
    elif role == "tool":
        icon, title, _ = TOOL_META.get(e["name"], ("🔧", e["name"], ""))
        mode = e["mode"]
        badge = {"approved": "approved", "auto": "auto", "denied": "denied"}[mode]
        label = {"approved": "Approved", "auto": "Auto-approved", "denied": "Denied"}[mode]
        args = ", ".join(f"{k}={v}" for k, v in e["args"].items())
        with st.chat_message("assistant", avatar="🔧"):
            st.markdown(
                f"<div class='tool-row {'denied' if mode == 'denied' else ''}'>"
                f"<div><span class='name'>{icon} {title}</span> "
                f"<span class='args'>{e['name']}({args})</span></div>"
                f"<span class='badge {badge}'>{label}</span></div>",
                unsafe_allow_html=True,
            )
            if mode != "denied":
                with st.expander("View tool result"):
                    st.markdown(e["result"])
    elif role == "error":
        st.error(e["content"], icon="🚨")


if not ss.ui_log and not ss.pending:
    st.markdown(
        "<div class='empty'><h3>What would you like to know?</h3>"
        "Pick a starting point or type your own question below.</div>",
        unsafe_allow_html=True,
    )
    examples = [
        "What's the weather in Lucknow?",
        "Latest news in Delhi",
        "Weather and news for Mumbai",
    ]
    cols = st.columns(len(examples))
    for col, text in zip(cols, examples):
        if col.button(text, use_container_width=True, key=f"ex_{text}"):
            ss.queued_prompt = text
            st.rerun()

for entry in ss.ui_log:
    render_entry(entry)

# ──────────────────────────────────────────────────────────────
# Human-in-the-loop approval panel
# ──────────────────────────────────────────────────────────────
if ss.pending:
    with st.chat_message("assistant", avatar="🏙️"):
        with st.container(border=True):
            n = len(ss.pending)
            st.markdown(
                f"<div class='approval-title'>Approval needed "
                f"<span class='badge pending'>{n} tool call{'s' if n > 1 else ''}</span></div>"
                f"<div class='approval-sub'>The agent can't continue until you decide.</div>",
                unsafe_allow_html=True,
            )
            for tc in ss.pending:
                icon, title, desc = TOOL_META.get(tc["name"], ("🔧", tc["name"], ""))
                state = ss.decisions.get(tc["id"])
                left, right = st.columns([3, 2])
                with left:
                    st.markdown(f"**{icon} {title}**  \n<span style='color:var(--muted);font-size:0.85rem'>{desc}</span>", unsafe_allow_html=True)
                    st.code(json.dumps(tc["args"]), language="json")
                with right:
                    if state:
                        st.markdown(
                            f"<span class='badge {state}'>{'Approved' if state == 'approved' else 'Denied'}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        b1, b2 = st.columns(2)
                        b1.button("Approve", key=f"ok_{tc['id']}", type="primary",
                                  on_click=decide, args=(tc["id"], "approved"), use_container_width=True)
                        b2.button("Deny", key=f"no_{tc['id']}",
                                  on_click=decide, args=(tc["id"], "denied"), use_container_width=True)
            if n > 1 and len(ss.decisions) < n:
                a1, a2 = st.columns(2)
                a1.button("Approve all", on_click=decide_all, args=("approved",), use_container_width=True)
                a2.button("Deny all", on_click=decide_all, args=("denied",), use_container_width=True)

    if all(tc["id"] in ss.decisions for tc in ss.pending):
        calls, decisions = ss.pending, dict(ss.decisions)
        ss.pending, ss.decisions = [], {}
        with st.spinner("Running approved tools..."):
            for tc in calls:
                execute_tool(tc, decisions[tc["id"]])
            safe_run_agent()
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Input
# ──────────────────────────────────────────────────────────────
typed = st.chat_input(
    "Ask about weather or news in a city...",
    disabled=bool(ss.pending),
)
prompt = typed or ss.queued_prompt
ss.queued_prompt = None

if prompt:
    ss.stats["turns"] += 1
    ss.ui_log.append({"role": "user", "content": prompt, "time": now()})
    ss.lc_messages.append(HumanMessage(content=prompt))
    with st.spinner("Thinking..."):
        safe_run_agent()
    st.rerun()