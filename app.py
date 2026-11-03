"""AI Study Planner - single-file Streamlit app.

Run:   pip install streamlit google-genai
       export GEMINI_API_KEY="your-key"
       streamlit run app.py
"""
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

# =============================================================== DATABASE
DB_PATH = Path(os.getenv("STUDY_DB_PATH", Path(__file__).parent / "study_planner.db"))
PBKDF2_ITERATIONS = 200_000
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash BLOB NOT NULL,
                salt BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, updated_at);
            CREATE INDEX IF NOT EXISTS idx_msgs_chat ON messages(chat_id, id);
            CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id);
            """
        )


# ---------------------------------------------------------------- accounts
def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ITERATIONS)


def register_user(name: str, email: str, password: str):
    """Returns (user_dict | None, error_message | None)."""
    name, email = name.strip(), email.strip().lower()
    if not name:
        return None, "Please enter your name."
    if not EMAIL_RE.match(email):
        return None, "Please enter a valid email address."
    if len(password) < 8:
        return None, "Password must be at least 8 characters."
    salt = os.urandom(16)
    try:
        with get_conn() as c:
            cur = c.execute(
                "INSERT INTO users (name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
                (name, email, _hash(password, salt), salt),
            )
            return {"id": cur.lastrowid, "name": name, "email": email}, None
    except sqlite3.IntegrityError:
        return None, "An account with this email already exists."


def authenticate(email: str, password: str):
    with get_conn() as c:
        row = c.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
    if row and hmac.compare_digest(row["password_hash"], _hash(password, row["salt"])):
        return {"id": row["id"], "name": row["name"], "email": row["email"]}
    return None


# ------------------------------------------------------------------- chats
def create_chat(user_id: int, title: str) -> int:
    with get_conn() as c:
        return c.execute(
            "INSERT INTO chats (user_id, title) VALUES (?, ?)", (user_id, title[:60])
        ).lastrowid


def list_chats(user_id: int):
    with get_conn() as c:
        return c.execute(
            "SELECT id, title FROM chats WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
            (user_id,),
        ).fetchall()


def get_messages(chat_id: int, user_id: int):
    """Returns messages only if the chat belongs to user_id."""
    with get_conn() as c:
        return c.execute(
            """SELECT m.role, m.content FROM messages m
               JOIN chats ch ON ch.id = m.chat_id
               WHERE m.chat_id = ? AND ch.user_id = ? ORDER BY m.id""",
            (chat_id, user_id),
        ).fetchall()


def add_message(chat_id: int, user_id: int, role: str, content: str) -> bool:
    with get_conn() as c:
        owned = c.execute(
            "SELECT 1 FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
        ).fetchone()
        if not owned:
            return False
        c.execute(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content),
        )
        c.execute("UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (chat_id,))
        return True


def delete_chat(chat_id: int, user_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))


# ------------------------------------------------------------------ topics
def add_topics(user_id: int, titles):
    titles = [t.strip() for t in titles if t.strip()]
    with get_conn() as c:
        c.executemany(
            "INSERT INTO topics (user_id, title) VALUES (?, ?)",
            [(user_id, t[:120]) for t in titles],
        )
    return len(titles)


def list_topics(user_id: int):
    with get_conn() as c:
        return c.execute(
            "SELECT id, title, completed FROM topics WHERE user_id = ? ORDER BY completed, id",
            (user_id,),
        ).fetchall()


def set_topic_done(topic_id: int, user_id: int, done: bool):
    with get_conn() as c:
        c.execute(
            """UPDATE topics SET completed = ?,
               completed_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE NULL END
               WHERE id = ? AND user_id = ?""",
            (int(done), int(done), topic_id, user_id),
        )


def delete_topic(topic_id: int, user_id: int):
    with get_conn() as c:
        c.execute("DELETE FROM topics WHERE id = ? AND user_id = ?", (topic_id, user_id))


# ============================================================ AI SERVICE
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


class AIError(Exception):
    """Raised with a user-friendly message when the AI call fails."""


def _api_key() :
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def build_system_prompt(student_name: str, topics) -> str:
    pending = [t["title"] for t in topics if not t["completed"]]
    done = [t["title"] for t in topics if t["completed"]]
    return f"""You are AI Study Planner, a friendly, encouraging study assistant for a student named {student_name}.
Today's date is {date.today():%A, %d %B %Y}.

Your jobs:
1. Create daily and weekly study timetables (use clear tables or time-blocked lists; ask for subjects, exam dates and available hours if missing, but give a useful draft first when you can).
2. Recommend what to study next, based on the student's pending topics, difficulty and deadlines.
3. Track progress. The student's tracker right now:
   - Pending topics: {pending or "none recorded"}
   - Completed topics: {done or "none recorded"}
   Refer to it, celebrate progress, and remind them they can add/tick topics in the Study Tracker panel.
4. When the student misses sessions, reschedule kindly: do not guilt-trip, shift missed topics into upcoming free slots, and prioritise the most important ones.
5. Answer study-related questions with simple, relevant, friendly explanations (short paragraphs, small examples, analogies).

Style: warm, concise, easy to read, light Markdown. If a question is unrelated to studying or learning, gently steer back to studying."""


def get_reply(history, student_name: str, topics) -> str:
    """history: [{'role': 'user'|'assistant', 'content': str}, ...] ending with the new user message."""
    key = _api_key()
    if not key:
        raise AIError(
            "Gemini API key not found. Set the GEMINI_API_KEY environment variable "
            "or add it to .streamlit/secrets.toml."
        )
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=build_system_prompt(student_name, topics),
                temperature=0.7,
            ),
        )
    except Exception as exc:  # network, quota, invalid key...
        raise AIError(f"The AI service returned an error: {exc}") from exc
    if not response.text:
        raise AIError("The AI returned an empty response. Please try rephrasing.")
    return response.text


# ===================================================================== UI
init_db()
st.set_page_config(page_title="AI Study Planner", page_icon="📖", layout="wide")

CSS = """
<style>
:root { --bg:#0f172a; --panel:#1e293b; --panel2:#273449; --line:#334155; --text:#e2e8f0; --muted:#94a3b8;
        --blue:#4a90e2; --blue-light:#7db8f5; --blue-dark:#3b7bd0; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg); color: var(--text); }
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] p { color: var(--text); }
.stApp [data-testid="stCaptionContainer"] { color: var(--muted); }
footer, #MainMenu { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] * { color: var(--text); }
.block-container { padding-top: 1.2rem; max-width: 980px; }
[data-testid="stSidebar"] { background: #111b2e; border-right: 1px solid var(--line); }
.brand { font-size: 1.6rem; font-weight: 700; color: var(--blue-light); white-space: nowrap; }
.brand small { font-size: .8rem; font-weight: 400; color: var(--muted); display:block; }
.hello { text-align: right; color: var(--muted); padding-top: .5rem; font-size: .95rem; }
.hello b { color: var(--text); }
div.stButton > button, div.stFormSubmitButton > button {
    border-radius: 10px; border: 1px solid var(--line); background: var(--panel); color: var(--text); }
div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    border-color: var(--blue); color: var(--blue-light); background: var(--panel2); }
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
    background: var(--blue); color: #fff; border: none; }
div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {
    background: var(--blue-dark); color: #fff; }
div.stButton > button[kind="primary"] p, div.stFormSubmitButton > button[kind="primary"] p { color: #fff; }
[data-testid="stChatMessage"] { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: .8rem 1rem; }
[data-testid="stExpander"] { background: var(--panel); border-radius: 12px; border: 1px solid var(--line); }
[data-testid="stExpander"] summary:hover { color: var(--blue-light); }
input, textarea, [data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"] {
    background: var(--bg) !important; color: var(--text) !important; }
[data-baseweb="input"], [data-baseweb="textarea"] { border: 1px solid var(--line) !important; border-radius: 10px; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: var(--bg); }
[data-testid="stChatInput"] { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; }
[data-testid="stChatInput"] textarea { background: transparent !important; }
div[role="dialog"] { background: var(--panel); color: var(--text); border: 1px solid var(--line); }
[data-baseweb="tab-list"] { background: transparent; }
[data-baseweb="tab"] { color: var(--muted); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--blue-light); }
[data-testid="stForm"] { border: 1px solid var(--line); background: transparent; }
[data-testid="stAlert"] { background: var(--panel2); color: var(--text); }
.empty { text-align:center; padding: 2rem 1rem 1rem; color: var(--muted); }
.empty h3 { color: var(--blue-light); margin-bottom:.2rem; }
@media (max-width: 640px) { .brand { font-size: 1.3rem; } .hello { text-align:left; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("user", None)
ss.setdefault("chat_id", None)


# ------------------------------------------------------------- callbacks
def open_chat(chat_id):
    ss.chat_id = chat_id


def remove_chat(chat_id):
    delete_chat(chat_id, ss.user["id"])
    if ss.chat_id == chat_id:
        ss.chat_id = None


def new_chat():
    ss.chat_id = None


def queue_prompt(text):
    ss.queued_prompt = text


def toggle_topic(topic_id):
    set_topic_done(topic_id, ss.user["id"], ss[f"topic_{topic_id}"])


def remove_topic(topic_id):
    delete_topic(topic_id, ss.user["id"])


def logout():
    ss.user = None
    ss.chat_id = None


# ----------------------------------------------------------- auth dialog
@st.dialog("Welcome to AI Study Planner")
def auth_dialog():
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", type="primary", use_container_width=True):
                user = authenticate(email, password)
                if user:
                    ss.user, ss.chat_id = user, None
                    st.rerun()
                else:
                    st.error("Incorrect email or password.")
    with tab_register:
        with st.form("register_form"):
            name = st.text_input("Name")
            email = st.text_input("Email", key="reg_email")
            password = st.text_input("Password (min. 8 characters)", type="password", key="reg_pw")
            if st.form_submit_button("Create account", type="primary", use_container_width=True):
                user, err = register_user(name, email, password)
                if err:
                    st.error(err)
                else:
                    ss.user, ss.chat_id = user, None
                    st.rerun()


# ----------------------------------------------------------------- header
h1, h2 = st.columns([3, 2], vertical_alignment="center")
with h1:
    st.markdown(
        '<div class="brand">📖 AI Study Planner<small>Your friendly study assistant</small></div>',
        unsafe_allow_html=True,
    )
with h2:
    if ss.user:
        a, b = st.columns([3, 2], vertical_alignment="center")
        a.markdown(f'<div class="hello">Hi, <b>{ss.user["name"]}</b></div>', unsafe_allow_html=True)
        b.button("Logout", on_click=logout, use_container_width=True)
    else:
        _, btn = st.columns([1, 2])
        if btn.button("Login / Register", type="primary", use_container_width=True):
            auth_dialog()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 💬 Chat History")
    if not ss.user:
        st.info("Log in to save and revisit your conversations.")
    else:
        st.button("＋ New chat", on_click=new_chat, type="primary", use_container_width=True)
        chats = list_chats(ss.user["id"])
        if not chats:
            st.caption("No conversations yet. Say hello to your assistant!")
        for ch in chats:
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.button(
                ch["title"], key=f"open_{ch['id']}", on_click=open_chat, args=(ch["id"],),
                use_container_width=True, type="primary" if ch["id"] == ss.chat_id else "secondary",
            )
            c2.button("🗑", key=f"del_{ch['id']}", on_click=remove_chat, args=(ch["id"],),
                      help="Delete chat")

# ------------------------------------------------------ study tracker panel
if ss.user:
    topics = list_topics(ss.user["id"])
    pending = [t for t in topics if not t["completed"]]
    with st.expander(f"📚 Study Tracker  ·  {len(topics) - len(pending)} done, {len(pending)} pending"):
        with st.form("add_topics", clear_on_submit=True):
            raw = st.text_area("Add topics (one per line)", height=80,
                               placeholder="Algebra: quadratic equations\nBiology: cell structure")
            if st.form_submit_button("Add topics", use_container_width=True):
                if add_topics(ss.user["id"], raw.splitlines()):
                    st.rerun()
        for t in topics:
            c1, c2 = st.columns([8, 1], vertical_alignment="center")
            c1.checkbox(t["title"], value=bool(t["completed"]), key=f"topic_{t['id']}",
                        on_change=toggle_topic, args=(t["id"],))
            c2.button("✕", key=f"rm_{t['id']}", on_click=remove_topic, args=(t["id"],),
                      help="Remove topic")
        if pending:
            st.button("😅 I missed a study session — adjust my plan", use_container_width=True,
                      on_click=queue_prompt,
                      args=("I missed my study session(s). Please adjust my schedule using my "
                            "pending topics and tell me what to focus on next.",))
else:
    topics = []

# ------------------------------------------------------------------- chat
history = get_messages(ss.chat_id, ss.user["id"]) if (ss.user and ss.chat_id) else []
for m in history:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "📖"):
        st.markdown(m["content"])

if not history:
    st.markdown(
        '<div class="empty"><h3>What shall we study today?</h3>'
        "<p>Ask for a timetable, a recommendation, or an explanation.</p></div>",
        unsafe_allow_html=True,
    )
    if ss.user:
        starters = [
            ("📅 Weekly timetable", "Create a weekly study timetable for me. Ask me anything you need to know."),
            ("🗓 Today's plan", "Create a study plan for today based on my pending topics."),
            ("🎯 What next?", "Based on my tracker, what should I study next and why?"),
            ("💡 Explain a concept", "Explain a tricky concept to me simply. Ask which one I'd like."),
        ]
        cols = st.columns(2)
        for i, (label, text) in enumerate(starters):
            cols[i % 2].button(label, key=f"starter_{i}", on_click=queue_prompt, args=(text,),
                               use_container_width=True)

prompt = st.chat_input(
    "Ask your study assistant..." if ss.user else "Please log in to start chatting",
    disabled=ss.user is None,
)
prompt = prompt or ss.pop("queued_prompt", None)

if prompt and ss.user:
    uid = ss.user["id"]
    created = ss.chat_id is None
    if created:
        ss.chat_id = create_chat(uid, prompt)
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    add_message(ss.chat_id, uid, "user", prompt)

    convo = [{"role": m["role"], "content": m["content"]} for m in history]
    convo.append({"role": "user", "content": prompt})
    with st.chat_message("assistant", avatar="📖"):
        with st.spinner("Thinking..."):
            try:
                reply = get_reply(convo, ss.user["name"], list_topics(uid))
            except AIError as exc:
                reply = None
                st.error(str(exc))
        if reply:
            st.markdown(reply)
            add_message(ss.chat_id, uid, "assistant", reply)
    if created and reply:
        st.rerun()  # refresh the sidebar with the new chat title