# deep_research_reddit.py
# ─────────────────────────────────────────────────────────────────────────────
# • Streamlit assistant for genre‑based Reddit deep research
# • Verdana 14 pt UI, live side‑bar timer, real‑time progress on summarisation
# • Pulls last N posts + comments, summarises with OpenAI o3, reflection pass
# • Shows elapsed time and reference list (title + URL) at the end
#
# DEPENDENCIES
#   pip install streamlit praw openai python-dotenv
#
# KEYS (env‑vars or .env file ‑‑ recommended)
#   OPENAI_API_KEY
#   REDDIT_CLIENT_ID
#   REDDIT_CLIENT_SECRET
#   REDDIT_USER_AGENT="DeepResearch/0.1"
# RUN
#   streamlit run deep_research_reddit.py
# ─────────────────────────────────────────────────────────────────────────────

import os, json, time, textwrap
from datetime import datetime, timezone
from typing import List, Dict, Callable

import streamlit as st
from dotenv import load_dotenv
import openai
import praw

# ── CSS: Verdana 14 pt everywhere ────────────────────────────────────────────
st.markdown(
    """
    <style>
    html, body, [class*=\"css\"], .stMarkdown, .stTextInput, .stButton, .stSlider label {
        font-family: Verdana, sans-serif !important;
        font-size: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── ENV / KEYS ───────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key        = os.getenv("OPENAI_API_KEY", "")
REDDIT_CLIENT_ID      = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET  = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT     = os.getenv("REDDIT_USER_AGENT", "DeepResearch/0.1")

if not all([openai.api_key, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET]):
    st.error("🚨 Set your OpenAI & Reddit credentials via env‑vars or a .env file.")
    st.stop()

# ── REDDIT CLIENT ────────────────────────────────────────────────────────────
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
GENRE_DEFAULT_SUB = {
    "horror": "horror",
    "sci-fi": "scifi",
    "rom-com": "romcom",
    "superhero": "marvelstudios",
    "documentary": "documentaries",
    "animation": "animation",
    "crime": "TrueFilm",
    "thriller": "Thrillers",
}

def fetch_threads(sub: str, limit: int, timer_cb: Callable[[float], None]) -> List[Dict]:
    """Pull newest <limit> threads and all comments."""
    threads = []
    for idx, post in enumerate(reddit.subreddit(sub).new(limit=limit), 1):
        post.comments.replace_more(limit=None)
        comments = " ".join(c.body for c in post.comments.list())
        threads.append({
            "id": post.id,
            "title": post.title,
            "body": post.selftext or "",
            "comments": comments,
            "url": post.url,
            "created": datetime.fromtimestamp(post.created_utc, tz=timezone.utc).strftime("%Y-%m-%d"),
        })
        timer_cb(0)  # update elapsed display
    return threads


def summarise_threads(threads: List[Dict], progress_bar, status_slot, timer_cb: Callable[[float], None], model: str = "o3", batch: int = 6) -> None:
    """Attach a `summary` dict to each thread in‑place, updating UI."""
    total = len(threads)
    done = 0
    for i in range(0, total, batch):
        chunk = threads[i:i + batch]
        payload = {
            t["id"]: t["title"] + "\n\n" + t["body"][:4000] + "\n\nComments:\n" + t["comments"][:6000]
            for t in chunk
        }
        status_slot.markdown(f"**Summarising:** {chunk[0]['title'][:80]}…")
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. For each Reddit thread JSON {id:text} return JSON with keys "
                    "gist (≤25 words), insight1, insight2, sentiment (positive/neutral/negative)."
                ),
            },
            {"role": "user", "content": json.dumps(payload)},
        ]
        resp = openai.chat.completions.create(model=model, messages=msgs)
        summaries = json.loads(resp.choices[0].message.content)
        for t in chunk:
            t["summary"] = summaries.get(t["id"], {})
        done += len(chunk)
        progress_bar.progress(done / total)
        timer_cb(0)
        time.sleep(1)  # politeness
    status_slot.markdown("**Summarising complete!**")


def reflect_and_verify(report: str, corpus: str, timer_cb: Callable[[float], None], model: str = "o3") -> str:
    msgs = [
        {
            "role": "system",
            "content": (
                "You are a critical reviewer. Verify that every claim in the ANSWER is supported by the CORPUS. "
                "If something is unsupported, correct it. Produce a concise, coherent report using the same style, "
                "adding citations [Title](URL) after each key insight."
            ),
        },
        {"role": "assistant", "content": f"CORPUS:\n{corpus}"},
        {"role": "assistant", "content": f"ANSWER:\n{report}"},
    ]
    resp = openai.chat.completions.create(model=model, messages=msgs)
    timer_cb(0)
    return resp.choices[0].message.content


def generate_report(threads: List[Dict], questions: List[str], timer_cb: Callable[[float], None]) -> str:
    corpus = "\n\n".join(
        f"{t['title']} – {t['summary'].get('gist','')} [URL]({t['url']})" for t in threads
    )[:15000]

    q_block = "\n".join(f"Q{i+1}. {q}" for i, q in enumerate(questions))
    msgs = [
        {
            "role": "system",
            "content": (
                "You are a senior story analyst. Answer each question in 1–2 concise paragraphs. "
                "After each key insight add a citation in the form [Title](URL). Create sections summarization of the entire reddit thread. Then provide 3 ACTIONABLE insights with evidence"
            ),
        },
        {"role": "assistant", "content": f"CORPUS ({len(threads)} threads):\n{corpus}"},
        {"role": "user", "content": q_block},
    ]
    resp = openai.chat.completions.create(model="o3", messages=msgs)
    draft = resp.choices[0].message.content
    timer_cb(0)
    final = reflect_and_verify(draft, corpus, timer_cb)
    return final

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🎬 Reddit Deep‑Research Assistant")

# live timer in sidebar
st.sidebar.header("⏱️ Elapsed")
clock_slot = st.sidebar.empty()
start_time_global = time.time()

def update_timer(_):
    elapsed = time.time() - start_time_global
    clock_slot.write(f"{elapsed:0.1f} s")

col1, col2 = st.columns([2, 1])
with col1:
    genre_input = st.text_input("Film/TV genre", value="horror").strip().lower()
with col2:
    n_posts = st.slider("Threads", 10, 200, 50, step=10)

def_sub = GENRE_DEFAULT_SUB.get(genre_input, "movies")
subreddit = st.text_input("Subreddit", value=def_sub).strip()

st.markdown("#### Research questions (1‑5, one per line)")
qs_text = st.text_area("", "What tropes feel over‑used?\nWhat excites this audience?")
questions = [q.strip() for q in qs_text.splitlines() if q.strip()][:5]

if st.button("Run research 🚀"):
    # reset timer
    global_start = time.time()
    start_time_global = global_start  # update for timer fn scope

    if not subreddit:
        st.error("Please specify a subreddit.")
        st.stop()
    if not questions:
        st.error("Enter at least one research question.")
        st.stop()

    # FETCH & SUMMARISE with progress
    with st.spinner("⛏️ Fetching threads + comments…"):
        threads = fetch_threads(subreddit, n_posts, update_timer)

    progress = st.progress(0.0)
    status_txt = st.empty()
    with st.spinner("📝 Summarising…"):
        summarise_threads(threads, progress, status_txt, update_timer)

    st.success(f"Summarised {len(threads)} threads from r/{subreddit}.")
    with st.expander("🔍 Gists & insights"):
        st.json([{"title": t["title"], **t["summary"], "url": t["url"]} for t in threads])

    # Report
    with st.spinner("🧠 Generating research report…"):
        report_md = generate_report(threads, questions, update_timer)

    st.markdown("## 📊 Research Report")
    st.markdown(report_md)

    total_elapsed = time.time() - global_start
    clock_slot.write(f"{total_elapsed:0.1f} s (done)")
