# deep_research_reddit.py
# ─────────────────────────────────────────────────────────────────────────────
# Streamlit assistant for genre‑based Reddit deep research
# • Verdana 14 pt UI, live side‑bar timer
# • Optional expander to browse raw PRAW JSON before summarisation
# • Progress bar & status while summarising
# • Final self‑contained report with separate 3‑point actionable insights for
#   Director, Storywriter, Producer/Marketer, plus a reference section.
# ─────────────────────────────────────────────────────────────────────────────

import os, json, time, random
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
    st.error("🚨 Set your OpenAI & Reddit credentials via env‑vars or a .env file.")
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

def fetch_threads(sub: str, limit: int, timer_cb: Callable[[], None]) -> List[Dict]:
    """Pull newest <limit> threads and all comments."""
    threads = []
    for post in reddit.subreddit(sub).new(limit=limit):
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
        timer_cb()
    return threads


def summarise_threads(threads: List[Dict], progress_bar, status_slot, sample_slot, timer_cb: Callable[[], None], model: str = "o3", batch: int = 6) -> None:
    total, done = len(threads), 0
    for i in range(0, total, batch):
        chunk = threads[i:i+batch]
        payload = {
            t["id"]: f"{t['title']}\n\n{t['body'][:4000]}\n\nComments:\n{t['comments'][:6000]}"
            for t in chunk
        }
        status_slot.markdown(f"**Summarising:** {chunk[0]['title'][:90]}…")
        sample_slot.markdown(random.choice(threads)['title'][:100])
        msgs = [
            {"role": "system", "content": "You are a research assistant. For each Reddit thread JSON {id:text} return JSON with keys gist (≤30 words), insight1, insight2, sentiment (positive/neutral/negative)."},
            {"role": "user",   "content": json.dumps(payload)},
        ]
        summaries = json.loads(openai.chat.completions.create(model=model, messages=msgs).choices[0].message.content)
        for t in chunk:
            t["summary"] = summaries.get(t["id"], {})
        done += len(chunk)
        progress_bar.progress(done/total)
        timer_cb()
        time.sleep(0.5)
    status_slot.markdown("**Summarising complete!**")


def build_references(threads: List[Dict]) -> str:
    refs = "\n".join(f"* [{t['title']}]({t['url']})" for t in threads)
    return f"### References\n{refs}"


def generate_report(genre: str, threads: List[Dict], questions: List[str], timer_cb: Callable[[], None]) -> str:
    corpus = "\n\n".join(
        f"{t['title']} – {t['summary'].get('gist','')} [URL]({t['url']})" for t in threads
    )[:15000]
    q_block = "\n".join(f"Q{i+1}. {q}" for i,q in enumerate(questions))

    prompt = (
        f"You are a senior story analyst assisting film writers and producers who are exploring the  **{genre}**. You have mined sub reddit content on Reddit audience data for **{genre}**. "
        "First, give a one-paragraph overall audience sentiment snapshot for the genre. Then, for EACH question provided, answer in ≤2 paragraphs with citations [Title](URL). "
        ""It is important to add citations in [Title](URL) form right after every key evidence point. "
        "Afterward, create three separate sections each with **3 actionable insights** (bullet list) backed by evidence: \n" 
        "* For Directors\n* For Storywriters / Script Developers\n* For Producer‑Investors & Marketers.\n" 
        "Conclude with a short 2‑sentence market‑fit summary."
        "Ensure a Readability sore of 60-70 for the report as measured by Flesch reading ease and write with citations"
    )

    msgs = [
        {"role": "system", "content": prompt},
        {"role": "assistant", "content": f"CORPUS ({len(threads)} threads):\n{corpus}"},
        {"role": "user", "content": q_block},
    ]
    report = openai.chat.completions.create(model="o3", messages=msgs).choices[0].message.content
    timer_cb()
    return report + "\n\n" + build_references(threads)

# ── UI ──────────────────────────────────────────────────────────────────────
st.title("🎬 Reddit Audience Deep‑Dive for Creatives")

# Digital timer
time_box = st.sidebar.empty()
start_time = time.time()

def tick():
    mins, secs = divmod(int(time.time() - start_time), 60)
    time_box.write(f"⏱️ {mins:02d}:{secs:02d}")

col1, col2 = st.columns([2,1])
with col1:
    genre = st.text_input("Genre", value="horror").strip().lower()
with col2:
    n_posts = st.slider("Threads", 10, 200, 50, 10)

subreddit = st.text_input("Subreddit", value=GENRE_DEFAULT_SUB.get(genre, "movies")).strip()

st.markdown("#### Research questions (1‑5, one per line)")
qs = st.text_area("Questions", "What tropes feel over‑used?\nWhat excites this audience?", label_visibility="collapsed")
questions = [q.strip() for q in qs.splitlines() if q.strip()][:5]

if st.button("Run research 🚀"):
    if not subreddit:
        st.error("Specify a subreddit."); st.stop()
    if not questions:
        st.error("Enter at least one research question."); st.stop()

    with st.spinner("⛏️ Fetching threads…"):
        threads = fetch_threads(subreddit, n_posts, tick)

    # Optional raw JSON view before summarising
    with st.expander("📄 Browse raw Reddit JSON (optional)"):
        st.json(threads)

    prog = st.progress(0.0)
    status = st.empty()
    sample = st.empty()
    with st.spinner("📝 Summarising threads…"):
        summarise_threads(threads, prog, status, sample, tick)

    st.success(f"Summaries ready: {len(threads)} threads from r/{subreddit}")
    with st.expander("🔍 Gists & insights"):
        st.json([{"title":t['title'], **t['summary']}";
