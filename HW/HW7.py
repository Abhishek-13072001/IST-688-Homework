# ---- SQLite fix for ChromaDB on Streamlit Cloud (must be at the very top) ----
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import re
import time
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# ---- Page title + description ----
st.title("HW 7: News Reporting Bot for a Law Firm")
st.write(
    "Ask about the news stories in the dataset. Use **top_news** for a ranked "
    "list of the most interesting stories (interesting = relevant to a law firm) "
    "or **search_news** to find news about a company or topic. "
    "Remembers the last 5 exchanges."
)

# ---- Settings ----
DATA = Path(__file__).parent / "HW7"
DB_PATH = DATA / "HW7_chroma_db"
NEWS_FILE = DATA / "news_clean.csv"
SCORES_FILE = DATA / "scores_gpt-5-mini.csv"
COLLECTION = "HW7News"
EMBED_MODEL = "text-embedding-3-small"
MODELS = ["gpt-5-mini", "gpt-5"]
SIM_CUTOFF = 0.75   # articles this similar count as the same story
MAX_DISTANCE = 0.65  # search results farther than this are ignored

# ---- OpenAI client from secrets ----
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API Key not found in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=openai_api_key)


# ---- Merge duplicate articles into one story ----
def group_stories(df, emb_by_id):
    story = {}   # article_id -> id of the lead article of its story
    size = {}    # lead article_id -> number of articles in the story
    for company, g in df.groupby("company"):
        g = g.sort_values("score", ascending=False)
        ids = g["article_id"].tolist()
        vecs = np.array([emb_by_id[i] for i in ids])
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        sim = vecs @ vecs.T
        for a, lead in enumerate(ids):
            if lead in story:
                continue
            members = [ids[b] for b in range(len(ids))
                       if ids[b] not in story and sim[a, b] >= SIM_CUTOFF]
            for m in members:
                story[m] = lead
            size[lead] = len(members)
    df["story_id"] = df["article_id"].map(story)
    df["story_size"] = df["story_id"].map(size)
    return df


# ---- Load data once (DB is built outside the app with build_db.py) ----
@st.cache_resource
def load_data():
    news = pd.read_csv(NEWS_FILE, encoding="utf-8").fillna("")
    scores = pd.read_csv(SCORES_FILE)[["article_id", "relevance", "about_company", "reason", "score"]]
    df = news.merge(scores, on="article_id", how="left")
    df["score"] = df["score"].fillna(0)
    df["reason"] = df["reason"].fillna("")
    collection = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(COLLECTION)
    data = collection.get(include=["embeddings"])
    emb_by_id = dict(zip(data["ids"], data["embeddings"]))
    df = group_stories(df, emb_by_id)
    return df, collection


with st.spinner("Loading news data..."):
    df, collection = load_data()

# same check as before: exact expected count, not just count() > 0
if collection.count() != len(df):
    st.error(f"Vector DB has {collection.count()} items but the CSV has {len(df)}. Rebuild it with build_db.py.")
    st.stop()

COMPANIES = sorted(df["company"].unique())


def get_embedding(text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text if text else " ")
    return resp.data[0].embedding


def resolve_company(name):
    # match what the model typed to a company name in the data
    if not name:
        return None
    n = name.strip().lower()
    for c in COMPANIES:
        if c.lower() == n:
            return c
    for c in COMPANIES:
        cl = c.lower()
        if re.search(r"\b" + re.escape(cl) + r"\b", n) or (len(n) >= 3 and n in cl):
            return c
    return None


def article_info(row):
    return {
        "title": row["title"],
        "company": row["company"],
        "source": row["source"],
        "date": str(row["date"])[:10],
        "url": row["URL"],
        "score": int(row["score"]),
        "why": row["reason"],
        "articles_on_story": int(row["story_size"]),
    }


# ---- Tool 1: ranked list of the most interesting news ----
def top_news(company=None, n=10):
    n = max(1, min(int(n or 10), 20))
    d = df[(df["article_id"] == df["story_id"]) & (df["score"] > 0)]   # one row per story
    if company:
        c = resolve_company(company)
        if not c:
            return f"No company matching '{company}' in the news set."
        d = d[d["company"] == c]
    d = d.sort_values(["score", "story_size", "date"], ascending=False).head(n)
    if d.empty:
        return "No interesting news found."
    return json.dumps([article_info(r) for _, r in d.iterrows()])


# ---- Tool 2: search news by topic or company ----
def search_news(topic, company=None):
    where = None
    if company:
        c = resolve_company(company)
        if not c:
            return f"No company matching '{company}' in the news set."
        where = {"company": c}
    res = collection.query(query_embeddings=[get_embedding(topic)], n_results=15,
                           where=where, include=["distances"])
    best = {}   # story id -> closest distance
    for aid, dist in zip(res["ids"][0], res["distances"][0]):
        if dist > MAX_DISTANCE:
            continue
        story = df.loc[df["article_id"] == aid, "story_id"].iloc[0]
        if story not in best or dist < best[story]:
            best[story] = dist
    if not best:
        return "No matching articles found."
    # sort by score, then by how close the match is
    leads = df[df["article_id"].isin(best)].copy()
    leads["dist"] = leads["article_id"].map(best)
    leads = leads.sort_values(["score", "dist"], ascending=[False, True]).head(5)
    out = []
    for _, lead in leads.iterrows():
        info = article_info(lead)
        info["snippet"] = lead["description"][:300]
        out.append(info)
    return json.dumps(out)


def run_tool(name, args):
    if name == "top_news":
        return top_news(args.get("company"), args.get("n", 10))
    if name == "search_news":
        return search_news(args.get("topic", ""), args.get("company"))
    return "Unknown tool."


# ---- Tool schemas given to the LLM ----
TOOLS = [
    {"type": "function", "function": {
        "name": "top_news",
        "description": ("Returns the most interesting news stories for a law firm as a ranked list "
                        "(best first). Use ONLY for general requests like 'most interesting', 'top' or "
                        "'biggest' news with no specific topic. Optionally limit to one company."),
        "parameters": {"type": "object", "properties": {
            "company": {"type": "string", "description": "Exact company name from the list. Leave out for all companies."},
            "n": {"type": "integer", "description": "How many stories to return (default 10)."}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "search_news",
        "description": ("Finds news about a specific company or topic. Use for 'find news about X' "
                        "and for ANY topic question such as patents, lawsuits, tax or mergers."),
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string", "description": "What to search for."},
            "company": {"type": "string", "description": "Exact company name from the list, if the question is about one company."}},
            "required": ["topic"]}}},
]

SYSTEM_PROMPT = (
    "You are a news assistant for a large global law firm that monitors news about its clients. "
    "Only report on articles returned by your tools and never use outside knowledge. "
    "For any news question, call a tool first. Use top_news for the most interesting news and "
    "search_news for news about a company or topic. "
    "When you list stories, use a numbered list in the order the tool returned them (best first). "
    "For each story give the bold title, company, source, date, the link, the interest score out of 100, "
    "and one or two sentences on why a lawyer should care. Take the legal angle only from the 'why' field "
    "and do not invent one. If the score is under 40, say it has low legal relevance. "
    "If articles_on_story is more than 1, mention that several articles cover it. "
    "A score of 0 means the article is probably not really about that company. "
    "If the tools find nothing, or the results are not about what was asked, say you could not find it "
    "and do not list unrelated articles. "
    "Do not end with a question or an offer to do more (you cannot fetch articles or monitor news). Stop after the last story. "
    "Companies in the data: " + "; ".join(COMPANIES)
)


# ---- Greetings: punctuation removed, then every word must be a greeting word ----
GREETING_WORDS = {"hi", "hello", "hey", "thanks", "thank", "you", "thx", "bye", "goodbye",
                  "how", "are", "whats", "up", "good", "morning", "afternoon", "evening",
                  "ok", "okay", "cool", "great", "nice", "doing", "there"}


def is_chitchat(text):
    words = re.sub(r"[^\w\s]", "", text).lower().split()
    return len(words) > 0 and all(w in GREETING_WORDS for w in words)


# ---- Run one question through the model and tools ----
def ask(model, messages, user_input):
    tools_used, tokens = [], 0

    def call(**kw):
        nonlocal tokens
        resp = client.chat.completions.create(model=model, messages=messages,
                                              reasoning_effort="low", **kw)
        tokens += resp.usage.total_tokens
        return resp.choices[0].message

    msg = call(tools=TOOLS, tool_choice="auto")
    # model skipped the tool on a real question -> force one
    if not msg.tool_calls and not is_chitchat(user_input):
        msg = call(tools=TOOLS, tool_choice="required")

    rounds = 0
    while msg.tool_calls and rounds < 3:
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = run_tool(tc.function.name, args)
            tools_used.append({"name": tc.function.name, "args": args, "result": result[:1500]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        msg = call(tools=TOOLS)
        rounds += 1

    if msg.tool_calls:   # still asking for tools after 3 rounds, answer without them
        msg = call()
    return msg.content or "Sorry, I could not produce an answer.", tools_used, tokens


# ---- Sidebar: model choice, sample questions, test log ----
st.sidebar.header("HW7 settings")
model = st.sidebar.selectbox("Model", MODELS)
st.sidebar.write("Sample questions")
for q in ["Find the most interesting news",
          "Find news about Tesla",
          "What is the latest on JPMorgan Chase?"]:
    if st.sidebar.button(q):
        st.session_state.hw7_pending = q

if "hw7_log" not in st.session_state:
    st.session_state.hw7_log = []
if st.session_state.hw7_log:
    log_csv = pd.DataFrame(st.session_state.hw7_log).to_csv(index=False)
    st.sidebar.download_button("Download test log", log_csv, "hw7_test_log.csv", "text/csv")

# ---- Chat memory: last 5 interactions (10 messages) ----
if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = [
        {"role": "assistant", "content": "Hi! Ask me for the most interesting news or news about a company.",
         "tools": [], "info": ""}
    ]

for m in st.session_state.hw7_messages:
    with st.chat_message(m["role"]):
        for t in m.get("tools", []):
            with st.expander(f"🔍 {t['name']}({json.dumps(t['args'])})"):
                st.text(t["result"])
        st.markdown(m["content"])
        if m.get("info"):
            st.caption(m["info"])

user_input = st.chat_input("Ask about the news...")
if st.session_state.get("hw7_pending"):
    user_input = st.session_state.pop("hw7_pending")

if user_input:
    st.session_state.hw7_messages.append(
        {"role": "user", "content": user_input, "tools": [], "info": ""})
    with st.chat_message("user"):
        st.markdown(user_input)

    # only role/content go to the API
    recent = [{"role": m["role"], "content": m["content"]}
              for m in st.session_state.hw7_messages[-10:]]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + recent

    with st.chat_message("assistant"):
        start = time.time()
        with st.spinner(f"Thinking ({model})..."):
            answer, tools_used, tokens = ask(model, messages, user_input)
        seconds = round(time.time() - start, 1)
        for t in tools_used:
            with st.expander(f"🔍 {t['name']}({json.dumps(t['args'])})"):
                st.text(t["result"])
        st.markdown(answer)
        info = f"{model} | {seconds}s | {tokens} tokens"
        st.caption(info)

    st.session_state.hw7_messages.append(
        {"role": "assistant", "content": answer, "tools": tools_used, "info": info})
    st.session_state.hw7_log.append(
        {"model": model, "question": user_input, "seconds": seconds, "tokens": tokens,
         "tools": ", ".join(t["name"] for t in tools_used), "answer": answer})