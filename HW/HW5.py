# ---- SQLite fix for ChromaDB on Streamlit Cloud (MUST be at the very top) ----
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import os
import re
import streamlit as st
from openai import OpenAI
import chromadb
from bs4 import BeautifulSoup

# ---- Page title + description ----
st.title("HW 5: iSchool Student Organizations Chatbot (Tool-Calling)")
st.write(
    "Ask a question about Syracuse iSchool student organizations. This version "
    "uses a **tool** (`relevant_club_info`) that the AI calls only when it "
    "decides it needs to search the document collection, rather than searching "
    "on every message. Remembers the last 5 exchanges."
)

# ---- OpenAI client from secrets ----
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API Key not found in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=openai_api_key)

# ---- Paths (same DB as HW4 -- reused, not rebuilt) ----
HTML_FOLDER = os.path.join(os.path.dirname(__file__), "su_orgs")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "HW4_chroma_db")


# ---- Helper: read one HTML file and extract clean text (same as HW4) ----
def read_html_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(c for c in chunks if c)


def chunk_text(text: str):
    """Same 2-chunk midpoint split used in HW4 -- kept identical so a rebuild
    (if ever needed) stays consistent with the existing persisted DB."""
    if len(text) < 50:
        return [text, ""]
    midpoint = len(text) // 2
    split_point = text.find(" ", midpoint)
    if split_point == -1:
        split_point = midpoint
    chunk1 = text[:split_point].strip()
    chunk2 = text[split_point:].strip()
    return [chunk1, chunk2]


def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text if text else " "
    )
    return response.data[0].embedding


def get_or_build_vectordb():
    """Reuses the HW4 collection if it exists; rebuilds only if missing/incomplete."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="HW4Collection")

    html_files = [f for f in os.listdir(HTML_FOLDER) if f.lower().endswith(".html")]
    expected_chunks = len(html_files) * 2

    if 0 < collection.count() < expected_chunks:
        chroma_client.delete_collection("HW4Collection")
        collection = chroma_client.get_or_create_collection(name="HW4Collection")

    if collection.count() >= expected_chunks:
        return collection

    progress = st.progress(0, text="Building vector database (first run only)...")
    for idx, filename in enumerate(html_files):
        file_path = os.path.join(HTML_FOLDER, filename)
        full_text = read_html_file(file_path)
        chunks = chunk_text(full_text)
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            embedding = get_embedding(chunk)
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{filename}_chunk{chunk_idx}"],
                metadatas=[{"filename": filename, "chunk": chunk_idx}]
            )
        progress.progress((idx + 1) / len(html_files),
                           text=f"Embedding {idx + 1}/{len(html_files)}: {filename}")
    progress.empty()
    return collection


# ---- Load / reuse vector DB, cached in session_state ----
if "hw5_VectorDB" not in st.session_state:
    with st.spinner("Loading vector database..."):
        st.session_state.hw5_VectorDB = get_or_build_vectordb()

collection = st.session_state.hw5_VectorDB


# ---- Part 3: the tool function itself ----
def relevant_club_info(query: str) -> str:
    """Runs a vector search against the org collection and returns formatted excerpts."""
    query_emb = get_embedding(query)
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]

    if not retrieved_docs:
        return "No relevant organization info found."

    context = ""
    for meta, doc in zip(retrieved_meta, retrieved_docs):
        context += f"\n\n--- From {meta['filename']} (chunk {meta['chunk']}) ---\n{doc[:2000]}"
    return context


# ---- Tool schema given to the LLM ----
club_info_tool = {
    "type": "function",
    "function": {
        "name": "relevant_club_info",
        "description": (
            "Searches the Syracuse iSchool student organizations document "
            "collection and returns the most relevant excerpts for a given "
            "query. Use this whenever the user asks about a specific club, "
            "organization, or activity you don't already have information on."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what info is needed."
                }
            },
            "required": ["query"],
        },
    },
}


# ---- Robust chit-chat detector (fixes the "Hi, how are you?" false-positive) ----
def is_chitchat(text: str) -> bool:
    """Strips ALL punctuation before comparing, so 'Hi, how are you?' matches
    correctly instead of falling through to a forced tool call."""
    cleaned = re.sub(r"[^\w\s]", "", text).strip().lower()
    greeting_phrases = {
        "hi", "hello", "hey", "hiya", "yo",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see you", "cya",
        "how are you", "whats up", "how are you doing",
        "good morning", "good afternoon", "good evening",
        "ok", "okay", "cool", "great", "nice",
    }
    if cleaned in greeting_phrases:
        return True
    # short messages with no organization-related keywords are treated as chit-chat
    org_keywords = ("club", "organization", "org", "society", "association",
                    "group", "team", "committee", "council")
    if len(cleaned.split()) <= 4 and not any(k in cleaned for k in org_keywords):
        return True
    return False


# ---- Chat memory: last 5 interactions (10 messages) ----
# Each stored message also carries tool_query/tool_result so history replay
# after a rerun shows the SAME expander that appeared live, instead of losing it.
if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = [
        {"role": "assistant", "content": "Hi! Ask me about iSchool student organizations.",
         "tool_query": None, "tool_result": None}
    ]

for msg in st.session_state.hw5_messages:
    with st.chat_message(msg["role"]):
        if msg.get("tool_query"):
            with st.expander(f"🔍 Searched for: \"{msg['tool_query']}\""):
                st.text(msg.get("tool_result", ""))
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about a student organization..."):

    st.session_state.hw5_messages.append(
        {"role": "user", "content": user_input, "tool_query": None, "tool_result": None}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    system_prompt = (
        "You are a helpful assistant that answers questions about Syracuse "
        "iSchool student organizations. You have a tool, relevant_club_info, "
        "that searches a document collection. If you use its results, clearly "
        "say so and name the source file. If the search genuinely finds "
        "nothing, say you could not find it."
    )

    # Only pass role/content to the API -- strip our internal tracking keys
    recent_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.hw5_messages[-10:]
    ]
    messages = [{"role": "system", "content": system_prompt}] + recent_messages

    chitchat = is_chitchat(user_input)
    used_query = None
    used_result = None

    with st.chat_message("assistant"):
        # --- First call: let the model decide whether to use the tool ---
        first_response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            tools=[club_info_tool],
            tool_choice="auto",
        )
        response_message = first_response.choices[0].message
        tool_calls = response_message.tool_calls

        # ---- Safety net: force the tool if the model skipped it on what
        # looks like a genuine organization question ----
        if not tool_calls and not chitchat:
            forced_response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                tools=[club_info_tool],
                tool_choice={"type": "function", "function": {"name": "relevant_club_info"}},
            )
            response_message = forced_response.choices[0].message
            tool_calls = response_message.tool_calls

        if tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls:
                args = json.loads(tool_call.function.arguments)
                search_query = args.get("query", user_input)
                tool_result = relevant_club_info(search_query)

                used_query = search_query
                used_result = tool_result[:1000] + ("..." if len(tool_result) > 1000 else "")

                with st.expander(f"🔍 Searched for: \"{search_query}\""):
                    st.text(used_result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            stream = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                stream=True,
            )
            response = st.write_stream(stream)
        else:
            response = response_message.content
            st.markdown(response)

    st.session_state.hw5_messages.append({
        "role": "assistant",
        "content": response,
        "tool_query": used_query,
        "tool_result": used_result,
    })