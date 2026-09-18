# ---- SQLite fix for ChromaDB on Streamlit Cloud (MUST be at the very top) ----
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI
import chromadb
from bs4 import BeautifulSoup
import os

# ---- Page title + description ----
st.title("HW 4: iSchool Student Organizations Chatbot")
st.write(
    "Ask a question about Syracuse iSchool student organizations. This chatbot "
    "uses a RAG (Retrieval-Augmented Generation) pipeline built from 513 student "
    "organization web pages. **Conversation memory:** the app remembers the last "
    "5 exchanges (10 messages) so you can ask follow-up questions."
)

# ---- OpenAI client from secrets ----
openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API Key not found in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=openai_api_key)

# ---- Paths ----
HTML_FOLDER = os.path.join(os.path.dirname(__file__), "su_orgs")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "HW4_chroma_db")

# ---- Helper: read one HTML file and extract clean text ----
def read_html_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(c for c in chunks if c)

# ----------------------------------------------------------------------------
# CHUNKING METHOD (explained per assignment step 2.a.ii.2):
# Each document is split into exactly 2 mini-documents ("chunks") by character
# count, cut at the nearest paragraph/sentence-ish boundary rather than at a
# hard character index. We chose a simple fixed-size (roughly-half) split
# instead of a heading-based split because the 513 org pages come from many
# different organizations and are not guaranteed to share a consistent HTML
# structure (headings, sections, etc.), so a structural split could fail
# silently on pages with unusual formatting. A character-based half-split is
# robust across all files, keeps each chunk a readable, self-contained piece
# of text, and still lets each half be embedded and retrieved independently,
# which is the core benefit of chunking (more precise retrieval than
# embedding one huge document per organization).
# ----------------------------------------------------------------------------
def chunk_text(text: str):
    """Splits text into 2 chunks, cutting near the midpoint at a space."""
    if len(text) < 50:
        return [text, ""]  # very short pages: just duplicate/skip second half
    midpoint = len(text) // 2
    # find the nearest space AFTER the midpoint so we don't cut a word
    split_point = text.find(" ", midpoint)
    if split_point == -1:
        split_point = midpoint
    chunk1 = text[:split_point].strip()
    chunk2 = text[split_point:].strip()
    return [chunk1, chunk2]

# ---- Helper: get an OpenAI embedding ----
def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text if text else " "
    )
    return response.data[0].embedding

# ---- Build the persistent vector DB (only if it doesn't already exist) ----
def get_or_build_vectordb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="HW4Collection")

    html_files = [f for f in os.listdir(HTML_FOLDER) if f.lower().endswith(".html")]
    expected_chunks = len(html_files) * 2  # 2 chunks per file

    # If the collection is INCOMPLETE (not just non-empty), wipe it and rebuild
    if 0 < collection.count() < expected_chunks:
        chroma_client.delete_collection("HW4Collection")
        collection = chroma_client.get_or_create_collection(name="HW4Collection")

    # If already fully built, reuse it
    if collection.count() >= expected_chunks:
        return collection

    # ---- build from scratch (unchanged from before) ----
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

# ---- Load or build once, cache in session_state to avoid repeat disk checks ----
if "HW4_VectorDB" not in st.session_state:
    with st.spinner("Loading vector database..."):
        st.session_state.HW4_VectorDB = get_or_build_vectordb()

collection = st.session_state.HW4_VectorDB

# ---- Chat memory: last 5 interactions (10 messages) ----
if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = [
        {"role": "assistant", "content": "Hi! Ask me about iSchool student organizations."}
    ]

for msg in st.session_state.hw4_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about a student organization..."):

    st.session_state.hw4_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- Retrieve relevant chunks from the vector DB ----
    query_emb = get_embedding(user_input)
    results = collection.query(query_embeddings=[query_emb], n_results=3)
    retrieved_docs = results["documents"][0]
    retrieved_meta = results["metadatas"][0]

    context = ""
    for meta, doc in zip(retrieved_meta, retrieved_docs):
        context += f"\n\n--- From {meta['filename']} (chunk {meta['chunk']}) ---\n{doc[:2000]}"

    system_prompt = (
        "You are a helpful assistant that answers questions about Syracuse "
        "iSchool student organizations, using the document excerpts below. "
        "If you use information from these excerpts, clearly say so and name "
        "the source file. If the answer is not in the excerpts, say you could "
        "not find it.\n\n"
        f"ORGANIZATION DOCUMENT EXCERPTS:{context}"
    )

    # ---- Buffer: last 10 messages (5 interactions) ----
    recent_messages = st.session_state.hw4_messages[-10:]

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "system", "content": system_prompt}] + recent_messages,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.hw4_messages.append({"role": "assistant", "content": response})