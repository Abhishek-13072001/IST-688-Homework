import streamlit as st
from openai import OpenAI
import anthropic
import requests
from bs4 import BeautifulSoup

# ---- Page title + description (item 6) ----
st.title("HW 3: Streaming Chatbot about URLs")
st.write(
    "This chatbot answers questions about the content of up to two web pages. "
    "Enter one or two URLs in the sidebar and pick an LLM (OpenAI or Anthropic). "
    "The page content is placed in a system prompt that is always kept, so the bot "
    "can discuss the URLs at any point in the conversation. "
    "**Conversation memory:** the app uses a 6-message buffer — it remembers only the "
    "last 3 user–assistant exchanges (6 messages) when talking to the LLM, so older "
    "turns drop off while the URL context and instructions are never removed."
)

# ---- URL reader (reused from HW2) ----
def read_url_content(url: str):
    """Reads and extracts clean text content from a URL."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return ' '.join(chunk for chunk in chunks if chunk)
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

# ---- Sidebar: URL inputs + model selection ----
with st.sidebar:
    st.header("Options")

    url1 = st.text_input("URL 1:", placeholder="https://example.com")
    url2 = st.text_input("URL 2 (optional):", placeholder="https://example.com")

    # 2 vendors, latest premium model each (item 3)
    llm_choice = st.selectbox(
        "Select LLM:",
        ("OpenAI (gpt-5)", "Anthropic (claude-opus-5)")
    )

# ---- Build URL context for the system prompt (item 4) ----
# We read the URLs once and store their text so it can be injected every turn.
url_context = ""
if url1:
    text1 = read_url_content(url1)
    if text1:
        url_context += f"\n\n--- CONTENT FROM URL 1 ({url1}) ---\n{text1[:8000]}"
if url2:
    text2 = read_url_content(url2)
    if text2:
        url_context += f"\n\n--- CONTENT FROM URL 2 ({url2}) ---\n{text2[:8000]}"

# System prompt = instructions + URL content. This is NEVER dropped by the buffer.
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the web page "
    "content provided below. If the answer is not in the content, say so honestly. "
    "Explain clearly and simply."
    + (url_context if url_context else "\n\n(No URL content provided yet.)")
)

# ---- Initialize conversation memory (Part A pattern from Lab 3) ----
if "hw3_messages" not in st.session_state:
    st.session_state.hw3_messages = [
        {"role": "assistant", "content": "Hi! Add a URL in the sidebar, then ask me about it."}
    ]

# ---- Show conversation so far ----
for msg in st.session_state.hw3_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
if user_input := st.chat_input("Ask a question about the URL(s)..."):

    # Warn if no URL is loaded yet
    if not url_context:
        st.warning("Please enter at least one URL in the sidebar first.")
    else:
        # Save + show user's message
        st.session_state.hw3_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ---- 6-message buffer (item 5): keep only last 6 messages (3 exchanges) ----
        # The system prompt is added separately below, so it's never counted/dropped.
        recent_messages = st.session_state.hw3_messages[-6:]

        # ---- Generate response based on selected vendor ----
        with st.chat_message("assistant"):
            try:
                if llm_choice.startswith("OpenAI"):
                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    buffer = [{"role": "system", "content": SYSTEM_PROMPT}] + recent_messages
                    stream = client.chat.completions.create(
                        model="gpt-5",          # premium OpenAI model (swap to gpt-4.1 if unavailable)
                        messages=buffer,
                        stream=True,
                    )
                    response = st.write_stream(stream)

                else:  # Anthropic
                    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
                    # Anthropic takes the system prompt as a separate argument
                    def claude_stream():
                        with client.messages.stream(
                            model="claude-opus-5",   # premium Anthropic model
                            max_tokens=1000,
                            system=SYSTEM_PROMPT,
                            messages=recent_messages,
                        ) as stream:
                            for text in stream.text_stream:
                                yield text
                    response = st.write_stream(claude_stream())

            except Exception as e:
                response = f"Error: {e}"
                st.error(response)

        # Save assistant reply to memory
        st.session_state.hw3_messages.append({"role": "assistant", "content": response})