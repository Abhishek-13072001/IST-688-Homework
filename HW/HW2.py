import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic
import pypdf

# ============================================================================
# MUST BE THE VERY FIRST STREAMLIT COMMAND
# ============================================================================
st.set_page_config(
    page_title="HW Manager",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_pdf(uploaded_file) -> str:
    """Reads a PDF file uploaded via Streamlit and extracts its text."""
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def read_url_content(url: str):
    """Reads and extracts clean text content from a URL."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Strip script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        return ' '.join(chunk for chunk in chunks if chunk)
    except requests.RequestException as e:
        st.error(f"Error reading URL: {e}")
        return None


def anthropic_text_stream(client, model, system_prompt, content):
    """Helper generator for Anthropic stream output"""
    with client.messages.stream(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": content}]
    ) as stream:
        for text in stream.text_stream:
            yield text


# ============================================================================
# MAIN PAGE ROUTING
# ============================================================================
st.title("📚 HW Manager")
st.markdown("---")

# Navigation
page = st.radio(
    "Select Assignment",
    ["HW 1", "HW 2: URL Summarizer"],
    label_visibility="collapsed",
    horizontal=True
)

# ============================================================================
# HW1 PAGE - DOCUMENT QUESTION ANSWERING APP
# ============================================================================
if page == "HW 1":
    st.title("My Document Question Answering App")
    st.write(
        "Upload a document below (.pdf or .txt) and ask a question about it – GPT will answer! "
        "To use this app, you need to provide an OpenAI API key, which you can get "
        "[here](https://platform.openai.com/api-keys)."
    )

    # OpenAI API Key input
    api_key = st.text_input("OpenAI API Key", type="password")

    if not api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    else:
        # Create OpenAI client
        client = OpenAI(api_key=api_key)

        # Validate API key immediately
        try:
            client.models.list()
        except Exception:
            st.error("Invalid OpenAI API Key. Please check your key and try again.")
            st.stop()

        # LLM Model selection
        model_options = {
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "gpt-4.1": "gpt-4.1",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-nano": "gpt-5-nano",
        }

        selected_model_name = st.selectbox(
            "Select an LLM model:",
            list(model_options.keys()),
            index=0
        )

        actual_model = model_options[selected_model_name]

        # Document upload
        uploaded_file = st.file_uploader("Upload a document (.pdf or .txt)", type=["pdf", "txt"])

        # Question input
        question = st.text_area(
            "Now ask a question about the document!",
            placeholder="Is this course hard?",
            disabled=not uploaded_file,
        )

        # Process and answer
        if question and uploaded_file:
            try:
                # Extract text based on file type
                file_extension = uploaded_file.name.split('.')[-1].lower()

                if file_extension == "txt":
                    document_content = uploaded_file.read().decode("utf-8", errors="ignore")
                elif file_extension == "pdf":
                    document_content = read_pdf(uploaded_file)
                else:
                    st.error("Unsupported file type.")
                    st.stop()

                # Build prompt and generate response with streaming
                messages = [
                    {
                        "role": "user",
                        "content": f"Here's a document: {document_content} \n\n---\n\n {question}",
                    }
                ]

                stream = client.chat.completions.create(
                    model=actual_model,
                    messages=messages,
                    stream=True,
                )
                st.write_stream(stream)

            except Exception as e:
                st.error(f"Error querying model {actual_model}: {e}")

# ============================================================================
# HW2 PAGE - URL SUMMARIZER
# ============================================================================
elif page == "HW 2: URL Summarizer":
    st.title("HW 2 – URL Summarizer")
    st.write("Enter a web page URL below to generate a customized summary.")

    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================

    with st.sidebar:
        st.header("Summarizer Options")

        # Summary type selection
        summary_type = st.selectbox(
            "Choose Summary Type:",
            (
                "Summarize the document in 100 words",
                "Summarize the document in 2 connecting paragraphs",
                "Summarize the document in 5 bullet points",
            ),
        )

        # Output language selection
        output_language = st.selectbox(
            "Select Output Language:",
            ("English", "Spanish", "French", "German", "Chinese")
        )

        # LLM Provider selection
        provider = st.selectbox(
            "Select LLM Provider:",
            ("OpenAI", "Anthropic (Claude)")
        )

        # Advanced model toggle
        use_advanced_model = st.checkbox("Use advanced model")

        # ====================================================================
        # MODEL CONFIGURATION
        # ====================================================================

        if provider == "OpenAI":
            api_key = st.secrets.get("OPENAI_API_KEY")

            if use_advanced_model:
                model_choice = "gpt-5-mini"
                st.caption(f"Model: {model_choice}")
            else:
                model_choice = st.selectbox(
                    "Select model:",
                    ("gpt-4o-mini", "gpt-4-turbo"),
                    index=0
                )

        elif provider == "Anthropic (Claude)":
            api_key = st.secrets.get("ANTHROPIC_API_KEY")

            if use_advanced_model:
                model_choice = "claude-opus-5"
                st.caption(f"Model: {model_choice}")
            else:
                model_choice = st.selectbox(
                    "Select model:",
                    ("claude-haiku-4-5-20251001", "claude-sonnet-4-6"),
                    index=0
                )

    # ========================================================================
    # URL INPUT
    # ========================================================================

    url_input = st.text_input("Web page URL:", placeholder="https://example.com")

    # ========================================================================
    # GENERATE SUMMARY
    # ========================================================================

    if st.button("Generate Summary"):
        if not url_input:
            st.warning("Please enter a URL first.")
        elif not api_key:
            st.error(
                f"API Key for {provider} not found in Streamlit secrets. "
                f"Please add it to .streamlit/secrets.toml"
            )
        else:
            with st.spinner("Extracting content from URL..."):
                web_text = read_url_content(url_input)

            if web_text:
                system_prompt = (
                    f"You are a helpful assistant. Follow this instruction strictly: {summary_type}. "
                    f"Output the response in {output_language} language."
                )

                st.subheader(f"Summary ({provider} - {model_choice})")

                try:
                    if provider == "OpenAI":
                        client = OpenAI(api_key=api_key)
                        stream = client.chat.completions.create(
                            model=model_choice,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Here is the URL content:\n\n{web_text[:12000]}"}
                            ],
                            stream=True,
                        )
                        st.write_stream(stream)

                    elif provider == "Anthropic (Claude)":
                        client = anthropic.Anthropic(api_key=api_key)
                        st.write_stream(
                            anthropic_text_stream(
                                client,
                                model_choice,
                                system_prompt,
                                f"Here is the URL content:\n\n{web_text[:12000]}"
                            )
                        )

                except Exception as e:
                    st.error(f"Error generating summary: {e}")