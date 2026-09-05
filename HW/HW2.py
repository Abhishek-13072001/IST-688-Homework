import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic

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

# App Title & Description
st.title("HW 2 – URL Summarizer")
st.write("Enter a web page URL below to generate a customized summary.")

# 1. Main Page Control: URL Input at the top (not in the sidebar)
url_input = st.text_input("Web page URL:", placeholder="https://example.com")

# 2. Sidebar Controls
st.sidebar.header("Summarizer Options")

# Summary type selection (from Lab 2)
summary_type = st.sidebar.selectbox(
    "Choose Summary Type:",
    (
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ),
)

# Output language selection
output_language = st.sidebar.selectbox(
    "Select Output Language:",
    ("English", "Spanish", "French", "German", "Chinese")
)

# LLM Provider selection
provider = st.sidebar.selectbox(
    "Select LLM Provider:",
    ("OpenAI", "Anthropic (Claude)")
)

# Checkbox for advanced model
use_advanced_model = st.sidebar.checkbox("Use advanced model")

# Configure provider-specific models and secrets
if provider == "OpenAI":
    api_key = st.secrets.get("OPENAI_API_KEY")
    advanced_model_name = "gpt-5-mini"
    
    basic_model = st.sidebar.selectbox(
        "Select model:",
        ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-5-nano"),
        index=0
    )
    model_choice = advanced_model_name if use_advanced_model else basic_model

elif provider == "Anthropic (Claude)":
    api_key = st.secrets.get("ANTHROPIC_API_KEY")
    advanced_model_name = "claude-sonnet-4-6"
    
    basic_model = st.sidebar.selectbox(
        "Select model:",
        ("claude-haiku-4-5-20251001", "claude-sonnet-4-6"),
        index=0
    )
    model_choice = advanced_model_name if use_advanced_model else basic_model

# Helper generator for Anthropic stream output
def anthropic_text_stream(client, model, system_prompt, content):
    with client.messages.stream(
        model=model,
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": content}]
    ) as stream:
        for text in stream.text_stream:
            yield text

# 3. Action Execution
if st.button("Generate Summary"):
    if not url_input:
        st.warning("Please enter a URL first.")
    elif not api_key:
        st.error(f"API Key for {provider} not found in Streamlit secrets. Please add it to secrets.toml.")
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