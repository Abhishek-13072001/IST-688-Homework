import streamlit as st
from openai import OpenAI
import pypdf

def read_pdf(uploaded_file) -> str:
    """Reads text from an uploaded PDF file."""
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Fetch key from Streamlit secrets
openai_api_key = st.secrets.get("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key not found in Streamlit secrets. Please add it to secrets.toml.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

st.title("Lab 2 – Document Summarizer")
st.write("Upload a PDF file to generate a customized summary.")

# Sidebar Controls
st.sidebar.header("Summarizer Options")

# 1. Summary Format Selection Dropdown
summary_type = st.sidebar.selectbox(
    "Choose Summary Type:",
    (
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ),
)

# 2. Basic Models Selection Dropdown (First option is default)
basic_model = st.sidebar.selectbox(
    "Select model:",
    (
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-nano",
    ),
    index=0
)

# 3. Advanced Model Checkbox with Model Name in brackets
advanced_model_name = "gpt-5-mini"
use_advanced_model = st.sidebar.checkbox(f"Use advanced model ({advanced_model_name})")

# Determine final model choice
model_choice = advanced_model_name if use_advanced_model else basic_model

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    document_text = read_pdf(uploaded_file)

    if st.button("Generate Summary"):
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. Follow this instruction strictly: {summary_type}.",
            },
            {
                "role": "user",
                "content": f"Here is the document content:\n\n{document_text}",
            },
        ]

        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        except Exception as e:
            st.error(f"Error generating summary: {e}")
