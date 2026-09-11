import streamlit as st
from openai import OpenAI
import pypdf

def read_pdf(uploaded_file) -> str:
    """Reads a PDF file uploaded via Streamlit and extracts its text."""
    pdf_reader = pypdf.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Title and description
st.title("My Document Question Answering App")
st.write(
    "Upload a document below (.pdf or .txt) and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys)."
)

# API key input
openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Create OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Validate API key immediately
    try:
        client.models.list()
    except Exception:
        st.error("Invalid OpenAI API Key. Please check your key and try again.")
        st.stop()

    # Model selector (Task 3b requirement)
    model_choice = st.selectbox(
        "Select an LLM model:",
        ("gpt-3.5-turbo", "gpt-4.1", "gpt-5-mini", "gpt-5-nano")
    )

    # File uploader (.pdf and .txt only)
    uploaded_file = st.file_uploader(
        "Upload a document (.pdf or .txt)", type=("pdf", "txt")
    )

    # Question text area
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Is this course hard?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:
        # File extension check (Task 3a requirement)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'txt':
            document = uploaded_file.read().decode("utf-8", errors="ignore")
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        # Build prompt
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate response
        try:
            stream = client.chat.completions.create(
                model=model_choice,
                messages=messages,
                stream=True,
            )
            st.write_stream(stream)
        except Exception as e:
            st.error(f"Error querying model {model_choice}: {e}")