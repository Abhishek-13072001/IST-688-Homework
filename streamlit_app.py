import streamlit as st

st.set_page_config(page_title="HW Manager", layout="wide")

hw1_page = st.Page("HW/HW1.py", title="HW 1", icon="1️⃣")
hw2_page = st.Page("HW/HW2.py", title="HW 2: URL Summarizer", icon="2️⃣")
hw3_page = st.Page("HW/HW3.py", title="HW 3: URL Chatbot", icon="3️⃣")
hw4_page = st.Page("HW/HW4.py", title="HW 4: iSchool Org RAG", icon="4️⃣")
hw5_page = st.Page("HW/HW5.py", title="HW 5: Tool-Calling RAG Bot", icon="5️⃣")

pg = st.navigation([hw1_page, hw2_page, hw3_page, hw4_page, hw5_page])
st.sidebar.title("HW Manager")
pg.run()