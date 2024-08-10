import streamlit as st
import os
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With files")
    st.header("ChatPDF BY YOUSSEF EMAD")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None

    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)
        process = st.button("Process")

    if process:
        files_text = get_files_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        st.session_state.text_chunks = text_chunks
        st.session_state.processComplete = True

    if st.session_state.processComplete:
        user_question = st.chat_input("Chat with your file")
        if user_question:
            handle_userinput(user_question)

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            st.error("Unsupported file type")
    return text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def handle_userinput(user_question):
    text_chunks = st.session_state.text_chunks

    # Vectorize the text chunks and the user question
    vectorizer = TfidfVectorizer().fit_transform([user_question] + text_chunks)
    vectors = vectorizer.toarray()

    # Calculate cosine similarities between the user question and text chunks
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    # Find the most similar chunk
    most_similar_chunk_index = cosine_similarities.argmax()
    most_similar_chunk = text_chunks[most_similar_chunk_index]

    # Store the conversation history
    st.session_state.chat_history.append({"user": user_question, "bot": most_similar_chunk})

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            message(messages["user"], is_user=True, key=str(i) + "_user")
            message(messages["bot"], key=str(i) + "_bot")

if __name__ == '__main__':
    main()
