from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import os


def generate_text(pdf_file_path, question):
    llm = ChatOllama(model="llama3.1")

    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    print(docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    db = FAISS.from_documents(documents, OllamaEmbeddings())
    retriver = db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriver, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\nLLM Response -->\n")
    response = rag_chain.invoke(question)
    return response


if not os.path.exists("uploads"):
    os.makedirs("uploads")


# Create a directory to save uploaded files
def save_uploaded_file(uploaded_file):
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join("uploads", uploaded_file.name)


st.title("PDF Upload, Save, and Read")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.write(f"File saved at: {file_path}")

st.divider()
prompt = st.chat_input("Chat with your pdf")
if prompt:
    st.write(prompt)
    st.write(generate_text(file_path, prompt))
