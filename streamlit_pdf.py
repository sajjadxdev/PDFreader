import streamlit as st
from PyPDF2 import PdfReader
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = ''

# Function to read PDF and extract text
def extract_text_from_pdf(pdf_file):
    pdfreader = PdfReader(pdf_file)
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# Function to split text into chunks
def split_text(raw_text):
    text_spliter = CharacterTextSplitter(
        separator='\n',
        chunk_size=800,
        chunk_overlap=200,
        length_function=len
    )
    return text_spliter.split_text(raw_text)

# Function to find the answer based on the question and text chunks
def find_answer(question, text_chunks):
    # Create embeddings for the text chunks
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    # Get the embedding for the question
    question_embedding = embeddings.embed_query(question)

    # Find the most similar chunk
    similar_chunks = vectorstore.similarity_search(question_embedding, k=1)
    
    # Return the most relevant chunk as the answer
    if similar_chunks:
        return similar_chunks[0].page_content  # Return the content of the most similar chunk
    else:
        return "No relevant information found."

# Streamlit UI
st.title("PDF Analyzer")
st.write("Upload a PDF file and ask questions about its content.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    raw_text = extract_text_from_pdf(uploaded_file)
    
    # Split the text into chunks
    text_chunks = split_text(raw_text)
    
    st.write("Text extracted and split into chunks.")
    
    # Display some of the chunks
    st.write("Here are some chunks from the PDF:")
    for i, chunk in enumerate(text_chunks[:3]):  # Display first 3 chunks
        st.write(f"**Chunk {i+1}:** {chunk[:200]}...")  # Show first 200 characters
    
    # Question input
    question = st.text_input("Ask a question about the PDF:")
    
    if st.button("Submit"):
        if uploaded_file is not None and question:
            # Find the answer based on the question and the extracted PDF text chunks
            answer = find_answer(question, text_chunks)
            
            # Display the results
            st.write("You asked:", question)
            st.write("Answer:", answer)
        elif uploaded_file is None:
            st.warning("Please upload a PDF file.")
        else:
            st.warning("Please enter a question.")