import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time  

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, file_index):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    faiss_index_path = os.path.join(os.getcwd(), f"faiss_index_{file_index}")
    try:
        os.makedirs(faiss_index_path, exist_ok=True)  
        vector_store.save_local(faiss_index_path)
        st.success(f"Index saved for file {file_index}!")
    except Exception as e:
        st.error(f"Error saving the index for file {file_index}: {e}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, index_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index_path = os.path.join(os.getcwd(), f"faiss_index_{index_key}")

    if not os.path.exists(faiss_index_path) or not os.path.exists(faiss_index_path + "/index.faiss"):
        st.error(f"FAISS index file not found for {index_key}. Please process the PDF file first.")
        return
    
    try:
        new_db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"An error occurred: {e}")

def main():
    st.set_page_config("Vcheck Global")
    st.header("Chat with Vcheck Chatbot 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    if user_question and 'file_index' in st.session_state:
        user_input(user_question, st.session_state.file_index)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []

        if pdf_docs:
            for pdf_file in pdf_docs:
                file_name = pdf_file.name
                timestamp = time.time() 
                file_index = f"{timestamp:.6f}"  
                
                st.session_state.uploaded_files.append(pdf_file)
                st.session_state.file_index = file_index 

                with st.spinner(f"Processing {file_name}..."):
                    raw_text = get_pdf_text([pdf_file])
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, file_index)
                    st.success(f"Processing of {file_name} done!")

if __name__ == "__main__":
    main()
