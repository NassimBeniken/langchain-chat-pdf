import os

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader  # Used to read the pages from the PDF
from langchain_text_splitters import (
    CharacterTextSplitter,
)  # Used to split the text into chunks
from langchain_community.embeddings import OpenAIEmbeddings  # Used to embed the chunks
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.openai import OpenAI


def init():
    load_dotenv()
    # Load the OPEN_API_KEY form the environment variables
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set in the environment variables")
        exit(1)

    # Set the page configuration
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")  # H1


def main():
    init()
    # Upload the pdf
    uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

    # Extract the pages from the pdf
    if uploaded_file is not None:  # User uploaded a pdf
        pdf_reader = PdfReader(uploaded_file)

        text = ""

        for page in pdf_reader.pages:
            # Extract the text from the page
            text += page.extract_text()

        # Split the text into chunks
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,  # 1000 characters per chunk
            # Each chunk overlaps with the previous one by 200 characters in order
            # to not miss any words
            chunk_overlap=200,
            length_function=len,
        )

        chunks = splitter.split_text(text)

        # Create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Ask a question
        user_question = st.text_input("Ask a question about your PDF :")
        if user_question:
            # Retrievd the most similar chunks (more relative to the question)
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Display the response
            st.write(response)


if __name__ == "__main__":
    main()
