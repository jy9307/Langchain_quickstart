from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from datetime import date

file_path = "sample/Langchain.pdf"

loader = PyPDFLoader(file_path)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    )

docs = loader.load_and_split(text_splitter=splitter)

embedder = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = FAISS.from_documents(docs, 
                                embedding= embedder)
vectorstore.save_local("./.cache/")