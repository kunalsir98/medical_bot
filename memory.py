#1.) load Raw pdf 
#2.) create chunks 
#3.)create  vector embeddings
#4.) store into vector Databases

from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ai.logger import logging
from ai.exception import CustomException
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
#load Raw pdf 
try:
    DATA_PATH='data/'
    def load_pdf_files(data):
        loader=DirectoryLoader(data,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)
        documents=loader.load()
        return documents
    documents=load_pdf_files(data=DATA_PATH)
    #print('length of pdf pages0,',len(documents))

    logging.info('Pdf file read sucssesfully')
except Exception as e:
    logging.info('error occured while reading the pdf')
    raise CustomException(e,sys)

#create chunks 
try:
    def create_chunks(extracted_data):
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                    chunk_overlap=50)
        text_chunk=text_splitter.split_documents(extracted_data)
        return text_chunk
    text_chunk=create_chunks(extracted_data=documents)
    #print('length of text chunks',len(text_chunk))
    logging.info('chunks created sucssefully')
except Exception as e:
    logging.info('Exception Occured while creating chunks')
    raise CustomException(e,sys)

## Create vector embeddings
try:
    def get_embedding_model():
        embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        return embedding_model
    embedding_model=get_embedding_model()
except Exception as e:
    logging.info('error occured while creating vector embeddings')
    raise CustomException(e,sys)

#store vector embeddings in vector database

DB_FAISS_PATH='vectorstore/db_faiss'
db=FAISS.from_documents(text_chunk,embedding_model)
db.save_local(DB_FAISS_PATH)
