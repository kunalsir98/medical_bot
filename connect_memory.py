#1) SetupLLM(mistral with huggingface)
#2) Connect LLM with FAISS
#3) Create chain
from langchain_huggingface import HuggingFaceEndpoint
from ai.logger import logging
from ai.exception import CustomException
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
import os 
import sys
from langchain_community.vectorstores import FAISS
## setup LLM
HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID='mistralai/Mistral-7B-Instruct-v0.3'
try:
    def load_llm(huggingface_repo_id):
        llm=HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            model_kwargs={'token':HF_TOKEN,
                        "max_length":"512"}
                        
        )
        return llm
    logging.info('setup created sucssesfully')

except Exception as e:
    logging.info('error occured while creating setup')
    raise CustomException(e,sys)

## connect LLM with FAISS and create chain
CUSTOM_PROMPT_TEMPLATE="""
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template,input_variables=['context','question'])
    return prompt
    
# load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db=FAISS.load_local(DB_FAISS_PATH,embedding_model,allow_dangerous_deserialization=True)

#Create q/a chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}


)

#chain active(invoke)
user_query=('write a query here:')
response=qa_chain.invoke({'query':user_query})
print('RESULT:',response['result'])
print('SOURCE DOCUMENTS:',response['source_documents'])

