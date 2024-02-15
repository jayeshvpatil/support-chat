import os
import faiss
from langchain.vectorstores import FAISS 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore  
from langchain.chains import RetrievalQAWithSourcesChain,LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.vectorstores import Chroma
import streamlit as st

    
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)
embedding_function=OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

def save_file(file):
    folder = 'data'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path

def load_search_retriever_chain():
    search = GoogleSearchAPIWrapper()   
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm, 
        search=search, 
        num_search_results=3
    )
    search_retriever_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)
    return search_retriever_chain

def load_qa_chain():
    docs = []
    loader = CSVLoader('./data/Google Support _ LLM From PSA Tickets - Sheet1.csv')
    docs.extend(loader.load())
    db = DocArrayInMemorySearch.from_documents(docs, embedding_function)
    retriever = db.as_retriever(
        search_type='mmr',
        search_kwargs={'k':2, 'fetch_k':4}
    )  
    qa_chain = RetrievalQAWithSourcesChain.from_llm(llm, retriever=retriever, verbose=True)
    return qa_chain

def load_summarize_issue_chain():
    summarize_chain = load_summarize_chain(llm, chain_type='map_reduce')
    return summarize_chain

def prepare_docs(doc, is_text=False):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    if is_text:
        texts = text_splitter.split_text(doc)
    else:
        texts = text_splitter.split_documents(doc)
    splits = [Document(page_content=t) for t in texts]
    return splits