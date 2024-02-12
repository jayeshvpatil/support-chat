import os
import faiss
from langchain.vectorstores import FAISS 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore  
from langchain.chains import RetrievalQAWithSourcesChain,LLMChain
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.chat_models import ChatOpenAI
import streamlit as st
import logging
    
logging.basicConfig()

os.environ['OPENAI_API_KEY'] = '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'
os.environ['GOOGLE_CSE_ID']='564d2fb8ad0a14540'
os.environ['GOOGLE_API_KEY']='AIzaSyCQlLrjXEWuhN2nz6_QFORyI9B6-Aqzzlg'

def save_file(file):
    folder = 'data'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    file_path = f'./{folder}/{file.name}'
    with open(file_path, 'wb') as f:
        f.write(file.getvalue())
    return file_path

def load_search_retriever_chain():
    # Vectorstore
    embeddings_model = OpenAIEmbeddings()  
    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # Search

    search = GoogleSearchAPIWrapper()   

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=3
    )

    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    search_retriever_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)
    return search_retriever_chain

def load_qa_chain():
    docs = []
    """
    uploaded_files = st.sidebar.file_uploader(label='Upload CSV files', type=['csv'], accept_multiple_files=True)
    if not uploaded_files:
        st.error("Please upload CSV documents to continue!")
        st.stop()
    for file in uploaded_files:
        file_path = save_file(file)
        print(file_path)
        loader = CSVLoader(file_path)
        docs.extend(loader.load())
        """  
    loader = CSVLoader('./data/Google Support _ LLM From PSA Tickets - Sheet1.csv')
    docs.extend(loader.load())
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = OpenAIEmbeddings()
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Define retriever
    retriever = vectordb.as_retriever(
        search_type='mmr',
        search_kwargs={'k':2, 'fetch_k':4}
    )

    # Setup memory for contextual conversation        
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # Setup LLM and QA chain
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, streaming=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
    return qa_chain