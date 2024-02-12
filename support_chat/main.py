import utils
import streamlit as st
from streaming import StreamHandler
import os

from langchain_openai import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SupportChatBot:

    def __init__(self):
        self.project_id = 'dce-gcp-training' # @param {type:"string"}
        self.location = 'us-central1'  # @param {type:"string"}
        self.model_name = "gemini-pro"
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"


    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, uploaded_files):
        # Load documents
        docs = []
        for file in uploaded_files:
            file_path = self.save_file(file)
            loader = CSVLoader(file_path)
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
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        return qa_chain
    
    @utils.enable_chat_history
    def main(self):

        # User Inputs
        uploaded_files = st.sidebar.file_uploader(label='Upload PDF files', type=['csv'], accept_multiple_files=True)
        if not uploaded_files:
            st.error("Please upload CSV documents to continue!")
            st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if uploaded_files and user_query:
            qa_chain = self.setup_qa_chain(uploaded_files)

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})    

if __name__ == "__main__":
    chat = SupportChatBot()
    chat.main()