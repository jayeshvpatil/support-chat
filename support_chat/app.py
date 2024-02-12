import streamlit as st
import os
import data_loader
import vertexai
from google.oauth2 import service_account
import streamlit as st
from vertexai.preview.generative_models import GenerativeModel
from langchain.chat_models.vertexai import ChatVertexAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings.vertexai import VertexAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel


docs=[]

PROJECT_ID = 'dce-gcp-training' # @param {type:"string"}
LOCATION = 'us-central1'  # @param {type:"string"}
MODEL_NAME = "gemini-pro"


def init_vertex():
    credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcs_connection"]
    )
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

def get_chat_model():
    chat_model = ChatVertexAI(
    model_name=MODEL_NAME, max_output_tokens=1048, temperature=0.2
) 
    return chat_model


def set_page_config():
    st.set_page_config(
        page_title="Google Support Chat",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)

def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path =os.path.join(directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
        st.toast(f"Saved file: {file.name} in {directory}")
        return file_path

set_page_config()

st.header("`Google Support Chat`")
st.info(
    "`I am an evaluation tool for question-answering using an existing vectorDB (currently Pinecone is supported) and an eval set. "
    "I will generate and grade an answer to each eval set question with the user-specific retrival setting, such as metadata filtering or self-querying retrieval." 
    " Experiments with different configurations are logged. For an example eval set, see eval_sets/lex-pod-eval.json.`")

tab1, tab2 = st.tabs(['Data','Chat'])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.form("user_input"):

            # Pinecone params 
            oai_api_key = st.text_input("`OpenAI API Key:`", type="password").strip()

            retriever_type = st.radio("`Choose retriever`",
                                    ("Pinecone",
                                    "Pinecone w/ self-querying",
                                    "Pinecone w/ metadata filtering",
                                    "Kor filtering"),
                                    index=0)

            num_neighbors = st.select_slider("`Choose # chunks to retrieve`",
                                            options=[3, 4, 5, 6, 7, 8])

            embeddings = st.radio("`Choose embeddings`",
                                ("HuggingFace",
                                "OpenAI"),
                                index=1)

            model = st.radio("`Choose model`",
                            ("gpt-3.5-turbo",
                            "gpt-4"),
                            index=0)
            submitted = st.form_submit_button("Submit Settings")
    with col2:
        with st.form(key='file_inputs'):
            uploaded_file= st.file_uploader("`Please upload eval set (.json):` ",
                                        type=['csv','txt','pdf'],
                                        accept_multiple_files=False)
            if uploaded_file is not None:
                saved_file_path= save_uploaded_file('data', uploaded_file)

            submitted = st.form_submit_button("Submit files")
            if submitted:
                docs = (data_loader.load_document(saved_file_path,'csv'))
        # Path to your data folder
        data_folder_path = 'data'

        # Check if the data folder exists
        if os.path.exists(data_folder_path):
            # List all files in the data folder
            files = os.listdir(data_folder_path)
            
            # Check if the folder is not empty
            if files:
                # Display each file in the list
                for file in files:
                    st.write(file)
            else:
                st.write('The data folder is empty.')
        else:
            st.write('Data folder does not exist.')
    with col3:
        with st.form(key='url_inputs'):
            # User input for a list of URLs
            urls = st.text_area("`Enter URLs, separated by commas:`") 
            submitted = st.form_submit_button("Submit Urls")
            if submitted:
                url_list = [url.strip() for url in urls.split(',')]
                st.write("Submitted URLs:")
                for url in url_list:
                    st.write(url)
                    docs.extend(data_loader.load_document(url, 'url'))

with tab2:
    # Split
    init_vertex()
    if docs:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = text_splitter.split_documents(docs)

        # Add to vectorDB
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            collection_name="ga4_support",
            embedding=VertexAIEmbeddings(),
        )
        retriever = vectorstore.as_retriever()

        # RAG prompt
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # LLM
        model = get_chat_model()

        # RAG chain
        chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | model
            | StrOutputParser()
        )
        # Add typing for input
        class Question(BaseModel):
            __root__: str
        chain = chain.with_types(input_type=Question)


