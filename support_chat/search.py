import streamlit as st
from chain import load_search_retriever_chain, load_qa_chain
from streaming import StreamHandler, PrintRetrievalHandler
import os
from jira import get_issues


os.environ['OPENAI_API_KEY'] = st.secrets["api_keys"]["OPENAI_API_KEY"]
os.environ['GOOGLE_CSE_ID']=st.secrets["api_keys"]["GOOGLE_CSE_ID"]
os.environ['GOOGLE_API_KEY']=st.secrets["api_keys"]["GOOGLE_API_KEY"]

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐",        layout="wide",
        initial_sidebar_state="expanded",)

st.sidebar.image("assets/further-logo.png",width=300)   
# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    "GA Product support helps you deep dive into your knowledgebase and google documentation and answer support issues "
)

st.sidebar.header("Resources")
st.sidebar.markdown("""
- [All Open Issues](https://apolloplatform.atlassian.net/jira/servicedesk/projects/PS/queues/custom/1)
- [GA4 Documentation](https://support.google.com/analytics/?hl=en&sjid=3086114336562628128-NA#topic=14090456)                  
"""
)


# Make retriever and llm
if 'search_retriever_chain' not in st.session_state:
    st.session_state['search_retriever_chain']= load_search_retriever_chain()
if 'qa_chain' not in st.session_state:
    st.session_state['qa_chain'] = load_qa_chain() 
search_retriever_chain = st.session_state.search_retriever_chain
qa_chain = st.session_state.qa_chain

tab1, tab2 = st.tabs(["Jira Tracker","Q&A"])
with tab1:
    df = get_issues()
    st.dataframe(df, use_container_width=True)
with tab2:
        # User input 
    question = st.text_area("`Ask a question:`")
    submit = st.button("Submit Question")
    if question and submit:
        # Write answer and sources
        tab1, tab2 = st.tabs(['Further Knowledgebase', 'GA4 Documentation'])
        with tab1:
            retrieval_streamer_cb = PrintRetrievalHandler(st.container())
            answer = st.empty()
            stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
            search_result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
            answer.markdown(search_result['answer'])
        with tab2:
            retrieval_streamer_cb = PrintRetrievalHandler(st.container())
            answer = st.empty()
            stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
            search_result = search_retriever_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
            answer.markdown(search_result['answer'])
            st.info('`Sources:`\n\n' + search_result['sources'])
