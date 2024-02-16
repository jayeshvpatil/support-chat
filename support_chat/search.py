import streamlit as st
from chain import load_search_retriever_chain, load_qa_chain,load_summarize_issue_chain, prepare_docs
from streaming import StreamHandler, PrintRetrievalHandler
import os
from jira import get_jira_tickets
import pandas as pd
from anonymizer import anonymize_text

def set_page_ui():
    st.set_page_config(page_title="GA Support Explorer", page_icon="🌐",        layout="wide",
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
def initialize_session_state():
    if 'search_retriever_chain' not in st.session_state:
        st.session_state['search_retriever_chain'] = load_search_retriever_chain()
    if 'qa_chain' not in st.session_state:
        st.session_state['qa_chain'] = load_qa_chain()
    if 'summarize_chain' not in st.session_state:
        st.session_state['summarize_chain'] = load_summarize_issue_chain()
def display_header():
    header_cols = st.columns([1, 4, 1, 1,1])
    header_cols[0].markdown("**Case#**")
    header_cols[1].markdown("**Summary**")
    header_cols[2].markdown("**Status**")
    header_cols[3].markdown("**Priority**")
    header_cols[4].markdown("**Action**")

def run_chain(input, chain, container_title, spinner_message, task_type='q&a'):
    with st.container():
        st.subheader(container_title)
        with st.spinner(spinner_message):
            retrieval_streamer_cb = PrintRetrievalHandler(st.container())
            answer_container = st.empty()
            stream_handler = StreamHandler(answer_container, initial_text="`Answer:`\n\n")
            if task_type=='q&a':
                final_input ={"question": input}
                result = chain(final_input, callbacks=[retrieval_streamer_cb, stream_handler])
                st.write(result['answer'])
            if task_type=='summarize':
                ticket = prepare_docs(input, is_text=True)
                final_input ={'input_documents':ticket}
                result = chain(final_input)
            if 'answer' in result:
                answer_container.markdown(result['answer'])
            if 'sources' in result:
                answer_container.info('`Sources:`\n\n' + result['sources'])
            return result

def perform_search(question):
    further_knowledgebase_tab, ga4_documentation_tab = st.tabs(['Further Knowledgebase', 'GA4 Documentation'])
    with further_knowledgebase_tab:
        run_chain(question, qa_chain, "Searching Further's knowledgebase...", "Please wait, searching...", 'q&a')
    with ga4_documentation_tab:
        ruun_chain(question, search_retriever_chain, "Searching the GA4 Documentation and web...", "Please wait, searching...",'q&a')


# Initialize session state
set_page_ui()
initialize_session_state()

search_retriever_chain = st.session_state.search_retriever_chain
qa_chain = st.session_state.qa_chain
summarize_chain = st.session_state.summarize_chain

tab1, tab2 = st.tabs(["Jira Tracker", "Q&A"])
with tab1:
    df = get_jira_tickets()
    display_header()
    for index, row in df.iterrows():
        cols = st.columns([1, 4, 1, 1,1])
        cols[0].write(row["id"])
        cols[1].write(row["summary"])
        cols[2].write(row["status"])
        cols[3].write(row["priority"])
        clicked = cols[4].button("Answer", key=index)
        if clicked:
            st.session_state["clicked_row"] = index
    if "clicked_row" in st.session_state:
        clicked_row_index = st.session_state["clicked_row"]
        description = df.iloc[clicked_row_index]['description']
    if description:
        anonymized_ticket = anonymize_text(description)
        if anonymized_ticket:
            with st.expander(f'Selected Ticket : {anonymized_ticket[:60]}'):
                st.markdown(anonymized_ticket)
            result =run_chain(anonymized_ticket, summarize_chain, "Summarizing Ticket...", "Please wait, summarizing...",'summarize')
            if 'output_text' in result:
                question = st.text_area(label="Answer issue",value = result['output_text'])
                submit = st.button("Answer the question", type='primary')
                if question and submit:
                    perform_search(question)
        
with tab2:
    question = st.text_area("Ask a question:")
    submit = st.button("Submit Question")
    if question and submit:
        perform_search(question)