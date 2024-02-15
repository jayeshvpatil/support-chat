import streamlit as st
from chain import load_search_retriever_chain, load_qa_chain,load_summarize_issue_chain, prepare_docs
from streaming import StreamHandler, PrintRetrievalHandler
import os
from jira import get_jira_tickets
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
from anonymizer import anonymize_text
import spacy.cli

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

def perform_search(input, chain, container_title, spinner_message, task_type='q&a'):
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

# Initialize session state
set_page_ui()
initialize_session_state()

search_retriever_chain = st.session_state.search_retriever_chain
qa_chain = st.session_state.qa_chain
summarize_chain = st.session_state.summarize_chain

tab1, tab2 = st.tabs(["Jira Tracker", "Q&A"])
with tab1:
    df = get_jira_tickets()
    gd = GridOptionsBuilder.from_dataframe(df)
    gd.configure_column(field="description", header_name="Description", height=50, wrapText=True, autoHeigh=True)
    gd.configure_pagination(enabled=True)
    gd.configure_selection(selection_mode='single', use_checkbox=True)
    grid_options= gd.build()
    grid_table = AgGrid(df, 
                        gridOptions=grid_options,
                        width='100%',
                        theme='material', 
                        update_mode=GridUpdateMode.GRID_CHANGED,
                        reload_data=True,
                        allow_unsafe_jscode=True,
                        columns_auto_size_mode=2,
                        enable_quicksearch=True,
                        editable=True)
    if grid_table['selected_rows']:
        selected_ticket= grid_table['selected_rows'][0]['description']
        anonymized_ticket = anonymize_text(selected_ticket)
        if anonymized_ticket:
            with st.expander(f'Selected Ticket : {anonymized_ticket[:60]}'):
                st.markdown(anonymized_ticket)
            result =perform_search(anonymized_ticket, summarize_chain, "Summarizing Ticket...", "Please wait, summarizing...",'summarize')
            if 'output_text' in result:
                question = st.text_area(label="Answer issue",value = result['output_text'])
                submit = st.button("Answer the question", type='primary')
                if question and submit:
                    further_knowledgebase_tab, ga4_documentation_tab = st.tabs(['Further Knowledgebase', 'GA4 Documentation'])
                    with further_knowledgebase_tab:
                        perform_search(question, qa_chain, "Searching Further's knowledgebase...", "Please wait, searching...", 'q&a')
                    with ga4_documentation_tab:
                        perform_search(question, search_retriever_chain, "Searching the GA4 Documentation and web...", "Please wait, searching...",'q&a')


        
with tab2:
    question = st.text_area("Ask a question:")
    submit = st.button("Submit Question")
    if question and submit:
        further_knowledgebase_tab, ga4_documentation_tab = st.tabs(['Further Knowledgebase', 'GA4 Documentation'])
        with further_knowledgebase_tab:
            perform_search(question, qa_chain, "Searching Further's knowledgebase...", "Please wait, searching...", 'q&a')
        with ga4_documentation_tab:
            perform_search(question, search_retriever_chain, "Searching the GA4 Documentation and web...", "Please wait, searching...",'q&a')
