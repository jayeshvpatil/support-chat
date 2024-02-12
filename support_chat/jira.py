import requests
import json
import pandas as pd
import streamlit as st
from typing import List, Tuple
from chain import load_search_retriever_chain, load_qa_chain
from streaming import StreamHandler, PrintRetrievalHandler


def init_chain():
    # Make retriever and llm
    if 'search_retriever_chain' not in st.session_state:
        st.session_state['search_retriever_chain']= load_search_retriever_chain()
    if 'qa_chain' not in st.session_state:   
        st.session_state['qa_chain'] = load_qa_chain() 
    search_retriever_chain = st.session_state.search_retriever_chain
    qa_chain = st.session_state.qa_chain
    return search_retriever_chain,qa_chain

def getJiraIssue(username, token, domain, projectKey, startAt):
    headers = {
            'Content-Type': 'application/json'
        }
    params = {
        "jql": "project = " + projectKey,
        "fieldsByKeys": "false",
        "fields": ["summary","status","assignee","created","issuetype","priority","creator","labels","updated", "description"],
        "startAt": startAt,
        "maxResults": 10
    }
    response = requests.get(f"https://{domain}/rest/api/3/search", auth=(username, token), headers=headers, params=params).json()
    return response

# Get all issue with 100 issues per API call
def getApiPagination(username, token, domain, projectKey):
    issueList = getJiraIssue(username, token, domain, projectKey, 0)
    startAt = issueList["startAt"]
    totalIssue = issueList["total"]

    while (totalIssue > 100) and (startAt < totalIssue):
        startAt += 100
        nextIssueList = getJiraIssue(username, token, domain, projectKey, startAt)
        issueList["issues"].extend(nextIssueList["issues"])
    
    return issueList

# Replace None with Empty Dictionary
def replace_none(dictionary):
    # checking for dictionary and replacing if None
    if isinstance(dictionary, dict):
        
        for key in dictionary:
            if dictionary[key] is None:
                dictionary[key] = {}
            else:
                replace_none(dictionary[key])
  
    # checking for list, and testing for each value
    elif isinstance(dictionary, list):
        for val in dictionary:
            replace_none(val)

def display_filters(df):
    priority = sorted(df['priority'].unique())
    selected_priority = st.sidebar.multiselect("Priority", priority,placeholder="Select a priority")
    status = sorted(df['status'].unique())
    selected_status = st.sidebar.multiselect("Status", status, placeholder="Select a status")
    creator = sorted(df['creator'].unique())
    selected_creator = st.sidebar.multiselect("Creator", creator,placeholder="Select a creator")
    assignee = df['assignee'].unique()
    selected_assignee =  st.sidebar.selectbox("Assignee",assignee,index=None,placeholder="Select assignee...")
    return selected_priority, selected_status, selected_creator, selected_assignee

def filter(data: pd.DataFrame, column: str, values: List[str]) -> pd.DataFrame:
    if not values:
        return data
    # Check if the column is a date type, then filter a range
    if pd.api.types.is_datetime64_any_dtype(data[column]):
        # Convert values to datetime objects
        start_date, end_date = pd.to_datetime(values, errors='coerce')
        # Filter data within the date range
        filtered_data = data[data[column].between(start_date, end_date, inclusive='both')]
    else:
        # For non-date columns, use isin
        filtered_data = data[data[column].isin(values)]
    return filtered_data

def load_data(username, token, domain, projectKey):
    rawData = getApiPagination(username, token, domain, projectKey)

    df = pd.DataFrame(columns=["id","key","summary","status","assignee","issuetype","priority","creator","labels", "description","created","updated"])

    issueList = rawData.get("issues")

    for issue in issueList:
        replace_none(issue)
        issueInfo = {
                        "id": issue.get("id"), 
                        "key": issue.get("key"),
                        "summary": issue.get("fields").get("summary", None),
                        "status": issue.get("fields").get("status", None).get("name", None),
                        "assignee": issue.get("fields").get("assignee", None).get("displayName", None),
                        "issuetype": issue.get("fields").get("issuetype", None).get("name", None),
                        "priority": issue.get("fields").get("priority", None).get("name", None),
                        "creator": issue.get("fields").get("creator", None).get("displayName", None),
                        "labels": issue.get("fields").get("labels", None),
                        "description": issue.get("fields").get("description", None),
                        "created": issue.get("fields").get("created", None),
                        "updated": issue.get("fields").get("updated", None),
                    }
        df = df.append(issueInfo, ignore_index=True)
        
    return df

def format_chat_history_to_markdown(chat_history):
    # Parse the JSON string into a Python dictionary
    # Initialize an empty list to hold Markdown lines
    markdown_lines = []

    # Function to process each content block recursively
    def process_content(content, indent_level=0):
        for item in content:
            if item['type'] == 'paragraph':
                # Add text content with proper indentation
                paragraph_lines = [c['text'] for c in item.get('content', []) if c['type'] == 'text']
                markdown_lines.append('    ' * indent_level + ' '.join(paragraph_lines))
            elif item['type'] == 'blockquote':
                # Process blockquote content recursively with increased indentation
                process_content(item['content'], indent_level + 1)
            elif item['type'] == 'link':
                # Format link
                text = item['text']
                href = item['marks'][0]['attrs']['href']
                markdown_lines.append(f"[{text}]({href})")

    # Start processing the top-level content
    if 'content' in chat_history:
        process_content(chat_history['content'])

    # Join the lines into a single Markdown string
    markdown_output = '\n\n'.join(markdown_lines)
    return markdown_output

def get_issues():

    df = load_data(st.secrets["jira_connection"]["username"],
                st.secrets["jira_connection"]["token"],
                st.secrets["jira_connection"]["domain"],
                st.secrets["jira_connection"]["project_key"]
                )
    priority, status, creator,assignee = display_filters(df)
    filtered_df = df.copy()
    filtered_df = filter(filtered_df, 'priority', priority)
    filtered_df = filter(filtered_df, 'status', status)
    filtered_df = filter(filtered_df, 'creator', creator)
    filtered_df = filter(filtered_df, 'assignee', assignee)
    filtered_df['description'] = df['description'].apply(format_chat_history_to_markdown)
    return filtered_df
