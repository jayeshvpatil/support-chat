import requests
import json
import pandas as pd
import streamlit as st
from typing import List, Tuple
from chain import load_summarize_issue_chain
from streaming import StreamHandler, PrintRetrievalHandler
import os



def getJiraIssue(username, token, domain, projectKey, startAt):
    headers = {
            'Content-Type': 'application/json'
        }
    params = {
        "jql": "project = " + projectKey,
        "fieldsByKeys": "false",
        "fields": ["summary","description","status","assignee","created","issuetype","priority","creator","labels","updated"],
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

    columns=["id","key","summary","description","status","assignee","issuetype","priority","creator","labels","created","updated"]
    issueList = rawData.get("issues")
    data = [] 
    for issue in issueList:
        replace_none(issue)
        issue_info = {
                        "id": issue.get("id"), 
                        "key": issue.get("key"),
                        "summary": issue.get("fields").get("summary", None),
                        "description": issue.get("fields").get("description", None),
                        "status": issue.get("fields").get("status", None).get("name", None),
                        "assignee": issue.get("fields").get("assignee", None).get("displayName", None),
                        "issuetype": issue.get("fields").get("issuetype", None).get("name", None),
                        "priority": issue.get("fields").get("priority", None).get("name", None),
                        "creator": issue.get("fields").get("creator", None).get("displayName", None),
                        "labels": issue.get("fields").get("labels", None),
                        "created": issue.get("fields").get("created", None),
                        "updated": issue.get("fields").get("updated", None),
                    }
        data.append(issue_info) 
    df = pd.DataFrame(data, columns=columns)
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

def get_jira_tickets(refresh=True):
    csv_file_path = 'data/issues_data.csv' 
    if refresh or not os.path.exists(csv_file_path):
        # Load fresh data
        df = load_data(st.secrets["jira_connection"]["username"],
                       st.secrets["jira_connection"]["token"],
                       st.secrets["jira_connection"]["domain"],
                       st.secrets["jira_connection"]["project_key"])
        df.to_csv(csv_file_path, index=False)
    else:
        df = pd.read_csv(csv_file_path, dtype={'description': str})
    priority, status, creator, assignee = display_filters(df)
    filtered_df = df.copy()
    filtered_df = filter(filtered_df, 'priority', priority)
    filtered_df = filter(filtered_df, 'status', status)
    filtered_df = filter(filtered_df, 'creator', creator)
    filtered_df = filter(filtered_df, 'assignee', assignee)
    filtered_df['description'] = filtered_df['description'].apply(format_chat_history_to_markdown)
    return filtered_df
