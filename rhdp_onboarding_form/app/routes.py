from flask import render_template, request, jsonify, redirect, url_for
from app import app
import requests
import os
import logging

# Constants
AGNOSTIC_V_URL = 'https://github.com/rhpds/agnosticv'
AGNOSTIC_D_URL = 'https://github.com/redhat-cop/agnosticd'
CATALOG_ITEM_POLICY_URL = 'https://spaces.redhat.com/display/RHPDS/Onboarding+Policy'
JIRA_ENDPOINT = "https://issues.redhat.com/rest/api/2/issue"
JIRA_TOKEN = os.getenv('JIRA_TOKEN')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')


logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        logging.debug("POST request received")
        return process_form_submission(request.form)
    logging.debug("GET request received")
    return render_template('index.html')

@app.route('/success')
def success():
    logging.debug("Success route called")
    jira_url = request.args.get('jiraUrl')
    jira_key = request.args.get('jiraKey')
    agnostic_v_url = request.args.get('agnosticVUrl')
    agnostic_d_url = request.args.get('agnosticDUrl')
    catalog_item_policy_url = request.args.get('catalogItemPolicyUrl')
    
    logging.debug(f"Rendering success.html with: jiraUrl={jira_url}, jiraKey={jira_key}")
    return render_template('success.html',
                           jira_url=jira_url,
                           jira_key=jira_key,
                           agnostic_v_url=agnostic_v_url,
                           agnostic_d_url=agnostic_d_url,
                           catalog_item_policy_url=catalog_item_policy_url)

@app.route('/get_agnostic_v_configs')
def get_agnostic_v_configs():
    url = "https://api.github.com/repos/rhpds/agnosticv/contents"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        configs = [item['name'] for item in contents if item['type'] == 'dir']
        return jsonify(configs)
    else:
        return jsonify([]), 500

@app.route('/get_agnostic_d_roles')
def get_agnostic_d_roles():
    url = "https://api.github.com/repos/redhat-cop/agnosticd/contents/ansible/roles_ocp_workloads"
    headers = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        roles = [item['name'] for item in contents if item['type'] == 'dir']
        return jsonify(roles)
    else:
        return jsonify([]), 500

def process_form_submission(form_data):
    logging.debug("Processing form submission")
    try:
        jira_response = create_jira_issue(form_data)
        
        if jira_response['success']:
           logging.debug("Jira issue created successfully")
           return redirect(url_for('success', 
               jiraUrl=jira_response['jiraUrl'],
               jiraKey=jira_response['jiraKey'],
               agnosticVUrl=AGNOSTIC_V_URL,
               agnosticDUrl=AGNOSTIC_D_URL,
               catalogItemPolicyUrl=CATALOG_ITEM_POLICY_URL
           ))
        else:
            logging.error(f"Jira issue creation failed: {jira_response['error']}")
            return jsonify({"success": False, "error": jira_response['error']})
     
    except Exception as e:
        logging.exception("Exception occurred during form submission")
        return jsonify({"success": False, "error": str(e)})
 
def create_jira_issue(form_data):
    summary = "Catalog Item Onboarding - " + form_data['requester'] + " - " + form_data['helpNeeded']
    
    # Add more fields to the summary if present
    if form_data.get('actionNeeded'):
        summary += " - " + form_data['actionNeeded']
    if form_data.get('contentType'):
        summary += " - " + form_data['contentType']
    if form_data.get('roverGroup'):
        summary += " - " + form_data['roverGroup'][0]
    if form_data.get('agnosticVAccess'):
        summary += " - " + form_data['agnosticVAccess'][0]
    if form_data.get('githubId'):
        summary += " - " + form_data['githubId']

    description = create_jira_description(form_data)
    
    payload = {
        "fields": {
            "project": {"key": "GPTEINFRA"},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"},
        }
    }
    
    headers = {
        "Authorization": f"Bearer {JIRA_TOKEN}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(JIRA_ENDPOINT, json=payload, headers=headers)
    
    if response.status_code == 201:
        data = response.json()
        jira_url = f"https://issues.redhat.com/browse/{data['key']}"
        return {
            "success": True,
            "jiraUrl": jira_url,
            "jiraKey": data['key']
        }
    else:
        error_message = f"Failed to create Jira issue. Status code: {response.status_code}"
        try:
            error_details = response.json()
            error_message += f". Details: {error_details}"
        except:
            pass
        return {
            "success": False,
            "error": error_message
        }

def create_jira_description(form_data):
    description = "h1. Catalog Item Onboarding Request\n\n"
    
    fields = [
        ("Requester", "requester"),
        ("Requester Email", "requesterEmail"),
        ("Help Needed", "helpNeeded"),
        ("Action Needed", "actionNeeded"),
        ("Rover Group Needed", "roverGroup"),
        ("AgnosticV Access Needed", "agnosticVAccess"),
        ("GitHub ID", "githubId"),
        ("Content Type", "contentType"),
        ("Audience", "audience"),
        ("AgnosticV Config", "agnosticVConfig")
    ]
    
    for label, key in fields:
        if key in form_data and form_data[key]:
            value = form_data[key]
            if isinstance(value, list):
                value = value[0]  # Take the first item if it's a list
            description += f"*{label}:* {value}\n"
    
    if form_data.get('agnosticDRolesToAdd') or form_data.get('agnosticDRolesToRemove'):
        description += "\nh3. AgnosticD Roles\n\n{code}\n"
        if form_data.get('agnosticDRolesToAdd'):
            roles_to_add = form_data['agnosticDRolesToAdd']
            if isinstance(roles_to_add, list):
                roles_to_add = ', '.join(roles_to_add)
            description += f"To Add: {roles_to_add}\n"
        if form_data.get('agnosticDRolesToRemove'):
            roles_to_remove = form_data['agnosticDRolesToRemove']
            if isinstance(roles_to_remove, list):
                roles_to_remove = ', '.join(roles_to_remove)
            description += f"To Remove: {roles_to_remove}\n"
        description += "{code}\n\n"
    
    additional_fields = [
        ("Github Fork URL", "githubForkUrl"),
        ("Developer Fork Branch", "developerForkBranch"),
        ("Google Drive Link to Developer Assets", "googleDriveLink"),
        ("Additional Details", "additionalDetails")
    ]
    
    for label, key in additional_fields:
        if key in form_data and form_data[key]:
            description += f"*{label}:* {form_data[key]}\n"
    
    return description
