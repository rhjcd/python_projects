document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('demoRequestForm');
    const helpNeededSelect = document.getElementById('helpNeeded');
    const dynamicQuestions = document.getElementById('dynamicQuestions');
    const additionalFields = document.getElementById('additionalFields');
    const githubForkUrlInput = document.getElementById('githubForkUrl');
    const developerForkBranchGroup = document.getElementById('developerForkBranchGroup');

    // Fetch and populate AgnosticV configs
    fetch('/get_agnostic_v_configs')
        .then(response => response.json())
        .then(configs => populateAgnosticVConfig(configs))
        .catch(error => console.error('Error fetching AgnosticV configs:', error));

    // Fetch and populate AgnosticD roles
    fetch('/get_agnostic_d_roles')
        .then(response => response.json())
        .then(roles => updateAgnosticDRoles(roles))
        .catch(error => console.error('Error fetching AgnosticD roles:', error));

    helpNeededSelect.addEventListener('change', function() {
        toggleQuestions(this.value);
    });

    githubForkUrlInput.addEventListener('input', function() {
        if (this.value.trim() !== '') {
            developerForkBranchGroup.style.display = 'block';
        } else {
            developerForkBranchGroup.style.display = 'none';
            document.getElementById('developerForkBranch').value = '';
        }
    });

form.addEventListener('submit', function(e) {
    e.preventDefault();
    console.log("Form submitted");
    
    const formData = new FormData(form);
    
    // Show loading overlay
    document.getElementById('submissionOverlay').style.display = 'flex';
    
    fetch('/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log("Response received:", response);
        console.log("Response URL:", response.url);
        console.log("Response type:", response.type);
        console.log("Is redirected:", response.redirected);
        if (response.redirected) {
            console.log("Redirecting to:", response.url);
            window.location.href = response.url;
            return;
        }
        return response.json();
    })
    .then(data => {
        console.log("Processed data:", data);
        // Hide loading overlay
        document.getElementById('submissionOverlay').style.display = 'none';
        
        if (data && data.success) {
            console.log("Success, redirecting to success page");
            window.location.href = '/success?' + new URLSearchParams({
                jiraUrl: data.jiraUrl,
                jiraKey: data.jiraKey,
                agnosticVUrl: data.agnosticVUrl,
                agnosticDUrl: data.agnosticDUrl,
                catalogItemPolicyUrl: data.catalogItemPolicyUrl
            }).toString();
        } else if (data && data.error) {
            console.error("Error received:", data.error);
            alert('Error: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        alert('An error occurred while submitting the form');
        // Hide loading overlay
        document.getElementById('submissionOverlay').style.display = 'none';
    });
});

function toggleQuestions(selectedValue) {
    const dynamicQuestions = document.getElementById('dynamicQuestions');
    const additionalFields = document.getElementById('additionalFields');
    
    dynamicQuestions.innerHTML = '';
    
    if (selectedValue === 'developerAccess') {
        dynamicQuestions.innerHTML = `
            <div class="form-group">
                <label for="roverGroupNeeded">Do you need to be added to the rhpds-devs Rover Group?</label>
                <select class="form-control custom-input" id="roverGroupNeeded" name="roverGroup">
                    <option value="">Select an option</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="form-group">
                <label for="agnosticVAccess">Do you need access to AgnosticV?</label>
                <select class="form-control custom-input" id="agnosticVAccess" name="agnosticVAccess">
                    <option value="">Select an option</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                </select>
            </div>
            <div class="form-group" id="githubIdQuestion" style="display: none;">
                <label for="githubId">What is your GitHub ID?</label>
                <input type="text" class="form-control custom-input" id="githubId" name="githubId" placeholder="Enter your GitHub ID">
            </div>
        `;
        
        document.getElementById('agnosticVAccess').addEventListener('change', function(event) {
            const githubIdQuestion = document.getElementById('githubIdQuestion');
            githubIdQuestion.style.display = event.target.value === 'Yes' ? 'block' : 'none';
        });
        
        additionalFields.style.display = 'block';
    } else if (selectedValue === 'catalogItem') {
        dynamicQuestions.innerHTML = `
            <div class="form-group">
                <label for="actionNeeded">What needs to be done?</label>
                <select class="form-control custom-input" id="actionNeeded" name="actionNeeded">
                    <option value="">Select an option</option>
                    <option value="onboard">Create a new catalog item for demo.redhat.com</option>
                    <option value="change">Change an existing catalog item in demo.redhat.com</option>
                    <option value="retire">Retire an existing catalog item in demo.redhat.com</option>
                    <option value="offline">Offline an existing catalog item temporarily for maintenance</option>
                </select>
            </div>
            <div class="form-group">
                <label for="contentType">Type of Content</label>
                <select class="form-control custom-input" id="contentType" name="contentType">
                    <option value="">Select an option</option>
                    <option value="communityContent">Community Content - an immutable Base; you add instructions</option>
                    <option value="standardContent">Standard Content - a "new" catalog item developed in agnosticD</option>
                </select>
            </div>
        `;
        additionalFields.style.display = 'block';
        } else {
            additionalFields.style.display = 'none';
        }
    }

function populateAgnosticVConfig(configs) {
    const datalist = document.getElementById('agnosticVConfigList');
    configs.forEach(function(config) {
        const option = document.createElement('option');
        option.value = config;
        datalist.appendChild(option);
    });
}

function updateAgnosticDRoles(roles) {
    const addSelect = document.getElementById('agnosticDRolesToAdd');
    const removeSelect = document.getElementById('agnosticDRolesToRemove');
    
    [addSelect, removeSelect].forEach(select => {
        select.innerHTML = '';
        roles.forEach(role => {
            const option = document.createElement('option');
            option.value = role;
            option.textContent = role;
            select.appendChild(option);
        });
    });
}

function updateSelectedRoles(selectId, displayId) {
    const select = document.getElementById(selectId);
    const display = document.getElementById(displayId);
    const selectedRoles = Array.from(select.selectedOptions).map(option => option.value);
    display.textContent = selectedRoles.length > 0 ? selectedRoles.join(', ') : 'No roles selected';
}

// Add event listeners for role selection
document.getElementById('agnosticDRolesToAdd').addEventListener('change', function() {
    updateSelectedRoles('agnosticDRolesToAdd', 'selectedRoles');
});

document.getElementById('agnosticDRolesToRemove').addEventListener('change', function() {
    updateSelectedRoles('agnosticDRolesToRemove', 'selectedRolesToRemove');
});
