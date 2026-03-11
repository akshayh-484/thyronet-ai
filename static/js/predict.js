// Predict page JavaScript
let selectedImage = null;

function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    if (tab === 'numerical') {
        document.querySelector('.tab:first-child').classList.add('active');
        document.getElementById('numerical-tab').classList.add('active');
    } else {
        document.querySelector('.tab:last-child').classList.add('active');
        document.getElementById('image-tab').classList.add('active');
    }
}

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        selectedImage = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('previewImage').style.display = 'block';
            document.getElementById('predictImageBtn').disabled = false;
        };
        reader.readAsDataURL(file);
    }
}

async function predictImage() {
    const formData = new FormData();
    formData.append('image', selectedImage);
    const response = await fetch('/predict-image', { method: 'POST', body: formData });
    const data = await response.json();
    console.log('Image prediction response:', data);
    console.log('Warning present:', 'warning' in data);
    console.log('Warning value:', data.warning);
    if (data.success) displayResult('imageResult', data);
}

async function predictNumerical(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const features = {};
    
    // CRITICAL FIX: Only parse numeric fields as float
    const numericFields = ['Age', 'TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size'];
    
    formData.forEach((value, key) => { 
        // Only parse numeric fields as float, keep text fields as strings
        if (numericFields.includes(key)) {
            features[key] = parseFloat(value);
        } else {
            features[key] = value;
        }
    });
    
    const method = document.getElementById('predictionMethod').value;
    let endpoint = '/predict-numerical';
    let requestBody = { features };
    
    // Route to correct endpoint based on method
    if (method === 'single' || method === 'best') {
        // Use numerical predictor (shows individual model votes)
        endpoint = '/predict-numerical';
    } else if (method.startsWith('ensemble-')) {
        // Use ensemble predictor (voting/stacking/weighted)
        endpoint = '/predict-numerical-ensemble';
        const ensembleMethod = method.replace('ensemble-', '');
        requestBody.method = ensembleMethod;
    }
    
    console.log('=== PREDICTION REQUEST ===');
    console.log('Endpoint:', endpoint);
    console.log('Features:', features);
    console.log('Method:', method);
    
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });
        const data = await response.json();
        console.log('=== PREDICTION RESPONSE ===');
        console.log('Response:', data);
        
        if (data.success) {
            displayResult('numericalResult', data);
        } else {
            alert('Prediction failed: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Fetch error:', error);
        alert('Network error: ' + error.message);
    }
}

function displayResult(elementId, data) {
    console.log('displayResult called with:', data);
    console.log('Warning check:', data.warning);
    
    const resultDiv = document.getElementById(elementId);
    resultDiv.className = `result ${data.prediction.toLowerCase()}`;
    
    let html = `
        <h2>${data.prediction === 'Benign' ? '✅' : '⚠️'} ${data.prediction}</h2>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${data.confidence}%"></div>
        </div>
        <div style="font-size: 1.5rem; margin: 1rem 0; font-weight: 600;">
            Confidence: ${data.confidence}%
        </div>
    `;
    
    // Display warning if present
    if (data.warning) {
        console.log('Adding warning to HTML:', data.warning);
        html += `<div style="background: rgba(255, 193, 7, 0.2); border: 2px solid rgba(255, 193, 7, 0.5); border-radius: 10px; padding: 1rem; margin: 1rem 0;">
            <strong style="color: #fff;">⚠️ Warning:</strong><br>
            <span style="color: #fff; font-size: 0.95rem;">${data.warning}</span>
        </div>`;
    } else {
        console.log('No warning in data');
    }
    
    if (data.method) {
        html += `<div style="font-size: 0.95rem; opacity: 0.95; margin-bottom: 1.5rem; padding: 0.75rem; background: rgba(255,255,255,0.2); border-radius: 8px;">
            📊 Method: ${data.method}
        </div>`;
    }
    
    html += `<div style="margin-top: 1.5rem;">`;
    
    if (data.probabilities) {
        html += `
            <div style="padding: 1rem; background: rgba(255,255,255,0.2); border-radius: 10px; max-width: 300px; margin: 0 auto;">
                <div style="font-size: 0.9rem; opacity: 0.9;">Benign Probability</div>
                <div style="font-size: 1.5rem; font-weight: 700;">${data.probabilities.Benign}%</div>
            </div>
        `;
    }
    
    html += `</div>`;
    
    if (data.individual_predictions) {
        html += `<div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 2px solid rgba(255,255,255,0.3);">
            <strong style="font-size: 1.1rem;">📊 Individual Model Predictions:</strong>
            <div style="margin-top: 1rem; display: grid; gap: 0.75rem;">`;
        
        for (const [model, pred] of Object.entries(data.individual_predictions)) {
            html += `<div style="padding: 0.75rem; background: rgba(255,255,255,0.15); border-radius: 8px;">
                <strong>${model}:</strong> ${pred.prediction} (${pred.confidence}%)
            </div>`;
        }
        
        html += `</div></div>`;
    }
    
    if (data.individual_results) {
        html += `<div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 2px solid rgba(255,255,255,0.3);">
            <strong style="font-size: 1.1rem;">📊 Individual Ensemble Results:</strong>
            <div style="margin-top: 1rem; display: grid; gap: 0.75rem;">
                <div style="padding: 0.75rem; background: rgba(255,255,255,0.15); border-radius: 8px;">
                    <strong>Voting:</strong> ${data.individual_results.voting.prediction} (${data.individual_results.voting.confidence}%)
                </div>
                <div style="padding: 0.75rem; background: rgba(255,255,255,0.15); border-radius: 8px;">
                    <strong>Stacking:</strong> ${data.individual_results.stacking.prediction} (${data.individual_results.stacking.confidence}%)
                </div>
                <div style="padding: 0.75rem; background: rgba(255,255,255,0.15); border-radius: 8px;">
                    <strong>Weighted:</strong> ${data.individual_results.weighted.prediction} (${data.individual_results.weighted.confidence}%)
                </div>
            </div>
        </div>`;
    }
    
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
