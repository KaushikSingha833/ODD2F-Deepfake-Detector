// Analytics Globals
let sessionHistory = [];
let historyChartInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    
    // Fetch global model metrics
    fetch('/metrics')
        .then(res => res.json())
        .then(data => {
            const gmAcc = document.getElementById('gm-acc');
            if (gmAcc) {
                gmAcc.textContent = data.accuracy + '%';
                document.getElementById('gm-prec').textContent = data.precision + '%';
                document.getElementById('gm-rec').textContent = data.recall + '%';
                document.getElementById('gm-f1').textContent = data.f1_score + '%';
            }
        })
        .catch(err => console.log('Error fetching metrics', err));
    
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    
    const sectionUpload = document.getElementById('upload-section');
    const sectionAnalyzing = document.getElementById('analyzing-section');
    const sectionResults = document.getElementById('results-section');
    
    const loaderText = document.getElementById('loader-text');
    const resetBtn = document.getElementById('reset-btn');
    
    // Result DOM elements
    const resultBadge = document.getElementById('result-badge');
    const confidenceScore = document.getElementById('confidence-score');
    const imgOriginal = document.getElementById('img-original');
    const imgEla = document.getElementById('img-ela');
    const imgHeatmap = document.getElementById('img-heatmap');

    // Setup Drag & Drop
    dropZone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover'), false);
    });

    dropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if(files.length) handleFiles(files[0]);
    }

    function handleFileSelect(e) {
        if(this.files.length) handleFiles(this.files[0]);
    }

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show Analyzing UI
        showSection(sectionAnalyzing);
        cycleLoaderText();

        const formData = new FormData();
        formData.append('file', file);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                populateResults(data);
                setTimeout(() => showSection(sectionResults), 500); // Small delay for smooth transition
            } else {
                alert('Error: ' + data.message);
                showSection(sectionUpload);
            }
        })
        .catch(err => {
            console.error(err);
            alert('An error occurred while analyzing the image.');
            showSection(sectionUpload);
        });
    }

    function populateResults(data) {
        // Set Badge
        resultBadge.textContent = data.result;
        resultBadge.className = 'result-badge ' + data.result.toLowerCase();
        
        // Set Confidence
        confidenceScore.textContent = data.confidence + '%';
        
        // Set Images
        imgOriginal.src = 'data:image/jpeg;base64,' + data.original_b64;
        imgEla.src = 'data:image/jpeg;base64,' + data.ela_b64;
        imgHeatmap.src = 'data:image/jpeg;base64,' + data.heatmap_b64;

        const isFake = data.result.toLowerCase() === 'fake';
        updateCharts(data, isFake, data.confidence);
    }

    resetBtn.addEventListener('click', () => {
        fileInput.value = ''; // Clear input
        showSection(sectionUpload);
    });

    function showSection(sectionToShow) {
        [sectionUpload, sectionAnalyzing, sectionResults].forEach(sec => {
            sec.classList.remove('active');
            setTimeout(() => {
                if(!sec.classList.contains('active')) sec.classList.add('hidden');
            }, 300); // Wait for opacity transition before display:none
        });

        sectionToShow.classList.remove('hidden');
        // Small delay to allow display block to apply before opacity transition
        setTimeout(() => sectionToShow.classList.add('active'), 50);
    }

    function cycleLoaderText() {
        const texts = ["Analyzing Pixels...", "Extracting ELA Map...", "Computing Spatial Attention...", "Finalizing Verdict..."];
        let i = 0;
        const interval = setInterval(() => {
            if(sectionAnalyzing.classList.contains('active')) {
                i = (i + 1) % texts.length;
                loaderText.textContent = texts[i];
            } else {
                clearInterval(interval);
            }
        }, 1500);
    }

    // Progress Polling Logic
    const progressSection = document.getElementById('progress-section');
    const progEpoch = document.getElementById('prog-epoch');
    const progBatch = document.getElementById('prog-batch');
    const progLoss = document.getElementById('prog-loss');
    const progAcc = document.getElementById('prog-acc');
    const progBar = document.getElementById('progress-bar-fill');
    // Fetch Progress Logic
    function fetchProgress() {
        fetch('/progress?t=' + Date.now())
            .then(res => res.json())
            .then(data => {
                if (data.status === 'inactive') {
                    // Do nothing if inactive
                    return;
                }
                
                // Show progress panel if it's hidden
                if (progressSection.classList.contains('hidden')) {
                    progressSection.classList.remove('hidden');
                    setTimeout(() => progressSection.classList.add('active'), 50);
                }
                
                if (data.status === 'training') {
                    progEpoch.textContent = `${data.epoch}/${data.total_epochs}`;
                    progBatch.textContent = `${data.batch}/${data.total_batches}`;
                    progLoss.textContent = data.loss.toFixed(4);
                    progAcc.textContent = (data.accuracy * 100).toFixed(2) + '%';
                    
                    const pct = (data.batch / data.total_batches) * 100;
                    progBar.style.width = pct + '%';
                    
                    const pctEl = document.getElementById('prog-pct');
                    if(pctEl) pctEl.textContent = `(${pct.toFixed(1)}%)`;
                    
                } else if (data.status === 'validation_done') {
                    progEpoch.textContent = `${data.epoch}/${data.total_epochs} (Validating...)`;
                    progLoss.textContent = `Val Loss: ${data.loss.toFixed(4)}`;
                    progAcc.textContent = `Val Acc: ${(data.accuracy * 100).toFixed(2)}% | Val AUC: ${data.auc.toFixed(4)}`;
                    progBar.style.width = '100%';
                    
                    const pctEl = document.getElementById('prog-pct');
                    if(pctEl) pctEl.textContent = `(100%)`;
                }
            })
            .catch(err => console.log('Error fetching progress:', err));
    }
    
    // Poll every 2 seconds
    setInterval(fetchProgress, 2000);
    // Initial fetch
    fetchProgress();

    // Control Buttons Logic
    const btnPause = document.getElementById('btn-pause');
    const btnCancel = document.getElementById('btn-cancel');
    let isPaused = false;

    if (btnPause) {
        btnPause.addEventListener('click', () => {
            isPaused = !isPaused;
            const action = isPaused ? 'pause' : 'resume';
            
            fetch('/control', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            });

            if (isPaused) {
                btnPause.textContent = 'Resume';
                btnPause.classList.remove('pause-btn');
                btnPause.classList.add('resume-btn');
                document.getElementById('train-status-indicator').style.animation = 'none';
                document.getElementById('train-status-indicator').style.backgroundColor = '#ffc800';
            } else {
                btnPause.textContent = 'Pause';
                btnPause.classList.remove('resume-btn');
                btnPause.classList.add('pause-btn');
                document.getElementById('train-status-indicator').style.animation = 'blink 1s infinite alternate';
                document.getElementById('train-status-indicator').style.backgroundColor = 'var(--accent)';
            }
        });
    }

    if (btnCancel) {
        btnCancel.addEventListener('click', () => {
            if(confirm("Are you sure you want to cancel the training process? This cannot be undone.")) {
                fetch('/control', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'cancel' })
                });
                document.getElementById('progress-section').classList.add('hidden');
            }
        });
    }

});

// Chart.js Initialization and Updating
function initCharts() {
    Chart.defaults.color = 'rgba(255, 255, 255, 0.7)';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';

    const ctxHistory = document.getElementById('historyChart').getContext('2d');
    historyChartInstance = new Chart(ctxHistory, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Fake Probability (%)',
                data: [],
                fill: true,
                backgroundColor: 'rgba(100, 255, 218, 0.1)',
                borderColor: '#64ffda',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 100 } }
        }
    });
}

function updateCharts(data, isFake, confidence) {
    document.getElementById('analytics-section').classList.remove('hidden');

    // Update Custom CSS Bar Chart
    if (data.forensics) {
        document.getElementById('bar-ela').style.width = data.forensics.ela_density + '%';
        document.getElementById('val-ela').textContent = data.forensics.ela_density.toFixed(1) + '%';
        
        document.getElementById('bar-noise').style.width = data.forensics.noise_variance + '%';
        document.getElementById('val-noise').textContent = data.forensics.noise_variance.toFixed(1) + '%';
        
        document.getElementById('bar-color').style.width = data.forensics.color_variance + '%';
        document.getElementById('val-color').textContent = data.forensics.color_variance.toFixed(1) + '%';
    }

    // Update Line Chart (Calculate raw fake probability)
    let fakeScore = isFake ? confidence : (100 - confidence);
    sessionHistory.push(fakeScore);
    
    historyChartInstance.data.labels.push(`Image ${sessionHistory.length}`);
    historyChartInstance.data.datasets[0].data.push(fakeScore);
    historyChartInstance.update();
}
