let selectedFile = null;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const verifyBtn = document.getElementById('verifyBtn');
const result = document.getElementById('result');
const loader = document.getElementById('loader');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file');
        return;
    }
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZone.classList.add('hidden');
        preview.classList.remove('hidden');
        verifyBtn.classList.remove('hidden');
        result.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedFile = null;
    fileInput.value = '';
    dropZone.classList.remove('hidden');
    preview.classList.add('hidden');
    verifyBtn.classList.add('hidden');
    result.classList.add('hidden');
}

async function verifyImage() {
    if (!selectedFile) return;
    
    loader.classList.remove('hidden');
    result.classList.add('hidden');
    verifyBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        const response = await fetch('/verify', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        loader.classList.add('hidden');
        verifyBtn.disabled = false;
        
        displayResult(data);
    } catch (error) {
        loader.classList.add('hidden');
        verifyBtn.disabled = false;
        alert('Error verifying image. Please try again.');
    }
}

function displayResult(data) {
    result.classList.remove('hidden', 'genuine', 'forged', 'adversarial');
    
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultMessage = document.getElementById('resultMessage');
    
    if (data.status === 'genuine') {
        result.classList.add('genuine');
        resultIcon.textContent = '✓';
        resultTitle.textContent = 'Genuine Signature';
        resultMessage.textContent = 'The signature appears to be authentic.';
    } else if (data.status === 'forged') {
        result.classList.add('forged');
        resultIcon.textContent = '✗';
        resultTitle.textContent = 'Forged Signature';
        resultMessage.textContent = 'The signature appears to be forged.';
    } else if (data.status === 'adversarial') {
        result.classList.add('adversarial');
        resultIcon.textContent = '⚠';
        resultTitle.textContent = 'Adversarial Attack Detected';
        resultMessage.textContent = 'The image contains adversarial manipulation.';
    }
}
