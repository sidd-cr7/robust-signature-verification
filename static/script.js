// ── Page transitions ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    const content = document.getElementById('page-content');
    if (content) requestAnimationFrame(() => content.classList.add('loaded'));

    document.querySelectorAll('.nav-link, .nav-cta, .btn-primary, .btn-ghost').forEach(link => {
        link.addEventListener('click', e => {
            const href = link.getAttribute('href');
            if (!href || href.startsWith('#')) return;
            e.preventDefault();
            if (content) content.classList.remove('loaded');
            setTimeout(() => window.location.href = href, 350);
        });
    });
});

// ── Drag & drop helpers ────────────────────────────────────
function onDrag(e, el) { e.preventDefault(); el.classList.add('drag-over'); }
function offDrag(el)   { el.classList.remove('drag-over'); }
function onDrop(e, target) {
    e.preventDefault();
    offDrag(e.currentTarget);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file, target);
}

// ── File handling ──────────────────────────────────────────
let selectedFile = null;
const siameseFiles = { 1: null, 2: null };

function handleFile(file, target) {
    if (!file.type.startsWith('image/')) return alert('Please upload an image file');
    const reader = new FileReader();
    reader.onload = e => {
        if (target === 'single') {
            selectedFile = file;
            document.getElementById('previewImage').src = e.target.result;
            document.getElementById('dropZone').classList.add('hidden');
            document.getElementById('preview').classList.remove('hidden');
            document.getElementById('verifyBtn').classList.remove('hidden');
        } else {
            siameseFiles[target] = file;
            document.getElementById('previewImage' + target).src = e.target.result;
            document.getElementById('dropZone' + target).classList.add('hidden');
            document.getElementById('preview' + target).classList.remove('hidden');
            if (siameseFiles[1] && siameseFiles[2])
                document.getElementById('siameseBtn').classList.remove('hidden');
        }
        hideResult();
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('dropZone').classList.remove('hidden');
    document.getElementById('preview').classList.add('hidden');
    document.getElementById('verifyBtn').classList.add('hidden');
    hideResult();
}

function removeSiamese(n) {
    siameseFiles[n] = null;
    document.getElementById('fileInput' + n).value = '';
    document.getElementById('dropZone' + n).classList.remove('hidden');
    document.getElementById('preview' + n).classList.add('hidden');
    document.getElementById('siameseBtn').classList.add('hidden');
    hideResult();
}

// ── Verify actions ─────────────────────────────────────────
async function verifyImage() {
    if (!selectedFile) return;
    const fd = new FormData();
    fd.append('image', selectedFile);
    await sendRequest('/verify', fd, false);
}

async function verifySiamese() {
    if (!siameseFiles[1] || !siameseFiles[2]) return;
    const fd = new FormData();
    fd.append('image1', siameseFiles[1]);
    fd.append('image2', siameseFiles[2]);
    await sendRequest('/verify-siamese', fd, true);
}

// ── Request ────────────────────────────────────────────────
async function sendRequest(url, formData, showScore) {
    const loader = document.getElementById('loader');
    const btn    = document.getElementById(url === '/verify' ? 'verifyBtn' : 'siameseBtn');
    loader.classList.remove('hidden');
    hideResult();
    btn.disabled = true;
    try {
        const res  = await fetch(url, { method: 'POST', body: formData });
        const data = await res.json();
        if (data.error) alert('Error: ' + data.error);
        else displayResult(data, showScore);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        loader.classList.add('hidden');
        btn.disabled = false;
    }
}

// ── Result display ─────────────────────────────────────────
const RESULT_MAP = {
    genuine:     { icon: '✓', label: 'Genuine Signature',           explanation: 'The signatures are highly similar based on learned features.' },
    forged:      { icon: '✗', label: 'Forged Signature',            explanation: 'The signatures are significantly different based on learned features.' },
    uncertain:   { icon: '⚠', label: 'Uncertain',                   explanation: 'The similarity score falls in an overlapping range. Manual verification recommended.' },
    adversarial: { icon: '⚠', label: 'Adversarial Attack Detected', explanation: 'The image contains adversarial manipulation.' },
};

function displayResult(data, showScore) {
    const result = document.getElementById('result');
    const entry  = RESULT_MAP[data.status] || { icon: '?', label: data.message || 'Unknown', explanation: '' };

    result.className = 'result ' + (data.status || '');
    document.getElementById('resultBadge').textContent       = (data.status || '').toUpperCase();
    document.getElementById('resultIcon').textContent         = entry.icon;
    document.getElementById('resultTitle').textContent        = entry.label;
    document.getElementById('resultExplanation').textContent  = entry.explanation;

    const scoreSection = document.getElementById('scoreSection');
    if (scoreSection) {
        if (showScore && data.distance !== undefined) {
            const dist       = parseFloat(data.distance);
            const confidence = Math.max(0, Math.min(100, Math.round((1 - dist / 2.0) * 100)));
            document.getElementById('scoreValue').textContent        = dist.toFixed(2);
            document.getElementById('confidenceBar').style.width     = confidence + '%';
            scoreSection.classList.remove('hidden');
        } else {
            scoreSection.classList.add('hidden');
        }
    }

    result.classList.remove('hidden');
}

function hideResult() {
    const r = document.getElementById('result');
    if (r) r.classList.add('hidden');
}
