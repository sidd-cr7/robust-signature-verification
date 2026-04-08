from flask import Flask, render_template, request, jsonify
import os, sys
from pathlib import Path
from datetime import datetime
from utils import predict_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.url_map.strict_slashes = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

@app.route('/analytics')
def analytics():
    total = len(history)
    genuine_count  = sum(1 for h in history if h['status'] == 'genuine')
    forged_count   = sum(1 for h in history if h['status'] == 'forged')
    uncertain_count= sum(1 for h in history if h['status'] == 'uncertain' or h['status'] == 'adversarial')
    distances = [h['distance'] for h in history if isinstance(h.get('distance'), (int, float))]
    d_low  = sum(1 for d in distances if d < 0.8)
    d_mid  = sum(1 for d in distances if 0.8 <= d <= 1.1)
    d_high = sum(1 for d in distances if d > 1.1)
    times  = [h['time'] for h in history]
    return render_template('analytics.html',
        total=total, genuine=genuine_count, forged=forged_count, uncertain=uncertain_count,
        d_low=d_low, d_mid=d_mid, d_high=d_high, times=times,
        history=history
    )

@app.route('/history')
def history_page():
    return render_template('history.html', history=history)

@app.route('/verify', methods=['POST'])
def verify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    result = predict_image(filepath)
    os.remove(filepath)
    history.append({
        'result': result.get('message', result.get('status', '')),
        'status': result.get('status', ''),
        'distance': result.get('distance', '-'),
        'time': datetime.now().strftime('%d %b %Y, %H:%M')
    })
    return jsonify(result)

@app.route('/verify-siamese', methods=['POST'])
def verify_siamese():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({'error': 'Two images required'}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    path1 = os.path.join(app.config['UPLOAD_FOLDER'], 'sig1_' + file1.filename)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], 'sig2_' + file2.filename)
    file1.save(path1)
    file2.save(path2)

    try:
        from verification.siamese_train import SiameseNetwork
        import torch
        from torchvision import transforms
        from PIL import Image

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SiameseNetwork().to(device)
        model.load_state_dict(torch.load(
            str(ROOT / 'siamese_model.pth'), map_location=device
        ))
        model.eval()

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((105, 105)),
            transforms.ToTensor()
        ])

        def load(p):
            img = Image.open(p).convert('RGB')
            return transform(img).unsqueeze(0).to(device)

        img1 = load(path1)
        img2 = load(path2)

        with torch.no_grad():
            out1, out2 = model(img1, img2)
            dist = torch.nn.functional.pairwise_distance(out1, out2).item()

        if dist < 0.95:
            status, message = 'genuine', 'Genuine Signature'
        elif dist > 1.1:
            status, message = 'forged', 'Forged Signature'
        else:
            status, message = 'uncertain', 'Uncertain — needs manual verification'

        history.append({
            'result': message,
            'status': status,
            'distance': round(dist, 2),
            'time': datetime.now().strftime('%d %b %Y, %H:%M')
        })
        return jsonify({'status': status, 'result': message, 'distance': round(dist, 2)})

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

    finally:
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
