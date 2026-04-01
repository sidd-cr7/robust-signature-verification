from flask import Flask, render_template, request, jsonify
import os, sys
from pathlib import Path
from utils import predict_image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare():
    return render_template('compare.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

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

        return jsonify({'status': status, 'result': message, 'distance': round(dist, 2)})

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

    finally:
        if os.path.exists(path1): os.remove(path1)
        if os.path.exists(path2): os.remove(path2)

if __name__ == '__main__':
    app.run(debug=True)
