from flask import Flask, request, jsonify 
from werkzeug.utils import secure_filename
from PIL import Image
from models.model import load_classification_model, classify_image
import os

feature_extractor, model = load_classification_model()

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({
            'error': 'No files uploaded'
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'error': 'No selected file'
        }), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    predicted_label = classify_image(filepath, feature_extractor, model)
    return jsonify({
        'label': predicted_label
    })

if __name__ == '__main__':
    app.run(debug=True)