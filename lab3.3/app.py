from flask import Flask, request, jsonify
import pytesseract
import numpy as np
import cv2
from PIL import Image 
import io 

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

@app.route("/ocr", methods=["POST"])
def recognize_text():
    if "file" not in request.files:
        return jsonify({
            "error": "no file uploaded"
        })
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = np.array(image)

    process_image = preprocess_image(image)

    text = pytesseract.image_to_string(process_image)

    return jsonify({
        "recognized_text": text
    })

if __name__ == "__main__":
    app.run(debug=True)