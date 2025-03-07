from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import os 

app = Flask(__name__)
CORS(app)

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
HUGGING_FACE_TRANSLATION_ENDPOINT = os.getenv("HUGGING_FACE_TRANSLATION_ENDPOINT")
HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
    "Content-Type": "application/octet-stream"
}

@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({
            "error": "No text provided"
        }, 400)

    response = requests.post(
        HUGGING_FACE_TRANSLATION_ENDPOINT,
        headers = HEADERS, 
        json={"inputs": text}
    )
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)