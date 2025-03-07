from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import os 

app = Flask(__name__)
CORS(app)

load_dotenv()

HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
HUGGING_FACE_SENTIMENT_ANALYSIS_ENDPOINT = os.getenv("HUGGING_FACE_SENTIMENT_ANALYSIS_ENDPOINT")
HEADERS = {
    "Authorization": f"Bearer {HUGGING_FACE_TOKEN}",
    "Content-Type": "application/octet-stream"
}

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json["text"]
    response = requests.post(
        HUGGING_FACE_SENTIMENT_ANALYSIS_ENDPOINT, 
        headers = HEADERS, 
        json={"inputs": text}
    )
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)