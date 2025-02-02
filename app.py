from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import torch

app = Flask(__name__)
CORS(app)  # Allow requests from frontend (Fixes CORS errors)

# Set up the model
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)


@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json
        text = data.get("text", "")
        length = int(data.get("length", 50))  # Summary length (default 50%)

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Generate summary
        summary = summarizer(text, max_length=length, min_length=30, do_sample=False)[0]['summary_text']
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
