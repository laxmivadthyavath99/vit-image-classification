# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from transformers import pipeline
# from PIL import Image
# import os

# app = Flask(__name__)
# CORS(app)

# # Load model + processor once
# classifier = pipeline("image-classification", model="./model", image_processor="./model")

# @app.route("/predict", methods=["POST"])



# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No file selected"}), 400
    
#     img = Image.open(file.stream)
#     preds = classifier(img)
#     return jsonify(preds)

# if __name__ == "__main__":
#     app.run(port=5000)
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from PIL import Image

app = Flask(__name__)
CORS(app)

# Load model + processor once
classifier = pipeline("image-classification", model="./model", image_processor="./model")

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No file selected"}), 400
    
#     img = Image.open(file.stream).convert("RGB")
#     preds = classifier(img)
# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "No file selected"}), 400

#     img = Image.open(file.stream)
#     preds = classifier(img)

#     # Only return labels (top 2 by score)
#     top_preds = sorted(preds, key=lambda x: x['score'], reverse=True)[:2]
#     result = [{"label": pred["label"]} for pred in top_preds]

#     return jsonify(result)
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    img = Image.open(file.stream)
    preds = classifier(img)

    # Filter predictions with a threshold
    threshold = 0.5
    top_preds = [p for p in preds if p["score"] >= threshold]

    if not top_preds:
        return jsonify([{"label": "others"}])

    # Return only labels (top 2 if available)
    top_preds = sorted(top_preds, key=lambda x: x['score'], reverse=True)[:2]
    result = [{"label": p["label"]} for p in top_preds]
    
    return jsonify(result)




    # # Get top 2 predictions only
    # top_preds = preds[:2]

    # return jsonify(top_preds)

if __name__ == "__main__":
    app.run(port=5000)
