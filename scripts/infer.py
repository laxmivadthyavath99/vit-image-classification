from transformers import pipeline
from PIL import Image
import os

# Path to the saved model
model_path = os.path.abspath("../model")

# Load classifier pipeline
classifier = pipeline(
    "image-classification",
    model=model_path,
    image_processor=model_path
)

# Classify function
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    result = classifier(image)
    return result
