from transformers import pipeline
import os

# Downloads and saves to your local model cache
classifier = pipeline("image-classification", model_path="users/testing/desktop/vit_project/model1")

# Run once, then copy cache to your 'model/' folder if needed.
