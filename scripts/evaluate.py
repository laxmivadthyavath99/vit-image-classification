from transformers import pipeline
from datasets import load_dataset
import os

model_path = os.path.abspath("model")
classifier = pipeline("image-classification", model=model_path)

dataset = load_dataset("imagefolder", data_dir="data", cache_dir="data/processed")
correct = 0

test_data = dataset["test"]
for example in test_data:
    prediction = classifier(example["image"])[0]["label"]
    if prediction.lower() == example["label"].lower():
        correct += 1

accuracy = correct / len(test_data)
print(f"Accuracy: {accuracy:.2%}")