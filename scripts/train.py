from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
dataset = load_dataset("imagefolder", data_dir="data/train")



label_names = dataset["train"].features["label"].names
print("Label names:", label_names)  # Should print ['nsfw', 'safe'] or ['safe', 'nsfw']
label2id = {name: i for i, name in enumerate(label_names)}
id2label = {i: name for i, name in enumerate(label_names)}
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
def transform(example):
    image = example["image"].convert("RGB").resize((224, 224))
    encoding = processor(images=image, return_tensors="pt")
    example["pixel_values"] = encoding["pixel_values"][0]
    example["label"] = example["label"]
    return example
dataset = dataset.map(transform)
dataset.set_format(type="torch", columns=["pixel_values", "label"])
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(label_names),
    id2label=id2label,
    label2id=label2id
)
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=4,
    num_train_epochs=4,
    logging_steps=10,
    save_steps=20,
    save_total_limit=1,
    remove_unused_columns=False
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
model.save_pretrained("./model")
processor.save_pretrained("./model")

