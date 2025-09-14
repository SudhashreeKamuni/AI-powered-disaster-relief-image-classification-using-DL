# eval.py

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print('entered eval file')

# --- Configuration ---
NUM_CLASSES = 7   # your model was trained on 7 classes
BATCH_SIZE = 32
NUM_WORKERS = 2
DATA_DIR = "/content/drive/MyDrive/disaster_classification/reorganized/dev"
MODEL_PATH = "saved_models/model_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dev_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"Loaded {len(dev_dataset)} images for evaluation.")
print(f"Class labels: {dev_dataset.classes}")

# --- Model ---
model = models.resnet18(weights=None)  # no pretrained
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

state = torch.load(MODEL_PATH, map_location=device)
if "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)

print(f"Model loaded successfully from {MODEL_PATH}")

model.to(device)
model.eval()

# --- Evaluation ---
all_preds, all_labels = [], []
print("Starting evaluation...")
with torch.no_grad():
    for i, (imgs, labels) in enumerate(dev_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(dev_loader)} batches...")

print("\n✅ Evaluation Complete!")

# --- Metrics ---
target_names = dev_dataset.classes

# Print classification report
print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=target_names))

# Save classification report as CSV
report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv", index=True)
print("✅ Classification report saved as classification_report.csv")

# Macro F1
macro_f1 = f1_score(all_labels, all_preds, average="macro")
print("Macro F1 Score:", macro_f1)

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=target_names,
    yticklabels=target_names,
    cmap="Blues"
)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")

# Save + Show
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("✅ Confusion matrix saved as confusion_matrix.png and displayed.")
