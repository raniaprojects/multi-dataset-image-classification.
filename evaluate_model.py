"""
Evaluate saved model and print accuracy, confusion matrix, and classification report
"""
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

DATASET_PATH = "/content/drive/MyDrive/datasets/Animals/val"
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

data = datasets.ImageFolder(DATASET_PATH, transform=test_tfms)
loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(data.classes)
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Classification Report:\n", classification_report(all_labels, all_preds, target_names=data.classes))
print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
