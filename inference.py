"""
Run inference on single image
"""
import torch
from torchvision import transforms, models
from PIL import Image
import os

MODEL_PATH = "best_model.pth"
CLASS_NAMES = ["cat", "dog", "lion", "tiger"]  # example, update per dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def predict_image(image_path):
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img = tfms(img).unsqueeze(0).to(device)
    outputs = model(img)
    _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

print(predict_image("/content/sample_image.jpg"))
