"""
Simple Flask API for image classification
"""
from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

CLASS_NAMES = ["cat", "dog", "lion", "tiger"]  # change per dataset
MODEL_PATH = "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
    return jsonify({"prediction": CLASS_NAMES[pred.item()]})

if __name__ == "__main__":
    app.run(debug=True)
