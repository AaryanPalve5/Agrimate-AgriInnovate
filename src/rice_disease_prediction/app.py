from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
from langchain.llms import Ollama

app = Flask(__name__)

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # Adjusted to 4 classes
model.load_state_dict(torch.load('model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the mean and std
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# Define the image transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def preprocess_image(image_bytes):
    """Preprocess the input image."""
    image = Image.open(io.BytesIO(image_bytes))
    return data_transforms(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file part')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction='No selected file')
    
    if file:
        img_bytes = file.read()
        tensor = preprocess_image(img_bytes).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()

        # Assuming you have a list of class names
        class_names = ['Bacterial Blight Disease', 'Blast Disease', 'Brown Spot Disease', 'False Smut Disease']
        predicted_class = class_names[class_idx]

        return render_template('index.html', prediction=f'Class = {predicted_class}')

if __name__ == '__main__':
    app.run(debug=True)
