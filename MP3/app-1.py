# Import required modules
import sqlite3
from flask import Flask, request, render_template, jsonify
from flask import Flask, request, redirect, url_for, flash
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torchvision import datasets, models, transforms
from torchvision.models import resnet50,ResNet50_Weights
import torch.nn.functional as Funct
device = "cuda" if torch.cuda.is_available() else "cpu"

# # Instantiate Flask object
app = Flask(__name__)


# Load the trained ViT model
#weights = ResNet50_Weights.DEFAULT
#model = resnet50().to(device)
#model.load_state_dict(torch.load('MP3/bloodcells_model.pth'))

num_classes = 8

# Define ResNet50 model without the final fully connected layer
model = resnet50(pretrained=False)
model.fc = torch.nn.Linear(2048, num_classes)

# Load the state dict excluding the final fully connected layer
state_dict = torch.load('MP3/bloodcells_model.pth')
state_dict.pop('fc.weight', None)
state_dict.pop('fc.bias', None)
model.load_state_dict(state_dict, strict=False)

model.eval()

class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Configuration
UPLOAD_FOLDER = 'MP3/uploads'
ALLOWED_EXTENSIONS = {'jpg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret_key"

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        #return redirect(request.url)
        return render_template('upload.html', prediction='No file part')

        # Get the image file from the request
    file = request.files['file']

    if file.filename == '':
        #return redirect(request.url)
        return render_template('upload.html', prediction='No selected file')
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Preprocess the image
        image = transform(filename).unsqueeze(0).to(device)
        
        # Make a prediction
        with torch.no_grad():
            output = model(image)
        
        # Get the predicted class
        _, predicted_class = torch.max(output, 1)
        
        return jsonify({'class': int(predicted_class)})
    
    return render_template('upload.html', prediction='Please upload image')


if __name__ == '__main__':
    app.run(debug=True)