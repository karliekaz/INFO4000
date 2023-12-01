# Import required modules
import sqlite3
from flask import Flask, request, render_template
from flask import Flask, request, redirect, url_for, flash
import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch.nn.functional as Funct

# # Instantiate Flask object
app = Flask(__name__)


# Load the trained ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=8, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load('MP3/vit_model.pth'))
model.eval()

class_names = ['Amoeba', 'Euglena', 'Hydra', 'Paramecium', 'Rod_bacteria', 'Spherical_bacteria', 'Spiral_bacteria', 'Yeast']

manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                         std=[0.229, 0.224, 0.225]) 
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
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        #return redirect(request.url)
        return render_template('upload.html', prediction='No file part')
    
    file = request.files['file']    #retrieve all files and read it in 
    
    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        #return redirect(request.url)
        return render_template('upload.html', prediction='No selected file')
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Load and preprocess the uploaded image
        input_image = Image.open(filename)
        input_tensor = manual_transforms(input_image)
        input_batch = input_tensor.unsqueeze(0)
        
        # Make prediction

        with torch.no_grad():
            outputs = model(input_batch).logits
            probabilities = Funct.softmax(outputs, dim=1)

        # get the prediction and probability
        predicted_class = torch.argmax(probabilities).item()
        prediction = class_names[predicted_class]
        probability = probabilities[0, predicted_class].item()
        

        # Make prediction

        return render_template('prediction.html', prediction=prediction, probability = probability)

    return render_template('upload.html', prediction='Please upload image')

if __name__ == '__main__':
    app.run(debug=True)
