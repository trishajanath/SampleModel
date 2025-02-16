from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import numpy as np
import joblib
import os
import io
import base64
from mainmodel import predict_subtype_with_heatmap
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from torchvision import models



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Subtype mapping
subtype_mapping = {
    "adenosis": 0, "fibroadenoma": 1, "tubular_adenoma": 2, "phyllodes_tumor": 3,
    "ductal_carcinoma": 4, "lobular_carcinoma": 5, "mucinous_carcinoma": 6, "papillary_carcinoma": 7
}
idx_to_subtype = {v: k for k, v in subtype_mapping.items()}

# Image Transformations (No Normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Trained Model
def load_model(model_path):
    model = models.resnet50(weights=None)  # Fixed deprecated warning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(subtype_mapping))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model_path = "breast_cancer_cnn.pth"
model = load_model(model_path)
print("✅ Model Loaded Successfully!")

app = FastAPI()

# Ensure upload directory exists
os.makedirs("uploads", exist_ok=True)

def extract_features(image):
    try:
        image = image.resize((64, 64))  # Resize image
        image_array = np.array(image)
        features = image_array.flatten()  # Flatten to 1D array
        return features
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head>
            <title>Upload Image</title>
        </head>
        <body>
            <h1>Upload an Image</h1>
            <form action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    """

@app.post("/upload", response_class=HTMLResponse)
async def upload(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Get prediction and heatmap
    predicted_subtype, confidence, heatmap = predict_subtype_with_heatmap(file_path)

    # Normalize heatmap if needed
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap)

    # Convert heatmap to base64
    buffered = io.BytesIO()
    heatmap.save(buffered, format="PNG")
    heatmap_str = base64.b64encode(buffered.getvalue()).decode()

    return f"""
    <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p>Predicted Subtype: {predicted_subtype}</p>
            <p>Confidence: {confidence:.2f}%</p>
            <h2>Heatmap</h2>
            <img src="data:image/png;base64,{heatmap_str}" alt="Heatmap">
            <br>
            <a href="/">Upload another image</a>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error opening image: {e}")

    features = extract_features(image)
    if features is None:
        raise HTTPException(status_code=400, detail="Error extracting features")

    features = np.array(features).reshape(1, -1)

    # Make prediction
    try:
        prediction = model.predict(features)
        predicted_subtype = list(subtype_mapping.keys())[list(subtype_mapping.values()).index(prediction[0])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    return {"predicted_subtype": predicted_subtype}

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)