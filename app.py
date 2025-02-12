from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import joblib
import os

# Load the trained model
model = joblib.load("breast_cancer_multiclass.pkl")

# Define tumor subtypes
subtype_mapping = {
    "adenosis": 0,
    "fibroadenoma": 1,
    "tubular_adenoma": 2,
    "phyllodes_tumor": 3,
    "ductal_carcinoma": 4,
    "lobular_carcinoma": 5,
    "mucinous_carcinoma": 6,
    "papillary_carcinoma": 7
}

# Initialize FastAPI app
app = FastAPI()

# Function to extract image features
def extract_features(image):
    try:
        image = image.resize((64, 64))  # Resize image
        image_array = np.array(image)
        features = image_array.flatten()  # Flatten to 1D array
        return features
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

# Define the prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read the image file
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error opening image: {e}")

    # Extract features
    features = extract_features(image)
    if features is None:
        raise HTTPException(status_code=400, detail="Error extracting features")

    # Reshape features for prediction
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)
    predicted_subtype = list(subtype_mapping.keys())[list(subtype_mapping.values()).index(prediction[0])]

    return {"predicted_subtype": predicted_subtype}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)