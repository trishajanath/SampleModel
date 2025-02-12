import os
import zipfile
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# ✅ Update this to the correct ZIP file path
zip_file_path = "/Users/trishajanath/Desktop/archive (2).zip"

# ✅ Define where to extract
extract_path = "BreakHis_Dataset"

# ✅ Extract ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# ✅ Define tumor subtypes
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

# ✅ Create folders for each subtype
for subtype in subtype_mapping.keys():
    os.makedirs(os.path.join(extract_path, subtype), exist_ok=True)

# ✅ Move images to subtype folders
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            for subtype in subtype_mapping.keys():
                if subtype in root.lower():
                    os.rename(file_path, os.path.join(extract_path, subtype, file))
print("✅ Dataset organized into subtypes.")

# ✅ Function to extract image features
def extract_features(image_path):
    try:
        image = Image.open(image_path).convert("RGB")  # Convert to RGB
        image = image.resize((64, 64))  # ✅ Smaller image for efficiency
        image_array = np.array(image)
        features = image_array.flatten()  # Flatten to 1D array
        return features
    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")
        return None

# ✅ Extract features and assign labels
X = []
y = []

for subtype, label in subtype_mapping.items():
    subtype_path = os.path.join(extract_path, subtype)
    for file in os.listdir(subtype_path):
        file_path = os.path.join(subtype_path, file)
        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)  # Assign the corresponding label

print("✅ Features extracted successfully.")

# ✅ Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train RandomForest model for multi-class classification
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Multi-Class Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Save the trained model
joblib.dump(model, "breast_cancer_multiclass.pkl")
print("✅ Multi-Class Model saved.")

# ✅ Function to predict the subtype of a new image
def predict_subtype(image_path):
    features = extract_features(image_path)
    if features is not None:
        features = np.array(features).reshape(1, -1)  # Reshape for single prediction
        prediction = model.predict(features)
        subtype = list(subtype_mapping.keys())[list(subtype_mapping.values()).index(prediction[0])]
        return subtype
    else:
        return "Error processing image."

# Example usage
new_image_path = "/Users/trishajanath/Desktop/BreakHis_Dataset/Benign/SOB_B_A-14-22549AB-40-006.png" # Replace with the path to your new image
predicted_subtype = predict_subtype(new_image_path)
print(f"Predicted Subtype: {predicted_subtype}")
