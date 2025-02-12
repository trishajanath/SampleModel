import os
import zipfile
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


zip_file_path = "/Users/trishajanath/Desktop/archive (2).zip"


extract_path = "BreakHis_Dataset"


with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


subtypes = [
    "Adenosis",
    "Fibroadenoma",
    "Tubular_Adenoma",
    "Phyllodes_Tumor",
    "Ductal_Carcinoma",
    "Lobular_Carcinoma",
    "Mucinous_Carcinoma",
    "Papillary_Carcinoma"
]


for subtype in subtypes:
    subtype_path = os.path.join(extract_path, subtype)
    os.makedirs(subtype_path, exist_ok=True)

# Move images to correct subfolders based on their subtype
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # ✅ Only move images
            file_path = os.path.join(root, file)
            for subtype in subtypes:
                if subtype.lower() in root.lower():
                    os.rename(file_path, os.path.join(extract_path, subtype, file))
                    break

print("✅ Dataset organized by tumor subtype successfully.")

# ✅ Image parameters
IMG_SIZE = (128, 128)  # Resize all images to 128x128
BATCH_SIZE = 32

# ✅ Data Augmentation for better generalization
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2  # 80% train, 20% validation
)

# ✅ Load training dataset
train_data = datagen.flow_from_directory(
    extract_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # Multi-class classification
    subset="training"
)

# ✅ Load validation dataset
val_data = datagen.flow_from_directory(
    extract_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# ✅ Print class mappings
print("Class Indices:", train_data.class_indices)

# ✅ CNN Model for Tumor Subtype Classification
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(8, activation='softmax')  # ✅ 8 classes
])

# ✅ Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Display model summary
model.summary()

# ✅ Train the CNN model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # You can adjust the number of epochs
)

# ✅ Evaluate the CNN model
val_loss, val_accuracy = model.evaluate(val_data)
print(f"✅ CNN Model Validation Accuracy: {val_accuracy * 100:.2f}%")

# ✅ Save the CNN model
cnn_model_path = "cnn_breast_cancer_model.h5"
model.save(cnn_model_path)
print(f"✅ CNN Model saved to {cnn_model_path}")