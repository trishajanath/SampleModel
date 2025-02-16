import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models, datasets
from torch.utils.data import DataLoader, Dataset, random_split
import cv2
import matplotlib.pyplot as plt
 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


zip_file_path = "D:\\breastcancerdataset.zip"
extract_path = "D:\\BreakHis_Dataset"
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])


class TumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for subtype, label in subtype_mapping.items():
            subtype_path = os.path.join(root_dir, subtype)
            if os.path.exists(subtype_path):
                for file in os.listdir(subtype_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(subtype_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


dataset = TumorDataset(extract_path, transform=transform)


train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"✅ Loaded {train_size} training samples and {test_size} testing samples.")

# Classifier
resnet_model = models.resnet50(pretrained=True)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, len(subtype_mapping))  # Adjust for 8 classes
resnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.0001)


num_epochs = 10
for epoch in range(num_epochs):
    resnet_model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("✅ Training Completed!")

# Eval
resnet_model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = (correct / total) * 100
print(f"✅ Model Accuracy: {accuracy:.2f}%")

torch.save(resnet_model.state_dict(), "breast_cancer_cnn.pth")
print("✅ CNN Model saved!")

# predict
def predict_subtype(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    resnet_model.eval()
    with torch.no_grad():
        output = resnet_model(image)
        _, predicted_class = torch.max(output, 1)
    
    predicted_subtype = list(subtype_mapping.keys())[predicted_class.item()]
    return predicted_subtype


# ✅ Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.size(2), input_tensor.size(3)))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# ✅ Function to Predict Tumor Subtype and Generate Heatmap
def predict_subtype_with_heatmap(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    resnet_model.eval()
    with torch.no_grad():
        output = resnet_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        predicted_subtype = list(subtype_mapping.keys())[predicted_class.item()]
        probabilities = torch.nn.functional.softmax(output, dim=1)[0] * 100

    grad_cam = GradCAM(resnet_model, resnet_model.layer4[-1])
    heatmap = grad_cam.__call__(image_tensor, predicted_class.item())
    grad_cam.remove_hooks()

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap + np.float32(image.resize((224, 224))) / 255
    heatmap = heatmap / np.max(heatmap)

    plt.imshow(heatmap)
    plt.title(f"Predicted Subtype: {predicted_subtype}\nConfidence: {probabilities[predicted_class.item()]:.2f}%")
    plt.axis('off')
    plt.show()

    return predicted_subtype, probabilities[predicted_class.item()]

# Example Usage
new_image_path = "C:\\Users\\Neelesh\\Downloads\\dfinal.png" 
predicted_subtype, confidence = predict_subtype_with_heatmap(new_image_path)
print(f"Predicted Subtype: {predicted_subtype}, Confidence: {confidence:.2f}%")

