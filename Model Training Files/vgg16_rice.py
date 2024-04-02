# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import glob
from pathlib import Path
from sklearn.metrics import confusion_matrix

train_path = '/Users/ananyashukla/Desktop/low-altitude-drone/paddy-disease-classification/train_images'
test_path = '/Users/ananyashukla/Desktop/low-altitude-drone/paddy-disease-classification/test_images'

for filepath in glob.glob(train_path + '/*/'):
    files = glob.glob(filepath + '*')
    print(f"{len(files)} \t {Path(filepath).name}")

files = glob.glob(test_path + '/*')
print(f"{len(files)} \t {Path(test_path).name}")

SEED = 123
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
n_classes = len(glob.glob(train_path + '/*/'))
print(f"Number of classes: {n_classes}")

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(data_path, '*', '*')))
        self.class_names = sorted(os.listdir(data_path))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.class_names.index(Path(image_path).parent.name)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = ImageDataset(train_path, transform=transform)
test_dataset = ImageDataset(test_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, n_classes)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}')

# Testing loop
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = sum(predictions == true_labels) / len(true_labels)
print(f'Test Accuracy: {accuracy:.4f}')

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print('Confusion Matrix:')
print(cm)

# Save the model
torch.save(model.state_dict(), 'vgg16_model.pth')