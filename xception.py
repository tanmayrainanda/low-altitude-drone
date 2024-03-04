import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from sklearn.metrics import confusion_matrix

# Define constants
SEED = 123
EPOCHS = 100
lr = 1e-4
input_size = 299  # InceptionV3 input size
batch_size = 32

# Define paths
train_path = 'paddy-disease-classification/train_images'  # Path to the directory containing training images
test_path = 'paddy-disease-classification/test_images'  # Path to the directory containing test images

# Define transforms
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = ImageFolder(train_path, transform=transform)
test_dataset = ImageFolder(test_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model instance
num_classes = len(train_dataset.classes)
model = inception_v3(pretrained=True, aux_logits=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

# Calculate accuracy and print confusion matrix
accuracy = np.mean(np.array(predictions) == np.array(true_labels))
print(f'Test Accuracy: {accuracy:.4f}')

cm = confusion_matrix(true_labels, predictions)
print('Confusion Matrix:')
print(cm)