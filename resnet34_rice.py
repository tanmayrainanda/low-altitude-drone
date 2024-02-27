import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import glob
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define paths
train_path = Path('/Users/tanmay/Documents/GitHub/low-altitude-drone/paddy-disease-classification/train_images')  # replace with your local path
test_path = Path('/Users/tanmay/Documents/GitHub/low-altitude-drone/paddy-disease-classification/test_images')  # replace with your local path

# Load train labels
train_df = pd.read_csv('/Users/tanmay/Documents/GitHub/low-altitude-drone/paddy-disease-classification/train.csv')  # replace with your local path
print(train_df.shape)
print(train_df.label.value_counts())

# Define transformations
transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define custom dataset
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

# Split data into train and validation sets
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Create data loaders
train_data = ImageDataset(df=train_df, img_dir=train_path, transform=transform)
valid_data = ImageDataset(df=valid_df, img_dir=train_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

# Define the model
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_df.label.unique()))
model = model.to('cpu')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Train the model
model.train()
for epoch in range(50):  # replace with the number of epochs you want to train for
    for inputs, labels in train_loader:
        inputs = inputs.to('cpu')
        labels = labels.to('cpu')

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

# Validate the model
model.eval()
with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs = inputs.to('cpu')
        labels = labels.to('cpu')

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        print(classification_report(labels.cpu(), preds.cpu()))
        print(confusion_matrix(labels.cpu(), preds.cpu()))
        print(accuracy_score(labels.cpu(), preds.cpu()))