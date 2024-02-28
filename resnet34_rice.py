import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
import torch
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
from sklearn.preprocessing import LabelEncoder
import wandb

# Initialize a new run
wandb.init(project="rice-disease-classification")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for device
device = torch.device("mps")

# Define paths
train_path = Path('paddy-disease-classification/Trainset')
test_path = Path('paddy-disease-classification/test_images')

# Load train labels and perform label encoding
train_df = pd.read_csv('paddy-disease-classification/train.csv')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])

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
        label = self.df.iloc[idx, 4]  # Adjusted to use the correct index for 'label_encoded'

        if self.transform:
            image = self.transform(image)

        return image, label

# Split data into train and validation sets
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Create data loaders
train_data = ImageDataset(df=train_df, img_dir=train_path, transform=transform)
valid_data = ImageDataset(df=valid_df, img_dir=train_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=200, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=200, shuffle=False)

# Define the model
model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_df['label'].unique()))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

# Wrap your model and optimizer with wandb
wandb.watch(model, log_freq=100)

# Train the model
model.train()
for epoch in range(50):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    wandb.log({"epoch": epoch, "loss": running_loss})

# Validate the model
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.view(-1).cpu().numpy())
        all_labels.extend(labels.view(-1).cpu().numpy())

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
print(f"Accuracy: {accuracy_score(all_labels, all_preds)}")

# Log final results with wandb
wandb.log({"classification_report": classification_report(all_labels, all_preds),
           "confusion_matrix": confusion_matrix(all_labels, all_preds),
           "accuracy": accuracy_score(all_labels, all_preds)})

# Close your wandb run
wandb.finish()