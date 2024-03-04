import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import inception_v3
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import wandb

# Initialize a new run
wandb.init(project="rice-disease-classification")

# Define constants
SEED = 123
EPOCHS = 100
lr = 1e-4
input_size = 299  # InceptionV3 input size
batch_size = 32

# Define paths
train_path = "paddy-disease-classification/Trainset" # Path to the directory containing training images
test_path =  "paddy-disease-classification/test_images" # Path to the directory containing test images

# Load train labels and perform label encoding
train_df = pd.read_csv('paddy-disease-classification/train.csv')

# Define custom dataset
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.classes = df['label'].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        image = Image.open(img_name)
        # label = self.df.iloc[idx, 4]  # Adjusted to use the correct index for 'label_encoded'
        label = self.df.iloc[idx, self.df.columns.get_loc('label')]

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Define transforms
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split data into train and validation sets
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Load datasets
# Create data loaders
train_data = ImageDataset(df=train_df, img_dir=train_path, transform=transform)
valid_data = ImageDataset(df=valid_df, img_dir=train_path, transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model instance
num_classes = len(train_data.classes)
# model = inception_v3(pretrained=True, aux_logits=False)
model = inception_v3(pretrained=True, aux_logits=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
# Training loop
# Training loop
# Training loop
for epoch in range(EPOCHS):
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = torch.tensor([list(train_data.classes).index(label) for label in labels]).to(device)
        optimizer.zero_grad()
        outputs, aux_outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4*loss2
        loss.backward()
        optimizer.step()

# Testing loop
model.eval()
predictions = []
true_labels = []
with torch.no_grad():
    for inputs, labels in valid_loader:
        labels = torch.tensor([list(valid_data.classes).index(label) for label in labels]).to(device)
        inputs = inputs.to(device)
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

# Initialize a new run
wandb.init(project="rice-disease-classification")