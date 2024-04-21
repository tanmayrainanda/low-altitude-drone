import torch
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import wandb

# Initialize a new run
wandb.init(project="paddy-disease-classification")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
train_path = Path('paddy-doctor-diseases-medium/train_images')
test_path = Path('paddy-disease-classification/test_images')

# Load train labels and perform label encoding
train_df = pd.read_csv('paddy-doctor-diseases-medium/metadata.csv')

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

# Adjusted ImageDataset class to use label encoding
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
        label = self.df.iloc[idx]['label_encoded']

        if self.transform:
            image = self.transform(image)

        return image, label

# Split data into train and validation sets
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Create data loaders
train_dataset = ImageDataset(df=train_df, img_dir=train_path, transform=transform)
valid_dataset = ImageDataset(df=valid_df, img_dir=train_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=200, shuffle=False)

# Define the model
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(np.unique(train_df['label_encoded'])))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training and validation functions
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    return train_loss, train_acc

def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_pred = []
    all_true = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_pred.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    valid_loss = running_loss / len(valid_loader)
    valid_acc = correct / total
    return valid_loss, valid_acc, all_true, all_pred

# Training loop
n_epochs = 100

for epoch in range(n_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc, all_true, all_pred = validate(model, valid_loader, criterion, device)

    # Log metrics to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Valid Loss": valid_loss,
        "Valid Accuracy": valid_acc
    })

    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

# Evaluation
class_names = label_encoder.classes_
all_true, all_pred = validate(model, valid_loader, criterion, device)[2:]
acc = accuracy_score(all_true, all_pred)
print("MobileNet Model Accuracy on Validation Set: {:.2f}%".format(acc * 100))

cls_report = classification_report(all_true, all_pred, target_names=class_names, digits=5)
print(cls_report)

# Save the model
torch.save(model.state_dict(), 'mobilenet_model.pth')

wandb.save('mobilenet_model.pth')
wandb.finish()