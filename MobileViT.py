import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import timm
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

wandb.init(project="paddy-disease-classification")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')

class CustomDataset(Dataset):
    def __init__(self, root_dir, metadata_file, transform=None):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.label_dict = dict(zip(self.metadata['image_id'], self.metadata['label']))

        # Iterate through each class (subfolder)
        self.classes = sorted(set(self.label_dict.values()))

        # Create a dictionary that maps each class to a unique integer
        self.class_to_int = {class_name: i for i, class_name in enumerate(self.classes)}

        # Create a list of tuples (image path, label) for each image
        self.data = [(os.path.join(root_dir, img_id), self.class_to_int[label]) for img_id, label in self.label_dict.items()]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to ensure 3 channels
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)


def train_model(model, num_epochs, train_loader, val_loader, criterion, optimizer, device, scheduler, early_stopper):
    best_accuracy = 0.0
    best_model_wts = None
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total_samples = 0.0, 0, 0

        # Training Loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

        train_loss = running_loss / total_samples
        train_acc = running_corrects.double() / total_samples
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item()) 

        # Validation Loop 
        model.eval()
        val_loss, val_corrects, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_samples += labels.size(0)

        val_loss /= val_samples
        val_acc = val_corrects.double() / val_samples
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())  # Convert to Standard Python Number
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        scheduler.step(val_loss)
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_wts = model.state_dict()

    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return y_true, y_pred

# Dataset Root Directory
root_dir = "paddy-doctor-diseases-medium/trainset_full"
metadata_file = "paddy-doctor-diseases-medium/metadata.csv"

# Set Computation Device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations (NumPy to Tensor and Normalization)
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((480, 480)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create Dataset instance
dataset = CustomDataset(root_dir, metadata_file, transform)

# Train and Test Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Apply transforms to datasets
train_dataset.dataset.transform = transform
test_dataset.dataset.transform = transform

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=None)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=None)

num_epochs = 20
model_name = 'mobilevit_s'

print(f"Training MobileViT_s...")
model = timm.create_model(model_name, pretrained=True, num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
early_stopper = EarlyStopping(patience=10, min_delta=0.001)

model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, num_epochs, train_loader, test_loader, criterion, optimizer, device, scheduler, early_stopper)

# Evaluate the model
y_true, y_pred = evaluate_model(model, test_loader, device)
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

torch.save(model.state_dict(), f'{model_name}_rice_disease_classifier.pth')

# Add evaluation results to 'results'
results = []
results.append({
    'Model': model_name,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})
wandb.save('mobilevit_model.pth')
wandb.finish()
