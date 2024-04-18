import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from yolov6.models.yolo import BuildYOLOX
from yolov6.utils.events import LOGGER
from yolov6.dataloaders.datasets import YOLODataset
from yolov6.utils.loss import ComputeLoss
from yolov6.utils.torch_utils import BBoxUtility
from yolov6.utils.general import one_cycle
from sklearn.preprocessing import LabelEncoder
import wandb

# Initialize a new run
wandb.init(project="paddy-disease-classification")

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths
train_path = Path('paddy-disease-classification/train_images')
test_path = Path('paddy-disease-classification/test_images')

# Load train labels and perform label encoding
train_df = pd.read_csv('paddy-disease-classification/train.csv')
label_encoder = LabelEncoder()
train_df['label_encoded'] = label_encoder.fit_transform(train_df['label'])

# Create a YOLODataset instance
train_dataset = YOLODataset(
    img_path=train_path,
    imgsz=(640, 640),
    batch_size=16,
    augment=True,
    hyp=None,
    rect=False,
    cache=False,
    single_cls=False,
    stride=32,
    pad=0.5,
    prefix=''
)

# Create a DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

# Load the YOLOv6 model
model = BuildYOLOX('yolov6s.pt', task='seg')
model = model.to(device)

# Define the loss function
compute_loss = ComputeLoss(model)

# Define the BBox utility
bbox_utility = BBoxUtility(model.model_type)

# Training function
def train(model, train_loader, device, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        imgs, targets, paths, _ = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward
        preds = model(imgs)

        # Loss
        loss, loss_items = compute_loss(preds, targets)

        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

# Validation function
def validate(model, valid_loader, device):
    model.eval()
    running_loss = 0.0
    for i, data in enumerate(valid_loader):
        imgs, targets, paths, _ = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward
        preds = model(imgs)

        # Loss
        loss, loss_items = compute_loss(preds, targets)

        running_loss += loss.item()

    epoch_loss = running_loss / len(valid_loader)
    return epoch_loss

# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937, nesterov=True)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    # Training
    train_loss = train(model, train_loader, device, epoch)

    # Validation
    valid_loss = validate(model, valid_loader, device)

    # Log metrics to wandb
    wandb.log({
        "Epoch": epoch + 1,
        "Train Loss": train_loss,
        "Valid Loss": valid_loss
    })

    # Update learning rate
    lr = one_cycle(epoch, n_epochs, 0.01, 0.001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

# Evaluation
class_names = label_encoder.classes_
# ... (Implement evaluation logic using YOLOv6 utilities)

# Save the model
model.save_model()

wandb.finish()