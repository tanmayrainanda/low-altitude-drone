import torch
from torchvision import models, transforms
from PIL import Image
from torch import nn

class_dict = {
    'bacterial_leaf_blight': 0,
    'bacterial_leaf_streak': 1,
    'bacterial_panicle_blight': 2,
    'blast': 3,
    'brown_spot': 4,
    'dead_heart': 5,
    'downy_mildew': 6,
    'hispa': 7,
    'normal': 8,
    'tungro': 9
}
# Define paths
model_path = 'Trained Models/mobilenet_model.pth'

# Define transformations
transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
def load_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Load the model
def load_model(model_path):
    model = models.mobilenet_v2(weights=None)
    num_classes = 10  # replace with your actual number of classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Classify the image
def classify(image_path, model_path):
    image = load_image(image_path)
    model = load_model(model_path)
    output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_index = predicted.item()
    reverse_class_dict = {v: k for k, v in class_dict.items()}
    return reverse_class_dict[predicted_index]

# Use the function
image_path = 'paddy-disease-classification/train_images/tungro/101759.jpg'
print(classify(image_path, model_path))