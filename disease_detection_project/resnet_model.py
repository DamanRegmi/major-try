# disease_detection/resnet_model.py

import torch
import torchvision.models as models
from torchvision import transforms

def load_resnet18():
    model = models.resnet18(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add a batch dimension
    return input_batch

def predict_image(image, model):
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class

def get_disease_name(prediction):
    # Map prediction index to disease name
    # Implement your logic here
    return "Tomato Leaf Blight"  # Placeholder
