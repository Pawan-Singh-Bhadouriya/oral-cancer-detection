import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same transformations as in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # Must match training
model.load_state_dict(torch.load("best_model2.pth", map_location=device))
model.to(device)
model.eval()

def predict(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # You can rename classes as needed
    classes = ["Cancer", "No Cancer"]
    return classes[predicted.item()]

# Example usage
if __name__ == "__main__":
    test_image = "/kaggle/input/dataset/OC Dataset kaggle new/test/0/471.jpeg"  # Replace with a real image path
    print("Prediction:", predict(test_image))


