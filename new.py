import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import ImageFile
from PIL import Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_loader(path):
    try:
        with Image.open(path) as img:
            return img.convert('RGB')
    except:
        return None

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if sample is None:
            # Skip or handle unreadable images
            return None, None
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def safe_collate_fn(batch):
    # Exclude unreadable samples
    batch = [(x, y) for (x, y) in batch if x is not None and y is not None]
    if len(batch) == 0:
        return None, None
    return default_collate(batch)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "data"
train_data = SafeImageFolder(
    root=data_dir + '/train',
    transform=transform,
    loader=safe_loader
)
validation_data = SafeImageFolder(
    root=data_dir + '/valid',
    transform=transform,
    loader=safe_loader
)

trainloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=safe_collate_fn)
valloader = DataLoader(validation_data, batch_size=32, shuffle=False, collate_fn=safe_collate_fn)

# Define model with 2 outputs and no sigmoid
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes
model = model.to(device)

# Use CrossEntropyLoss for two-class output
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in trainloader:
        # Labels: shape [batch_size], values 0 or 1
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)     # shape [batch_size, 2]
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(dim=1)        # get predicted class
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_data)
    epoch_acc = correct / total if total > 0 else 0
    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Validation loop
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            
            _, preds = outputs.max(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(validation_data)
    val_acc = val_correct / val_total if val_total > 0 else 0
    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        
print("Training complete.")
