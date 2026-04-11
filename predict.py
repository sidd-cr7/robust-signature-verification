import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("sealguard_resnet.pth", map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# CHANGE IMAGE NAME HERE
image_path = "adversarial_fake.jpg"

img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)

if predicted.item() == 0:
    print("Prediction: FORGED")
else:
    print("Prediction: GENUINE")