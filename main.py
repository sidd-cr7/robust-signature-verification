import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Adversarial Detector --------
adv_model = models.resnet18(weights=None)
adv_model.fc = nn.Linear(adv_model.fc.in_features, 2)
adv_model.load_state_dict(torch.load("adv_detector.pth", map_location=device))
adv_model = adv_model.to(device)
adv_model.eval()

# -------- Load Forgery Detector --------
forg_model = models.resnet18(weights=None)
forg_model.fc = nn.Linear(forg_model.fc.in_features, 2)
forg_model.load_state_dict(torch.load("sealguard_resnet.pth", map_location=device))
forg_model = forg_model.to(device)
forg_model.eval()

# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# CHANGE THIS IMAGE NAME TO TEST
image_path = "adversarial_real.jpg"

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# -------- Step 1: Check Adversarial --------
with torch.no_grad():
    adv_output = adv_model(img_tensor)
    _, adv_pred = torch.max(adv_output, 1)

# 0 = adversarial, 1 = normal
if adv_pred.item() == 0:
    print("⚠ Adversarial Manipulation Detected!")
else:
    print("✓ No Adversarial Attack Detected")

    # -------- Step 2: Check Forgery --------
    with torch.no_grad():
        forg_output = forg_model(img_tensor)
        _, forg_pred = torch.max(forg_output, 1)

    if forg_pred.item() == 0:
        print("❌ Forged Signature")
    else:
        print("✔ Genuine Signature")