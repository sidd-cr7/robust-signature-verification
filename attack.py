import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import foolbox as fb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained ResNet
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("sealguard_resnet.pth", map_location=device))
model = model.to(device)
model.eval()

fmodel = fb.PyTorchModel(model, bounds=(0, 1))

# Transform (no normalization for attack input)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Normalization for prediction
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

source_folder = "adv_dataset/normal"
save_folder = "adv_dataset/adversarial"

os.makedirs(save_folder, exist_ok=True)

attack = fb.attacks.FGSM()

count = 0

for filename in os.listdir(source_folder):
    img_path = os.path.join(source_folder, filename)

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Get correct label
    with torch.no_grad():
        output = model(normalize(img_tensor.squeeze()).unsqueeze(0))
        _, label = torch.max(output, 1)

    adversarial, _, success = attack(fmodel, img_tensor, label, epsilons=0.02)

    adv_img = adversarial.squeeze().cpu().numpy().transpose(1,2,0)
    adv_img = adv_img.clip(0,1)

    save_path = os.path.join(save_folder, filename)
    Image.fromarray((adv_img*255).astype("uint8")).save(save_path)

    count += 1

print(f"{count} adversarial images generated.")