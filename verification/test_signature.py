import os
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

# Ensure the project root is on sys.path so this script can be run from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verification.siamese_train import SiameseNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((105,105)),
    transforms.ToTensor()
])

def resolve_path(path: str) -> str:
    """Resolve a path relative to the project root.

    This allows running the script from any current working directory.
    """

    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return str(p)


def load_image(path: str):
    filepath = resolve_path(path)
    if not Path(filepath).is_file():
        raise FileNotFoundError(
            f"Image file not found: {path}\n" \
            f"Resolved path: {filepath}\n" \
            "Please provide a valid image path or place test images under the project root."
        )

    # Model expects 3-channel RGB input (pretrained encoder uses 3-channel conv weights).
    img = Image.open(filepath).convert("RGB")
    img = transform(img)
    return img.unsqueeze(0).to(device)


img1 = load_image("verification/sign1.jpg")
img2 = load_image("verification/sign4.jpg")

with torch.no_grad():
    output1, output2 = model(img1, img2)
    distance = torch.nn.functional.pairwise_distance(output1, output2)

print("Distance:", distance.item())

dist = distance.item()

dist = distance.item()
print("Distance:", dist)

if dist < 1:
    print("Genuine Signature")
elif dist > 1.1:
    print("Forged Signature")
else:
    print("Uncertain (needs manual verification)")