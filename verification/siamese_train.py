import os
import random
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# -------------------------------
# Dataset that generates pairs
# -------------------------------

class SignaturePairDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        
        real_path = os.path.join(dataset_path, "real")
        forged_path = os.path.join(dataset_path, "forged")
        
        self.real_images = [os.path.join(real_path, img) for img in os.listdir(real_path) if img.endswith(('.jpg', '.png'))]
        self.forged_images = [os.path.join(forged_path, img) for img in os.listdir(forged_path) if img.endswith(('.jpg', '.png'))]
        self.all_images = self.real_images + self.forged_images

    def __len__(self):
        if len(self.all_images) == 0:
            raise RuntimeError(f"No images found in dataset path: {self.dataset_path}. "
                               "Make sure 'real' and 'forged' subfolders exist and contain .jpg/.png files.")
        return len(self.all_images)

    def __getitem__(self, idx):
        same_class = random.randint(0, 1)
        
        if same_class:
            class_images = random.choice([self.real_images, self.forged_images])
            img1_path = random.choice(class_images)
            img2_path = random.choice(class_images)
            label = 1
        else:
            img1_path = random.choice(self.real_images)
            img2_path = random.choice(self.forged_images)
            label = 0
        
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# -------------------------------
# Siamese Network
# -------------------------------

class SiameseNetwork(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

    def forward_once(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


# -------------------------------
# Contrastive Loss
# -------------------------------

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        distance = torch.nn.functional.pairwise_distance(out1, out2)

        loss = torch.mean(
            label * torch.pow(distance,2) +
            (1-label) * torch.pow(torch.clamp(self.margin-distance, min=0.0),2)
        )

        return loss


# -------------------------------
# Training
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Siamese network on signature pairs.")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "..", "dataset"),
                        help="Path to the dataset folder that contains 'real' and 'forged' subfolders.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained ResNet18 weights.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (e.g., cuda or cpu).")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after this many batches per epoch (for quick tests).")
    parser.add_argument("--save-path", default="siamese_model.pth",
                        help="Path to save the trained model state dict.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    print(f"Dataset path: {args.dataset}")

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = SignaturePairDataset(args.dataset, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SiameseNetwork(pretrained=args.pretrained).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_batches = len(loader)
    max_batches = args.max_batches or num_batches
    n_batches = min(num_batches, max_batches)
    print(f"Dataset size: {len(dataset)} pairs, {num_batches} batches per epoch (running up to {n_batches} per epoch).")

    for epoch in range(args.epochs):
        total_loss = 0.0

        for batch_idx, (img1, img2, label) in enumerate(loader, start=1):
            if batch_idx > max_batches:
                break

            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0 or batch_idx == num_batches:
                print(f"Epoch {epoch+1}/{args.epochs}  batch {batch_idx}/{num_batches}  loss {loss.item():.4f}")

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{args.epochs} finished. avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Siamese model saved to: {args.save_path}")
