from pathlib import Path


path = Path(r"c:\Users\l\OneDrive\Documents\PROJECTS\dl project\verification\siamese_train.py")
text = path.read_text(encoding='utf-8')
marker = "# -------------------------------\n# Training\n# -------------------------------\n"
if marker not in text:
    raise RuntimeError(f"Marker not found in {path}")

head, _sep, tail = text.partition(marker)

# Build the new training block
new_block = marker + """if __name__ == \"__main__\":
    import argparse

    parser = argparse.ArgumentParser(description=\"Train a Siamese network on signature pairs.\")
    parser.add_argument(\"--dataset\", default=os.path.join(os.path.dirname(__file__), \"..\", \"dataset\"),
                        help=\"Path to the dataset folder that contains 'real' and 'forged' subfolders.\")
    parser.add_argument(\"--epochs\", type=int, default=5, help=\"Number of training epochs.\")
    parser.add_argument(\"--batch-size\", type=int, default=16, help=\"Training batch size.\")
    parser.add_argument(\"--lr\", type=float, default=1e-4, help=\"Learning rate.\")
    parser.add_argument(\"--pretrained\", action=\"store_true\", help=\"Use pretrained ResNet18 weights.\")
    parser.add_argument(\"--device\", default=\"cuda\" if torch.cuda.is_available() else \"cpu\",\n                        help=\"Device to train on (e.g., cuda or cpu).\")
    parser.add_argument(\"--save-path\", default=\"siamese_model.pth\",\n                        help=\"Path to save the trained model state dict.\")
    args = parser.parse_args()

    print(f\"Using device: {args.device}\")
    print(f\"Dataset path: {args.dataset}\")

    device = torch.device(args.device)

    transform = transforms.Compose([\n        transforms.Resize((224, 224)),\n        transforms.ToTensor()\n    ])

    dataset = SignaturePairDataset(args.dataset, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = SiameseNetwork(pretrained=args.pretrained).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_batches = len(loader)
    print(f\"Dataset size: {len(dataset)} pairs, {num_batches} batches per epoch.\")

    for epoch in range(args.epochs):
        total_loss = 0.0

        for batch_idx, (img1, img2, label) in enumerate(loader, start=1):
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
                print(f\"Epoch {epoch+1}/{args.epochs}  batch {batch_idx}/{num_batches}  loss {loss.item():.4f}\")

        avg_loss = total_loss / num_batches
        print(f\"Epoch {epoch+1}/{args.epochs} finished. avg loss: {avg_loss:.4f}\")

    torch.save(model.state_dict(), args.save_path)
    print(f\"Siamese model saved to: {args.save_path}\")\n"""

path.write_text(head + new_block, encoding='utf-8')
print('patch applied')
