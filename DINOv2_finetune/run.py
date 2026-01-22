import os, tarfile, requests
from os.path import isdir
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from model import PatchCoreDINOv2

# -----------------------------
# Dataset helper
# -----------------------------
class MVTecTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        for defect_type in os.listdir(root):
            defect_dir = os.path.join(root, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            for img_name in os.listdir(defect_dir):
                if img_name.endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(defect_dir, img_name), defect_type))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label

# -----------------------------
# Fine-tuning loop (projection head)
# -----------------------------
def train_projection_head(model, dataloader, epochs=5, lr=1e-4, device="cuda"):
    optimizer = torch.optim.Adam(model.proj_head.parameters(), lr=lr)
    criterion = torch.nn.CosineEmbeddingLoss()
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (img, _) in dataloader:
            img = img.to(device)
            # create two augmented views
            img1 = aug(transforms.ToPILImage()(img[0].cpu())).unsqueeze(0).to(device)
            img2 = aug(transforms.ToPILImage()(img[0].cpu())).unsqueeze(0).to(device)

            z1 = model._extract_global(img1)
            z2 = model._extract_global(img2)

            target = torch.ones(z1.size(0)).to(device)
            loss = criterion(z1, z2, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Fine-tune] Epoch {epoch+1}/{epochs}, Loss {total_loss/len(dataloader):.4f}")

    model.eval()

# -----------------------------
# Main loop
# -----------------------------
objects = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
           "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
           "transistor", "wood", "zipper"]

output_dir = "results_dinov2"
os.makedirs(output_dir, exist_ok=True)

directory = "/content/mvtec_anomaly_detection"
device = "cuda" if torch.cuda.is_available() else "cpu"

for obj in objects:
    print(f"\n=== Running {obj} ===")
    start_time = datetime.now()
    dataset_path = os.path.join(directory, obj)

    if not isdir(dataset_path):
        print(f"Dataset {obj} not found.")
        continue

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=transform)
    test_dataset = MVTecTestDataset(root=os.path.join(dataset_path, "test"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Model
    model = PatchCoreDINOv2(device=device, backbone="dinov2_vits14", proj_dim=128)

    # Fine-tune projection head (self-supervised, only "good" images)
    train_projection_head(model, train_loader, epochs=3, lr=1e-4, device=device)

    # Build memory bank
    model.fit(train_loader, f_coreset=0.25)

    # Evaluate
    scores = model.predict(test_loader)
    labels = [1 if lbl != "good" else 0 for _, lbl in test_dataset.samples]
    auc = roc_auc_score(labels, scores)
    print(f"AUC: {auc:.4f}")
    print("Execution time:", datetime.now() - start_time)

    # Save a few heatmaps
    save_dir = os.path.join(output_dir, obj)
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(min(10, len(test_dataset))):
        img, lbl = test_dataset[idx]
        heatmap = model.anomaly_map(img)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        heatmap_np = heatmap.numpy()

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title(f"Original ({lbl})")

        plt.subplot(1, 2, 2)
        plt.imshow(img_np)
        plt.imshow(heatmap_np, cmap="jet", alpha=0.5)
        plt.title("Anomaly Heatmap")
        plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"))
        plt.close()

    print(f"Saved heatmaps to {save_dir}")
