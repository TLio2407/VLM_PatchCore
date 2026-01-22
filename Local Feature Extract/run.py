import os
import requests
import tarfile
from os.path import isdir
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from vlm_local_extract_patchcore import PatchCoreCLIP
from datetime import datetime

# -----------------------------
# Dataset URLs (MVTec AD benchmark)
# -----------------------------
_DATASET_URL = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz"
}
# -----------------------------
# Custom dataset for /test
# -----------------------------
class MVTecTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []

        for defect_type in os.listdir(root):
            defect_dir = os.path.join(root, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            for img_name in os.listdir(defect_dir):
                if img_name.endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(defect_dir, img_name), defect_type))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# Main run loop
# -----------------------------
objects = [
    "bottle","cable","capsule","carpet","grid","hazelnut",
    "leather","metal_nut","pill","screw","tile","toothbrush",
    "transistor","wood","zipper"
]

directory = r"/content/mvtec_anomaly_detection"

device = "cuda" if torch.cuda.is_available() else "cpu"

for obj in objects:
    start_time = datetime.now()
    dataset_path = os.path.join(directory, obj)

    # -----------------------------
    # Download if missing
    # -----------------------------
    if not isdir(dataset_path):
        print(f"Dataset {obj} not found, downloading...")
        url = _DATASET_URL[obj]
        tar_filename = obj + ".tar.xz"
        response = requests.get(url, stream=True)
        with open(tar_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        with tarfile.open(tar_filename, 'r:xz') as tar:
            print(f"Extracting {tar_filename} ...")
            tar.extractall(directory)
        os.remove(tar_filename)
        print(f"Download complete: {obj}")

    # -----------------------------
    # Transforms
    # -----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    # Train (good only) + Test (all defects)
    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=transform
    )
    test_dataset = MVTecTestDataset(
        root=os.path.join(dataset_path, "test"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # -----------------------------
    # Run PatchCoreCLIP
    # -----------------------------
    model = PatchCoreCLIP(device=device, max_memory=100_000, proj_dim=128)

    # Full memory (auto-subsampled if > max_memory)
    model.fit(train_loader, f_coreset=1.0)
    scores = model.predict(test_loader)
    labels = [1 if lbl != "good" else 0 for _, lbl in test_dataset.samples]
    auc = roc_auc_score(labels, scores)

    # Reduced memory 10%
    model.fit(train_loader, f_coreset=0.1)
    scores_01 = model.predict(test_loader)
    auc_01 = roc_auc_score(labels, scores_01)

    # Reduced memory 25%
    model.fit(train_loader, f_coreset=0.25)
    scores_025 = model.predict(test_loader)
    auc_025 = roc_auc_score(labels, scores_025)

    # -----------------------------
    # Report results
    # -----------------------------
    print(f"Metrics for {obj}:")
    print(f"  AUC full:     {auc}")
    print(f"  AUC coreset 0.1: {auc_01}")
    print(f"  AUC coreset 0.25: {auc_025}")
    print(f"  Execution time:", datetime.now() - start_time)

    # pick one test image
    sample_img, _ = test_dataset[0]  # tensor
    heatmap = model.anomaly_map(sample_img)

    # Convert to numpy for visualization
    import numpy as np
    img_np = sample_img.permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize
    heatmap_np = heatmap.numpy()

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img_np)
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(img_np)
    plt.imshow(heatmap_np, cmap="jet", alpha=0.5)  # overlay
    plt.title("Anomaly Heatmap")
    plt.show()
    # -----------------------------
    # Optional: Plot ROC (per object)
    # -----------------------------
    # fpr, tpr, _ = roc_curve(labels, scores)
    # fpr01, tpr01, _ = roc_curve(labels, scores_01)
    # fpr025, tpr025, _ = roc_curve(labels, scores_025)
    # plt.figure()
    # plt.plot(fpr, tpr, label=f\"Coreset=1.0 (AUC={auc:.3f})\")
    # plt.plot(fpr01, tpr01, label=f\"Coreset=0.1 (AUC={auc_01:.3f})\")
    # plt.plot(fpr025, tpr025, label=f\"Coreset=0.25 (AUC={auc_025:.3f})\")
    # plt.plot([0,1],[0,1],'k--')
    # plt.xlabel(\"False Positive Rate\")
    # plt.ylabel(\"True Positive Rate\")
    # plt.title(f\"ROC Curves - {obj}\")
    # plt.legend()
    # plt.show()
