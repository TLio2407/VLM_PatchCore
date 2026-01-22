import os
import requests
import tarfile
import cv2
import csv
import torch
import numpy as np
import tifffile  # Ensure this is installed: pip install tifffile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
from datetime import datetime
from model import PatchCoreDINOv3

# --- Configuration for Evaluation ---
ANOMALY_MAPS_ROOT = "anomaly_maps_eval"  # Base directory for the evaluation script
os.makedirs(ANOMALY_MAPS_ROOT, exist_ok=True)

_DATASET_URL = {
    "bottle": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937413-1629958794/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937484-1629959013/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938129-1629960351/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938130-1629960389/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938166-1629960554/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938383-1629960649/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938385-1629960680/zipper.tar.xz"
}

datasets_list = {
    "mvtechAD_1": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
}

# -------- Main Execution --------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "mvtec_anomaly_detection"

os.makedirs("heatmaps", exist_ok=True)

dataset_metrics_file = Path("dataset_metrics.csv")
write_header_ds = not dataset_metrics_file.exists()
csv_fp_ds = open(dataset_metrics_file, "a", newline="")
csv_writer_ds = csv.writer(csv_fp_ds)

if write_header_ds:
    csv_writer_ds.writerow(["dataset", "object", "best_alpha", "AUC", "exec_time"])

for dataset_name, ds in datasets_list.items():
    dataset_aucs = []

    for obj in ds:
        # --- Download Logic ---
        obj_path = os.path.join(data_dir, obj)
        if not os.path.isdir(obj_path):
            print(f"Downloading {obj}...")
            url = _DATASET_URL[obj]
            tarname = Path(url).name
            with open(tarname, "wb") as f, requests.get(url, stream=True) as r:
                for chunk in r.iter_content(8192): f.write(chunk)
            with tarfile.open(tarname, "r:*") as tar:
                tar.extractall(data_dir)
            os.remove(tarname)

        print(f"\nProcessing object: {obj}")
        start = datetime.now()
        
        # MVTec AD standard resolution is 224x224 for many backbones
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = datasets.ImageFolder(os.path.join(obj_path, "train"), transform)
        test_dataset = datasets.ImageFolder(os.path.join(obj_path, "test"), transform)
        class_to_idx = test_dataset.class_to_idx
        good_idx = class_to_idx.get("good")
        
        if good_idx is None:
            continue

        train_loader = DataLoader(train_dataset, batch_size=14, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=14, shuffle=False)

        model = PatchCoreDINOv3(device=device, backbone="dinov3_vits16plus", coreset_method="kmeans")
        model.fit(train_loader)

        # Create evaluation structure: <anomaly_maps_dir>/<object_name>/test/
        eval_obj_test_dir = os.path.join(ANOMALY_MAPS_ROOT, obj, "test")
        os.makedirs(eval_obj_test_dir, exist_ok=True)

        print(f"Generating .tiff anomaly maps for evaluation...")
        for idx in range(len(test_dataset)):
            img_tensor, _ = test_dataset[idx]
            img_path, label_idx = test_dataset.samples[idx]
            
            # Determine directory and filename for evaluation
            defect_name = [name for name, i in class_to_idx.items() if i == label_idx][0]
            image_id = Path(img_path).stem
            
            # Create specific defect folder
            defect_dir = os.path.join(eval_obj_test_dir, defect_name)
            os.makedirs(defect_dir, exist_ok=True)

            # Generate and save anomaly map
            with torch.no_grad():
                # anomaly_map returns a tensor which we convert to float32
                heatmap = model.anomaly_map(img_tensor).cpu().numpy().astype(np.float32)

            # Resize heaptmap to match original image size
            original_image = cv2.imread(img_path)
            if original_image is not None:
                h, w = original_image.shape[:2]
                heatmap = cv2.resize(heatmap, (w, h))

            # Save as .tiff
            tif_save_path = os.path.join(defect_dir, f"{image_id}.tiff")
            tifffile.imwrite(tif_save_path, heatmap)

        # (Optional) Log standard image-level metrics
        local_global = model.predict(test_loader, alpha=None)
        labels = np.array([1 if lbl != good_idx else 0 for _, lbl in test_dataset.samples])
        l_norm = (local_global["local"] - np.mean(local_global["local"])) / (np.std(local_global["local"]) + 1e-12)
        g_norm = (local_global["global"] - np.mean(local_global["global"])) / (np.std(local_global["global"]) + 1e-12)

        best_auc, best_alpha = 0, 0
        for a in np.arange(0, 1.01, 0.01):
            scores = a * l_norm + (1 - a) * g_norm
            auc = roc_auc_score(labels, scores)
            if auc > best_auc:
                best_auc, best_alpha = auc, a
        
        exec_time = (datetime.now() - start).total_seconds()
        csv_writer_ds.writerow([dataset_name, obj, round(best_alpha, 2), round(best_auc, 4), round(exec_time, 2)])
        csv_fp_ds.flush()

csv_fp_ds.close()
print(f"✅ Anomaly maps exported to {ANOMALY_MAPS_ROOT}")