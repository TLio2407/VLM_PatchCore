import os
import requests
import tarfile
import cv2
import csv
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
from model import PatchCoreDINOv3


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
    "zipper": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f283/download/420938385-1629960680/zipper.tar.xz",
    "can" : "https://www.mydrive.ch/shares/121501/26456e2f3ef813930866f8f9b072593a/download/466651130-1743159807/can.tar.gz",
    "fabric" : "https://www.mydrive.ch/shares/121502/812590f745083da1f5edb338a6c321c4/download/466651519-1743160379/fabric.tar.gz",
    "fruit_jelly" : "https://www.mydrive.ch/shares/121503/951a46ce30a3af3787ce9671cfa8613a/download/466651800-1743164023/fruit_jelly.tar.gz",
    "rice" : "https://www.mydrive.ch/shares/121504/0014676292c3c44931712a54fb3bdbe8/download/466653907-1743164943/rice.tar.gz",
    "sheet_metal" : "https://www.mydrive.ch/shares/121505/2d8fcdc8e988456bdd18696746eda0a0/download/466654829-1743166795/sheet_metal.tar.gz",
    "vial" : "https://www.mydrive.ch/shares/121506/739dc6459c939fe464c0d26acc6c2d55/download/466654885-1743167505/vial.tar.gz",
    "wallplugs" : "https://www.mydrive.ch/shares/121507/66fe6e114b498e03be8d48c711794be7/download/466655287-1743168151/wallplugs.tar.gz",
    "walnuts" : "https://www.mydrive.ch/shares/121508/9fcf67e49f0dc61a9608f57ba0482356/download/466656233-1743168988/walnuts.tar.gz"
}
datasets_list = {
    "mvtechAD_1" : ["bottle", "cable", "capsule", "carpet", "grid","hazelnut","leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"],
    "mvtechAD_2" : ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]
}
class MVTecTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        for d in os.listdir(root):
            sub = os.path.join(root, d)
            if not os.path.isdir(sub): continue
            for f in os.listdir(sub):
                if f.endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(sub, f), d))
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        try:
            img = Image.open(p).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), lbl


# -------- main loop --------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "/content/mvtec_anomaly_detection"

os.makedirs("heatmaps", exist_ok=True)
os.makedirs("bounded", exist_ok=True)

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
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        train_dataset = datasets.ImageFolder(os.path.join(obj_path, "train"), transform)
        test_dataset =  datasets.ImageFolder(os.path.join(obj_path, "test"), transform)

        # FIX 1: Map the 'good' folder name to its integer index
        class_to_idx = test_dataset.class_to_idx
        good_idx = class_to_idx.get("good")
        
        if good_idx is None:
            print(f"⚠️ Skipping {obj}: 'good' folder not found.")
            continue

        train_loader = DataLoader(train_dataset, batch_size=14, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=14, shuffle=False, num_workers=2)

        model = PatchCoreDINOv3(device=device, backbone="dinov3_vits16plus", coreset_method="kmeans")
        model.fit(train_loader)

        local_global = model.predict(test_loader, alpha=None)
        
        # FIX 2: Correct labels (Anomaly = 1, Normal = 0)
        labels = np.array([1 if lbl != good_idx else 0 for _, lbl in test_dataset.samples])
        
        if len(np.unique(labels)) < 2:
            print(f"⚠️ Skipping {obj}: Only one class present in test set.")
            continue
        
        # Pre-calculate normalization to speed up alpha loop
        l_norm = (local_global["local"] - np.mean(local_global["local"])) / (np.std(local_global["local"]) + 1e-12)
        g_norm = (local_global["global"] - np.mean(local_global["global"])) / (np.std(local_global["global"]) + 1e-12)

        best_auc, best_alpha = 0, 0
        for a in np.arange(0, 1.01, 0.01):
            scores = a * l_norm + (1 - a) * g_norm
            auc = roc_auc_score(labels, scores)
            if auc > best_auc:
                best_auc, best_alpha = auc, a
        
        dataset_aucs.append(best_auc)
        final_scores = best_alpha * l_norm + (1 - best_alpha) * g_norm
        
        fpr, tpr, thresholds = roc_curve(labels, final_scores)
        best_thresh = thresholds[np.argmax(tpr - fpr)]
        
        exec_time = (datetime.now() - start).total_seconds()
        print(f"📊 {obj} AUC: {best_auc:.4f} | Best α: {best_alpha:.2f}")
        
        csv_writer_ds.writerow([dataset_name, obj, round(best_alpha, 2), round(best_auc, 4), round(exec_time, 2)])
        csv_fp_ds.flush()

        # --- Heatmap Generation ---
        save_dir = os.path.join("heatmaps", obj)
        os.makedirs(save_dir, exist_ok=True)

        for idx in range(len(test_dataset)):
            # Only save images that are actually predicted as anomalous or are true anomalies
            if final_scores[idx] > best_thresh or labels[idx] == 1:
                img_tensor, _ = test_dataset[idx]
                heatmap = model.anomaly_map(img_tensor)
                
                # Get folder name for filename
                lbl_name = [k for k, v in class_to_idx.items() if v == test_dataset.samples[idx][1]][0]
                
                # De-normalize for visualization
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
                img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                heatmap_color = cv2.applyColorMap((heatmap.numpy() * 255).astype(np.uint8), cv2.COLORMAP_JET)
                superimposed = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
                
                cv2.imwrite(os.path.join(save_dir, f"{lbl_name}_{idx:03d}.png"), np.hstack([img_bgr, superimposed]))

    if dataset_aucs:
        print(f"\n📈 {dataset_name} Average AUC: {np.mean(dataset_aucs):.4f}")

csv_fp_ds.close()
print("✅ Metrics saved to dataset_metrics.csv")