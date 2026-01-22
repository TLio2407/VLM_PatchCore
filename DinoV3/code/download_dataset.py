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
    "mvtechAD_1" : ["bottle", "cable", "capsule", "carpet", "grid","hazelnut","leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]
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
data_dir = "D:/Cuong/Project/VLM PatchCore/DinoV3/datasets"
# data_dir = "/content/datasets"
# ===== Metrics CSV setup =====
dataset_metrics_file = Path("dataset_metrics.csv")
write_header_ds = not dataset_metrics_file.exists()

csv_fp_ds = open(dataset_metrics_file, "a", newline="")
csv_writer_ds = csv.writer(csv_fp_ds)

if write_header_ds:
    csv_writer_ds.writerow(["dataset", "avg_AUC", "num_objects"])

for dataset_name, ds in datasets_list.items():
    dataset_aucs = []

    print("Running on datasets:", dataset_name)
    for obj in ds:
        obj_path = os.path.join(data_dir, obj)
        if not os.path.isdir(obj_path):
            print(f"Downloading {obj}...")
            url = _DATASET_URL[obj]
            tarname = Path(url).name
            with open(tarname, "wb") as f, requests.get(url, stream=True) as r:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            
            with tarfile.open(tarname, "r:*") as tar:
                tar.extractall(data_dir)
            os.remove(tarname)
            print(f"Downloaded and extracted {obj}.")