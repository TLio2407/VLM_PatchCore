import torch
import numpy as np
import torch.nn as nn
import open_clip
from sklearn.random_projection import SparseRandomProjection

class PatchCoreCLIP(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # Load CLIP backbone
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.to(self.device)
        self.model.eval()

        self.memory_bank = None  # will hold training embeddings

    def fit(self, dataloader, f_coreset=1.0):
        """Extract embeddings from training set and store them in memory_bank"""
        features = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device)
                feats = self.model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                features.append(feats.cpu())
        features = torch.cat(features, dim=0)

        # ---------- Apply coreset subsampling ----------
        if f_coreset < 1.0:
            n_keep = int(len(features) * f_coreset)
            rand_idx = np.random.choice(len(features), n_keep, replace=False)
            self.memory_bank = features[rand_idx]
            print(f"✅ Memory bank reduced to {len(self.memory_bank)} embeddings (coreset {f_coreset})")
        else:
            self.memory_bank = features
            print(f"✅ Memory bank built with {len(self.memory_bank)} embeddings")
    def predict(self, dataloader):
        """Compute anomaly scores for test set"""
        scores = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device)

                image_features = self.model.encode_image(x)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Distance to nearest neighbor in memory bank
                dists = torch.cdist(image_features.cpu(), self.memory_bank)
                min_dists, _ = torch.min(dists, dim=1)
                scores.extend(min_dists.numpy())

        return scores

    def score(self, image):
        """Compute anomaly score for a single image tensor"""
        self.model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            dists = torch.cdist(image_features.cpu(), self.memory_bank)
            score, _ = torch.min(dists, dim=1)
        return score.item()
