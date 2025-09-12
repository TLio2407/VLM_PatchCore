import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.random_projection import SparseRandomProjection


class PatchCoreDINOv2(nn.Module):
    """
    PatchCore-style anomaly detector using DINOv2 ViT backbone.
    Extracts patch-level tokens (local) + CLS token (global).
    """

    def __init__(self, device: str = "cuda", backbone: str = "dinov2_vits14",
                 max_memory: int = 100_000, proj_dim: int = 128, seed: int = 0):
        super().__init__()
        self.device = device
        self.max_memory = int(max_memory)
        self.proj_dim = int(proj_dim)
        self.rng = np.random.default_rng(seed)

        # Load pretrained DINOv2 backbone
        if backbone.startswith("dinov2"):
            # Load via torch.hub
            self.model = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True)
        else:
            # fallback to torchvision or your other backbones
            self.model = models.get_model(backbone, pretrained=True)        
        self.model.to(device).eval()

        # Hook storage for patch tokens
        self._cached_tokens = None

        def save_patch_tokens(module, input, output):
            # output: [B, N, D] (CLS + patches)
            self._cached_tokens = output.detach()

        # register hook on last block
        self.model.blocks[-1].register_forward_hook(save_patch_tokens)

        # Memory banks
        self.memory_bank_local = None
        self.memory_bank_global = None

    # -----------------------------
    # Feature extraction
    # -----------------------------
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _ = self.model(x)
            if self._cached_tokens is None:
                raise RuntimeError("Hook did not capture patch tokens.")
            patches = self._cached_tokens[:, 1:, :]  # drop CLS
            patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-12)
        return patches

    def _extract_global(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _ = self.model(x)
            if self._cached_tokens is None:
                raise RuntimeError("Hook did not capture patch tokens.")
            cls_token = self._cached_tokens[:, 0, :]  # CLS embedding
            cls_token = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-12)
        return cls_token.cpu()

    def _batch_collect_patches(self, dataloader) -> torch.Tensor:
        all_feats = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device, non_blocking=True)
                patches = self._extract_patch_tokens(x)
                B, P, D = patches.shape
                all_feats.append(patches.reshape(B * P, D).cpu())
        return torch.cat(all_feats, dim=0) if all_feats else torch.empty(0)

    # -----------------------------
    # Coreset selection
    # -----------------------------
    def _kcenter_greedy(self, X: np.ndarray, n_select: int) -> np.ndarray:
        N = X.shape[0]
        if n_select >= N or N == 0:
            return np.arange(N)

        start = self.rng.integers(0, N)
        selected = [start]
        dists = np.sum((X - X[start]) ** 2, axis=1)

        for _ in range(1, n_select):
            idx = int(np.argmax(dists))
            selected.append(idx)
            new_d = np.sum((X - X[idx]) ** 2, axis=1)
            dists = np.minimum(dists, new_d)
        return np.array(selected, dtype=np.int64)

    def _coreset_select(self, feats: torch.Tensor, target_n: int) -> torch.Tensor:
        N = feats.size(0)
        if target_n >= N:
            return feats

        rp_dim = min(self.proj_dim, max(2, feats.size(1)))
        projector = SparseRandomProjection(n_components=rp_dim, random_state=0)
        X = projector.fit_transform(feats.numpy()).astype(np.float32, copy=False)

        idx = self._kcenter_greedy(X, target_n)
        return feats[idx]

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(self, dataloader, f_coreset: float = 1.0):
        # Local features
        feats = self._batch_collect_patches(dataloader)
        N = feats.size(0)
        if N == 0:
            raise RuntimeError("No local features extracted.")

        if f_coreset < 1.0:
            target_n = max(1, int(N * f_coreset))
        else:
            target_n = min(N, self.max_memory)

        if target_n < N:
            mb_local = self._coreset_select(feats, target_n)
            print(f"Local memory bank reduced to {len(mb_local)} patches (from {N}) via coreset.")
        else:
            mb_local = feats
            print(f"Local memory bank built with {len(mb_local)} patches.")

        self.memory_bank_local = mb_local.contiguous()

        # Global features
        global_feats = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device)
                g = self._extract_global(x)
                global_feats.append(g)
        self.memory_bank_global = torch.cat(global_feats, dim=0)
        print(f"Global memory bank built with {len(self.memory_bank_global)} embeddings.")

    def predict(self, dataloader, alpha: float = 0.7):
        if self.memory_bank_local is None or self.memory_bank_global is None:
            raise RuntimeError("Memory banks are empty. Call fit() first.")

        scores = []
        MB_local = self.memory_bank_local
        MB_global = self.memory_bank_global

        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device, non_blocking=True)

                patches = self._extract_patch_tokens(x)
                B, P, D = patches.shape
                patches = patches.cpu()
                globals_x = self._extract_global(x)

                for i in range(B):
                    # local anomaly
                    pi = patches[i]
                    dists_local = torch.cdist(pi, MB_local)
                    min_patch_dists, _ = torch.min(dists_local, dim=1)
                    local_score = torch.max(min_patch_dists).item()

                    # global anomaly
                    gi = globals_x[i:i+1]
                    gdist = torch.cdist(gi, MB_global).min().item()

                    # combine
                    score = alpha * local_score + (1 - alpha) * gdist
                    scores.append(score)
        return scores

    def score(self, image: torch.Tensor, alpha: float = 0.7) -> float:
        if self.memory_bank_local is None or self.memory_bank_global is None:
            raise RuntimeError("Memory banks are empty. Call fit() first.")

        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)

            patches = self._extract_patch_tokens(x)[0].cpu()
            dists_local = torch.cdist(patches, self.memory_bank_local)
            local_score = torch.max(torch.min(dists_local, dim=1)[0]).item()

            g = self._extract_global(x)
            gdist = torch.cdist(g, self.memory_bank_global).min().item()

            score = alpha * local_score + (1 - alpha) * gdist
        return score

    def anomaly_map(self, image: torch.Tensor, upsample_size=(224, 224)):
        if self.memory_bank_local is None:
            raise RuntimeError("Local memory bank is empty. Call fit() first.")

        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)
            patches = self._extract_patch_tokens(x)[0].cpu()
            dists = torch.cdist(patches, self.memory_bank_local)
            min_patch_dists, _ = torch.min(dists, dim=1)

            side = int((min_patch_dists.shape[0]) ** 0.5)
            heatmap = min_patch_dists.view(side, side)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)

            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=upsample_size,
                mode="bilinear",
                align_corners=False
            )[0, 0]

        return heatmap
