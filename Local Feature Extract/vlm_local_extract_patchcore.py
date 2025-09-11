import torch
import numpy as np
import torch.nn as nn
import open_clip
from sklearn.random_projection import SparseRandomProjection


class PatchCoreCLIP(nn.Module):
    """
    PatchCore-style anomaly detector using CLIP ViT as backbone.

    Upgrades vs original:
      • Extracts LOCALLY-AWARE features (patch tokens) from CLIP ViT instead of only global pooled embedding.
      • Builds a large patch-level memory bank from train "good" images.
      • Coreset subsampling via k-center greedy on a low-dim random projection, like PatchCore.
      • Automatic safety: if memory bank would exceed `max_memory`, we auto-subsample even if f_coreset=1.0.

    API is drop-in compatible with your existing vlm_run.py:
      - fit(dataloader, f_coreset=1.0)
      - predict(dataloader) -> list of per-image anomaly scores
      - score(image) -> single-image score
    """

    def __init__(self, device: str = "cuda", max_memory: int = 100_000, proj_dim: int = 128, seed: int = 0):
        super().__init__()
        self.device = device
        self.max_memory = int(max_memory)
        self.proj_dim = int(proj_dim)
        self.rng = np.random.default_rng(seed)

        # Load CLIP backbone (ViT-B/32 pretrained on LAION2B)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.to(self.device).eval()

        # Hook storage for patch tokens
        self._cached_tokens = None

        def save_patch_tokens(module, input, output):
            # output: [B, 1+P, D] (CLS + patch tokens)
            self._cached_tokens = output.detach()

        # Register hook on last transformer block
        self.model.visual.transformer.resblocks[-1].register_forward_hook(save_patch_tokens)

        self.memory_bank = None

    # -----------------------------
    # Feature extraction helpers
    # -----------------------------
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized patch embeddings for a batch (no CLS token).
        Output shape: [B, P, D].
        """
        with torch.no_grad():
            _ = self.model.encode_image(x)  # forward pass triggers hook
            if self._cached_tokens is None:
                raise RuntimeError("Hook did not capture patch tokens. Check model internals.")
            patches = self._cached_tokens[:, 1:, :]  # drop CLS token
            patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-12)
        return patches

    def _batch_collect_patches(self, dataloader) -> torch.Tensor:
        all_feats = []
        self.model.eval()
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device, non_blocking=True)
                patches = self._extract_patch_tokens(x)  # [B, P, D]
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
        feats = self._batch_collect_patches(dataloader)
        N = feats.size(0)
        if N == 0:
            raise RuntimeError("No features extracted. Check your training data and transforms.")

        if f_coreset <= 0.0 or f_coreset > 1.0:
            raise ValueError("f_coreset must be in (0, 1].")

        if f_coreset < 1.0:
            target_n = max(1, int(N * f_coreset))
        else:
            target_n = N

        if target_n > self.max_memory:
            target_n = self.max_memory

        if target_n < N:
            mb = self._coreset_select(feats, target_n)
            print(f"✅ Local memory bank reduced to {len(mb)} patches (from {N}) via coreset.")
        else:
            mb = feats
            print(f"✅ Local memory bank built with {len(mb)} patches.")

        self.memory_bank = mb.contiguous()

    def predict(self, dataloader):
        if self.memory_bank is None or len(self.memory_bank) == 0:
            raise RuntimeError("Memory bank is empty. Call fit() first.")

        scores = []
        MB = self.memory_bank
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device, non_blocking=True)
                patches = self._extract_patch_tokens(x)
                B, P, D = patches.shape
                patches = patches.cpu()
                for i in range(B):
                    pi = patches[i]
                    dists = torch.cdist(pi, MB)
                    min_patch_dists, _ = torch.min(dists, dim=1)
                    score = torch.max(min_patch_dists).item()
                    scores.append(score)
        return scores

    def score(self, image: torch.Tensor) -> float:
        self.model.eval()
        if self.memory_bank is None or len(self.memory_bank) == 0:
            raise RuntimeError("Memory bank is empty. Call fit() first.")
        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)
            patches = self._extract_patch_tokens(x)[0].cpu()
            dists = torch.cdist(patches, self.memory_bank)
            score = torch.max(torch.min(dists, dim=1)[0]).item()
        return score

    def anomaly_map(self, image: torch.Tensor, upsample_size=(224, 224)):
        """
        Generate anomaly heatmap for a single image.
        Args:
            image: torch.Tensor (3xHxW), already preprocessed.
            upsample_size: tuple, size to upsample heatmap to.
        Returns:
            heatmap: torch.Tensor (HxW) anomaly intensity.
        """
        if self.memory_bank is None or len(self.memory_bank) == 0:
            raise RuntimeError("Memory bank is empty. Call fit() first.")

        self.model.eval()
        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)
            patches = self._extract_patch_tokens(x)[0].cpu()  # [P, D]

            dists = torch.cdist(patches, self.memory_bank)  # [P, M]
            min_patch_dists, _ = torch.min(dists, dim=1)    # [P]

            # Reshape into grid: ViT-B/32 → 7x7 patches (224/32 = 7)
            side = int((min_patch_dists.shape[0]) ** 0.5)
            heatmap = min_patch_dists.view(side, side)

            # Normalize to [0,1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)

            # Upsample to match image size
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=upsample_size,
                mode="bilinear",
                align_corners=False
            )[0, 0]

        return heatmap
