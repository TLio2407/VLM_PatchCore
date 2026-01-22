import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection
from sklearn.cluster import MiniBatchKMeans
from torchvision import transforms
from PIL import Image
import cv2
import math

class PatchCoreDINOv3(nn.Module):
    def __init__(self,
                 device: str = "cuda",
                 backbone=None,
                 max_memory: int = 100_000,
                 proj_dim: int = 128,
                 seed: int = 0,
                 mv_layers=None,
                 fusion_weights=None,
                 use_pca: bool = True,
                 pca_dim: int = 256,
                 coreset_method: str = "kmeans"):
        super().__init__()
        self.device = device
        self.max_memory = int(max_memory)
        self.proj_dim = int(proj_dim)
        self.rng = np.random.default_rng(seed)
        self.use_pca = use_pca
        self.pca_dim = int(pca_dim)
        self.coreset_method = coreset_method
        assert coreset_method in ("kcenter", "kmeans", "minibatchkmeans")

        if mv_layers is None:
            mv_layers = [-1, -3, -5]
        self.mv_layers = mv_layers

        if fusion_weights is None:
            init_w = np.ones(len(mv_layers), dtype=np.float32) / len(mv_layers)
        else:
            init_w = np.array(fusion_weights, dtype=np.float32)
            init_w = init_w / (init_w.sum() + 1e-12)
        self.fusion_logits = nn.Parameter(torch.tensor(np.log(init_w + 1e-12), dtype=torch.float32))

        if backbone is None:
            raise ValueError("Please provide a backbone string (e.g., 'dinov3_vits16').")
        if backbone.startswith("dinov2"):
            self.model = torch.hub.load("facebookresearch/dinov2", backbone, pretrained=True)
        elif backbone.startswith("dinov3"):
            repo_dir = "/content/dinov3"
            self.model = torch.hub.load(
                repo_dir,
                'dinov3_vits16',
                source='local',
                weights="https://huggingface.co/Fanqi-Lin-IR/dinov3_vits16_pretrain/resolve/main/dinov3_vits16_pretrain.pth"
            )
        else:
            import torchvision.models as models
            self.model = models.get_model(backbone, pretrained=True)

        self.model.to(device).eval()

        self._cached_tokens = {}
        self._registered_hooks = []
        for layer_idx in self.mv_layers:
            def make_hook(lidx):
                def save_patch_tokens(module, input, output):
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    self._cached_tokens[lidx] = output.detach()
                return save_patch_tokens
            blk = self.model.blocks[layer_idx]
            hook = blk.register_forward_hook(make_hook(layer_idx))
            self._registered_hooks.append(hook)

        self.memory_bank_local = None
        self.memory_bank_global = None
        self._pca = None

    def _clear_cache(self):
        self._cached_tokens = {}

    def _get_fusion_weights(self):
        w = F.softmax(self.fusion_logits, dim=0)
        return w.cpu().numpy()

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self._clear_cache()
            if self.device.startswith("cuda"):
                with torch.amp.autocast('cuda'):
                    _ = self.model(x.to(self.device))
            else:
                _ = self.model(x.to(self.device))
            collected = []
            fusion_w = torch.tensor(self._get_fusion_weights(), dtype=torch.float32)
            for i, lidx in enumerate(self.mv_layers):
                if lidx not in self._cached_tokens:
                    raise RuntimeError(f"Hook for layer {lidx} didn't fire.")
                out = self._cached_tokens[lidx]
                patches = out[:, 5:, :]
                patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-12)
                patches = patches.cpu() * float(fusion_w[i])
                collected.append(patches)
            fused = torch.cat(collected, dim=-1)
            fused = fused / (fused.norm(dim=-1, keepdim=True) + 1e-12)
        return fused

    def _extract_global(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self._clear_cache()
            if self.device.startswith("cuda"):
                with torch.amp.autocast('cuda'):
                    _ = self.model(x.to(self.device))
            else:
                _ = self.model(x.to(self.device))
            lidx = self.mv_layers[0]
            cls_token = self._cached_tokens[lidx][:, 0, :]
            cls_token = cls_token / (cls_token.norm(dim=-1, keepdim=True) + 1e-12)
        return cls_token.cpu()

    def _batch_collect_patches(self, dataloader) -> torch.Tensor:
        all_feats = []
        total_patches = 0
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device, non_blocking=True)
                patches = self._extract_patch_tokens(x)
                B, P, D = patches.shape
                feats = patches.reshape(B * P, D).cpu()
                all_feats.append(feats)
                total_patches += feats.size(0)
                if total_patches >= self.max_memory:
                    break
        feats = torch.cat(all_feats, dim=0)
        return feats[:self.max_memory]

    def _kcenter_greedy(self, X: np.ndarray, n_select: int) -> np.ndarray:
        N = X.shape[0]
        if n_select >= N:
            return np.arange(N)
        start = int(self.rng.integers(0, N))
        selected = [start]
        dists = np.sum((X - X[start]) ** 2, axis=1)
        for _ in range(1, n_select):
            idx = int(np.argmax(dists))
            selected.append(idx)
            new_d = np.sum((X - X[idx]) ** 2, axis=1)
            dists = np.minimum(dists, new_d)
        return np.array(selected)

    def _coreset_select(self, feats: torch.Tensor, target_n: int) -> torch.Tensor:
        N, D = feats.shape
        if target_n >= N:
            return feats

        if self.use_pca:
            X = feats.numpy()
            if self._pca is None:
                pca = PCA(n_components=min(self.pca_dim, D), whiten=True, random_state=0)
                Xp = pca.fit_transform(X)
                self._pca = pca
            else:
                Xp = self._pca.transform(X)
            X_red = Xp
        else:
            rp = SparseRandomProjection(n_components=min(self.proj_dim, D))
            X_red = rp.fit_transform(feats.numpy())

        if self.coreset_method == "minibatchkmeans":
            mbk = MiniBatchKMeans(n_clusters=target_n, batch_size=2048, random_state=0)
            mbk.fit(X_red)
            centers = torch.tensor(mbk.cluster_centers_, dtype=feats.dtype)
            return centers
        elif self.coreset_method == "kmeans":
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=target_n, random_state=0)
            kmeans.fit(X_red)
            centers = torch.tensor(kmeans.cluster_centers_, dtype=feats.dtype)
            return centers
        else:
            idx = self._kcenter_greedy(X_red, target_n)
            return feats[idx]

    def fit(self, dataloader, f_coreset: float = 1.0):
        feats = self._batch_collect_patches(dataloader)
        target_n = int(min(len(feats), self.max_memory * f_coreset))
        self.memory_bank_local = F.normalize(self._coreset_select(feats, target_n), dim=-1).to(torch.float32)

        globals_ = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device)
                g = self._extract_global(x)
                globals_.append(g)
        self.memory_bank_global = F.normalize(torch.cat(globals_, dim=0), dim=-1).to(torch.float32)

        print(f"✅ Memory banks built: local={len(self.memory_bank_local)}, global={len(self.memory_bank_global)}")

    def predict(self, dataloader, alpha: float = None):
        MB_local = self.memory_bank_local.to(self.device)
        MB_global = self.memory_bank_global.to(self.device)

        local_scores = []
        global_scores = []
        with torch.no_grad():
            for (x, _) in dataloader:
                x = x.to(self.device)
                patches = self._extract_patch_tokens(x)
                patches = patches.to(self.device)
                B, P, D = patches.shape
                patches_reshaped = patches.reshape(B * P, D)
                dists = torch.cdist(patches_reshaped, MB_local)
                min_patch_dists = dists.min(dim=1)[0]
                min_patch_dists = min_patch_dists.reshape(B, P)
                per_image_local = min_patch_dists.max(dim=1)[0].cpu().numpy()
                local_scores.extend(per_image_local.tolist())

                globals_ = self._extract_global(x).to(self.device)
                gdists = torch.cdist(globals_, MB_global)
                per_image_global = gdists.min(dim=1)[0].cpu().numpy()
                global_scores.extend(per_image_global.tolist())

        local_scores = np.array(local_scores)
        global_scores = np.array(global_scores)

        if alpha is None:
            return {"local": local_scores, "global": global_scores}

        ln = (local_scores - local_scores.mean()) / (local_scores.std() + 1e-12)
        gn = (global_scores - global_scores.mean()) / (global_scores.std() + 1e-12)
        return alpha * ln + (1 - alpha) * gn
    
    def anomaly_map(self, image: torch.Tensor, upsample_size=(224, 224)):
        if self.memory_bank_local is None:
            raise RuntimeError("Local memory bank is empty. Call fit() first.")

        with torch.no_grad():
            x = image.unsqueeze(0).to(self.device)
            patches = self._extract_patch_tokens(x)[0].cpu()
            
            dists = torch.cdist(patches, self.memory_bank_local)
            min_patch_dists, _ = torch.min(dists, dim=1)

            num_patches = min_patch_dists.shape[0]
            side = int(num_patches ** 0.5)
            
            heatmap_np = min_patch_dists.view(side, side).numpy()

            heatmap_np = cv2.resize(heatmap_np, upsample_size, interpolation=cv2.INTER_CUBIC)

            heatmap_np = cv2.GaussianBlur(heatmap_np, (0, 0), sigmaX=4.0, sigmaY=4.0)

            min_val, max_val = heatmap_np.min(), heatmap_np.max()
            if max_val - min_val > 1e-12:
                heatmap_np = (heatmap_np - min_val) / (max_val - min_val)
            else:
                heatmap_np = np.zeros_like(heatmap_np)

            heatmap_np = heatmap_np ** 2

            return torch.tensor(heatmap_np)

    @staticmethod
    def generate_heatmap(img_tensor, patch_scores, save_path, alpha=0.5):
        import cv2
        import numpy as np
        import math

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = img_tensor.clone() * std + mean
        img = (img * 255).clamp(0, 255).byte().permute(1,2,0).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        side = int(math.sqrt(len(patch_scores)))
        heatmap = patch_scores.reshape(side, side)

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=4.0, sigmaY=4.0)

        min_val, max_val = heatmap.min(), heatmap.max()
        heatmap = (heatmap - min_val) / (max_val - min_val + 1e-10)

        heatmap = heatmap ** 2 

        heatmap_8bit = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 1.0, heatmap_color, alpha, 0)
        
        combined = np.concatenate([img, overlay], axis=1)
        cv2.imwrite(save_path, combined)
