from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.pointnet import PointNet
from src.sample import farthest_point_sampling


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape (N, 3), got {points.shape}")
    points = np.asarray(points, dtype=np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if radius <= 1e-12:
        raise ValueError("Point cloud radius is too small after centering.")
    return centered / radius


def downsample_point_cloud(points: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    if n_points <= 0:
        raise ValueError("n_points must be > 0.")
    if points.shape[0] == n_points:
        return points.astype(np.float32, copy=False)
    if points.shape[0] > n_points:
        idx = farthest_point_sampling(points, n_points, seed=seed)
        return points[idx].astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    extra = rng.choice(points.shape[0], size=n_points - points.shape[0], replace=True)
    return np.concatenate([points, points[extra]], axis=0).astype(np.float32, copy=False)


class PointCloudDataset(Dataset):
    def __init__(self, point_clouds: np.ndarray):
        if point_clouds.ndim != 3 or point_clouds.shape[-1] != 3:
            raise ValueError(f"point_clouds must have shape (B, N, 3), got {point_clouds.shape}")
        self.point_clouds = torch.from_numpy(point_clouds.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return int(self.point_clouds.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.point_clouds[index]


class PointCloudAutoencoder(nn.Module):
    def __init__(self, num_points: int, feature_dim: int):
        super().__init__()
        self.num_points = int(num_points)
        self.feature_dim = int(feature_dim)
        self.encoder = PointNet(point_feature_dim=3, pc_feature_dim=self.feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, self.num_points * 3),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        features, _ = self.encoder(points)
        return features

    def forward(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(points)
        recon = self.decoder(features).view(points.shape[0], self.num_points, 3)
        return features, recon


def chamfer_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(pred, target, p=2)
    distances_sq = distances * distances
    pred_to_target = distances_sq.min(dim=2)[0]
    target_to_pred = distances_sq.min(dim=1)[0]
    return pred_to_target.mean() + target_to_pred.mean()


@dataclass
class TrainingResult:
    model: PointCloudAutoencoder
    history: List[float]


def train_autoencoder(
    point_clouds: np.ndarray,
    feature_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    scheduler_eta_min: float,
    device: torch.device,
    seed: int,
) -> TrainingResult:
    if epochs <= 0:
        raise ValueError("epochs must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if lr <= 0.0:
        raise ValueError("lr must be > 0.")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be >= 0.")
    if scheduler_eta_min < 0.0:
        raise ValueError("scheduler_eta_min must be >= 0.")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = PointCloudDataset(point_clouds)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    model = PointCloudAutoencoder(num_points=point_clouds.shape[1], feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_steps = max(epochs * len(loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=scheduler_eta_min,
    )

    history: List[float] = []
    for _ in range(epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        for batch in loader:
            batch = batch.to(device)
            _, recon = model(batch)
            loss = chamfer_distance(recon, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.item()) * int(batch.shape[0])
            count += int(batch.shape[0])
        history.append(epoch_loss / max(count, 1))

    return TrainingResult(model=model, history=history)


def extract_embeddings(
    model: PointCloudAutoencoder,
    point_clouds: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = PointCloudDataset(point_clouds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    outputs: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model.encode(batch)
            outputs.append(feats.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0)


def standardize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = (features - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


@dataclass
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float


def _assign_clusters(features: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1).astype(np.int32)
    min_dist = distances[np.arange(features.shape[0]), labels]
    return labels, min_dist


def _compute_centers(features: np.ndarray, labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    centers = np.empty((k, features.shape[1]), dtype=np.float32)
    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.any(mask):
            centers[cluster_id] = features[mask].mean(axis=0)
        else:
            centers[cluster_id] = features[rng.integers(features.shape[0])]
    return centers


def run_kmeans(
    features: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 10,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> KMeansResult:
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {features.shape}")
    if k <= 0:
        raise ValueError("k must be > 0.")
    if k > features.shape[0]:
        raise ValueError(f"k={k} cannot exceed number of samples={features.shape[0]}.")

    best_labels = None
    best_centers = None
    best_inertia = None

    for init_idx in range(n_init):
        rng = np.random.default_rng(seed + init_idx)
        init_indices = rng.choice(features.shape[0], size=k, replace=False)
        centers = features[init_indices].copy()

        for _ in range(max_iter):
            labels, min_dist = _assign_clusters(features, centers)
            new_centers = _compute_centers(features, labels, k, rng)
            shift = np.linalg.norm(new_centers - centers, axis=1).max()
            centers = new_centers
            if shift <= tol:
                break

        labels, min_dist = _assign_clusters(features, centers)
        inertia = float(min_dist.sum())
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels
            best_centers = centers.copy()
            best_inertia = inertia

    if best_labels is None or best_centers is None or best_inertia is None:
        raise RuntimeError("KMeans failed to produce a valid result.")

    return KMeansResult(labels=best_labels, centers=best_centers, inertia=best_inertia)


def build_cluster_tag(version: str, feature_dim: int, k: int, seed: int) -> str:
    return f"{version}_ae{int(feature_dim)}_k{int(k)}_seed{int(seed)}"


def reorder_clusters_by_global_center_distance(
    labels: np.ndarray,
    centers: np.ndarray,
    features: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32)
    centers = np.asarray(centers, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)

    global_center = features.mean(axis=0)
    center_distances = np.linalg.norm(centers - global_center[None, :], axis=1)
    sorted_old_ids = np.argsort(center_distances, kind="stable").astype(np.int32)

    remap = np.empty(centers.shape[0], dtype=np.int32)
    remap[sorted_old_ids] = np.arange(centers.shape[0], dtype=np.int32)
    reordered_labels = remap[labels]
    reordered_centers = centers[sorted_old_ids]
    reordered_distances = center_distances[sorted_old_ids].astype(np.float32)
    return reordered_labels.astype(np.int32), reordered_centers.astype(np.float32), reordered_distances


def load_object_point_clouds(
    objdata_root: Path,
    scale_tag: str,
    pc_subdir: str,
    fps_points: int,
    seed: int,
) -> Tuple[List[str], List[Path], np.ndarray]:
    object_names: List[str] = []
    object_dirs: List[Path] = []
    point_clouds: List[np.ndarray] = []

    for object_dir in sorted(p for p in objdata_root.iterdir() if p.is_dir() and p.name != "_meta"):
        pc_path = object_dir / scale_tag / pc_subdir / "global_pc.npy"
        if not pc_path.exists():
            continue
        points = np.load(pc_path)
        normalized = normalize_point_cloud(points)
        sampled = downsample_point_cloud(normalized, fps_points, seed=seed + len(object_names))
        object_names.append(object_dir.name)
        object_dirs.append(object_dir)
        point_clouds.append(sampled)

    if not point_clouds:
        raise RuntimeError(
            f"No valid point clouds found under {objdata_root} for scale_tag={scale_tag} pc_subdir={pc_subdir}."
        )

    return object_names, object_dirs, np.stack(point_clouds, axis=0)


def save_cluster_artifacts(
    output_dir: Path,
    object_names: Sequence[str],
    labels: np.ndarray,
    normalized_embeddings: np.ndarray,
    centers: np.ndarray,
    center_global_distances: np.ndarray,
    train_history: Sequence[float],
    cluster_tag: str,
    scale_tag: str,
    pc_subdir: str,
    fps_points: int,
    feature_dim: int,
    kmeans_k: int,
    seed: int,
    model_state_dict: Dict[str, torch.Tensor],
    extra_meta: Dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.asarray(labels, dtype=np.int32)
    normalized_embeddings = np.asarray(normalized_embeddings, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    center_global_distances = np.asarray(center_global_distances, dtype=np.float32)

    torch.save(model_state_dict, output_dir / "ae_state_dict.pt")

    history_payload = {"train_loss": [float(v) for v in train_history]}
    (output_dir / "train_history.json").write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    distances = np.linalg.norm(normalized_embeddings - centers[labels], axis=1)
    global_center = normalized_embeddings.mean(axis=0)
    global_distances = np.linalg.norm(normalized_embeddings - global_center[None, :], axis=1)

    cluster_labels_payload: Dict[str, Dict] = {}
    object_labels_payload: Dict[str, Dict] = {}

    unique_cluster_ids = sorted(int(cluster_id) for cluster_id in np.unique(labels))
    for cluster_id in unique_cluster_ids:
        member_indices = np.where(labels == cluster_id)[0]
        ordered_indices = member_indices[np.argsort(distances[member_indices], kind="stable")]

        members_payload: List[Dict] = []
        for rank_in_cluster, member_index in enumerate(ordered_indices):
            object_name = str(object_names[int(member_index)])
            distance_to_center = float(distances[int(member_index)])
            distance_to_global_center = float(global_distances[int(member_index)])
            members_payload.append(
                {
                    "object_name": object_name,
                    "object_scale_key": f"{object_name}__{scale_tag}",
                    "distance_to_center": distance_to_center,
                    "distance_to_global_center": distance_to_global_center,
                    "rank_in_cluster": int(rank_in_cluster),
                }
            )

        cluster_labels_payload[str(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "distance_to_global_center": float(center_global_distances[cluster_id]),
            "member_count": len(members_payload),
            "members": members_payload,
        }

        for rank_in_cluster, member_index in enumerate(ordered_indices):
            object_name = str(object_names[int(member_index)])
            object_labels_payload[object_name] = {
                "object_name": object_name,
                "object_scale_key": f"{object_name}__{scale_tag}",
                "cluster_id": int(cluster_id),
                "distance_to_center": float(distances[int(member_index)]),
                "distance_to_global_center": float(global_distances[int(member_index)]),
                "rank_in_cluster": int(rank_in_cluster),
                "scale_tag": scale_tag,
            }

    meta = {
        "cluster_tag": cluster_tag,
        "scale_tag": scale_tag,
        "pc_subdir": pc_subdir,
        "fps_points": int(fps_points),
        "feature_dim": int(feature_dim),
        "kmeans_k": int(kmeans_k),
        "seed": int(seed),
        "num_objects": len(object_names),
        "cluster_order_rule": "cluster ids are sorted by ascending distance from cluster center to global feature center",
        "files": {
            "history": "train_history.json",
            "model": "ae_state_dict.pt",
            "object_labels": "object_labels.json",
            "cluster_labels": "cluster_labels.json",
        },
    }
    if extra_meta:
        meta["training"] = extra_meta
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "object_labels.json").write_text(
        json.dumps(
            {
                "cluster_tag": cluster_tag,
                "scale_tag": scale_tag,
                "objects": object_labels_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "cluster_labels.json").write_text(
        json.dumps(
            {
                "cluster_tag": cluster_tag,
                "scale_tag": scale_tag,
                "clusters": cluster_labels_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
