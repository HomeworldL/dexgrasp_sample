#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from train.pointnet import PointNet
from train.train_policy import (
    STD_EPS,
    TARGET_DIM,
    _load_qpos_squeeze,
    choose_grasp_index,
    deterministic_resample_points,
    load_manifest,
    target_from_qpos,
)
from utils.utils_seed import set_seed


LOGGER = logging.getLogger(__name__)


EXPERIMENT: Dict[str, object] = {
    "name": "minimal_grasp_retrieval",
    "run_config_path": "configs/run_YCB_liberhand.json",
    "manifest_path": "datasets/graspdata_YCB_liberhand/train.json",
    "artifact_dir": "train/artifacts/minimal_grasp_retrieval",
    "seed": 7,
    "point_count": 1024,
    "epochs": 400,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "num_workers": 0,
    "log_every": 10,
    "early_stop_loss": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "point_feature_dim": 3,
    "local_conv_hidden_layers_dim": [64, 128, 256],
    "global_mlp_hidden_layers_dim": [256],
    "pc_feature_dim": 128,
    "head_hidden_dim": 128,
    "activation": "leaky_relu",
    "grasp_selection_mode": "per_view_random",
    "distance_temperature": 1.0,
}


@dataclass(frozen=True)
class RetrievalSampleRecord:
    sample_index: int
    object_scale_key: str
    object_name: str
    view_index: int
    grasp_index: int
    partial_pc_path: str
    grasp_h5_path: str
    mjcf_path: str
    scale: float


def normalize_candidates(
    targets: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return ((targets - mean) / std).astype(np.float32)


def compute_candidate_logits(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if query_embeddings.ndim != 2:
        raise ValueError(f"Expected query embeddings shape (B, D), got {tuple(query_embeddings.shape)}")
    if candidate_embeddings.ndim != 2:
        raise ValueError(
            f"Expected candidate embeddings shape (K, D), got {tuple(candidate_embeddings.shape)}"
        )
    distances = torch.cdist(
        query_embeddings.unsqueeze(0),
        candidate_embeddings.unsqueeze(0),
    ).squeeze(0)
    return -(distances.square() / float(temperature))


def select_candidate_indices(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = compute_candidate_logits(query_embeddings, candidate_embeddings, temperature=temperature)
    return torch.argmax(logits, dim=1)


def build_retrieval_samples(
    manifest_path: str | Path,
    point_count: int,
    seed: int,
    grasp_selection_mode: str,
) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    dataset_root = Path(manifest_path).resolve().parent

    object_scale_to_qpos: Dict[str, np.ndarray] = {}
    object_scale_to_targets: Dict[str, np.ndarray] = {}
    all_candidate_targets: List[np.ndarray] = []

    for item in manifest:
        object_scale_key = str(item["object_scale_key"])
        qpos_squeeze = _load_qpos_squeeze(dataset_root / str(item["grasp_h5_path"]))
        candidate_targets = np.stack([target_from_qpos(qpos) for qpos in qpos_squeeze], axis=0)
        object_scale_to_qpos[object_scale_key] = qpos_squeeze.astype(np.float32)
        object_scale_to_targets[object_scale_key] = candidate_targets.astype(np.float32)
        all_candidate_targets.append(candidate_targets.astype(np.float32))

    candidate_targets_all = np.concatenate(all_candidate_targets, axis=0).astype(np.float32)
    target_mean = candidate_targets_all.mean(axis=0).astype(np.float32)
    target_std = np.clip(candidate_targets_all.std(axis=0), STD_EPS, None).astype(np.float32)

    object_scale_to_targets_norm = {
        key: normalize_candidates(value, target_mean, target_std)
        for key, value in object_scale_to_targets.items()
    }

    point_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    records: List[RetrievalSampleRecord] = []
    sample_index = 0

    for item_index, item in enumerate(manifest):
        object_scale_key = str(item["object_scale_key"])
        qpos_squeeze = object_scale_to_qpos[object_scale_key]
        candidate_targets_norm = object_scale_to_targets_norm[object_scale_key]
        view_paths = list(item["partial_pc_path"])
        for view_index, partial_pc_rel in enumerate(view_paths):
            sample_seed = int(seed + item_index * 1000 + view_index)
            grasp_index = choose_grasp_index(
                qpos_squeeze=qpos_squeeze,
                item_index=item_index,
                view_index=view_index,
                seed=seed,
                selection_mode=grasp_selection_mode,
            )
            point_cloud = np.load(dataset_root / str(partial_pc_rel)).astype(np.float32)
            sampled_points = deterministic_resample_points(
                point_cloud,
                point_count=point_count,
                seed=sample_seed,
            )
            point_batches.append(sampled_points)
            target_batches.append(candidate_targets_norm[grasp_index].astype(np.float32))
            records.append(
                RetrievalSampleRecord(
                    sample_index=sample_index,
                    object_scale_key=object_scale_key,
                    object_name=str(item["object_name"]),
                    view_index=int(view_index),
                    grasp_index=int(grasp_index),
                    partial_pc_path=str(partial_pc_rel),
                    grasp_h5_path=str(item["grasp_h5_path"]),
                    mjcf_path=str(item["mjcf_path"]),
                    scale=float(item["scale"]),
                )
            )
            sample_index += 1

    if not point_batches:
        raise RuntimeError(f"No retrieval samples built from {manifest_path}.")

    object_scale_to_indices: Dict[str, List[int]] = {}
    for record in records:
        object_scale_to_indices.setdefault(record.object_scale_key, []).append(record.sample_index)

    return {
        "dataset_root": str(dataset_root),
        "points": np.stack(point_batches, axis=0).astype(np.float32),
        "normalized_targets": np.stack(target_batches, axis=0).astype(np.float32),
        "records": records,
        "target_mean": target_mean,
        "target_std": target_std,
        "object_scale_to_qpos": object_scale_to_qpos,
        "object_scale_to_targets_norm": object_scale_to_targets_norm,
        "object_scale_to_indices": object_scale_to_indices,
    }


class RetrievalDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        point_count: int,
        seed: int,
        grasp_selection_mode: str,
    ) -> None:
        payload = build_retrieval_samples(
            manifest_path=manifest_path,
            point_count=point_count,
            seed=seed,
            grasp_selection_mode=grasp_selection_mode,
        )
        self.dataset_root = str(payload["dataset_root"])
        self.points = np.asarray(payload["points"], dtype=np.float32)
        self.normalized_targets = np.asarray(payload["normalized_targets"], dtype=np.float32)
        self.records: List[RetrievalSampleRecord] = list(payload["records"])
        self.target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(payload["target_std"], dtype=np.float32)
        self.object_scale_to_qpos = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(payload["object_scale_to_qpos"]).items()
        }
        self.object_scale_to_targets_norm = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in dict(payload["object_scale_to_targets_norm"]).items()
        }
        self.object_scale_to_indices = {
            key: list(indices)
            for key, indices in dict(payload["object_scale_to_indices"]).items()
        }

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[int(index)]
        return {
            "points": torch.from_numpy(self.points[index]),
            "target": torch.from_numpy(self.normalized_targets[index]),
            "candidate_local_index": int(record.grasp_index),
            "sample_index": int(record.sample_index),
            "object_scale_key": record.object_scale_key,
            "object_name": record.object_name,
            "view_index": int(record.view_index),
            "grasp_index": int(record.grasp_index),
            "partial_pc_path": record.partial_pc_path,
            "grasp_h5_path": record.grasp_h5_path,
            "mjcf_path": record.mjcf_path,
            "scale": float(record.scale),
        }


class ObjectScaleBatchSampler(BatchSampler):
    def __init__(self, groups: Dict[str, Sequence[int]], shuffle: bool, seed: int):
        self.groups = {key: list(indices) for key, indices in groups.items()}
        self.group_keys = sorted(self.groups)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        order = list(self.group_keys)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(order)
        for key in order:
            yield list(self.groups[key])

    def __len__(self) -> int:
        return len(self.group_keys)


class PointNetRetrievalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = PointNet(
            point_feature_dim=int(EXPERIMENT["point_feature_dim"]),
            local_conv_hidden_layers_dim=list(EXPERIMENT["local_conv_hidden_layers_dim"]),
            global_mlp_hidden_layers_dim=list(EXPERIMENT["global_mlp_hidden_layers_dim"]),
            pc_feature_dim=int(EXPERIMENT["pc_feature_dim"]),
            activation=str(EXPERIMENT["activation"]),
        )
        self.head = nn.Sequential(
            nn.Linear(int(EXPERIMENT["pc_feature_dim"]), int(EXPERIMENT["head_hidden_dim"])),
            nn.LeakyReLU(),
            nn.Linear(int(EXPERIMENT["head_hidden_dim"]), TARGET_DIM),
        )
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        global_feature, _ = self.backbone(points)
        return self.head(global_feature)


def build_model(device: str | torch.device) -> PointNetRetrievalModel:
    return PointNetRetrievalModel().to(device)


def build_dataloader(dataset: RetrievalDataset, shuffle: bool) -> tuple[DataLoader, ObjectScaleBatchSampler]:
    batch_sampler = ObjectScaleBatchSampler(
        groups=dataset.object_scale_to_indices,
        shuffle=shuffle,
        seed=int(EXPERIMENT["seed"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(EXPERIMENT["num_workers"]),
    )
    return dataloader, batch_sampler


def artifact_dir() -> Path:
    path = Path(str(EXPERIMENT["artifact_dir"])).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_payload(
    model: nn.Module,
    dataset: RetrievalDataset,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
) -> Dict[str, object]:
    return {
        "experiment": dict(EXPERIMENT),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_accuracy": float(train_accuracy),
        "target_mean": dataset.target_mean.copy(),
        "target_std": dataset.target_std.copy(),
        "model_state_dict": model.state_dict(),
        "sample_count": len(dataset),
        "batch_count": len(dataset.object_scale_to_indices),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    dataset: RetrievalDataset,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
) -> None:
    torch.save(
        _checkpoint_payload(
            model,
            dataset,
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
        ),
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, object]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device,
) -> tuple[PointNetRetrievalModel, Dict[str, object]]:
    checkpoint = load_checkpoint(path, map_location=device)
    model = build_model(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def train_policy_retrieval() -> Dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_seed(int(EXPERIMENT["seed"]))
    device = str(EXPERIMENT["device"])
    dataset = RetrievalDataset(
        manifest_path=str(EXPERIMENT["manifest_path"]),
        point_count=int(EXPERIMENT["point_count"]),
        seed=int(EXPERIMENT["seed"]),
        grasp_selection_mode=str(EXPERIMENT["grasp_selection_mode"]),
    )
    dataloader, batch_sampler = build_dataloader(dataset, shuffle=True)
    steps_per_epoch = len(dataloader)
    model = build_model(device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(EXPERIMENT["learning_rate"]),
        weight_decay=float(EXPERIMENT["weight_decay"]),
    )
    out_dir = artifact_dir()
    best_checkpoint_path = out_dir / "checkpoint_best.pt"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"
    summary_path = out_dir / "train_summary.json"

    LOGGER.info(
        "Start minimal grasp retrieval training: samples=%d object_batches=%d point_count=%d device=%s",
        len(dataset),
        steps_per_epoch,
        int(EXPERIMENT["point_count"]),
        device,
    )

    best_loss = math.inf
    best_epoch = -1
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(EXPERIMENT["epochs"]) + 1):
        batch_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch in dataloader:
            object_scale_key = str(batch["object_scale_key"][0])
            points = batch["points"].to(device)
            labels = batch["candidate_local_index"].to(device)
            candidate_embeddings = torch.from_numpy(
                dataset.object_scale_to_targets_norm[object_scale_key]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            queries = model(points)
            logits = compute_candidate_logits(
                queries,
                candidate_embeddings,
                temperature=float(EXPERIMENT["distance_temperature"]),
            )
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(points.shape[0])
            epoch_loss += float(loss.item()) * batch_size
            epoch_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
            epoch_total += batch_size

        train_loss = float(epoch_loss / max(1, epoch_total))
        train_accuracy = float(epoch_correct / max(1, epoch_total))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_accuracy),
            }
        )
        save_checkpoint(
            last_checkpoint_path,
            model,
            dataset,
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
        )

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            save_checkpoint(
                best_checkpoint_path,
                model,
                dataset,
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
            )

        if (
            epoch == 1
            or epoch % int(EXPERIMENT["log_every"]) == 0
            or epoch == int(EXPERIMENT["epochs"])
        ):
            LOGGER.info(
                "epoch=%d train_loss=%.8f train_acc=%.4f best_loss=%.8f",
                epoch,
                train_loss,
                train_accuracy,
                best_loss,
            )

        if train_loss <= float(EXPERIMENT["early_stop_loss"]):
            LOGGER.info(
                "Early stop at epoch=%d because train_loss=%.8f <= %.8f",
                epoch,
                train_loss,
                float(EXPERIMENT["early_stop_loss"]),
            )
            break

    summary = {
        "experiment": dict(EXPERIMENT),
        "sample_count": len(dataset),
        "object_batches": steps_per_epoch,
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_loss),
        "checkpoint_best": str(best_checkpoint_path),
        "checkpoint_last": str(last_checkpoint_path),
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Training finished. best_epoch=%d best_loss=%.8f", best_epoch, best_loss)
    LOGGER.info("Artifacts written to %s", out_dir)
    return summary


def main() -> None:
    train_policy_retrieval()


if __name__ == "__main__":
    main()
