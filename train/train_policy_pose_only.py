#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from train.pointnet import PointNet
from train.train_policy import (
    STD_EPS,
    _load_qpos_squeeze,
    _se3_log_6d,
    _transform_from_qpos,
    _transform_from_se3_log,
    choose_grasp_index,
    deterministic_resample_points,
    load_manifest,
)
from utils.utils_seed import set_seed


LOGGER = logging.getLogger(__name__)


EXPERIMENT: Dict[str, object] = {
    "name": "minimal_grasp_regression_pose_only",
    "run_config_path": "configs/run_YCB_liberhand.json",
    "manifest_path": "datasets/graspdata_YCB_liberhand/train.json",
    "artifact_dir": "train/artifacts/minimal_grasp_regression_pose_only",
    "seed": 7,
    "point_count": 1024,
    "batch_size": 32,
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
    "scheduler_name": "none",
    "scheduler_eta_min": 1e-5,
}

POSE_TARGET_DIM = 6


@dataclass(frozen=True)
class PoseOnlySampleRecord:
    sample_index: int
    object_scale_key: str
    object_name: str
    view_index: int
    grasp_index: int
    partial_pc_path: str
    grasp_h5_path: str
    mjcf_path: str
    scale: float


def pose_target_from_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    return _se3_log_6d(_transform_from_qpos(qpos)).astype(np.float32)


def transform_from_pose_target(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if target.shape[0] != POSE_TARGET_DIM:
        raise ValueError(f"Expected pose target dimension {POSE_TARGET_DIM}, got {target.shape[0]}.")
    return _transform_from_se3_log(target)


def build_pose_only_samples(
    manifest_path: str | Path,
    point_count: int,
    seed: int,
    grasp_selection_mode: str,
) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    dataset_root = Path(manifest_path).resolve().parent

    point_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    target_qpos_batches: List[np.ndarray] = []
    records: List[PoseOnlySampleRecord] = []
    sample_index = 0

    for item_index, item in enumerate(manifest):
        qpos_squeeze = _load_qpos_squeeze(dataset_root / str(item["grasp_h5_path"]))
        for view_index, partial_pc_rel in enumerate(list(item["partial_pc_path"])):
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
            target_qpos = qpos_squeeze[grasp_index].astype(np.float32)
            point_batches.append(sampled_points)
            target_batches.append(pose_target_from_qpos(target_qpos))
            target_qpos_batches.append(target_qpos)
            records.append(
                PoseOnlySampleRecord(
                    sample_index=sample_index,
                    object_scale_key=str(item["object_scale_key"]),
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
        raise RuntimeError(f"No pose-only samples built from {manifest_path}.")

    targets = np.stack(target_batches, axis=0).astype(np.float32)
    target_mean = targets.mean(axis=0).astype(np.float32)
    target_std = np.clip(targets.std(axis=0), STD_EPS, None).astype(np.float32)
    normalized_targets = ((targets - target_mean) / target_std).astype(np.float32)

    return {
        "dataset_root": str(dataset_root),
        "points": np.stack(point_batches, axis=0).astype(np.float32),
        "targets": targets,
        "normalized_targets": normalized_targets,
        "target_qpos": np.stack(target_qpos_batches, axis=0).astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
        "records": records,
    }


class PoseOnlyDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        point_count: int,
        seed: int,
        grasp_selection_mode: str,
    ) -> None:
        payload = build_pose_only_samples(
            manifest_path=manifest_path,
            point_count=point_count,
            seed=seed,
            grasp_selection_mode=grasp_selection_mode,
        )
        self.dataset_root = str(payload["dataset_root"])
        self.points = np.asarray(payload["points"], dtype=np.float32)
        self.targets = np.asarray(payload["targets"], dtype=np.float32)
        self.normalized_targets = np.asarray(payload["normalized_targets"], dtype=np.float32)
        self.target_qpos = np.asarray(payload["target_qpos"], dtype=np.float32)
        self.target_mean = np.asarray(payload["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(payload["target_std"], dtype=np.float32)
        self.records: List[PoseOnlySampleRecord] = list(payload["records"])

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[int(index)]
        return {
            "points": torch.from_numpy(self.points[index]),
            "target": torch.from_numpy(self.normalized_targets[index]),
            "target_raw": torch.from_numpy(self.targets[index]),
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


class PoseOnlyRegressor(nn.Module):
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
            nn.Linear(int(EXPERIMENT["head_hidden_dim"]), POSE_TARGET_DIM),
        )
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        global_feature, _ = self.backbone(points)
        return self.head(global_feature)


def build_model(device: str | torch.device) -> PoseOnlyRegressor:
    return PoseOnlyRegressor().to(device)


def create_dataloader(dataset: PoseOnlyDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(EXPERIMENT["batch_size"]),
        shuffle=True,
        num_workers=int(EXPERIMENT["num_workers"]),
        drop_last=False,
    )


def artifact_dir() -> Path:
    path = Path(str(EXPERIMENT["artifact_dir"])).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_scheduler(optimizer: torch.optim.Optimizer):
    scheduler_name = str(EXPERIMENT["scheduler_name"]).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(EXPERIMENT["epochs"]),
            eta_min=float(EXPERIMENT["scheduler_eta_min"]),
        )
    raise ValueError(
        f"Unsupported scheduler_name '{EXPERIMENT['scheduler_name']}'. "
        "Expected 'none' or 'cosine'."
    )


def denormalize_target(
    normalized_target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (normalized_target * std + mean).astype(np.float32)


def evaluate_pose_metrics(
    model: nn.Module,
    dataset: PoseOnlyDataset,
    device: str | torch.device,
) -> Dict[str, float]:
    model.eval()
    predicted_targets: List[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            points = torch.from_numpy(dataset.points[index]).unsqueeze(0).to(device)
            pred = model(points).squeeze(0).cpu().numpy().astype(np.float32)
            predicted_targets.append(denormalize_target(pred, dataset.target_mean, dataset.target_std))

    pred_targets = np.stack(predicted_targets, axis=0).astype(np.float32)
    target_transforms = np.stack(
        [_transform_from_qpos(qpos) for qpos in dataset.target_qpos],
        axis=0,
    ).astype(np.float32)
    pred_transforms = np.stack(
        [transform_from_pose_target(target) for target in pred_targets],
        axis=0,
    ).astype(np.float32)

    trans_err = np.linalg.norm(pred_transforms[:, :3, 3] - target_transforms[:, :3, 3], axis=1)
    angle_err: List[float] = []
    for pred_transform, target_transform in zip(pred_transforms, target_transforms):
        rel_rot = pred_transform[:3, :3] @ target_transform[:3, :3].T
        angle_err.append(float(np.degrees(Rotation.from_matrix(rel_rot).magnitude())))

    model.train()
    return {
        "translation_err_mean_m": float(trans_err.mean()),
        "translation_err_median_m": float(np.median(trans_err)),
        "rotation_err_mean_deg": float(np.mean(angle_err)),
        "rotation_err_median_deg": float(np.median(angle_err)),
    }


def _checkpoint_payload(
    model: nn.Module,
    dataset: PoseOnlyDataset,
    epoch: int,
    train_loss: float,
    pose_metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "experiment": dict(EXPERIMENT),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "target_mean": dataset.target_mean.copy(),
        "target_std": dataset.target_std.copy(),
        "pose_metrics": dict(pose_metrics),
        "model_state_dict": model.state_dict(),
        "sample_count": len(dataset),
        "target_dim": POSE_TARGET_DIM,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    dataset: PoseOnlyDataset,
    epoch: int,
    train_loss: float,
    pose_metrics: Dict[str, float],
) -> None:
    torch.save(
        _checkpoint_payload(
            model,
            dataset,
            epoch=epoch,
            train_loss=train_loss,
            pose_metrics=pose_metrics,
        ),
        path,
    )


def train_policy_pose_only() -> Dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_seed(int(EXPERIMENT["seed"]))
    device = str(EXPERIMENT["device"])
    dataset = PoseOnlyDataset(
        manifest_path=str(EXPERIMENT["manifest_path"]),
        point_count=int(EXPERIMENT["point_count"]),
        seed=int(EXPERIMENT["seed"]),
        grasp_selection_mode=str(EXPERIMENT["grasp_selection_mode"]),
    )
    dataloader = create_dataloader(dataset)
    steps_per_epoch = len(dataloader)
    model = build_model(device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(EXPERIMENT["learning_rate"]),
        weight_decay=float(EXPERIMENT["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer)
    criterion = nn.MSELoss()
    out_dir = artifact_dir()
    best_checkpoint_path = out_dir / "checkpoint_best.pt"
    last_checkpoint_path = out_dir / "checkpoint_last.pt"
    summary_path = out_dir / "train_summary.json"

    LOGGER.info(
        "Start pose-only grasp regression training: samples=%d point_count=%d batch_size=%d "
        "steps_per_epoch=%d device=%s",
        len(dataset),
        int(EXPERIMENT["point_count"]),
        int(EXPERIMENT["batch_size"]),
        steps_per_epoch,
        device,
    )

    best_loss = float("inf")
    best_epoch = -1
    best_pose_metrics: Dict[str, float] = {}
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(EXPERIMENT["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for batch in dataloader:
            points = batch["points"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(points)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            batch_size = int(points.shape[0])
            epoch_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        if scheduler is not None:
            scheduler.step()
        train_loss = float(epoch_loss / max(1, sample_count))
        history.append({"epoch": float(epoch), "train_loss": float(train_loss)})

        pose_metrics = evaluate_pose_metrics(model, dataset, device=device)
        save_checkpoint(
            last_checkpoint_path,
            model,
            dataset,
            epoch=epoch,
            train_loss=train_loss,
            pose_metrics=pose_metrics,
        )

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_pose_metrics = dict(pose_metrics)
            save_checkpoint(
                best_checkpoint_path,
                model,
                dataset,
                epoch=epoch,
                train_loss=train_loss,
                pose_metrics=pose_metrics,
            )

        if (
            epoch == 1
            or epoch % int(EXPERIMENT["log_every"]) == 0
            or epoch == int(EXPERIMENT["epochs"])
        ):
            LOGGER.info(
                "epoch=%d train_loss=%.8f best_loss=%.8f trans_mean=%.6f rot_mean=%.3f",
                epoch,
                train_loss,
                best_loss,
                pose_metrics["translation_err_mean_m"],
                pose_metrics["rotation_err_mean_deg"],
            )
            LOGGER.info("epoch=%d lr=%.8e", epoch, optimizer.param_groups[0]["lr"])

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
        "batch_size": int(EXPERIMENT["batch_size"]),
        "steps_per_epoch": int(steps_per_epoch),
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_loss),
        "best_pose_metrics": dict(best_pose_metrics),
        "scheduler_name": str(EXPERIMENT["scheduler_name"]),
        "checkpoint_best": str(best_checkpoint_path),
        "checkpoint_last": str(last_checkpoint_path),
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Training finished. best_epoch=%d best_loss=%.8f", best_epoch, best_loss)
    LOGGER.info("Artifacts written to %s", out_dir)
    return summary


def main() -> None:
    train_policy_pose_only()


if __name__ == "__main__":
    main()
