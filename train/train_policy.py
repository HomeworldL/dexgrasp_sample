#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.transforms import se3_exp_map, se3_log_map
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from train.pointnet import PointNet
from utils.utils_seed import set_seed


LOGGER = logging.getLogger(__name__)


EXPERIMENT: Dict[str, object] = {
    "name": "minimal_grasp_regression_shared_label",
    "run_config_path": "configs/run_YCB_liberhand.json",
    "manifest_path": "datasets/graspdata_YCB_liberhand/train.json",
    "artifact_dir": "train/artifacts/minimal_grasp_regression_shared_label",
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
    "grasp_selection_mode": "shared_object_scale",
    "scheduler_name": "none",
    "scheduler_eta_min": 1e-5,
}

POSE_TARGET_DIM = 6
QPOS_DIM = 27
JOINT_DIM = 20
TARGET_DIM = POSE_TARGET_DIM + JOINT_DIM
STD_EPS = 1e-6


@dataclass(frozen=True)
class SampleRecord:
    sample_index: int
    object_scale_key: str
    object_name: str
    view_index: int
    grasp_index: int
    partial_pc_path: str
    grasp_h5_path: str
    mjcf_path: str
    scale: float


def _xyzw_from_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32)
    return np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float32,
    )


def _wxyz_from_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )


def _transform_from_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if qpos.shape[0] != QPOS_DIM:
        raise ValueError(f"Expected qpos dimension {QPOS_DIM}, got {qpos.shape[0]}.")
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Rotation.from_quat(_xyzw_from_wxyz(qpos[3:7])).as_matrix()
    transform[:3, 3] = qpos[:3]
    return transform.astype(np.float32)


def _qpos_from_transform_and_joints(
    transform: np.ndarray,
    joints: np.ndarray,
) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float32)
    joints = np.asarray(joints, dtype=np.float32).reshape(-1)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected SE(3) transform shape (4, 4), got {transform.shape}.")
    if joints.shape[0] != JOINT_DIM:
        raise ValueError(f"Expected joint dimension {JOINT_DIM}, got {joints.shape[0]}.")
    quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat().astype(np.float32)
    qpos = np.concatenate(
        [
            transform[:3, 3].astype(np.float32),
            _wxyz_from_xyzw(quat_xyzw),
            joints.astype(np.float32),
        ]
    )
    return qpos.astype(np.float32)


def _se3_log_6d(transform: np.ndarray) -> np.ndarray:
    transform_tensor = torch.from_numpy(transform.astype(np.float32).T.copy()).unsqueeze(0)
    return se3_log_map(transform_tensor).squeeze(0).cpu().numpy().astype(np.float32)


def _transform_from_se3_log(log6d: np.ndarray) -> np.ndarray:
    log6d_tensor = torch.from_numpy(np.asarray(log6d, dtype=np.float32).reshape(1, 6))
    transform_tensor = se3_exp_map(log6d_tensor).squeeze(0).cpu().numpy().astype(np.float32)
    return transform_tensor.T.copy()


def target_from_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    return np.concatenate([_se3_log_6d(_transform_from_qpos(qpos)), qpos[7:]], axis=0).astype(
        np.float32
    )


def qpos_from_target(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if target.shape[0] != TARGET_DIM:
        raise ValueError(f"Expected target dimension {TARGET_DIM}, got {target.shape[0]}.")
    transform = _transform_from_se3_log(target[:POSE_TARGET_DIM])
    return _qpos_from_transform_and_joints(transform, target[POSE_TARGET_DIM:])


def normalize_target(target: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((target - mean) / std).astype(np.float32)


def denormalize_target(
    normalized_target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return (normalized_target * std + mean).astype(np.float32)


def load_manifest(path: str | Path) -> List[Dict]:
    manifest_path = Path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def deterministic_resample_points(
    points: np.ndarray,
    point_count: int,
    seed: int,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N, 3), got {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty.")
    rng = np.random.default_rng(int(seed))
    replace = points.shape[0] < int(point_count)
    indices = rng.choice(points.shape[0], size=int(point_count), replace=replace)
    return points[indices].astype(np.float32)


def _load_qpos_squeeze(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as handle:
        if "qpos_squeeze" not in handle:
            raise KeyError(f"Missing qpos_squeeze in {h5_path}")
        qpos = np.asarray(handle["qpos_squeeze"][:], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
        raise ValueError(f"Expected qpos_squeeze shape (N, {QPOS_DIM}), got {qpos.shape}.")
    if qpos.shape[0] <= 0:
        raise ValueError(f"qpos_squeeze is empty in {h5_path}")
    return qpos


def choose_grasp_index(
    qpos_squeeze: np.ndarray,
    item_index: int,
    view_index: int,
    seed: int,
    selection_mode: str,
) -> int:
    if selection_mode == "shared_object_scale":
        rng = np.random.default_rng(int(seed + item_index * 1000))
        return int(rng.integers(0, qpos_squeeze.shape[0]))
    if selection_mode == "per_view_random":
        rng = np.random.default_rng(int(seed + item_index * 1000 + view_index))
        return int(rng.integers(0, qpos_squeeze.shape[0]))
    raise ValueError(
        f"Unsupported grasp_selection_mode '{selection_mode}'. "
        "Expected 'shared_object_scale' or 'per_view_random'."
    )


def build_flattened_samples(
    manifest_path: str | Path,
    point_count: int,
    seed: int,
    grasp_selection_mode: str,
) -> Dict[str, object]:
    manifest = load_manifest(manifest_path)
    dataset_root = Path(manifest_path).resolve().parent

    point_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    qpos_batches: List[np.ndarray] = []
    records: List[SampleRecord] = []
    sample_index = 0

    for item_index, item in enumerate(manifest):
        qpos_squeeze = _load_qpos_squeeze(dataset_root / str(item["grasp_h5_path"]))
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
            target_qpos = qpos_squeeze[grasp_index].astype(np.float32)
            point_batches.append(sampled_points)
            qpos_batches.append(target_qpos)
            target_batches.append(target_from_qpos(target_qpos))
            records.append(
                SampleRecord(
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
        raise RuntimeError(f"No flattened training samples built from {manifest_path}.")

    targets = np.stack(target_batches, axis=0).astype(np.float32)
    target_mean = targets.mean(axis=0).astype(np.float32)
    target_std = np.clip(targets.std(axis=0), STD_EPS, None).astype(np.float32)
    normalized_targets = ((targets - target_mean) / target_std).astype(np.float32)

    return {
        "dataset_root": str(dataset_root),
        "points": np.stack(point_batches, axis=0).astype(np.float32),
        "targets": targets,
        "normalized_targets": normalized_targets,
        "target_qpos": np.stack(qpos_batches, axis=0).astype(np.float32),
        "target_mean": target_mean,
        "target_std": target_std,
        "records": records,
    }


class FlattenedGraspDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        point_count: int,
        seed: int,
        grasp_selection_mode: str,
    ) -> None:
        super().__init__()
        payload = build_flattened_samples(
            manifest_path,
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
        self.records: List[SampleRecord] = list(payload["records"])

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.records[int(index)]
        return {
            "points": torch.from_numpy(self.points[index]),
            "target": torch.from_numpy(self.normalized_targets[index]),
            "target_raw": torch.from_numpy(self.targets[index]),
            "target_qpos": torch.from_numpy(self.target_qpos[index]),
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


class PointNetRegressor(nn.Module):
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


def build_model(device: str | torch.device) -> PointNetRegressor:
    model = PointNetRegressor()
    return model.to(device)


def create_dataloader(dataset: FlattenedGraspDataset) -> DataLoader:
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


def evaluate_qpos_metrics(
    model: nn.Module,
    dataset: FlattenedGraspDataset,
    device: str | torch.device,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            points = torch.from_numpy(dataset.points[index]).unsqueeze(0).to(device)
            pred = model(points).squeeze(0).cpu().numpy().astype(np.float32)
            predictions.append(denormalize_target(pred, target_mean, target_std))
    pred_targets = np.stack(predictions, axis=0).astype(np.float32)
    pred_qpos = np.stack([qpos_from_target(target) for target in pred_targets], axis=0)
    true_qpos = dataset.target_qpos
    trans_err = np.linalg.norm(pred_qpos[:, :3] - true_qpos[:, :3], axis=1)
    rot_err: List[float] = []
    for pred_quat, true_quat in zip(pred_qpos[:, 3:7], true_qpos[:, 3:7]):
        dot = np.clip(abs(float(np.dot(pred_quat, true_quat))), 0.0, 1.0)
        rot_err.append(float(np.degrees(2.0 * np.arccos(dot))))
    joint_l1 = np.abs(pred_qpos[:, 7:] - true_qpos[:, 7:]).mean(axis=1)
    model.train()
    return {
        "translation_err_mean_m": float(trans_err.mean()),
        "translation_err_median_m": float(np.median(trans_err)),
        "rotation_err_mean_deg": float(np.mean(rot_err)),
        "rotation_err_median_deg": float(np.median(rot_err)),
        "joint_l1_mean": float(joint_l1.mean()),
        "joint_l1_median": float(np.median(joint_l1)),
    }


def _checkpoint_payload(
    model: nn.Module,
    dataset: FlattenedGraspDataset,
    epoch: int,
    train_loss: float,
    qpos_metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "experiment": dict(EXPERIMENT),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "target_mean": dataset.target_mean.copy(),
        "target_std": dataset.target_std.copy(),
        "qpos_metrics": dict(qpos_metrics),
        "model_state_dict": model.state_dict(),
        "sample_count": len(dataset),
        "target_dim": TARGET_DIM,
        "qpos_dim": QPOS_DIM,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    dataset: FlattenedGraspDataset,
    epoch: int,
    train_loss: float,
    qpos_metrics: Dict[str, float],
) -> None:
    torch.save(
        _checkpoint_payload(
            model,
            dataset,
            epoch=epoch,
            train_loss=train_loss,
            qpos_metrics=qpos_metrics,
        ),
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, object]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: str | Path,
    device: str | torch.device,
) -> tuple[PointNetRegressor, Dict[str, object]]:
    checkpoint = load_checkpoint(path, map_location=device)
    model = build_model(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_train_loss(
    model: nn.Module,
    dataset: FlattenedGraspDataset,
    device: str | torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in create_dataloader(dataset):
            points = batch["points"].to(device)
            targets = batch["target"].to(device)
            predictions = model(points)
            loss = criterion(predictions, targets)
            batch_size = int(points.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    model.train()
    return float(total_loss / max(1, total_count))


def _write_summary(path: Path, summary: Dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def train_policy() -> Dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    set_seed(int(EXPERIMENT["seed"]))
    device = str(EXPERIMENT["device"])
    dataset = FlattenedGraspDataset(
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
    last_checkpoint_path = out_dir / "checkpoint_last.pt"
    best_checkpoint_path = out_dir / "checkpoint_best.pt"
    summary_path = out_dir / "train_summary.json"

    LOGGER.info(
        "Start minimal grasp regression training: samples=%d point_count=%d batch_size=%d "
        "steps_per_epoch=%d device=%s",
        len(dataset),
        int(EXPERIMENT["point_count"]),
        int(EXPERIMENT["batch_size"]),
        steps_per_epoch,
        device,
    )

    best_loss = float("inf")
    best_epoch = -1
    best_qpos_metrics: Dict[str, float] = {}
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
        qpos_metrics = evaluate_qpos_metrics(
            model,
            dataset,
            device=device,
            target_mean=dataset.target_mean,
            target_std=dataset.target_std,
        )
        save_checkpoint(
            last_checkpoint_path,
            model,
            dataset,
            epoch=epoch,
            train_loss=train_loss,
            qpos_metrics=qpos_metrics,
        )

        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            best_qpos_metrics = dict(qpos_metrics)
            save_checkpoint(
                best_checkpoint_path,
                model,
                dataset,
                epoch=epoch,
                train_loss=train_loss,
                qpos_metrics=qpos_metrics,
            )

        if (
            epoch == 1
            or epoch % int(EXPERIMENT["log_every"]) == 0
            or epoch == int(EXPERIMENT["epochs"])
        ):
            LOGGER.info(
                "epoch=%d train_loss=%.8f best_loss=%.8f trans_mean=%.6f rot_mean=%.3f joint_l1=%.6f",
                epoch,
                train_loss,
                best_loss,
                qpos_metrics["translation_err_mean_m"],
                qpos_metrics["rotation_err_mean_deg"],
                qpos_metrics["joint_l1_mean"],
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
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_loss),
        "batch_size": int(EXPERIMENT["batch_size"]),
        "steps_per_epoch": int(steps_per_epoch),
        "checkpoint_best": str(best_checkpoint_path),
        "checkpoint_last": str(last_checkpoint_path),
        "best_qpos_metrics": dict(best_qpos_metrics),
        "scheduler_name": str(EXPERIMENT["scheduler_name"]),
        "history": history,
    }
    _write_summary(summary_path, summary)
    LOGGER.info("Training finished. best_epoch=%d best_loss=%.8f", best_epoch, best_loss)
    LOGGER.info("Artifacts written to %s", out_dir)
    return summary


def main() -> None:
    train_policy()


if __name__ == "__main__":
    main()
