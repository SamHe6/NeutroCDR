from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from neutro_cdr.data import NeutralizationDataset
from neutro_cdr.models import build_model
from neutro_cdr.training.metrics import classification_metrics, regression_metrics, threshold_tuned_classification_metrics
from neutro_cdr.training.train import move_to_device, multitask_loss, sync_feature_dims


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neutro_cdr_formal_esm_igbert.yaml")
    parser.add_argument("--out-dir", default="results/neutro_cdr_5fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-column", default="antibody_id")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume each fold from its existing history/checkpoint when available.")
    parser.add_argument("--patience", type=int, default=0, help="Early-stop a fold after this many epochs without validation-score improvement. 0 disables it.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation-score improvement required to reset early stopping.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        base_cfg = yaml.safe_load(handle)
    sync_feature_dims(base_cfg)
    if args.epochs is not None:
        base_cfg["train"]["epochs"] = args.epochs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(base_cfg["data"]["pair_table"]).reset_index(drop=True)
    labels = frame["label"].astype(int).to_numpy()
    groups = get_groups(frame, args.group_column)

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows = []
    for fold_idx, (train_valid_idx, test_idx) in enumerate(
        splitter.split(frame, labels, groups), start=1
    ):
        fold_metrics = run_fold(
            fold_idx=fold_idx,
            train_valid_idx=train_valid_idx,
            test_idx=test_idx,
            frame=frame,
            base_cfg=base_cfg,
            out_dir=out_dir,
            args=args,
        )
        fold_rows.append(fold_metrics)
        write_summary(out_dir, fold_rows)

    write_summary(out_dir, fold_rows)


def run_fold(
    fold_idx: int,
    train_valid_idx: np.ndarray,
    test_idx: np.ndarray,
    frame: pd.DataFrame,
    base_cfg: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, float]:
    fold_dir = out_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_idx, valid_idx = split_train_valid(
        train_valid_idx=train_valid_idx,
        labels=frame["label"].astype(int).to_numpy(),
        groups=get_groups(frame, args.group_column),
        valid_fraction=args.valid_fraction,
        seed=args.seed + fold_idx,
    )
    fold_table = make_fold_table(
        frame, train_idx, valid_idx, test_idx, group_column=args.group_column
    )
    fold_table_path = fold_dir / "pairs.csv"
    fold_table.to_csv(fold_table_path, index=False)

    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["pair_table"] = str(fold_table_path)
    cfg["paths"]["results_dir"] = str(fold_dir)
    with open(fold_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)

    device = torch.device(args.device)
    train_loader = make_loader(cfg, "train", shuffle=True)
    valid_loader = make_loader(cfg, "valid", shuffle=False)
    test_loader = make_loader(cfg, "test", shuffle=False)

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))
    use_amp = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history = load_history(fold_dir / "history.json") if args.resume else []
    start_epoch = len(history) + 1

    checkpoint_path = fold_dir / "neutro_cdr.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        best_score = float(checkpoint.get("valid_score", best_score))
        best_threshold = float(checkpoint.get("classification_threshold", best_threshold))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None and use_amp:
            scaler.load_state_dict(checkpoint["scaler"])

    if history and len(history) >= cfg["train"]["epochs"] and (fold_dir / "metrics.json").exists():
        with open(fold_dir / "metrics.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, scaler, use_amp, fold_idx)
        valid_metrics = evaluate_with_optional_threshold(model, valid_loader, device)
        row = {"fold": fold_idx, "epoch": epoch, "train_loss": train_loss, **{f"valid_{key}": value for key, value in valid_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with open(fold_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        score = valid_metrics.get("auprc", 0.0) + valid_metrics.get("spearman", 0.0)
        if score > best_score + args.min_delta:
            best_score = score
            epochs_without_improvement = 0
            best_threshold = float(valid_metrics.get("tuned_threshold", 0.5))
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if use_amp else None,
                    "config": cfg,
                    "classification_threshold": best_threshold,
                    "fold": fold_idx,
                    "valid_score": best_score,
                    "epoch": epoch,
                },
                fold_dir / "neutro_cdr.pt",
            )
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    json.dumps(
                        {
                            "fold": fold_idx,
                            "early_stop_epoch": epoch,
                            "best_valid_score": best_score,
                            "patience": args.patience,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break

    checkpoint = torch.load(fold_dir / "neutro_cdr.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate_with_optional_threshold(model, test_loader, device, threshold=float(checkpoint["classification_threshold"]))
    fold_metrics = {
        "fold": fold_idx,
        "best_valid_score": float(checkpoint["valid_score"]),
        "selected_threshold": float(checkpoint["classification_threshold"]),
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(fold_metrics, handle, indent=2)
    print(json.dumps(fold_metrics, ensure_ascii=False), flush=True)
    return fold_metrics


def load_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def split_train_valid(
    train_valid_idx: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    valid_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be between 0 and 1")
    local_labels = labels[train_valid_idx]
    local_groups = groups[train_valid_idx]
    n_splits = max(2, round(1.0 / valid_fraction))
    if np.unique(local_groups).size < n_splits:
        raise ValueError(f"need at least {n_splits} groups for valid_fraction={valid_fraction}")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_local, valid_local = next(
        splitter.split(train_valid_idx, local_labels, local_groups)
    )
    return train_valid_idx[train_local], train_valid_idx[valid_local]


def get_groups(frame: pd.DataFrame, group_column: str) -> np.ndarray:
    if group_column not in frame.columns:
        raise KeyError(f"group column {group_column!r} is not present in the pair table")
    return frame[group_column].astype(str).to_numpy()


def make_fold_table(
    frame: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
    group_column: str = "antibody_id",
) -> pd.DataFrame:
    fold_table = frame.copy()
    fold_table["split"] = "unused"
    fold_table.loc[train_idx, "split"] = "train"
    fold_table.loc[valid_idx, "split"] = "valid"
    fold_table.loc[test_idx, "split"] = "test"
    split_counts = fold_table.groupby(group_column, dropna=False)["split"].nunique()
    if (split_counts > 1).any():
        raise RuntimeError("antibody leakage detected across train/valid/test")
    return fold_table


def make_loader(cfg: dict, split: str, shuffle: bool) -> DataLoader:
    dataset = NeutralizationDataset(
        cfg["data"]["pair_table"],
        split,
        cfg["data"]["max_antibody_len"],
        cfg["data"]["max_antigen_len"],
        esm_antigen_path=cfg["data"].get("esm_antigen_path"),
        igbert_antibody_path=cfg["data"].get("igbert_antibody_path"),
        cdr_mask_path=cfg["data"].get("cdr_mask_path"),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["train"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    fold_idx: int,
) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc=f"fold {fold_idx} train", leave=False):
        batch = move_to_device(batch, device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(batch)
            loss = multitask_loss(output, batch, cfg)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu()) * batch["label"].size(0)
    return total / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate_with_optional_threshold(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = None,
) -> dict[str, float]:
    model.eval()
    labels: list[float] = []
    probs: list[float] = []
    targets: list[float] = []
    preds: list[float] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_to_device(batch, device)
        output = model(batch)
        labels.extend(batch["label"].detach().cpu().tolist())
        probs.extend(torch.sigmoid(output["logits"]).detach().cpu().tolist())
        exact = batch.get("censor_code", torch.zeros_like(batch["log_ic50"])) == 0
        targets.extend(batch["log_ic50"][exact].detach().cpu().tolist())
        preds.extend(output["log_ic50"][exact].detach().cpu().tolist())

    cls_threshold = 0.5 if threshold is None else threshold
    out = {
        **classification_metrics(labels, probs, threshold=cls_threshold),
        **regression_metrics(targets, preds),
    }
    if threshold is None:
        out.update(threshold_tuned_classification_metrics(labels, probs))
    return out


def write_summary(out_dir: Path, fold_rows: list[dict[str, float]]) -> None:
    frame = pd.DataFrame(fold_rows)
    numeric = frame.select_dtypes(include=[np.number])
    summary = {
        "folds": fold_rows,
        "mean": numeric.mean(numeric_only=True).to_dict(),
        "std": numeric.std(ddof=1, numeric_only=True).fillna(0.0).to_dict(),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
