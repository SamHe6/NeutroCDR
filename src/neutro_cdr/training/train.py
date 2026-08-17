from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from neutro_cdr.data import NeutralizationDataset
from neutro_cdr.models import build_model
from neutro_cdr.training.metrics import classification_metrics, regression_metrics, threshold_tuned_classification_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/neutro_cdr.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    sync_feature_dims(cfg)

    device = torch.device(args.device)
    out_dir = Path(cfg["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_dataset(cfg, "train")
    valid_ds = build_dataset(cfg, "valid")
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"].get("num_workers", 0))
    valid_loader = DataLoader(valid_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"].get("num_workers", 0))

    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 0.0))
    use_amp = bool(cfg["train"].get("amp", False) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    history = []
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, scaler, use_amp)
        valid_metrics = evaluate(model, valid_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        with open(out_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        score = selection_score(valid_metrics, cfg["train"].get("selection_metric", "auprc_plus_spearman"))
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "classification_threshold": valid_metrics.get("tuned_threshold", 0.5),
                },
                out_dir / "neutro_cdr.pt",
            )

    with open(out_dir / "history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
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
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[float] = []
    probs: list[float] = []
    targets: list[float] = []
    preds: list[float] = []
    for batch in loader:
        batch = move_to_device(batch, device)
        output = model(batch)
        labels.extend(batch["label"].detach().cpu().tolist())
        probs.extend(torch.sigmoid(output["logits"]).detach().cpu().tolist())
        targets.extend(batch["log_ic50"].detach().cpu().tolist())
        preds.extend(output["log_ic50"].detach().cpu().tolist())
    return {
        **classification_metrics(labels, probs),
        **threshold_tuned_classification_metrics(labels, probs),
        **regression_metrics(targets, preds),
    }


def multitask_loss(output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    cls_loss = F.binary_cross_entropy_with_logits(output["logits"], batch["label"])
    reg_loss = F.smooth_l1_loss(output["log_ic50"], batch["log_ic50"])
    threshold = cfg["loss"].get("log_ic50_threshold", 1.0)
    temperature = cfg["loss"].get("consistency_temperature", 0.25)
    p_from_ic50 = torch.sigmoid((threshold - output["log_ic50"].detach()) / temperature)
    consistency = F.binary_cross_entropy_with_logits(output["logits"], p_from_ic50)
    return cls_loss + cfg["loss"].get("reg_weight", 0.5) * reg_loss + cfg["loss"].get("consistency_weight", 0.1) * consistency


def selection_score(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "auprc_plus_spearman":
        return metrics.get("auprc", 0.0) + metrics.get("spearman", 0.0)
    if metric_name == "tuned_f1":
        return metrics.get("tuned_f1", metrics.get("f1", 0.0))
    if metric_name == "tuned_mcc":
        return metrics.get("tuned_mcc", metrics.get("mcc", 0.0))
    return metrics.get(metric_name, 0.0)


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_dataset(cfg: dict, split: str) -> NeutralizationDataset:
    return NeutralizationDataset(
        cfg["data"]["pair_table"],
        split,
        cfg["data"]["max_antibody_len"],
        cfg["data"]["max_antigen_len"],
        esm_antigen_path=cfg["data"].get("esm_antigen_path"),
        igbert_antibody_path=cfg["data"].get("igbert_antibody_path"),
        cdr_mask_path=cfg["data"].get("cdr_mask_path"),
    )


def sync_feature_dims(cfg: dict) -> None:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    if model_cfg.get("use_esm_antigen") and data_cfg.get("esm_antigen_path"):
        payload = torch.load(data_cfg["esm_antigen_path"], map_location="cpu")
        if isinstance(payload, dict) and "embedding_dim" in payload:
            model_cfg["esm_dim"] = int(payload["embedding_dim"])
    if model_cfg.get("use_igbert_antibody") and data_cfg.get("igbert_antibody_path"):
        payload = torch.load(data_cfg["igbert_antibody_path"], map_location="cpu")
        if isinstance(payload, dict) and "embedding_dim" in payload:
            model_cfg["igbert_dim"] = int(payload["embedding_dim"])


if __name__ == "__main__":
    main()
