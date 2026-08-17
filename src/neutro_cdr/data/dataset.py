from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from neutro_cdr.data.tokenizer import SEP_ID, encode_sequence
from neutro_cdr.features.biophysics import physicochemical_pair_bias, surface_propensity
from neutro_cdr.features.cdr import cdr_mask_from_segments, standard_cdr_mask


class NeutralizationDataset(Dataset):
    def __init__(
        self,
        table_path: str | Path,
        split: str,
        max_antibody_len: int = 280,
        max_antigen_len: int = 1280,
        esm_antigen_path: str | Path | None = None,
        igbert_antibody_path: str | Path | None = None,
        cdr_mask_path: str | Path | None = None,
    ) -> None:
        frame = pd.read_csv(table_path)
        if "split" in frame.columns:
            frame = frame[frame["split"].eq(split)].reset_index(drop=True)
        for column in ["heavy_seq", "light_seq", "antigen_seq", "variant"]:
            if column in frame.columns:
                frame[column] = frame[column].fillna("")
        self.frame = frame
        self.max_antibody_len = max_antibody_len
        self.max_antigen_len = max_antigen_len
        self.esm_antigens = load_esm_antigens(esm_antigen_path)
        self.esm_dim = infer_esm_dim(self.esm_antigens)
        self.igbert_antibodies = load_igbert_antibodies(igbert_antibody_path)
        self.igbert_dim = infer_igbert_dim(self.igbert_antibodies)
        self.cdr_masks = load_cdr_masks(cdr_mask_path)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        heavy = clean_cell(row["heavy_seq"])
        light = clean_cell(row["light_seq"])
        antigen = clean_cell(row["antigen_seq"])
        antibody = f"{heavy}X{light}"

        ab_ids, ab_mask = encode_sequence(antibody, self.max_antibody_len)
        sep_pos = min(len(heavy), self.max_antibody_len - 1)
        ab_ids[sep_pos] = SEP_ID
        ag_ids, ag_mask = encode_sequence(antigen, self.max_antigen_len)

        cache_key = antibody_cache_key(heavy, light)
        if self.cdr_masks:
            if cache_key not in self.cdr_masks:
                raise KeyError(f"CDR mask cache has no entry for antibody {cache_key}")
            cached_mask = self.cdr_masks[cache_key]
            cdr_mask = torch.zeros(self.max_antibody_len, dtype=torch.bool)
            length = min(self.max_antibody_len, cached_mask.numel())
            cdr_mask[:length] = cached_mask[:length]
        else:
            segments = {
                key: row[key]
                for key in ["H_CDR1", "H_CDR2", "H_CDR3", "L_CDR1", "L_CDR2", "L_CDR3"]
                if key in row.index and isinstance(row[key], str)
            }
            if segments:
                cdr_mask = cdr_mask_from_segments(heavy, light, self.max_antibody_len, segments)
            else:
                cdr_mask = standard_cdr_mask(heavy, light, self.max_antibody_len)

        item = {
            "antibody_ids": ab_ids,
            "antibody_mask": ab_mask,
            "antigen_ids": ag_ids,
            "antigen_mask": ag_mask,
            "cdr_mask": cdr_mask & ab_mask,
            "ab_surface": surface_propensity(antibody, self.max_antibody_len) * ab_mask.float(),
            "ag_surface": surface_propensity(antigen, self.max_antigen_len) * ag_mask.float(),
            "mutation_weight": parse_mutation_weight(row, self.max_antigen_len) * ag_mask.float(),
            "physchem_bias": physicochemical_pair_bias(antibody, antigen, self.max_antibody_len, self.max_antigen_len),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),

            "log_ic50": torch.tensor(float(row.get("log10_ic50", 0.0)), dtype=torch.float32),
        }
        if self.esm_antigens:
            item["ag_esm"] = pad_esm_embedding(self.esm_antigens[clean_cell(row["variant"])], self.max_antigen_len, self.esm_dim)
        if self.igbert_antibodies:
            key = antibody_cache_key(heavy, light)
            item["ab_igbert"] = self.igbert_antibodies[key].float()
        return item


def parse_mutation_weight(row: pd.Series, max_len: int) -> torch.Tensor:
    values = torch.zeros(max_len, dtype=torch.float32)
    if "mutation_positions" not in row.index or pd.isna(row["mutation_positions"]):
        if "mutation_vector" not in row.index or pd.isna(row["mutation_vector"]):
            return values
        raw = ast.literal_eval(str(row["mutation_vector"]))
        vector = torch.as_tensor(raw, dtype=torch.float32).flatten()
        length = min(max_len, vector.numel())
        values[:length] = vector[:length]
        return values.clamp(0.0, 1.0)
    positions = str(row["mutation_positions"]).strip()
    weights = str(row.get("mutation_weights", "")).strip()
    if not positions:
        return values
    position_parts = positions.split(";")
    weight_parts = weights.split(";") if weights else []
    for idx, pos in enumerate(position_parts):
        if not pos:
            continue
        position = int(float(pos))
        if position >= max_len:
            continue
        weight = float(weight_parts[idx]) if idx < len(weight_parts) and weight_parts[idx] else 1.0
        values[position] = weight
    return values.clamp(0.0, 1.0)


def clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def load_esm_antigens(path: str | Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu")
    variants = payload.get("variants", payload)
    return {str(key): value.float() for key, value in variants.items()}


def load_igbert_antibodies(path: str | Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu")
    antibodies = payload.get("antibodies", payload)
    return {str(key): value.float() for key, value in antibodies.items()}


def load_cdr_masks(path: str | Path | None) -> dict[str, torch.Tensor]:
    if path is None:
        return {}
    payload = torch.load(path, map_location="cpu")
    masks = payload.get("masks", payload)
    return {str(key): torch.as_tensor(value, dtype=torch.bool).flatten() for key, value in masks.items()}


def infer_esm_dim(embeddings: dict[str, torch.Tensor]) -> int:
    if not embeddings:
        return 0
    return int(next(iter(embeddings.values())).size(-1))


def infer_igbert_dim(embeddings: dict[str, torch.Tensor]) -> int:
    if not embeddings:
        return 0
    return int(next(iter(embeddings.values())).numel())


def pad_esm_embedding(embedding: torch.Tensor, max_len: int, esm_dim: int) -> torch.Tensor:
    values = torch.zeros(max_len, esm_dim, dtype=torch.float32)
    length = min(max_len, embedding.size(0))
    values[:length] = embedding[:length].float()
    return values


def antibody_cache_key(heavy: str, light: str) -> str:
    joined = f"{heavy}|{light}"
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
