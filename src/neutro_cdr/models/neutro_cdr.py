from __future__ import annotations

import math

import torch
from torch import nn

from neutro_cdr.data.tokenizer import VOCAB


class NeutroCDR(nn.Module):
    def __init__(
        self,
        vocab_size: int = len(VOCAB),
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_cdr_mask: bool = True,
        use_physchem_bias: bool = True,
        use_surface_bias: bool = True,
        use_mutation_bias: bool = True,
        use_mutation_embedding: bool = True,
        use_mutation_context: bool = True,
        use_esm_antigen: bool = False,
        esm_dim: int = 320,
        use_igbert_antibody: bool = False,
        igbert_dim: int = 1024,
        use_attention_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.use_cdr_mask = use_cdr_mask
        self.use_physchem_bias = use_physchem_bias
        self.use_surface_bias = use_surface_bias
        self.use_mutation_bias = use_mutation_bias
        self.use_mutation_embedding = use_mutation_embedding
        self.use_mutation_context = use_mutation_context
        self.use_esm_antigen = use_esm_antigen
        self.use_igbert_antibody = use_igbert_antibody
        self.use_attention_fusion = use_attention_fusion
        self.physchem_scale = nn.Parameter(torch.tensor(0.5))
        self.surface_scale = nn.Parameter(torch.tensor(0.5))
        self.mutation_scale = nn.Parameter(torch.tensor(0.5))

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.mutation_projection = nn.Linear(1, hidden_dim)
        self.esm_antigen_projection = nn.Linear(esm_dim, hidden_dim)
        self.igbert_antibody_projection = nn.Sequential(
            nn.Linear(igbert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.ab_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ag_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.local_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_attention_fusion:
            self.fusion_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.cls_head = nn.Linear(hidden_dim, 1)
        self.reg_head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ab_ids = batch["antibody_ids"]
        ag_ids = batch["antigen_ids"]
        ab_mask = batch["antibody_mask"].bool()
        ag_mask = batch["antigen_mask"].bool()
        cdr_mask = batch["cdr_mask"].bool() & ab_mask
        if not self.use_cdr_mask:
            cdr_mask = ab_mask

        ab = self.embedding(ab_ids)
        mutation_weight = batch.get("mutation_weight")
        ag = self.embedding(ag_ids)
        ag_esm = batch.get("ag_esm")
        if ag_esm is not None and self.use_esm_antigen:
            ag = ag + self.esm_antigen_projection(ag_esm.float())
        if mutation_weight is not None and self.use_mutation_embedding:
            ag = ag + self.mutation_projection(mutation_weight.unsqueeze(-1).float())
        ab = self.ab_encoder(ab, src_key_padding_mask=~ab_mask)
        ag = self.ag_encoder(ag, src_key_padding_mask=~ag_mask)

        q = self.query(ab)
        k = self.key(ag)
        v = self.value(ag)
        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.size(-1))

        if self.use_physchem_bias:
            logits = logits + self.physchem_scale * batch["physchem_bias"]
        if self.use_surface_bias:
            surface_bias = batch["ab_surface"].unsqueeze(-1) * batch["ag_surface"].unsqueeze(1)
            logits = logits + self.surface_scale * surface_bias
        if mutation_weight is not None and self.use_mutation_bias:
            mutation_bias = mutation_weight.unsqueeze(1).expand(-1, ab_ids.size(1), -1)
            logits = logits + self.mutation_scale * mutation_bias

        pair_mask = cdr_mask.unsqueeze(-1) & ag_mask.unsqueeze(1)
        logits = logits.masked_fill(~pair_mask, -1e4)
        attn = torch.softmax(logits, dim=-1)
        local_tokens = torch.matmul(attn, v)
        local_tokens = self.local_proj(local_tokens)
        local = masked_mean(local_tokens, cdr_mask)

        ab_pool = masked_mean(ab, ab_mask)
        ab_igbert = batch.get("ab_igbert")
        if ab_igbert is not None and self.use_igbert_antibody:
            ab_pool = ab_pool + self.igbert_antibody_projection(ab_igbert.float())
        ag_pool = masked_mean(ag, ag_mask)
        global_context = self.global_proj(torch.cat([ab_pool, ag_pool, ab_pool * ag_pool, (ab_pool - ag_pool).abs()], dim=-1))

        mutation_context = (
            mutation_masked_mean(ag, mutation_weight, ag_mask)
            if mutation_weight is not None and self.use_mutation_context
            else torch.zeros_like(ag_pool)
        )
        if self.use_attention_fusion:
            fusion_tokens = torch.stack([local, global_context, mutation_context], dim=1)
            fusion_weights = torch.softmax(self.fusion_attention(fusion_tokens).squeeze(-1), dim=-1)
            fused = self.fusion_norm((fusion_tokens * fusion_weights.unsqueeze(-1)).sum(dim=1))
        else:
            fusion_weights = None
            fused = self.fusion(torch.cat([local, global_context], dim=-1))
        return {
            "logits": self.cls_head(fused).squeeze(-1),
            "log_ic50": self.reg_head(fused).squeeze(-1),
            "attention": attn,
            "fusion_weights": fusion_weights,
        }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (values * weights).sum(dim=1) / denom


def mutation_masked_mean(values: torch.Tensor, mutation_weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mutation_weight.float() * mask.float()
    denom = weights.sum(dim=1, keepdim=True)
    pooled = (values * weights.unsqueeze(-1)).sum(dim=1) / denom.clamp_min(1.0)
    return torch.where(denom > 0, pooled, torch.zeros_like(pooled))
