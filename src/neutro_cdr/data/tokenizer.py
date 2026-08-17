from __future__ import annotations

import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SPECIALS = ["<pad>", "<unk>", "<sep>"]
VOCAB = {token: idx for idx, token in enumerate(SPECIALS + list(AMINO_ACIDS))}
PAD_ID = VOCAB["<pad>"]
UNK_ID = VOCAB["<unk>"]
SEP_ID = VOCAB["<sep>"]


def encode_sequence(sequence: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [VOCAB.get(aa, UNK_ID) for aa in str(sequence).upper()]
    ids = ids[:max_len]
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    if pad > 0:
        ids.extend([PAD_ID] * pad)
        mask.extend([0] * pad)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)
