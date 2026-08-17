from __future__ import annotations

from functools import lru_cache

import torch
from abnumber import Chain


def cdr_mask_from_segments(heavy: str, light: str, max_len: int, segments: dict[str, str]) -> torch.Tensor:
    mask = torch.zeros(max_len, dtype=torch.bool)
    offset_light = len(str(heavy)) + 1
    for name, segment in segments.items():
        if not isinstance(segment, str) or not segment:
            continue
        if name.startswith("H_"):
            start = str(heavy).find(segment)
            offset = 0
        else:
            start = str(light).find(segment)
            offset = offset_light
        if start >= 0:
            left = offset + start
            right = min(offset + start + len(segment), max_len)
            if left < max_len:
                mask[left:right] = True
    return mask[:max_len]


def standard_cdr_mask(heavy: str, light: str, max_len: int) -> torch.Tensor:
 
    mask = torch.zeros(max_len, dtype=torch.bool)
    heavy = str(heavy)
    light = str(light)
    _project_numbered_cdrs(mask, heavy, offset=0, expected_chain="H")
    _project_numbered_cdrs(mask, light, offset=len(heavy) + 1, expected_chain="L")
    return mask


def _project_numbered_cdrs(
    mask: torch.Tensor,
    sequence: str,
    offset: int,
    expected_chain: str,
) -> None:
    if not sequence or offset >= mask.numel():
        return

    numbered_sequence, cdr_flags, chain_type = _number_with_anarcii(sequence)
    if expected_chain == "H" and chain_type != "H":
        raise ValueError(f"Expected a heavy chain, but ANARCII classified it as {chain_type!r}")
    if expected_chain == "L" and chain_type not in {"K", "L"}:
        raise ValueError(f"Expected a light chain, but ANARCII classified it as {chain_type!r}")

    domain_start = sequence.find(numbered_sequence)
    if domain_start < 0:
        raise ValueError("ANARCII-numbered domain could not be mapped back to its input sequence")
    for raw_index, is_cdr in enumerate(cdr_flags):
        model_index = offset + domain_start + raw_index
        if model_index >= mask.numel():
            break
        if is_cdr:
            mask[model_index] = True


@lru_cache(maxsize=None)
def _number_with_anarcii(sequence: str) -> tuple[str, tuple[bool, ...], str]:
    try:
        chain = Chain(
            sequence,
            scheme="imgt",
            cdr_definition="imgt",
            use_anarcii=True,
            anarcii_args={"cpu": True, "ncpu": 1},
        )
    except Exception as exc:
        raise ValueError(f"ANARCII could not number antibody chain: {exc}") from exc

    positions = tuple(chain.positions.items())
    numbered_sequence = "".join(amino_acid for _, amino_acid in positions)
    cdr_flags = tuple(position.is_in_cdr() for position, _ in positions)
    return numbered_sequence, cdr_flags, chain.chain_type
