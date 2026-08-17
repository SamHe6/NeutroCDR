from __future__ import annotations

import torch

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

CHARGE = {aa: 0.0 for aa in AA_ORDER}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.4})

POLAR = {aa: 0.0 for aa in AA_ORDER}
for aa in "RNDQEHKSTY":
    POLAR[aa] = 1.0

HBOND = {aa: 0.0 for aa in AA_ORDER}
for aa in "RNDQEHKSTY":
    HBOND[aa] = 1.0

VOLUME = {
    "A": 88.6,
    "C": 108.5,
    "D": 111.1,
    "E": 138.4,
    "F": 189.9,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "K": 168.6,
    "L": 166.7,
    "M": 162.9,
    "N": 114.1,
    "P": 112.7,
    "Q": 143.8,
    "R": 173.4,
    "S": 89.0,
    "T": 116.1,
    "V": 140.0,
    "W": 227.8,
    "Y": 193.6,
}


def surface_propensity(sequence: str, max_len: int) -> torch.Tensor:
    values = []
    for aa in str(sequence).upper()[:max_len]:
        polar = POLAR.get(aa, 0.0)
        charge = abs(CHARGE.get(aa, 0.0))
        hydrophobic_penalty = max(HYDROPATHY.get(aa, 0.0), 0.0) / 4.5
        values.append(max(0.0, min(1.0, 0.55 * polar + 0.35 * charge + 0.10 * (1.0 - hydrophobic_penalty))))
    if len(values) < max_len:
        values.extend([0.0] * (max_len - len(values)))
    return torch.tensor(values, dtype=torch.float32)


def physicochemical_pair_bias(ab_sequence: str, ag_sequence: str, max_ab_len: int, max_ag_len: int) -> torch.Tensor:
    ab = str(ab_sequence).upper()[:max_ab_len]
    ag = str(ag_sequence).upper()[:max_ag_len]
    ab_props = _property_matrix(ab, max_ab_len)
    ag_props = _property_matrix(ag, max_ag_len)

    hbond = torch.minimum(ab_props["hbond"].unsqueeze(1), ag_props["hbond"].unsqueeze(0))
    electrostatic = torch.clamp(-(ab_props["charge"].unsqueeze(1) * ag_props["charge"].unsqueeze(0)), min=0.0)
    hydrophobic = 1.0 - torch.clamp((ab_props["hydropathy"].unsqueeze(1) - ag_props["hydropathy"].unsqueeze(0)).abs() / 9.0, max=1.0)
    volume = 1.0 - torch.clamp((ab_props["volume"].unsqueeze(1) - ag_props["volume"].unsqueeze(0)).abs() / 170.0, max=1.0)
    valid = ab_props["valid"].unsqueeze(1) * ag_props["valid"].unsqueeze(0)
    return (0.30 * hbond + 0.30 * electrostatic + 0.25 * hydrophobic + 0.15 * volume) * valid


def _property_matrix(sequence: str, max_len: int) -> dict[str, torch.Tensor]:
    seq = str(sequence).upper()[:max_len]
    props = {
        "hbond": torch.zeros(max_len, dtype=torch.float32),
        "charge": torch.zeros(max_len, dtype=torch.float32),
        "hydropathy": torch.zeros(max_len, dtype=torch.float32),
        "volume": torch.full((max_len,), 120.0, dtype=torch.float32),
        "valid": torch.zeros(max_len, dtype=torch.float32),
    }
    for i, aa in enumerate(seq):
        props["hbond"][i] = HBOND.get(aa, 0.0)
        props["charge"][i] = CHARGE.get(aa, 0.0)
        props["hydropathy"][i] = HYDROPATHY.get(aa, 0.0)
        props["volume"][i] = VOLUME.get(aa, 120.0)
        props["valid"][i] = 1.0 if aa in AA_ORDER else 0.0
    return props
