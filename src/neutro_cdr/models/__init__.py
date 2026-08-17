from __future__ import annotations

from neutro_cdr.models.neutro_cdr import NeutroCDR

try:
    from neutro_cdr.models.neutro_cdr_clean import NeutroCDRClean
except ModuleNotFoundError:
    NeutroCDRClean = None


def build_model(model_cfg: dict) -> NeutroCDR:
    cfg = dict(model_cfg)
    architecture = cfg.pop("architecture", "full")
    if architecture == "full":
        return NeutroCDR(**cfg)
    if architecture == "clean":
        if NeutroCDRClean is None:
            raise RuntimeError("The clean model architecture is not available in this checkout.")
        return NeutroCDRClean(**cfg)
    raise ValueError(f"Unknown model architecture: {architecture}")


__all__ = ["NeutroCDR", "NeutroCDRClean", "build_model"]
