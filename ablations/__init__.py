"""Ablation variants for impostor-based detectors."""

from ablations.impostor_asgalf import ASGALFImpostorDetector
from ablations.impostor_homotopy import HBCImpostorDetector
from ablations.impostor_potha2017 import Potha2017ImpostorDetector

__all__ = [
    "ASGALFImpostorDetector",
    "HBCImpostorDetector",
    "Potha2017ImpostorDetector",
]
