"""Ablation variants for impostor-based detectors."""

from ablations.impostor_asgalf import ASGALFImpostorDetector
from ablations.impostor_bdi import BDIImpostorDetector
from ablations.impostor_homotopy import HBCImpostorDetector
from ablations.impostor_potha2017 import Potha2017ImpostorDetector
from ablations.impostor_permutation_calibrated import (
    PermutationCalibratedImpostorDetector,
)

__all__ = [
    "ASGALFImpostorDetector",
    "BDIImpostorDetector",
    "HBCImpostorDetector",
    "Potha2017ImpostorDetector",
    "PermutationCalibratedImpostorDetector",
]
