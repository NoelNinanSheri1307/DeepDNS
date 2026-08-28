"""
Deterministic label parsing and metadata extraction for CIC-Bell DNS Exfiltration datasets.

Guarantees label consistency without inferring labels from numerical feature values.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union
from dataclasses import dataclass


VALID_MODALITIES = {"audio", "compressed", "exe", "image", "text", "video"}
VALID_INTENSITIES = {"benign", "standard", "light", "heavy"}


@dataclass(frozen=True)
class CaptureMetadata:
    """Immutable metadata container for a dataset capture session."""
    label: int
    label_name: str
    attack_modality: Optional[str]
    intensity: str
    capture_id: str
    source_file: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_name": self.label_name,
            "attack_modality": self.attack_modality,
            "intensity": self.intensity,
            "capture_id": self.capture_id,
            "source_file": self.source_file,
        }


def parse_cic_bell_label(file_path: Union[str, Path]) -> CaptureMetadata:
    """
    Deterministically parses binary label, attack modality, intensity, and capture_id
    from the standardized directory structure and filename of a CIC-Bell file.

    Args:
        file_path: Relative or absolute path to a CIC-Bell CSV file.

    Returns:
        CaptureMetadata object with verified ground-truth fields.

    Raises:
        ValueError: If the file path structure does not conform to known patterns.
    """
    path_obj = Path(file_path)
    filename = path_obj.name
    path_str = str(path_obj).replace("\\", "/")
    path_lower = path_str.lower()
    fname_lower = filename.lower()

    # Determine binary label
    # In CIC-Bell: if 'benign' appears in filename or parent subfolder 'benign', it is benign.
    # If in 'attacks' subfolder without benign, it is an attack.
    is_benign = "benign" in fname_lower or "/benign/" in path_lower or path_lower.endswith("/benign")
    is_attack = "attack" in path_lower and not is_benign

    if not is_benign and not is_attack:
        # Check if root directory is Benign
        if "benign" in path_lower:
            is_benign = True
        else:
            raise ValueError(f"Ambiguous label classification for path: '{file_path}'")

    label = 0 if is_benign else 1
    label_name = "benign" if is_benign else "attack"

    # Determine Intensity
    if "light" in path_lower or "light" in fname_lower:
        intensity = "light"
    elif "heavy" in path_lower or "heavy" in fname_lower:
        intensity = "heavy"
    elif is_benign:
        intensity = "standard"
    else:
        intensity = "standard"

    # Determine Modality
    modality = None
    if label == 1:
        for mod in VALID_MODALITIES:
            if f"_{mod}." in fname_lower or f"-{mod}." in fname_lower or f"_{mod}_" in fname_lower:
                modality = mod
                break
        if modality is None:
            # Check without separator
            for mod in VALID_MODALITIES:
                if mod in fname_lower:
                    modality = mod
                    break
        if modality is None:
            raise ValueError(f"Unable to resolve attack modality for attack file: '{file_path}'")

    # Generate unique canonical capture_id
    # e.g., 'stateless_features-heavy_audio.pcap.csv' -> 'heavy_audio'
    clean_name = filename
    for prefix in ["stateless_features-", "stateful_features-", "stateless_features-_", "stateful_features-_"]:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break
            
    for suffix in [".pcap.csv", ".csv"]:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
            break

    # Strip leading/trailing underscores
    capture_id = clean_name.strip("_")
    
    # Prepend folder intensity if capture_id doesn't include it and it's ambiguous
    if capture_id in ["1", "2", "3"]:
        if "heavy" in path_lower:
            capture_id = f"benign_heavy_{capture_id}"
        else:
            capture_id = f"benign_{capture_id}"

    return CaptureMetadata(
        label=label,
        label_name=label_name,
        attack_modality=modality,
        intensity=intensity,
        capture_id=capture_id,
        source_file=filename,
    )
