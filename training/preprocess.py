from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def _load_nifti(path: str | Path) -> np.ndarray:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)


def _bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (slice(None), slice(None), slice(None))
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _resize(volume: np.ndarray, target_shape: tuple[int, int, int], order: int) -> np.ndarray:
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order).astype(np.float32, copy=False)


def preprocess_volume(
    image_path: str | Path,
    mask_path: str | Path | None = None,
    target_shape: tuple[int, int, int] = (128, 128, 96),
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
    crop_margin: int = 12,
    use_brain_mask: bool = True,
    final_clip: tuple[float, float] = (-5.0, 5.0),
    final_scale: float = 5.0,
) -> np.ndarray:
    vol = _load_nifti(image_path)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

    mask = None
    if use_brain_mask and mask_path and Path(mask_path).exists():
        mask = _load_nifti(mask_path) > 0
        if mask.shape != vol.shape:
            mask = None
    if mask is None:
        mask = vol != 0

    slices = _bbox(mask, crop_margin)
    vol = vol[slices]
    mask = mask[slices]

    foreground = vol[mask]
    if foreground.size == 0:
        foreground = vol.reshape(-1)
    if foreground.size == 0:
        return np.zeros(target_shape, dtype=np.float32)

    lo, hi = np.percentile(foreground, clip_percentiles)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        vol = np.clip(vol, lo, hi)

    foreground = vol[mask] if mask.any() else vol.reshape(-1)
    mean = float(foreground.mean())
    std = float(foreground.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    vol = (vol - mean) / std

    vol = np.clip(vol, final_clip[0], final_clip[1])
    vol = vol / float(final_scale)
    vol = _resize(vol, target_shape, order=1)
    return vol.astype(np.float32, copy=False)

