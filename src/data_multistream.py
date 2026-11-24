"""Leak-free multi-stream X-ray dataset and DataLoader builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}
MASK_DIR_NAMES = {"ground_truth", "masks"}
NORMAL_DIR_NAMES = {"good", "normal"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _is_mask_file(path: Path, root_dir: Path) -> bool:
    """Return True if a path belongs to a mask directory or uses a *_mask suffix."""
    try:
        rel_parts = path.relative_to(root_dir).parts[:-1]
    except ValueError:
        rel_parts = path.parts[:-1]
    if any(part.lower() in MASK_DIR_NAMES for part in rel_parts):
        return True
    return path.stem.lower().endswith("_mask")


def _load_dicom_as_rgb(path: Path) -> Image.Image:
    """Load a DICOM image with 1-99 percentile intensity scaling as RGB PIL Image."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)

    # Handle multi-frame or multi-channel arrays gracefully.
    if arr.ndim == 3 and arr.shape[-1] not in (1, 3):
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    p1, p99 = np.percentile(arr, [1.0, 99.0])
    if p99 > p1:
        arr = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0)
    else:
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr_u8).convert("RGB")


def _load_image_as_rgb(path: Path) -> Image.Image:
    """Load a standard image or DICOM file as an RGB PIL Image."""
    if path.suffix.lower() == ".dcm":
        return _load_dicom_as_rgb(path)
    with Image.open(path) as img:
        return img.convert("RGB")


def _find_mask_path(image_path: Path, root_dir: Path) -> Path | None:
    """Search sibling mask directories or *_mask.* files for a matching ground-truth mask."""
    stem = image_path.stem
    mask_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    # 1. Sibling *_mask.* next to the image.
    for ext in mask_exts:
        candidate = image_path.with_name(f"{stem}_mask{ext}")
        if candidate.is_file():
            return candidate

    # 2. Sibling ground_truth/ or masks/ directory at split root or image parent's parent.
    try:
        rel = image_path.relative_to(root_dir)
        rel_parent = rel.parent
    except ValueError:
        rel_parent = Path(".")

    candidate_roots = [root_dir, root_dir.parent, image_path.parent.parent]
    for base in candidate_roots:
        for mask_dir_name in ("ground_truth", "masks"):
            mask_base = base / mask_dir_name
            if not mask_base.is_dir():
                continue
            search_dirs = [mask_base / rel_parent, mask_base / image_path.parent.name, mask_base]
            for sdir in search_dirs:
                for name in (stem, f"{stem}_mask"):
                    for ext in mask_exts:
                        candidate = sdir / f"{name}{ext}"
                        if candidate.is_file():
                            return candidate
    return None


class XRaySplitDataset(Dataset):
    """Dataset for a single X-ray split (train_normal, val_normal, or test)."""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 448,
        split: str = "train_normal",
        patch_size: int | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = int(image_size)
        self.split = split
        self.patch_size = int(patch_size) if patch_size is not None else None

        if self.patch_size is not None:
            if self.patch_size < 1:
                raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")
            if self.image_size % self.patch_size != 0:
                raise ValueError(
                    f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})."
                )

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Split directory does not exist: {self.root_dir}")

        all_files = sorted(
            p
            for p in self.root_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_EXTENSIONS
            and not _is_mask_file(p, self.root_dir)
        )
        self.paths: list[str] = [str(p) for p in all_files]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.mask_resize = transforms.Resize(
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def _infer_is_anomalous(self, path: Path) -> int:
        if self.split != "test":
            return 0
        try:
            rel_parts = path.relative_to(self.root_dir).parts[:-1]
        except ValueError:
            rel_parts = ()
        top_or_parent = {path.parent.name.lower()}
        if rel_parts:
            top_or_parent.add(rel_parts[0].lower())
        if top_or_parent & NORMAL_DIR_NAMES:
            return 0
        return 1

    def _load_mask(self, image_path: Path) -> torch.Tensor:
        mask_path = _find_mask_path(image_path, self.root_dir)
        if mask_path is None:
            return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        with Image.open(mask_path) as m_img:
            m_gray = m_img.convert("L")
            m_resized = self.mask_resize(m_gray)
            m_arr = np.asarray(m_resized, dtype=np.float32) / 255.0
        binary = (m_arr > 0.5).astype(np.float32)
        return torch.from_numpy(binary).unsqueeze(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        path_str = self.paths[idx]
        path = Path(path_str)

        rgb_img = _load_image_as_rgb(path)
        image_tensor = self.image_transform(rgb_img)
        mask_tensor = self._load_mask(path)
        is_anomalous = self._infer_is_anomalous(path)

        return image_tensor, mask_tensor, is_anomalous, path_str


def verify_zero_split_overlap(
    train_paths: Iterable[str | Path],
    val_paths: Iterable[str | Path],
    test_paths: Iterable[str | Path],
) -> dict[str, int]:
    """Ensure train, val, and test splits share zero resolved paths or file IDs (stems)."""
    splits: dict[str, list[Path]] = {
        "train_normal": [Path(p) for p in train_paths],
        "val_normal": [Path(p) for p in val_paths],
        "test": [Path(p) for p in test_paths],
    }

    resolved_sets = {k: {str(p.resolve()) for p in paths} for k, paths in splits.items()}
    stem_sets = {k: {p.stem for p in paths} for k, paths in splits.items()}

    pairs = [
        ("train_normal", "val_normal"),
        ("train_normal", "test"),
        ("val_normal", "test"),
    ]

    overlap_messages: list[str] = []
    for a, b in pairs:
        path_overlap = sorted(resolved_sets[a] & resolved_sets[b])
        if path_overlap:
            overlap_messages.append(
                f"Resolved path overlap between {a} and {b} ({len(path_overlap)}): {path_overlap[:10]}"
            )
        stem_overlap = sorted(stem_sets[a] & stem_sets[b])
        if stem_overlap:
            overlap_messages.append(
                f"File ID (stem) overlap between {a} and {b} ({len(stem_overlap)}): {stem_overlap[:10]}"
            )

    if overlap_messages:
        raise ValueError("Data leakage detected across splits:\n" + "\n".join(overlap_messages))

    return {
        "train_count": len(splits["train_normal"]),
        "val_count": len(splits["val_normal"]),
        "test_count": len(splits["test"]),
        "path_overlaps": 0,
        "id_overlaps": 0,
    }


def build_multistream_dataloaders(
    cfg: dict[str, Any],
    patch_size: int | None = None,
) -> dict[str, Any]:
    """Build train_normal, val_normal, and test DataLoaders with leak-free verification."""
    data_cfg = cfg["data"]
    exp_cfg = cfg.get("experiment") or {}
    backbone_cfg = cfg.get("backbone") or {}

    raw_image_size = exp_cfg.get("image_size")
    if raw_image_size is None:
        raw_image_size = backbone_cfg.get("image_size", 448)
    image_size = int(raw_image_size)

    batch_size = int(data_cfg.get("batch_size", 12))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    train_ds = XRaySplitDataset(
        data_cfg["train_normal_dir"],
        image_size=image_size,
        split="train_normal",
        patch_size=patch_size,
    )
    val_ds = XRaySplitDataset(
        data_cfg["val_normal_dir"],
        image_size=image_size,
        split="val_normal",
        patch_size=patch_size,
    )
    test_ds = XRaySplitDataset(
        data_cfg["test_dir"],
        image_size=image_size,
        split="test",
        patch_size=patch_size,
    )

    split_audit = verify_zero_split_overlap(train_ds.paths, val_ds.paths, test_ds.paths)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train_normal": train_loader,
        "val_normal": val_loader,
        "test": test_loader,
        "split_audit": split_audit,
    }
