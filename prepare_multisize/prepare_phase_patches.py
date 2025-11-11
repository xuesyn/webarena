#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_phase_patches.py

Utility to pre-sample 2D CT patches for the AbdomenAtlas phase
classification task.  The script mirrors the segmentation patch
preparation pipeline but stores per-patch phase labels instead of
per-pixel masks.  Output shards are written as
```
X_00001.npy  # float16, (N, C, H, W)
y_00001.npy  # uint8,  (N,)
meta_00001.json
```
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Phase taxonomy
# ---------------------------------------------------------------------------
PHASE_ORDER_DEFAULT: Tuple[str, ...] = ("Arterial", "Venous", "Delay")


@dataclass(frozen=True)
class CaseInfo:
    case_id: str
    phase_name: str
    phase_id: int
    metadata: Mapping[str, Optional[str]]
    ct_path: Path
    label_path: Optional[Path]


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------
def _read_excel_with_pandas(path: Path, sheet_name: Optional[str] = None):
    try:
        import pandas as pd  # type: ignore
    except ImportError:  # pragma: no cover - pandas might be unavailable
        return None

    sheet_arg = sheet_name if sheet_name is not None else 0
    df = pd.read_excel(path, sheet_name=sheet_arg)
    if isinstance(df, dict):  # pragma: no cover - defensive guard
        if sheet_name and sheet_name in df:
            return df[sheet_name]
        # fall back to the first sheet when pandas returns a mapping
        try:
            first_key = next(iter(df))
        except StopIteration:  # pragma: no cover - empty workbook edge case
            raise ValueError(f"Excel file {path} does not contain any sheets")
        return df[first_key]
    return df


def _read_excel_with_openpyxl(path: Path, sheet_name: Optional[str] = None):
    try:
        from openpyxl import load_workbook  # type: ignore
    except ImportError:  # pragma: no cover - openpyxl might be unavailable
        return None

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb.active

    rows = ws.iter_rows(min_row=1, values_only=True)
    header = None
    data = []
    for row in rows:
        if header is None:
            header = [str(cell).strip() if cell is not None else "" for cell in row]
            continue
        data.append(row)
    wb.close()
    if header is None:
        raise ValueError(f"Excel file {path} appears to be empty")
    return header, data


def load_phase_metadata(
    excel_path: Path,
    phase_order: Sequence[str],
    id_column: str = "BDMAP",
    phase_column: str = "phase label",
    extra_columns: Sequence[str] = ("spacing", "shape"),
    sheet_name: Optional[str] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    """Return mapping case_id -> metadata with phase label.

    The phase string is stored under the key ``phase_column``.
    Extra columns are preserved when present; missing columns are
    recorded with ``None``.
    """

    excel_path = excel_path.expanduser().resolve()
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel metadata file not found: {excel_path}")

    df = _read_excel_with_pandas(excel_path, sheet_name)
    records: Dict[str, Dict[str, Optional[str]]] = {}
    if df is not None:
        df_columns = {str(col).strip(): col for col in df.columns}
        if id_column not in df_columns or phase_column not in df_columns:
            available = ", ".join(df_columns.keys())
            raise KeyError(
                f"Excel columns must include '{id_column}' and '{phase_column}'. Found: {available}"
            )
        for _, row in df.iterrows():
            case_id = str(row[df_columns[id_column]]).strip()
            if not case_id:
                continue
            phase_value = row[df_columns[phase_column]]
            if isinstance(phase_value, str):
                phase_value = phase_value.strip()
            if not phase_value:
                continue
            rec: Dict[str, Optional[str]] = {
                phase_column: str(phase_value),
            }
            for col in extra_columns:
                if col in df_columns:
                    val = row[df_columns[col]]
                    rec[col] = None if val is None or (isinstance(val, float) and math.isnan(val)) else str(val)
            records[case_id] = rec
        return records

    header_data = _read_excel_with_openpyxl(excel_path, sheet_name)
    if header_data is None:
        raise RuntimeError(
            "Neither pandas nor openpyxl is available to parse Excel files. "
            "Install one of them to proceed."
        )
    header, data_rows = header_data
    header_map = {name: idx for idx, name in enumerate(header)}
    if id_column not in header_map or phase_column not in header_map:
        available = ", ".join(header_map.keys())
        raise KeyError(
            f"Excel columns must include '{id_column}' and '{phase_column}'. Found: {available}"
        )

    for row in data_rows:
        idx_case = header_map[id_column]
        idx_phase = header_map[phase_column]
        if idx_case >= len(row) or idx_phase >= len(row):
            continue
        case_val = row[idx_case]
        phase_val = row[idx_phase]
        case_id = str(case_val).strip() if case_val is not None else ""
        if not case_id:
            continue
        phase_value = str(phase_val).strip() if phase_val is not None else ""
        if not phase_value:
            continue
        rec = {phase_column: phase_value}
        for col in extra_columns:
            idx_extra = header_map.get(col)
            rec[col] = str(row[idx_extra]).strip() if idx_extra is not None and idx_extra < len(row) and row[idx_extra] is not None else None
        records[case_id] = rec
    return records


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------
def hu_window_to01(img: np.ndarray, center: float, width: float) -> np.ndarray:
    lo = center - width / 2.0
    hi = center + width / 2.0
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def stack_slices(volume: np.ndarray, z: int, stack: int) -> np.ndarray:
    if stack == 1:
        return volume[..., z][None, ...]
    half = stack // 2
    chans = []
    for dz in range(-half, half + 1):
        zz = int(np.clip(z + dz, 0, volume.shape[2] - 1))
        chans.append(volume[..., zz])
    return np.stack(chans, axis=0)


def clamp_center_to_window(cy: int, cx: int, th: int, tw: int, H: int, W: int) -> Tuple[int, int]:
    y0 = int(np.clip(cy - th // 2, 0, max(0, H - th)))
    x0 = int(np.clip(cx - tw // 2, 0, max(0, W - tw)))
    return y0, x0


def random_crop_coords(H: int, W: int, th: int, tw: int, rng: np.random.RandomState) -> Tuple[int, int]:
    if H <= th and W <= tw:
        return 0, 0
    y0 = 0 if H <= th else rng.randint(0, H - th + 1)
    x0 = 0 if W <= tw else rng.randint(0, W - tw + 1)
    return int(y0), int(x0)


def choose_crop_size(crop_sizes: Sequence[int], rng: np.random.RandomState) -> int:
    if len(crop_sizes) == 1:
        return int(crop_sizes[0])
    weights = np.full(len(crop_sizes), 1.0 / len(crop_sizes), dtype=np.float64)
    weights /= weights.sum()
    return int(rng.choice(crop_sizes, p=weights))


def resize_patch(x: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected patch shape (C,H,W), got {x.shape}")
    if np.any(~np.isfinite(x)):
        raise ValueError("Patch contains NaN or Inf values")
    hwc = x.transpose(1, 2, 0)
    resized = cv2.resize(hwc, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return resized.transpose(2, 0, 1).astype(np.float32, copy=False)


def pad_or_resize_label(mask: np.ndarray, target_hw: Tuple[int, int], dilate_radius: int) -> np.ndarray:
    if dilate_radius > 0 and mask.size > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
        mask = cv2.dilate(mask.astype(np.uint8), ker, iterations=1).astype(bool)
    if mask.shape == target_hw:
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def choose_z_list(nz: int, max_slices: int) -> List[int]:
    zs = list(range(nz))
    if max_slices < nz:
        step = max(1, math.ceil(nz / max_slices))
        zs = zs[::step]
    return zs


def pick_from_mask(mask: np.ndarray, rng: np.random.RandomState, k: int = 8) -> List[Tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    idx = rng.randint(0, len(ys), size=min(k, len(ys)))
    return [(int(ys[i]), int(xs[i])) for i in idx]


# ---------------------------------------------------------------------------
# Per-case worker
# ---------------------------------------------------------------------------
@dataclass
class WorkerArgs:
    case: CaseInfo
    slice_stack: int
    crop_sizes: Sequence[int]
    patches_per_slice: int
    patch_multiplier: float
    fg_sample_prob: float
    fg_dilate_radius: int
    target_hw: Tuple[int, int]
    hu_center: float
    hu_width: float
    max_slices_per_vol: int
    seed: int
    split_name: str
    max_patches: Optional[int] = None


def _process_case(args: WorkerArgs):
    rng = np.random.RandomState(args.seed)

    case = args.case
    img = nib.load(str(case.ct_path))
    volume = np.asarray(img.dataobj, dtype=np.float32)

    label_volume = None
    if case.label_path and case.label_path.exists():
        lbl_img = nib.load(str(case.label_path))
        label_volume = np.asarray(lbl_img.dataobj)
    H, W, Z = volume.shape

    if label_volume is not None and label_volume.shape != volume.shape:
        raise ValueError(
            f"Label volume shape {label_volume.shape} does not match CT volume {volume.shape} for case {case.case_id}"
        )

    mask_volume = np.ones((H, W, Z), dtype=bool)
    has_mask_guidance = False
    if label_volume is not None:
        mask_volume = label_volume > 0
        has_mask_guidance = bool(mask_volume.any())

    zs = choose_z_list(Z, max(1, min(args.max_slices_per_vol, Z)))

    per_slice = max(1, int(math.ceil(args.patches_per_slice * args.patch_multiplier)))
    target_hw = args.target_hw

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta: List[Dict[str, object]] = []

    split_name = args.split_name

    max_patches = args.max_patches if args.max_patches is not None else None
    for z in zs:
        if max_patches is not None and len(X_list) >= max_patches:
            break
        mask2d = mask_volume[..., z]
        use_fg_guidance = has_mask_guidance and args.fg_sample_prob > 0.0
        x_stack = stack_slices(volume, z, args.slice_stack)

        for _ in range(per_slice):
            if max_patches is not None and len(X_list) >= max_patches:
                break
            attempts = 0
            accepted = False
            while attempts < 8 and not accepted:
                attempts += 1
                size = choose_crop_size(args.crop_sizes, rng)
                th = tw = int(size)

                if use_fg_guidance and mask2d.any() and rng.rand() < args.fg_sample_prob:
                    ctx_mask = mask2d
                    if args.fg_dilate_radius > 0:
                        ker = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * args.fg_dilate_radius + 1, 2 * args.fg_dilate_radius + 1)
                        )
                        ctx_mask = cv2.dilate(mask2d.astype(np.uint8), ker, iterations=1).astype(bool)
                    centers = pick_from_mask(ctx_mask, rng, k=8)
                    if centers:
                        cy, cx = centers[rng.randint(0, len(centers))]
                    else:
                        y0r, x0r = random_crop_coords(H, W, th, tw, rng)
                        cy, cx = y0r + th // 2, x0r + tw // 2
                else:
                    y0r, x0r = random_crop_coords(H, W, th, tw, rng)
                    cy, cx = y0r + th // 2, x0r + tw // 2

                y0, x0 = clamp_center_to_window(cy, cx, th, tw, H, W)
                y1, x1 = y0 + th, x0 + tw
                x_patch = x_stack[:, y0:y1, x0:x1]

                if x_patch.shape[1] != th or x_patch.shape[2] != tw:
                    continue
                accepted = True

            if not accepted:
                continue

            patch = resize_patch(x_patch, target_hw)
            patch = hu_window_to01(patch, args.hu_center, args.hu_width).astype(np.float16, copy=False)

            X_list.append(patch)
            y_list.append(case.phase_id)
            meta.append(
                {
                    "case_id": case.case_id,
                    "phase": case.phase_name,
                    "phase_id": case.phase_id,
                    "split": split_name,
                    "z": int(z),
                    "center_y": int(cy),
                    "center_x": int(cx),
                    "y0": int(y0),
                    "x0": int(x0),
                    "h": int(th),
                    "w": int(tw),
                    **{k: v for k, v in case.metadata.items() if k not in {"phase label"}},
                }
            )

    if not X_list:
        X = np.empty((0, args.slice_stack, target_hw[0], target_hw[1]), dtype=np.float16)
        y = np.empty((0,), dtype=np.uint8)
    else:
        X = np.stack(X_list, axis=0)
        y = np.asarray(y_list, dtype=np.uint8)

    return X, y, meta


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def parse_crop_sizes(expr: Optional[str], default: Iterable[int] = (256,)) -> List[int]:
    if expr is None:
        return sorted({int(v) for v in default})
    values = [int(tok) for tok in expr.split(",") if tok.strip()]
    if not values:
        raise ValueError("crop-sizes must contain at least one positive integer")
    if any(v <= 0 for v in values):
        raise ValueError("crop-sizes must be positive integers")
    return sorted(set(values))


def parse_class_multiplier(expr: Optional[str], phase_order: Sequence[str]) -> Dict[str, float]:
    multipliers = {phase.lower(): 1.0 for phase in phase_order}
    if not expr:
        return multipliers
    expr = expr.replace("：", ":")
    for token in expr.split(","):
        token = token.strip()
        if not token or ":" not in token:
            continue
        key, val = token.split(":", 1)
        key = key.strip().lower()
        try:
            value = float(val.strip())
        except ValueError:
            continue
        if key in multipliers and value > 0:
            multipliers[key] = value
    return multipliers


def read_case_list(path: Optional[str]) -> Optional[Sequence[str]]:
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Case list file not found: {p}")
    items: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_case_infos(
    dataset_root: Path,
    metadata: Mapping[str, Mapping[str, Optional[str]]],
    phase_order: Sequence[str],
    cases_filter: Optional[Sequence[str]],
    phase_column: str,
) -> List[CaseInfo]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    phase_to_id = {name.lower(): idx for idx, name in enumerate(phase_order)}

    case_infos: List[CaseInfo] = []
    wanted = set(case.lower() for case in cases_filter) if cases_filter else None

    for case_dir in sorted(dataset_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        if wanted and case_id.lower() not in wanted:
            continue
        case_meta = metadata.get(case_id)
        if not case_meta:
            print(f"[warn] Case {case_id} missing metadata; skipping")
            continue
        phase_value = str(case_meta.get(phase_column, "")).strip()
        if not phase_value:
            print(f"[warn] Case {case_id} has empty phase label; skipping")
            continue
        phase_key = phase_value.lower()
        if phase_key not in phase_to_id:
            print(f"[warn] Case {case_id} has unknown phase '{phase_value}'; skipping")
            continue
        phase_id = phase_to_id[phase_key]
        ct_path = case_dir / "ct.nii.gz"
        if not ct_path.exists():
            print(f"[warn] Missing CT volume for case {case_id}; skipping")
            continue
        label_path = case_dir / "combined_labels.nii.gz"
        info = CaseInfo(
            case_id=case_id,
            phase_name=phase_order[phase_id],
            phase_id=phase_id,
            metadata=case_meta,
            ct_path=ct_path,
            label_path=label_path if label_path.exists() else None,
        )
        case_infos.append(info)

    return case_infos


def split_cases(
    case_infos: Sequence[CaseInfo],
    phase_order: Sequence[str],
    test_ratio: float,
    seed: int,
) -> Tuple[List[CaseInfo], List[CaseInfo]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test-ratio must be in the interval (0, 1)")

    rng = np.random.RandomState(seed)

    phase_to_cases: Dict[int, List[CaseInfo]] = {idx: [] for idx in range(len(phase_order))}
    for case in case_infos:
        phase_to_cases.setdefault(case.phase_id, []).append(case)

    train: List[CaseInfo] = []
    test: List[CaseInfo] = []

    for phase_id, phase_name in enumerate(phase_order):
        items = phase_to_cases.get(phase_id, [])
        if not items:
            raise RuntimeError(f"No cases found for phase '{phase_name}'")
        items = items[:]  # copy before shuffling
        rng.shuffle(items)

        desired = int(math.ceil(len(items) * test_ratio))
        if desired <= 0:
            desired = 1
        if len(items) > 1:
            desired = min(len(items) - 1, desired)
        else:
            desired = 1

        test_items = items[:desired]
        train_items = items[desired:]

        if not train_items and len(items) > 1:
            # Edge case: rounding consumed all items; move one back to train
            train_items = [test_items.pop()]

        test.extend(test_items)
        train.extend(train_items)

    # Keep deterministic ordering by case_id for reproducibility of downstream processing
    train.sort(key=lambda c: c.case_id)
    test.sort(key=lambda c: c.case_id)

    return train, test


def save_label_mapping(outdir: Path, phase_order: Sequence[str]) -> None:
    mapping = {int(idx): name for idx, name in enumerate(phase_order)}
    with (outdir / "phase_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def summarize_case_split(name: str, cases: Sequence[CaseInfo], phase_order: Sequence[str]) -> None:
    counts = np.zeros(len(phase_order), dtype=np.int64)
    for case in cases:
        counts[case.phase_id] += 1
    total = int(counts.sum())
    print(f"===== {name.upper()} CASE SPLIT =====")
    print(f"Total cases: {total}")
    for idx, phase_name in enumerate(phase_order):
        print(f" - {phase_name}: {int(counts[idx])} cases")


def write_case_split_list(outdir: Path, split_name: str, cases: Sequence[CaseInfo]) -> None:
    path = outdir / f"{split_name}_cases.txt"
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(f"{case.case_id}\n")


def scan_existing_outputs(split_dir: Path) -> Tuple[int, Set[str]]:
    """Return next shard id and completed case IDs from existing metadata."""

    completed: Set[str] = set()
    max_id = 0
    for meta_path in sorted(split_dir.glob("meta_*.json")):
        stem = meta_path.stem
        try:
            _, idx_str = stem.split("_", 1)
            max_id = max(max_id, int(idx_str))
        except Exception:
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        for rec in payload.get("meta", []):
            case_id = rec.get("case_id") if isinstance(rec, dict) else None
            if case_id:
                completed.add(str(case_id))
    next_shard = max_id + 1 if max_id > 0 else 1
    return next_shard, completed


def load_existing_class_counts(split_dir: Path, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for y_path in sorted(split_dir.glob("y_*.npy")):
        try:
            arr = np.load(y_path, mmap_mode="r")
        except Exception:
            continue
        try:
            bc = np.bincount(np.asarray(arr).ravel(), minlength=num_classes)
            counts[: len(bc)] += bc[:num_classes]
        finally:
            base = getattr(arr, "base", None)
            if hasattr(base, "close"):
                try:
                    base.close()  # type: ignore[call-arg]
                except Exception:
                    pass
    return counts


def estimate_case_patch_count(case: CaseInfo, args: argparse.Namespace, multiplier: float) -> int:
    img = nib.load(str(case.ct_path))
    _, _, Z = img.shape
    max_slices = max(1, min(args.max_slices_per_vol, Z))
    zs = choose_z_list(Z, max_slices)
    per_slice = max(1, int(math.ceil(args.patches_per_slice * multiplier)))
    return len(zs) * per_slice


def run_sampling_for_split(
    split_name: str,
    cases: Sequence[CaseInfo],
    outdir: Path,
    args: argparse.Namespace,
    phase_order: Sequence[str],
    class_multipliers: Mapping[str, float],
    crop_sizes: Sequence[int],
    seed_offset: int,
    resume: bool = False,
    dry_run_patches: int = 0,
) -> np.ndarray:
    if not cases:
        print(f"[warn] No cases assigned to {split_name} split; skipping patch generation")
        return np.zeros(len(phase_order), dtype=np.int64)

    if not dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    resume = bool(resume)
    dry_run = dry_run_patches > 0

    completed_cases: Set[str] = set()
    shard_id = 1
    class_counts = np.zeros(len(phase_order), dtype=np.int64)

    if resume and not dry_run:
        shard_id, completed_cases = scan_existing_outputs(outdir)
        if completed_cases:
            print(
                f"[resume] Detected {len(completed_cases)} cases already saved for {split_name} split; "
                "they will be skipped"
            )
        if shard_id > 1:
            class_counts += load_existing_class_counts(outdir, len(phase_order))
            print(f"[resume] Next shard id for {split_name} will start at {shard_id:05d}")
    else:
        existing = list(outdir.glob("X_*.npy")) + list(outdir.glob("y_*.npy")) + list(outdir.glob("meta_*.json"))
        if existing and not dry_run:
            raise RuntimeError(
                f"Output directory {outdir} already contains shards. Use --resume to append or remove old files first."
            )

    jobs: List[WorkerArgs] = []
    for idx, case in enumerate(cases):
        if case.case_id in completed_cases:
            continue
        mult = class_multipliers.get(case.phase_name.lower(), 1.0)
        jobs.append(
            WorkerArgs(
                case=case,
                slice_stack=args.slice_stack,
                crop_sizes=crop_sizes,
                patches_per_slice=args.patches_per_slice,
                patch_multiplier=mult,
                fg_sample_prob=args.fg_sample_prob,
                fg_dilate_radius=max(0, args.fg_dilate_radius),
                target_hw=(args.target_size, args.target_size),
                hu_center=args.hu_center,
                hu_width=args.hu_width,
                max_slices_per_vol=args.max_slices_per_vol,
                seed=args.seed + seed_offset + idx,
                split_name=split_name,
                max_patches=None,
            )
        )

    if dry_run:
        if not jobs:
            print(f"[dry-run] No cases available for {split_name} split after filtering; nothing to estimate")
            return class_counts
        total_expected = 0
        sample_X: List[np.ndarray] = []
        sample_y: List[int] = []
        sample_meta: List[Dict[str, object]] = []

        for job in jobs:
            total_expected += estimate_case_patch_count(job.case, args, job.patch_multiplier)
            remaining = dry_run_patches - len(sample_X)
            if remaining <= 0:
                continue
            job_dry = WorkerArgs(
                case=job.case,
                slice_stack=job.slice_stack,
                crop_sizes=job.crop_sizes,
                patches_per_slice=job.patches_per_slice,
                patch_multiplier=job.patch_multiplier,
                fg_sample_prob=job.fg_sample_prob,
                fg_dilate_radius=job.fg_dilate_radius,
                target_hw=job.target_hw,
                hu_center=job.hu_center,
                hu_width=job.hu_width,
                max_slices_per_vol=job.max_slices_per_vol,
                seed=job.seed,
                split_name=job.split_name,
                max_patches=remaining,
            )
            X, y, meta = _process_case(job_dry)
            if X.shape[0] == 0:
                continue
            take = min(remaining, X.shape[0])
            sample_X.extend(list(X[:take]))
            sample_y.extend(list(y[:take]))
            sample_meta.extend(meta[:take])

        if not sample_X:
            print(f"[dry-run] No patches could be sampled for {split_name} split; unable to estimate storage")
            return class_counts

        import io

        stacked_x = np.stack(sample_X, axis=0)
        buf_x = io.BytesIO()
        np.save(buf_x, stacked_x)
        x_bytes = buf_x.tell()

        stacked_y = np.asarray(sample_y, dtype=np.uint8)
        buf_y = io.BytesIO()
        np.save(buf_y, stacked_y)
        y_bytes = buf_y.tell()

        meta_payload = {"count": len(sample_meta), "meta": sample_meta}
        meta_bytes = len(json.dumps(meta_payload, ensure_ascii=False).encode("utf-8"))

        per_patch_x = x_bytes / len(sample_X)
        per_patch_y = y_bytes / len(sample_y)
        per_patch_meta = meta_bytes / len(sample_meta)

        total_patches = total_expected
        est_total_bytes = total_patches * (per_patch_x + per_patch_y + per_patch_meta)
        est_gb = est_total_bytes / (1024 ** 3)

        print(f"===== {split_name.upper()} DRY RUN ESTIMATE =====")
        print(f"Sampled patches: {len(sample_X)}")
        print(f"Estimated total patches: {total_patches}")
        print(
            f"Average bytes per patch -> X: {per_patch_x:.1f}, y: {per_patch_y:.1f}, meta: {per_patch_meta:.1f}"
        )
        print(f"Estimated total storage: {est_total_bytes:.0f} bytes (~{est_gb:.2f} GiB)")

        return class_counts

    bufX: List[np.ndarray] = []
    bufY: List[np.ndarray] = []
    meta_all: List[Dict[str, object]] = []

    def flush(force: bool = False) -> None:
        nonlocal shard_id, bufX, bufY, meta_all
        n_in_buf = sum(arr.shape[0] for arr in bufX)
        if n_in_buf == 0:
            return
        if not force and n_in_buf < args.shard_size:
            return

        parts_x: List[np.ndarray] = []
        parts_y: List[np.ndarray] = []
        to_take = n_in_buf if force else args.shard_size
        while bufX and to_take > 0:
            x0, y0 = bufX[0], bufY[0]
            if x0.shape[0] <= to_take:
                parts_x.append(bufX.pop(0))
                parts_y.append(bufY.pop(0))
                to_take -= x0.shape[0]
            else:
                take = to_take
                parts_x.append(x0[:take])
                parts_y.append(y0[:take])
                bufX[0] = x0[take:]
                bufY[0] = y0[take:]
                to_take = 0

        if not parts_x:
            return

        Xout = np.concatenate(parts_x, axis=0)
        Yout = np.concatenate(parts_y, axis=0)
        for cls in range(len(phase_order)):
            class_counts[cls] += int((Yout == cls).sum())

        m_take = Xout.shape[0]
        m_this = meta_all[:m_take]
        meta_all[:] = meta_all[m_take:]

        xpath = outdir / f"X_{shard_id:05d}.npy"
        ypath = outdir / f"y_{shard_id:05d}.npy"
        mpath = outdir / f"meta_{shard_id:05d}.json"
        np.save(xpath, Xout)
        np.save(ypath, Yout.astype(np.uint8, copy=False))
        with mpath.open("w", encoding="utf-8") as f:
            json.dump({"count": int(Xout.shape[0]), "meta": m_this}, f, ensure_ascii=False)
        print(f"[write] {split_name} shard {shard_id} with {Xout.shape[0]} patches")
        shard_id += 1

    if jobs:
        if args.num_workers > 0:
            from multiprocessing import Pool

            with Pool(processes=args.num_workers) as pool:
                for X, y, meta in pool.imap_unordered(_process_case, jobs):
                    if X.shape[0] == 0:
                        continue
                    bufX.append(X)
                    bufY.append(y)
                    meta_all.extend(meta)
                    flush(force=False)
        else:
            for job in jobs:
                X, y, meta = _process_case(job)
                if X.shape[0] == 0:
                    continue
                bufX.append(X)
                bufY.append(y)
                meta_all.extend(meta)
                flush(force=False)
    else:
        if completed_cases:
            print(f"[resume] No remaining cases to process for {split_name} split.")

    flush(force=True)

    total = int(class_counts.sum())
    print(f"===== {split_name.upper()} PATCH SUMMARY =====")
    if total == 0:
        print("No patches were generated.")
    else:
        for idx, name in enumerate(phase_order):
            count = int(class_counts[idx])
            ratio = count / total if total > 0 else 0.0
            print(f" - {name}: {count} patches ({ratio:.4%})")
        print(f"Total patches: {total}")

    return class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AbdomenAtlas phase classification patches")
    parser.add_argument("--dataset-root", required=True, help="Root directory of AbdomenAtlas cases")
    parser.add_argument("--metadata-xlsx", required=True, help="Excel file containing phase labels")
    parser.add_argument("--outdir", required=True, help="Output directory for shards")
    parser.add_argument("--slice-stack", type=int, default=3, help="Number of adjacent slices to stack (must be odd)")
    parser.add_argument("--crop-sizes", type=str, default=None, help="Comma separated crop sizes (e.g. '128,192,256')")
    parser.add_argument("--patches-per-slice", type=int, default=3, help="Base number of patches per slice")
    parser.add_argument(
        "--class-patch-multipliers",
        type=str,
        default=None,
        help="Comma separated overrides like 'Arterial:1.0,Venous:1.0,Delay:4.0'",
    )
    parser.add_argument(
        "--fg-sample-prob",
        type=float,
        default=0.0,
        help="Probability of sampling patch centers from foreground masks (only used when masks are available)",
    )
    parser.add_argument(
        "--fg-dilate-radius",
        type=int,
        default=0,
        help="Radius (in pixels) for dilating foreground masks when guidance is enabled",
    )
    parser.add_argument("--target-size", type=int, default=256, help="Final square patch size")
    parser.add_argument("--hu-center", type=float, default=60.0, help="HU window center")
    parser.add_argument("--hu-width", type=float, default=400.0, help="HU window width")
    parser.add_argument("--max-slices-per-vol", type=int, default=999999, help="Max slices sampled per volume")
    parser.add_argument("--shard-size", type=int, default=2048, help="Number of patches per output shard")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of worker processes (0 = single process)")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed")
    parser.add_argument(
        "--phase-order",
        type=str,
        default=",".join(PHASE_ORDER_DEFAULT),
        help="Comma separated list defining phase order",
    )
    parser.add_argument("--case-list", type=str, default=None, help="Optional text file containing case IDs to include")
    parser.add_argument("--excel-sheet", type=str, default=None, help="Optional sheet name inside the Excel workbook")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of cases reserved for the test split (default 0.2 for 80/20 split)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume patch generation by appending to existing shards instead of overwriting",
    )
    parser.add_argument(
        "--dry-run-patches",
        type=int,
        default=0,
        help="Sample this many patches per split to estimate storage without writing shards",
    )

    args = parser.parse_args()

    if args.slice_stack < 1 or args.slice_stack % 2 == 0:
        raise ValueError("slice-stack must be an odd integer >= 1")

    if args.dry_run_patches < 0:
        raise ValueError("dry-run-patches must be >= 0")

    phase_order = tuple([tok.strip() for tok in args.phase_order.split(",") if tok.strip()])
    if not phase_order:
        raise ValueError("phase-order must contain at least one class")

    crop_sizes = parse_crop_sizes(args.crop_sizes, default=(args.target_size,))

    metadata = load_phase_metadata(
        Path(args.metadata_xlsx),
        phase_order=phase_order,
        sheet_name=args.excel_sheet,
    )
    case_list = read_case_list(args.case_list)

    case_infos = build_case_infos(
        Path(args.dataset_root),
        metadata=metadata,
        phase_order=phase_order,
        cases_filter=case_list,
        phase_column="phase label",
    )
    if not case_infos:
        raise RuntimeError("No cases matched the provided filters and metadata")

    class_multipliers = parse_class_multiplier(args.class_patch_multipliers, phase_order)

    train_cases, test_cases = split_cases(case_infos, phase_order, args.test_ratio, args.seed)

    outdir = Path(args.outdir).expanduser().resolve()
    dry_run = args.dry_run_patches > 0

    if not dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
        save_label_mapping(outdir, phase_order)

        summarize_case_split("train", train_cases, phase_order)
        summarize_case_split("test", test_cases, phase_order)

        write_case_split_list(outdir, "train", train_cases)
        write_case_split_list(outdir, "test", test_cases)
    else:
        summarize_case_split("train", train_cases, phase_order)
        summarize_case_split("test", test_cases, phase_order)

    train_counts = run_sampling_for_split(
        split_name="train",
        cases=train_cases,
        outdir=outdir / "train",
        args=args,
        phase_order=phase_order,
        class_multipliers=class_multipliers,
        crop_sizes=crop_sizes,
        seed_offset=0,
        resume=args.resume,
        dry_run_patches=args.dry_run_patches,
    )

    test_counts = run_sampling_for_split(
        split_name="test",
        cases=test_cases,
        outdir=outdir / "test",
        args=args,
        phase_order=phase_order,
        class_multipliers=class_multipliers,
        crop_sizes=crop_sizes,
        seed_offset=len(train_cases),
        resume=args.resume,
        dry_run_patches=args.dry_run_patches,
    )

    total_counts = train_counts + test_counts
    total = int(total_counts.sum())
    print("===== OVERALL PATCH SUMMARY =====")
    if total == 0:
        print("No patches were generated across splits.")
    else:
        for idx, name in enumerate(phase_order):
            count = int(total_counts[idx])
            ratio = count / total if total > 0 else 0.0
            print(f" - {name}: {count} patches ({ratio:.4%})")
        print(f"Total patches (train+test): {total}")

    if dry_run:
        print("[dry-run] Estimation complete. No shards were written.")


if __name__ == "__main__":
    main()
