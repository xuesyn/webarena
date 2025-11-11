#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_phase_classifier.py

Distributed training entry point for the AbdomenAtlas phase classification
task using pre-sampled 2D patches produced by ``prepare_phase_patches.py``.

Key features
------------
* Supports sharded ``.npy`` patch datasets with memmap loading to minimise
  host RAM usage.
* Wraps a frozen (by default) DINOv3 teacher backbone and attaches a
  multi-layer multi-head attention classification head.
* Provides gradient accumulation, bf16 autocast, deterministic seeding and
  DDP-friendly logging/metric aggregation.
* Designed for highly imbalanced class distributions by optionally
  providing class weights to the CrossEntropy loss and reporting per-class
  accuracies.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def _set_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    torch.backends.cudnn.benchmark = True


_set_env()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log_on_main(message: str) -> None:
    if is_main_process():
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ddp_setup() -> Tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def ddp_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# NumPy -> torch utilities (ensure contiguous, writable views)
# ---------------------------------------------------------------------------
def _np_to_torch_f32(arr: np.ndarray) -> torch.Tensor:
    out = np.asarray(arr, dtype=np.float32)
    if (not out.flags["C_CONTIGUOUS"]) or (not out.flags["WRITEABLE"]):
        out = np.ascontiguousarray(out).copy()
    return torch.from_numpy(out)


def _np_to_torch_long(arr: np.ndarray) -> torch.Tensor:
    out = np.asarray(arr, dtype=np.int64)
    if (not out.flags["C_CONTIGUOUS"]) or (not out.flags["WRITEABLE"]):
        out = np.ascontiguousarray(out).copy()
    return torch.from_numpy(out)


# ---------------------------------------------------------------------------
# Dataset: sharded patch storage
# ---------------------------------------------------------------------------
def _list_shard_pairs(folder: Path) -> List[Tuple[Path, Path]]:
    """Return sorted (X, y) shard pairs."""

    xs = sorted(folder.glob("X_*.npy"))
    ys = sorted(folder.glob("y_*.npy"))
    if xs and ys:
        if len(xs) != len(ys):
            raise RuntimeError(f"Shard mismatch in {folder}: {len(xs)} X vs {len(ys)} y files")
        return list(zip(xs, ys))

    x_single = folder / "X.npy"
    y_single = folder / "y.npy"
    if x_single.exists() and y_single.exists():
        return [(x_single, y_single)]
    raise FileNotFoundError(f"No shard files found in {folder}")


class PhasePatchDataset(Dataset):
    """Memmap-backed dataset that iterates over (patch, label) pairs."""

    def __init__(self, shard_pairs: Sequence[Tuple[Path, Path]], memmap: bool = True) -> None:
        self.shard_pairs = list(shard_pairs)
        if not self.shard_pairs:
            raise ValueError("PhasePatchDataset requires at least one shard")

        self._xs: List[np.ndarray] = []
        self._ys: List[np.ndarray] = []
        self._offsets: List[Tuple[int, int]] = []
        count = 0
        for xp, yp in self.shard_pairs:
            X = np.load(xp, mmap_mode="r" if memmap else None)
            y = np.load(yp, mmap_mode="r" if memmap else None)
            if X.shape[0] != y.shape[0]:
                raise RuntimeError(f"Shard length mismatch: {xp.name} vs {yp.name}")
            if y.ndim != 1:
                raise ValueError(f"Classification shards must store 1D labels; got {yp} with shape {y.shape}")
            self._xs.append(X)
            self._ys.append(y)
            self._offsets.append((count, count + X.shape[0]))
            count += X.shape[0]

        self.num_samples = count
        first_patch = self._xs[0][0]
        if first_patch.ndim == 2:
            self.channels, self.height, self.width = 1, first_patch.shape[0], first_patch.shape[1]
        elif first_patch.ndim == 3:
            self.channels, self.height, self.width = first_patch.shape
        else:
            raise ValueError(f"Unexpected patch shape: {first_patch.shape}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_samples

    def _locate(self, index: int) -> Tuple[int, int]:
        lo, hi = 0, len(self._offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, end = self._offsets[mid]
            if index < start:
                hi = mid - 1
            elif index >= end:
                lo = mid + 1
            else:
                return mid, index - start
        raise IndexError(index)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_idx, inner_idx = self._locate(index)
        x_np = self._xs[shard_idx][inner_idx]
        if x_np.ndim == 2:
            x_np = x_np[None, ...]
        y_np = self._ys[shard_idx][inner_idx]
        return _np_to_torch_f32(x_np), _np_to_torch_long(y_np)


# ---------------------------------------------------------------------------
# Backbone adapter (DINOv3 teacher)
# ---------------------------------------------------------------------------
class DINOv3Backbone(nn.Module):
    """Loads a DINOv3 ViT teacher checkpoint and returns patch-grid features."""

    def __init__(self, repo: str, weights: str, arch: str = "dinov3_vitb16") -> None:
        super().__init__()
        self.embed_dim = 768
        self.patch_stride = 16
        self.model: Optional[nn.Module] = None
        self._load_model(repo, weights, arch)

    def _load_model(self, repo: str, weights: str, arch: str) -> None:
        try:
            import sys

            repo = repo.rstrip("/")
            if repo and repo not in sys.path:
                sys.path.insert(0, repo)
            from dinov3.models.vision_transformer import vit_base as _vit_base  # type: ignore

            model = _vit_base()
            state = torch.load(weights, map_location="cpu")
            if isinstance(state, dict) and "teacher" in state:
                state = state["teacher"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing and is_main_process():
                print(f"[dinov3] missing keys: {len(missing)}")
            if unexpected and is_main_process():
                print(f"[dinov3] unexpected keys: {len(unexpected)}")
            self.model = model.eval()
            self.embed_dim = getattr(model, "embed_dim", self.embed_dim)
            return
        except Exception as exc:  # pragma: no cover - fallback path
            if is_main_process():
                print(f"[dinov3] local repo import failed: {exc}")

        try:
            import timm  # type: ignore

            model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, dynamic_img_size=True)
            state = torch.load(weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing and is_main_process():
                print(f"[timm] missing keys: {len(missing)}")
            if unexpected and is_main_process():
                print(f"[timm] unexpected keys: {len(unexpected)}")
            self.model = model.eval()
            self.embed_dim = getattr(model, "embed_dim", self.embed_dim)
        except Exception as exc:  # pragma: no cover - fatal path
            raise RuntimeError(f"Cannot load DINOv3 weights: {exc}") from exc

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3, ...]

        out = None
        if hasattr(self.model, "forward_features"):
            out = self.model.forward_features(x)
            if isinstance(out, dict):
                for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state", "tokens"):
                    if key in out and isinstance(out[key], torch.Tensor):
                        out = out[key]
                        break

        if isinstance(out, torch.Tensor):
            tensor = out
            if tensor.ndim == 3:
                return self._tokens_to_grid(tensor, x.shape[-2], x.shape[-1])
            if tensor.ndim == 4:
                return tensor

        if hasattr(self.model, "get_intermediate_layers"):
            try:
                layers = self.model.get_intermediate_layers(x, n=1, reshape=True)
                tensor = layers[0]
                if tensor.ndim == 4:
                    return tensor
                if tensor.ndim == 3:
                    return self._tokens_to_grid(tensor, x.shape[-2], x.shape[-1])
            except Exception:  # pragma: no cover - fallback
                pass

        raise RuntimeError("Backbone forward did not produce usable features")

    def _tokens_to_grid(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = tokens.shape
        if N == (H * W) // (self.patch_stride ** 2) + 1:
            tokens = tokens[:, 1:, :]
            N -= 1
        size = int(round(math.sqrt(N)))
        h = size
        w = N // h if h > 0 else 0
        if h * w != N:
            h = max(1, int(round(H / self.patch_stride)))
            w = max(1, int(round(W / self.patch_stride)))
            if h * w != N:
                w = max(1, N // h)
        grid = tokens.transpose(1, 2).reshape(B, C, h, w).contiguous()
        return grid


# ---------------------------------------------------------------------------
# Multi-attention classification head
# ---------------------------------------------------------------------------
def _build_2d_sincos_pos_embed(embed_dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Return (grid_h * grid_w, embed_dim) sin/cos positional embeddings."""

    def _pos_embed_from_coords(pos: np.ndarray) -> np.ndarray:
        omega = np.arange(embed_dim // 4, dtype=np.float64)
        omega /= embed_dim / 4.0
        omega = 1.0 / (10000 ** omega)
        pos = np.asarray(pos, dtype=np.float64)
        if pos.ndim == 2:
            if pos.shape[1] != 1:
                raise ValueError(
                    f"Expected single-column coordinate array, got shape {pos.shape}"
                )
            pos = pos[:, 0]
        out = np.einsum("n,d->nd", pos, omega)
        return np.concatenate((np.sin(out), np.cos(out)), axis=1)

    grid_y = np.linspace(-1.0, 1.0, grid_h, dtype=np.float64)
    grid_x = np.linspace(-1.0, 1.0, grid_w, dtype=np.float64)
    coords = np.stack(np.meshgrid(grid_y, grid_x, indexing="ij"), axis=-1).reshape(-1, 2)
    emb_y = _pos_embed_from_coords(coords[:, [0]])
    emb_x = _pos_embed_from_coords(coords[:, [1]])
    pos = np.concatenate((emb_y, emb_x), axis=1)
    if pos.shape[1] < embed_dim:
        pad = np.zeros((pos.shape[0], embed_dim - pos.shape[1]), dtype=np.float64)
        pos = np.concatenate((pos, pad), axis=1)
    return torch.from_numpy(pos.astype(np.float32, copy=False))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MultiAttentionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("MultiAttentionHead depth must be >= 1")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_cache: dict[Tuple[int, int], torch.Tensor] = {}
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, N, C)

        key = (H, W)
        if key not in self.pos_cache:
            pos = _build_2d_sincos_pos_embed(C, H, W)
            pos = pos.to(feat.device)
            self.pos_cache[key] = pos
        else:
            pos = self.pos_cache[key]
            if pos.device != feat.device:
                pos = pos.to(feat.device)
                self.pos_cache[key] = pos

        tokens = tokens + pos.unsqueeze(0)
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, tokens), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls = x[:, 0]
        return self.fc(cls)


# ---------------------------------------------------------------------------
# Phase classifier module
# ---------------------------------------------------------------------------
class PhaseClassifier(nn.Module):
    def __init__(
        self,
        repo: str,
        weights: str,
        arch: str,
        num_classes: int,
        head_depth: int,
        head_heads: int,
        head_mlp_ratio: float,
        head_dropout: float,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = DINOv3Backbone(repo, weights, arch)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            embed_dim = self.backbone(dummy).shape[1]
        self.head = MultiAttentionHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            depth=head_depth,
            num_heads=head_heads,
            mlp_ratio=head_mlp_ratio,
            dropout=head_dropout,
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)


def wrap_ddp(module: nn.Module, local_rank: int) -> DDP:
    return DDP(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class MetricState:
    loss: float = 0.0
    correct: int = 0
    total: int = 0
    per_class_correct: Optional[np.ndarray] = None
    per_class_total: Optional[np.ndarray] = None

    def update(self, logits: torch.Tensor, target: torch.Tensor, loss_value: float, num_classes: int) -> None:
        pred = logits.argmax(dim=1)
        correct = (pred == target).sum().item()
        total = target.numel()
        self.loss += loss_value
        self.correct += correct
        self.total += total
        if self.per_class_correct is None:
            self.per_class_correct = np.zeros(num_classes, dtype=np.int64)
            self.per_class_total = np.zeros(num_classes, dtype=np.int64)
        target_np = target.cpu().numpy()
        pred_np = pred.cpu().numpy()
        for c in range(num_classes):
            mask = target_np == c
            if mask.any():
                self.per_class_total[c] += int(mask.sum())
                self.per_class_correct[c] += int((pred_np[mask] == c).sum())

    def as_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_tensor = torch.tensor(self.loss, dtype=torch.float64, device=device)
        corr_tensor = torch.tensor(self.correct, dtype=torch.float64, device=device)
        tot_tensor = torch.tensor(self.total, dtype=torch.float64, device=device)
        return loss_tensor, corr_tensor, tot_tensor

    def class_tensors(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.per_class_correct is None or self.per_class_total is None:
            raise RuntimeError("Per-class statistics were not initialised")
        correct = torch.from_numpy(self.per_class_correct).to(device=device, dtype=torch.float64)
        total = torch.from_numpy(self.per_class_total).to(device=device, dtype=torch.float64)
        return correct, total


def _reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ---------------------------------------------------------------------------
# Training / evaluation loop
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_accum: int = 1,
    train: bool = True,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float, List[float]]:
    state = MetricState()
    if train:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)

    autocast_enabled = torch.cuda.is_available()
    progress = tqdm(total=len(loader), disable=not is_main_process(), leave=False)
    progress.set_description("train" if train else "valid")

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=torch.bfloat16):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            if train and grad_accum > 1:
                loss = loss / grad_accum

        if train:
            loss.backward()
            if scaler is not None:
                raise RuntimeError("GradScaler should not be passed when using manual autocast")
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        state.update(logits.detach(), labels.detach(), float(loss.item()), num_classes)

        if is_main_process():
            acc = state.correct / max(1, state.total)
            progress.set_postfix(loss=state.loss / max(1, step + 1), acc=f"{acc:.3f}")
            progress.update(1)

    if is_main_process():
        progress.close()

    loss_tensor, corr_tensor, tot_tensor = state.as_tensors(device)
    loss_tensor = _reduce_tensor(loss_tensor)
    corr_tensor = _reduce_tensor(corr_tensor)
    tot_tensor = _reduce_tensor(tot_tensor)

    class_correct, class_total = state.class_tensors(device)
    class_correct = _reduce_tensor(class_correct)
    class_total = _reduce_tensor(class_total)

    avg_loss = float(loss_tensor.item() / max(1.0, len(loader)))
    overall_acc = float((corr_tensor / torch.clamp(tot_tensor, min=1.0)).item())
    per_class = []
    for c in range(num_classes):
        denom = max(1.0, class_total[c].item())
        per_class.append(float(class_correct[c].item() / denom))

    return avg_loss, overall_acc, per_class


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best: float, args: dict) -> None:
    state = {
        "model": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best": best,
        "args": args,
    }
    torch.save(state, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Tuple[int, float]:
    state = torch.load(path, map_location="cpu")
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), float(state.get("best", -1e9))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DINOv3-backed phase classifier")
    parser.add_argument("--train-shards", required=True, help="Directory containing training X_*.npy / y_*.npy shards")
    parser.add_argument("--val-shards", default="", help="Optional validation shards directory")

    parser.add_argument("--dinov3-repo", required=True, help="Path to the local DINOv3 repository")
    parser.add_argument("--dinov3-ckpt", required=True, help="Teacher checkpoint (.pth) for the backbone")
    parser.add_argument("--dinov3-arch", default="dinov3_vitb16", help="Backbone architecture identifier (informational)")

    parser.add_argument("--num-classes", type=int, default=3, help="Number of output classes (default: 3 phases)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze DINOv3 weights (recommended)")

    parser.add_argument("--head-depth", type=int, default=6, help="Number of transformer blocks in the classification head")
    parser.add_argument("--head-heads", type=int, default=8, help="Multi-head attention heads per block")
    parser.add_argument("--head-mlp-ratio", type=float, default=4.0, help="Expansion ratio for MLP inside transformer blocks")
    parser.add_argument("--head-dropout", type=float, default=0.1, help="Dropout probability for attention/MLP layers")

    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument(
        "--ckpt-interval",
        type=int,
        default=100,
        help="Save an additional checkpoint every N epochs (default: 100)",
    )
    parser.add_argument("--batch", type=int, default=128, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--wd", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999), help="AdamW beta coefficients")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")

    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per process")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--persistent-workers", action="store_true", help="Enable persistent workers")
    parser.add_argument("--pin-memory", action="store_true", default=True, help="Pin CUDA memory for DataLoader")

    parser.add_argument("--class-weights", default="", help="Optional CSV of class weights for CrossEntropy loss")
    parser.add_argument("--workdir", required=True, help="Output directory for checkpoints/logs")
    parser.add_argument("--resume", default="", help="Resume checkpoint path")
    parser.add_argument("--seed", type=int, default=2025, help="Base random seed")

    return parser


def parse_class_weights(text: str, num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    if not text:
        return None
    values = [float(tok) for tok in text.split(",") if tok.strip()]
    if len(values) != num_classes:
        raise ValueError(f"Expected {num_classes} class weights, received {len(values)}")
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return tensor / tensor.sum() * num_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    local_rank, rank, world_size, device = ddp_setup()
    set_seed(args.seed + rank)
    if is_main_process():
        print(f"[DDP] world={world_size} rank={rank} device={device}")

    train_pairs = _list_shard_pairs(Path(args.train_shards))
    train_dataset = PhasePatchDataset(train_pairs, memmap=True)

    val_loader = None
    if args.val_shards:
        val_pairs = _list_shard_pairs(Path(args.val_shards))
        val_dataset = PhasePatchDataset(val_pairs, memmap=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        drop_last=True,
    )

    model = PhaseClassifier(
        repo=args.dinov3_repo,
        weights=args.dinov3_ckpt,
        arch=args.dinov3_arch,
        num_classes=args.num_classes,
        head_depth=args.head_depth,
        head_heads=args.head_heads,
        head_mlp_ratio=args.head_mlp_ratio,
        head_dropout=args.head_dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    model = wrap_ddp(model, local_rank)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
        betas=tuple(args.betas),
    )

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ckpt_last = workdir / "last.pth"
    ckpt_best = workdir / "best.pth"

    start_epoch = 0
    best_metric = -1e9
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            if is_main_process():
                print(f"[resume] Loading checkpoint from {resume_path}")
            start_epoch, best_metric = load_checkpoint(resume_path, model, optimizer)
            start_epoch += 1
        elif is_main_process():
            print(f"[resume] Checkpoint not found: {resume_path}")

    class_weights = parse_class_weights(args.class_weights, args.num_classes, device)

    if args.ckpt_interval <= 0:
        raise ValueError("--ckpt-interval must be a positive integer")

    for epoch in range(start_epoch, args.epochs):
        log_on_main(f"Epoch {epoch + 1}/{args.epochs}")
        train_sampler.set_epoch(epoch + 1)
        train_loss, train_acc, train_class = run_epoch(
            model,
            train_loader,
            device,
            args.num_classes,
            optimizer=optimizer,
            grad_accum=max(1, args.grad_accum),
            train=True,
            class_weights=class_weights,
        )

        message = f"[Ep{epoch + 1:03d}] train_loss {train_loss:.4f} | train_acc {train_acc:.4f}"
        message += " | train_cls " + ",".join(f"{acc:.3f}" for acc in train_class)

        monitor = train_acc
        if val_loader is not None:
            val_loss, val_acc, val_class = run_epoch(
                model,
                val_loader,
                device,
                args.num_classes,
                optimizer=None,
                train=False,
                class_weights=class_weights,
            )
            message += f" | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
            message += " | val_cls " + ",".join(f"{acc:.3f}" for acc in val_class)
            monitor = val_acc

        if is_main_process():
            print(message, flush=True)
            with open(workdir / "train_log.txt", "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
            save_checkpoint(ckpt_last, model, optimizer, epoch, best_metric, vars(args))
            if monitor > best_metric:
                best_metric = monitor
                save_checkpoint(ckpt_best, model, optimizer, epoch, best_metric, vars(args))
                print(f"  -> saved best ({best_metric:.4f})", flush=True)

            if (epoch + 1) % args.ckpt_interval == 0:
                epoch_path = workdir / f"epoch_{epoch + 1:04d}.pth"
                save_checkpoint(epoch_path, model, optimizer, epoch, best_metric, vars(args))

        ddp_barrier()

    if is_main_process():
        print("[done]", flush=True)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()

