"""Datasets.

CleanDepthDataset: clean image + metric depth + shared (V, A) draws.
PairedDataset:     pre-synthesised hazy/clean pairs (B0 = RESIDE-OTS's own hazy images).

Both are indexed by a "global" index  g = epoch * N + i  produced by EpochSampler.
Crop, flip and the (V, A) draw are seeded by (seed, i, epoch), so every arm trained
with the same seed sees exactly the same crops and the same transmission maps, and a
single DataLoader serves the whole run (no worker re-spawn per epoch, which is slow
on Windows).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class EpochSampler(Sampler):
    """Yields epoch * n + i for a fresh seeded permutation of range(n) in every epoch."""

    def __init__(self, n, n_epochs, seed):
        self.n, self.n_epochs, self.seed = n, n_epochs, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for e in range(self.n_epochs):
            for i in torch.randperm(self.n, generator=g).tolist():
                yield e * self.n + i

    def __len__(self):
        return self.n * self.n_epochs


def _to_tensor(img):
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def _crop_params(rng, H, W, c):
    y = int(rng.integers(0, H - c + 1)); x = int(rng.integers(0, W - c + 1))
    return y, x, bool(rng.random() < 0.5)


class CleanDepthDataset(Dataset):
    def __init__(self, clean_dir, depth_dir, params_npz, crop=256, seed=0, n_train=0):
        p = np.load(params_npz)
        names = [str(n) for n in p["names"]]
        order = np.argsort(names)
        if n_train:
            order = order[:n_train]
        self.names = [names[i] for i in order]
        self.V, self.A = p["V"][order], p["A"][order]
        self.K = self.V.shape[1]
        lookup = {f.stem: f for f in Path(clean_dir).iterdir() if f.suffix.lower() in IMG_EXT}
        missing = [n for n in self.names if n not in lookup]
        assert not missing, f"{len(missing)} clean images listed in {params_npz} are missing, e.g. {missing[:3]}"
        self.files = [lookup[n] for n in self.names]
        self.depth_dir = Path(depth_dir)
        self.crop, self.seed = crop, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, g):
        epoch, i = divmod(int(g), len(self.files))
        rng = np.random.default_rng([self.seed, i, epoch])
        img = Image.open(self.files[i]).convert("RGB")
        d = np.load(self.depth_dir / (self.names[i] + ".npy")).astype(np.float32)
        W, H = img.size
        assert d.shape == (H, W), f"depth {d.shape} does not match image {(H, W)} for {self.names[i]}"
        c = self.crop
        if min(H, W) < c:
            s = c / min(H, W)
            img = img.resize((int(np.ceil(W * s)), int(np.ceil(H * s))), Image.BICUBIC)
            d = np.asarray(Image.fromarray(d).resize(img.size, Image.BILINEAR))
            W, H = img.size
        J = _to_tensor(img)
        D = torch.from_numpy(np.ascontiguousarray(d))[None]
        y, x, flip = _crop_params(rng, H, W, c)
        J = J[:, y:y + c, x:x + c]; D = D[:, y:y + c, x:x + c]
        if flip:
            J = J.flip(-1); D = D.flip(-1)
        k = int(rng.integers(0, self.K))
        return {"J": J.contiguous(), "depth": D.contiguous(), "V": torch.tensor(float(self.V[i, k])),
                "A": torch.from_numpy(self.A[i, k].astype(np.float32).copy()), "idx": i}


def _num(name):
    m = re.match(r"(\d+)", name)
    return int(m.group(1)) if m else None


def pair_files(hazy_dir, gt_dir=None, hazy_key="hazy", gt_key="GT"):
    """Pair hazy/GT files by leading integer id (O-HAZE, NH-HAZE, Dense-Haze, RESIDE).

    Two folders: every image in each folder is used. One folder: files are split by
    `hazy_key` / `gt_key` in the file name (NH-HAZE: 01_hazy.png, 01_GT.png).
    """
    hd = Path(hazy_dir); gd = Path(gt_dir) if gt_dir else hd
    hz = [f for f in hd.iterdir() if f.suffix.lower() in IMG_EXT and (gt_dir or hazy_key.lower() in f.name.lower())]
    gts = [f for f in gd.iterdir() if f.suffix.lower() in IMG_EXT and (gt_dir or gt_key.lower() in f.name.lower())]
    gmap = {_num(f.name): f for f in gts}
    return [(h, gmap[_num(h.name)]) for h in sorted(hz) if _num(h.name) in gmap]


class PairedDataset(Dataset):
    """B0: RESIDE-6K OTS hazy images vs their clean images (same crops as CleanDepthDataset)."""

    def __init__(self, pairs, crop=256, seed=0, n_train=0):
        self.pairs = pairs[:n_train] if n_train else pairs
        self.crop, self.seed = crop, seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, g):
        epoch, i = divmod(int(g), len(self.pairs))
        rng = np.random.default_rng([self.seed, i, epoch])
        h, gt = self.pairs[i]
        H_ = Image.open(h).convert("RGB"); G = Image.open(gt).convert("RGB")
        if H_.size != G.size:
            H_ = H_.resize(G.size, Image.BICUBIC)
        W, Hh = G.size
        c = self.crop
        if min(Hh, W) < c:
            s = c / min(Hh, W); sz = (int(np.ceil(W * s)), int(np.ceil(Hh * s)))
            H_ = H_.resize(sz, Image.BICUBIC); G = G.resize(sz, Image.BICUBIC); W, Hh = sz
        y, x, flip = _crop_params(rng, Hh, W, c)
        a = _to_tensor(H_)[:, y:y + c, x:x + c]; b = _to_tensor(G)[:, y:y + c, x:x + c]
        if flip:
            a = a.flip(-1); b = b.flip(-1)
        return {"hazy": a.contiguous(), "J": b.contiguous(), "idx": i}


# --------------------------------------------------------------------------
# Real paired test sets, resized once and cached (decoding 13-megapixel JPEGs is slow)
# --------------------------------------------------------------------------
TEST_SETS = {
    # name: (hazy_dir, gt_dir or None if both in one folder, expected pairs)
    "OHAZE": ("data/test/OHAZE/hazy", "data/test/OHAZE/GT", 45),
    "NHHAZE": ("data/test/NHHAZE", None, 55),
    "DENSE": ("data/test/DENSE/hazy", "data/test/DENSE/GT", 55),
}


def _load_resized(p, long_side):
    img = Image.open(p).convert("RGB")
    W, H = img.size
    s = long_side / max(W, H) if long_side else 1.0
    if s < 1:
        img = img.resize((round(W * s), round(H * s)), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def load_test_set(name, long_side=1024, cache_dir="data/cache", allow_partial=False):
    """Returns [(file_name, hazy_uint8 HxWx3, gt_uint8 HxWx3)], bicubic-resized to `long_side`."""
    hz, gt, expected = TEST_SETS[name]
    pairs = pair_files(hz, gt)
    assert allow_partial or len(pairs) == expected, f"{name}: found {len(pairs)} pairs, expected {expected} (check {hz})"
    cache = Path(cache_dir) / f"{name}_{long_side}_{len(pairs)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        if names == [h.name for h, _ in pairs]:
            return [(n, z[f"h{i}"], z[f"g{i}"]) for i, n in enumerate(names)]
    out, store = [], {}
    for i, (h, g) in enumerate(pairs):
        x, y = _load_resized(h, long_side), _load_resized(g, long_side)
        if x.shape != y.shape:
            x = np.asarray(Image.fromarray(x).resize((y.shape[1], y.shape[0]), Image.BICUBIC), dtype=np.uint8)
        out.append((h.name, x, y)); store[f"h{i}"] = x; store[f"g{i}"] = y
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array([n for n, _, _ in out]), **store)
    return out
