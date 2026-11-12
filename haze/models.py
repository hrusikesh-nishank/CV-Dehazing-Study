"""One registry for every dehazer used in the project.

Every model is wrapped as `Dehazer`: input and output are RGB in [0, 1], any size
(reflect-padded internally to the architecture's required multiple). This removes
the most common silent bug in cross-paper comparisons: feeding [0,1] images to
a network trained on [-1,1] (DehazeFormer/gUNet) or vice versa (DEA-Net).

  name              source (official code in third_party/)     native range  pad
  dehazeformer-t/s/b/m   IDKiro/DehazeFormer (TIP 2023)          [-1, 1]       32
  gunet-t/s              IDKiro/gUNet (arXiv 2022)               [-1, 1]       16
  ffanet                 FFA-Net (AAAI 2020), via gUNet baselines [-1, 1]       4
  deanet                 DEA-Net (TIP 2024), inference Backbone   [0, 1]        4
  dcp                    Dark Channel Prior (He et al., CVPR 2009) – classical, no weights

Run `python setup_third_party.py` once to fetch the official model code.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TP = ROOT / "third_party"


# --------------------------------------------------------------------------
# loading official code without package-name clashes (both repos use `models`)
# --------------------------------------------------------------------------
def _timm_compat():
    """The official repos import `timm.models.layers`, deprecated in timm 1.x. Alias it to
    `timm.layers` if a future timm removes it."""
    try:
        import timm.models.layers  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name == "timm":
            raise ModuleNotFoundError("timm is required: pip install -r requirements.txt") from e
        import timm.layers as tl
        sys.modules["timm.models.layers"] = tl


def _load_file(alias, path):
    if alias in sys.modules:
        return sys.modules[alias]
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing - run `python setup_third_party.py`")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_gunet():
    alias = "tp_gunet"
    if alias in sys.modules:
        return sys.modules[alias]
    path = TP / "gUNet" / "models" / "gunet.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python setup_third_party.py`")
    src = path.read_text(encoding="utf-8").replace("from .norm_layer import *", "")
    mod = types.ModuleType(alias)
    mod.__file__ = str(path)
    sys.modules[alias] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def _load_deanet():
    alias = "tp_deanet"
    if alias in sys.modules:
        return sys.modules[alias]
    pkg = TP / "DEA-Net" / "code" / "model"
    if not pkg.exists():
        raise FileNotFoundError(f"{pkg} missing - run `python setup_third_party.py`")
    spec = importlib.util.spec_from_file_location(alias, pkg / "__init__.py",
                                                  submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_net(name):
    if name != "dcp":
        _timm_compat()
    if name.startswith("dehazeformer-"):
        m = _load_file("tp_dehazeformer", TP / "DehazeFormer" / "models" / "dehazeformer.py")
        return getattr(m, "dehazeformer_" + name.split("-")[1])(), "pm1", 32
    if name.startswith("gunet-"):
        m = _load_gunet()
        return getattr(m, "gunet_" + name.split("-")[1])(), "pm1", 16
    if name == "ffanet":
        m = _load_file("tp_ffanet", TP / "gUNet" / "models" / "baselines" / "ffanet" / "network.py")
        return m.FFA(gps=3, blocks=19), "pm1", 4
    if name == "deanet":
        m = _load_deanet()
        return m.Backbone(), "01", 4
    if name == "dcp":
        from .dcp import DarkChannelPrior
        return DarkChannelPrior(), "01", 1
    raise KeyError(f"unknown model {name}")


MODELS = ["dehazeformer-t", "dehazeformer-s", "dehazeformer-b", "dehazeformer-m",
          "gunet-t", "gunet-s", "ffanet", "deanet", "dcp"]


class Dehazer(nn.Module):
    """[0,1] in -> [0,1] out wrapper with internal reflect padding."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.net, self.range, self.mult = _build_net(name)

    def forward(self, x01):
        H, W = x01.shape[-2:]
        ph, pw = (-H) % self.mult, (-W) % self.mult
        if ph or pw:
            x01 = F.pad(x01, (0, pw, 0, ph), mode="reflect")
        x = x01 * 2 - 1 if self.range == "pm1" else x01
        y = self.net(x)
        y = (y + 1) / 2 if self.range == "pm1" else y
        return y[..., :H, :W]


def build(name) -> Dehazer:
    return Dehazer(name)


def load_weights(model: Dehazer, path, strict=True):
    """Loads official checkpoints (DehazeFormer {'state_dict': {'module.x': ..}},
    DEA-Net plain state_dict) and our own (Dehazer.net.state_dict())."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}
    res = model.net.load_state_dict(sd, strict=strict)
    return res


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# inference helpers (full image, or tiled for large models / images)
# --------------------------------------------------------------------------
@torch.no_grad()
def infer(model, x01, tile=0, overlap=64, amp=True):
    """x01: (1,3,H,W) on the model's device, values in [0,1]. tile=0 -> whole image at once.

    Robustness: on CUDA out-of-memory it retries with 768-px, then 512-px tiles (64-px
    overlap, averaged); if half precision produces non-finite values it re-runs in float32.
    DCP is never tiled (its airlight estimate is global).
    """
    if model.name == "dcp":
        tile = 0
    for attempt_tile in [tile] + [t for t in (768, 512) if tile == 0 or t < tile]:
        oom = False
        try:
            out = _infer(model, x01, attempt_tile, overlap, amp)
            if not torch.isfinite(out).all():
                out = _infer(model, x01, attempt_tile, overlap, False)
            return out.nan_to_num(0.0).clamp(0, 1)
        except torch.cuda.OutOfMemoryError:
            oom = True                       # retry outside the except block so the traceback
        if oom:                              # (which pins the failed tensors) is released first
            torch.cuda.empty_cache()
    raise RuntimeError(f"out of GPU memory even with 512-px tiles (image {tuple(x01.shape[-2:])})")


def _infer(model, x01, tile, overlap, amp):
    dev_type = x01.device.type
    use_amp = amp and dev_type == "cuda" and model.name != "dcp"
    if tile <= 0 or max(x01.shape[-2:]) <= tile:
        with torch.autocast(dev_type, dtype=torch.float16, enabled=use_amp):
            return model(x01).float()
    _, C, H, W = x01.shape
    out = torch.zeros_like(x01); wsum = torch.zeros_like(x01[:, :1])
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step)); xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    win = torch.ones(1, 1, min(tile, H), min(tile, W), device=x01.device)
    for y in ys:
        for x in xs:
            patch = x01[..., y:y + tile, x:x + tile]
            with torch.autocast(dev_type, dtype=torch.float16, enabled=use_amp):
                p = model(patch).float()
            out[..., y:y + tile, x:x + tile] += p * win
            wsum[..., y:y + tile, x:x + tile] += win
    return out / wsum
