"""Tests for the model registry, DCP, tiling, metrics and the VOC mAP code.  Run: python tests/test_models.py"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from haze.models import MODELS, build, infer, load_weights  # noqa: E402


def test_all_models_odd_sizes():
    for n in MODELS:
        m = build(n).eval()
        y = infer(m, torch.rand(1, 3, 67, 93), amp=False)
        assert y.shape == (1, 3, 67, 93) and 0 <= y.min() and y.max() <= 1, n


def test_weight_roundtrip(tmp=Path("tests/_tmp.pth")):
    for n in ["dehazeformer-t", "gunet-t"]:
        a, b = build(n).eval(), build(n).eval()
        torch.save(a.net.state_dict(), tmp)
        load_weights(b, tmp)
        x = torch.rand(1, 3, 64, 64)
        assert torch.allclose(a(x), b(x), atol=1e-6), n
    tmp.unlink()


def test_same_seed_same_initial_weights():
    torch.manual_seed(3); a = build("dehazeformer-t")
    torch.manual_seed(3); b = build("dehazeformer-t")
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_official_checkpoints_if_present():
    for n, p in [("dehazeformer-t", "ckpt/outdoor/dehazeformer-t.pth"), ("dehazeformer-s", "ckpt/outdoor/dehazeformer-s.pth"),
                 ("dehazeformer-b", "ckpt/outdoor/dehazeformer-b.pth"), ("dehazeformer-m", "ckpt/outdoor/dehazeformer-m.pth"),
                 ("dehazeformer-t", "ckpt/reside6k/dehazeformer-t.pth"), ("dehazeformer-b", "ckpt/reside6k/dehazeformer-b.pth"),
                 ("deanet", "ckpt/deanet/deanet-ots.pth")]:
        if (ROOT / p).exists():
            r = load_weights(build(n), ROOT / p)
            assert not r.missing_keys and not r.unexpected_keys, (n, r)
            print("   official checkpoint OK:", p)


def test_dcp_dehazes_synthetic_haze():
    torch.manual_seed(0)
    J = torch.rand(1, 3, 1, 1) * 0.5 + torch.nn.functional.interpolate(torch.rand(1, 3, 16, 16), size=(128, 128), mode="bicubic").clamp(0, 1) * 0.5
    t = torch.linspace(0.9, 0.3, 128).view(1, 1, 128, 1).expand(1, 1, 128, 128)
    I = J * t + 0.9 * (1 - t)
    out = infer(build("dcp"), I, amp=False)
    err_in, err_out = ((I - J) ** 2).mean(), ((out - J) ** 2).mean()
    print(f"   DCP mse hazy {err_in:.4f} -> {err_out:.4f}")
    assert err_out < err_in


def test_tiled_close_to_full():
    m = build("gunet-t").eval()
    x = torch.rand(1, 3, 200, 300)
    full = infer(m, x, amp=False)
    tiled = infer(m, x, tile=128, overlap=64, amp=False)
    d = (full - tiled).abs().mean().item()
    print(f"   tiled vs full mean abs diff {d:.4f} (receptive-field edge effects only)")
    assert d < 0.05


def test_ssim_psnr_match_skimage():
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from haze.metrics import psnr, ssim
    rng = np.random.default_rng(0)
    base = np.clip(np.kron(rng.random((12, 16, 3)), np.ones((8, 8, 1))) + rng.normal(0, 0.03, (96, 128, 3)), 0, 1).astype(np.float32)
    for noise in [0.02, 0.15]:
        other = np.clip(base * 0.8 + 0.1 + rng.normal(0, noise, base.shape), 0, 1).astype(np.float32)
        ref = structural_similarity(base, other, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False)
        mine = ssim(base, other, "cpu")
        assert abs(ref - mine) < 1e-5, (ref, mine)
        assert abs(peak_signal_noise_ratio(base, other, data_range=1.0) - psnr(base, other)) < 1e-4
    print(f"   SSIM matches skimage ({mine:.5f} vs {ref:.5f})")


def test_voc_map():
    from eval_rtts import voc_map
    gts = {"a": [(0, [10, 10, 50, 50], 0), (2, [60, 60, 100, 100], 0), (2, [0, 0, 5, 5], 1)]}
    perfect = [("a", 0, 0.9, [10, 10, 50, 50]), ("a", 2, 0.8, [60, 60, 100, 100])]
    m, aps = voc_map(gts, perfect)
    assert abs(aps["person"] - 100) < 1e-6 and abs(aps["car"] - 100) < 1e-6
    # a detection on a 'difficult' object is ignored (not a false positive)
    m2, aps2 = voc_map(gts, perfect + [("a", 2, 0.95, [0, 0, 5, 5])])
    assert abs(aps2["car"] - 100) < 1e-6
    # a confident false positive halves precision at full recall
    m3, aps3 = voc_map(gts, perfect + [("a", 0, 0.99, [200, 200, 220, 220])])
    assert 40 < aps3["person"] < 60, aps3["person"]


def test_pairing_and_test_set_cache(tmp=Path("tests/_pairs")):
    from PIL import Image
    from haze import data
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "h").mkdir(parents=True); (tmp / "g").mkdir(); (tmp / "nh").mkdir()
    rng = np.random.default_rng(0)

    def img(p, hw=(60, 90)):
        Image.fromarray((rng.random((*hw, 3)) * 255).astype(np.uint8)).save(p)
    for n in ["01_outdoor_hazy.jpg", "03_outdoor_hazy.jpg"]:
        img(tmp / "h" / n)
    for n in ["01_outdoor_GT.jpg", "03_outdoor_GT.JPG"]:
        img(tmp / "g" / n)
    for n in ["01_hazy.png", "01_GT.png", "47_hazy.png", "47_GT.png"]:
        img(tmp / "nh" / n, (120, 200))
    p1 = data.pair_files(tmp / "h", tmp / "g"); p2 = data.pair_files(tmp / "nh")
    assert [(a.name, b.name) for a, b in p1] == [("01_outdoor_hazy.jpg", "01_outdoor_GT.jpg"), ("03_outdoor_hazy.jpg", "03_outdoor_GT.JPG")]
    assert [(a.name, b.name) for a, b in p2] == [("01_hazy.png", "01_GT.png"), ("47_hazy.png", "47_GT.png")]
    old = dict(data.TEST_SETS)
    data.TEST_SETS["_T"] = (str(tmp / "nh"), None, 2)
    try:
        a = data.load_test_set("_T", 100, cache_dir=tmp / "cache")
        b = data.load_test_set("_T", 100, cache_dir=tmp / "cache")           # second call reads the cache
        assert len(a) == 2 and a[0][1].shape == (60, 100, 3) and a[0][1].dtype == np.uint8
        assert all(np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]) and x[0] == y[0] for x, y in zip(a, b))
    finally:
        data.TEST_SETS.clear(); data.TEST_SETS.update(old)
    shutil.rmtree(tmp, ignore_errors=True)


def test_gpu_amp_train_step_if_available():
    if not torch.cuda.is_available():
        print("   (no GPU: skipped)"); return
    for n in ["dehazeformer-t", "gunet-t", "deanet"]:
        m = build(n).cuda().train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.rand(2, 3, 128, 128, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            y = m(x)
        loss = torch.nn.functional.l1_loss(y.float(), x)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        assert torch.isfinite(loss), n
        out = infer(m.eval(), torch.rand(1, 3, 200, 264, device="cuda"))
        assert torch.isfinite(out).all() and out.shape[-2:] == (200, 264), n
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    os.chdir(ROOT)
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
    print("ALL MODEL TESTS PASSED")
