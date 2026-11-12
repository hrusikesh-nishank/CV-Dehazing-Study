"""Unit tests for the synthesis pipeline, parameters and data loading (CPU; a few also on GPU if present).

Run:  python tests/test_synth.py
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from haze import isp  # noqa: E402
from haze.synth import ARMS, Arm, synthesize, synthesize_pair, transmission  # noqa: E402


def _batch(B=4, H=64, W=64, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    J = torch.rand(B, 3, H, W, generator=g) * 0.9 + 0.05
    d = 5 + 300 * torch.rand(B, 1, H, W, generator=g)
    V = torch.tensor([50.0, 100.0, 300.0, 800.0])[:B]
    A = torch.tensor([[0.80, 0.85, 0.92]]).repeat(B, 1)
    return J.to(device), d.to(device), V.to(device), A.to(device)


def G(seed=0):
    return torch.Generator().manual_seed(seed)


def test_tone_roundtrip():
    J, *_ = _batch()
    assert (isp.to_display(isp.to_linear(J)) - J).abs().max() < 1e-4


def test_identity_camera_roundtrip():
    J, *_ = _batch()
    cam = isp.random_camera(J.shape[0], G(1))
    out = isp.image_through_camera(isp.to_linear(J), cam, use_mosaic=False, noise=False)
    assert (out - J).abs().max() < 1e-3, (out - J).abs().max()


def test_camera_never_clips_white_or_saturated_colours():
    # 4,000 random cameras: a white / saturated patch must survive RAW conversion + sensor clipping
    cam = isp.random_camera(4000, G(11))
    assert cam.rgb_gain.min() >= 1.0
    cols = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.05, 0.05], [0.05, 1.0, 0.05], [0.05, 0.05, 1.0], [0.9, 0.9, 0.1]])
    for c in cols:
        J = c.view(1, 3, 1, 1).expand(4000, 3, 2, 2)
        raw = isp.invert_gains(isp.apply_ccm(isp.to_linear(J), cam.rgb2cam), cam)
        assert raw.max() <= 1.0 + 1e-5, (c, raw.max())
    # with (tiny) noise switched on, the clipped round trip of white stays white on average
    camq = isp.random_camera(256, G(12), shot_range=(1e-9, 1e-9))
    W = torch.ones(256, 3, 16, 16)
    out = isp.image_through_camera(isp.to_linear(W), camq, use_mosaic=False, noise=True, generator=G(2))
    assert (out - 1).abs().max() < 5e-3, (out - 1).abs().max()


def test_no_haze_is_identity_for_sRGB_and_linear_arms():
    J, d, V, A = _batch()
    huge_V = torch.full_like(V, 1e9)
    for arm in ["N0", "R0", "R1", "H"]:
        out, tgt = synthesize_pair(J, d, huge_V, A, ARMS[arm], generator=G())
        assert (out - J).abs().max() < 2e-3, (arm, (out - J).abs().max())
        assert torch.equal(tgt, J)


def test_linear_vs_srgb_worked_example():
    # J = 0.1, A = 0.9, t = 0.5  ->  0.50 in sRGB, ~0.65 in linear light
    J = torch.full((1, 3, 8, 8), 0.1); A = torch.full((1, 3), 0.9)
    V = torch.tensor([2.996 / 0.6931]); d = torch.ones(1, 1, 8, 8)
    r0 = synthesize(J, d, V, A, ARMS["R0"]).mean().item(); r1 = synthesize(J, d, V, A, ARMS["R1"]).mean().item()
    print(f"   R0 {r0:.4f}  R1 {r1:.4f}")
    assert abs(r0 - 0.5) < 1e-3 and abs(r1 - 0.646) < 2e-3


def test_all_arms_run_bounded_and_deterministic():
    J, d, V, A = _batch()
    for name, arm in ARMS.items():
        a, ta = synthesize_pair(J, d, V, A, arm, generator=G(3))
        b, tb = synthesize_pair(J, d, V, A, arm, generator=G(3))
        assert a.shape == J.shape and a.min() >= 0 and a.max() <= 1 and torch.isfinite(a).all(), name
        assert ta.shape == J.shape and ta.min() >= 0 and ta.max() <= 1, name
        assert torch.equal(a, b) and torch.equal(ta, tb), f"{name} is not deterministic for a fixed seed"


def test_target_is_tone_consistent():
    # camera with tone/WB randomisation but (almost) no noise, no JPEG, no haze: hazy == target != J
    J, d, V, A = _batch()
    arm = Arm(linear=True, camera=True, noise=False, rand_isp=True)
    out, tgt = synthesize_pair(J, d, torch.full_like(V, 1e9), A, arm, generator=G(), wb_sigma=0.0)
    assert (out - tgt).abs().max() < 3e-3, (out - tgt).abs().max()
    assert (tgt - J).abs().max() > 0.01          # the tone curve was really changed
    # targets stay J when the ISP is not randomised
    _, t2 = synthesize_pair(J, d, V, A, ARMS["R2"], generator=G())
    assert torch.equal(t2, J)


def test_shared_random_streams_across_arms():
    # with zero depth, heterogeneity cannot change anything: R3 and R3H must be identical,
    # which proves they draw the same camera, ISP, JPEG and noise numbers
    J, d, V, A = _batch()
    z = torch.zeros_like(d)
    a = synthesize(J, z, V, A, ARMS["R3"], generator=G(5)); b = synthesize(J, z, V, A, ARMS["R3H"], generator=G(5))
    assert torch.equal(a, b)
    # and the sRGB pair R0 / H likewise
    assert torch.equal(synthesize(J, z, V, A, ARMS["R0"], generator=G(5)), synthesize(J, z, V, A, ARMS["H"], generator=G(5)))


def test_hetero_preserves_mean_extinction():
    _, d, V, _ = _batch(H=128, W=128)
    d = torch.full_like(d, 100.0)
    t0 = transmission(d, V); t1 = transmission(d, V, hetero=True, generator=G())
    tau0, tau1 = -torch.log(t0).mean((1, 2, 3)), -torch.log(t1).mean((1, 2, 3))
    assert torch.allclose(tau0, tau1, rtol=1e-4) and (t1.std((1, 2, 3)) > 0).all()


def test_neutral_airlight_arm():
    J, d, V, A = _batch()
    dense = torch.full_like(V, 1.0)                      # t ~ 0 everywhere -> image = airlight
    n0 = synthesize(J, d, dense, A, ARMS["N0"]); r0 = synthesize(J, d, dense, A, ARMS["R0"])
    assert (n0[:, 0] - n0[:, 2]).abs().max() < 1e-4      # grey
    assert (r0[:, 2] - r0[:, 0]).mean() > 0.05           # bluish, as A


def test_noise_is_signal_dependent_and_applied_after_haze():
    J, d, V, A = _batch(H=128, W=128)
    arm = Arm(linear=True, camera=True, noise=True)
    cam = isp.random_camera(4, G(1), shot_range=(8e-4, 8e-4))
    lin = torch.full((4, 3, 128, 128), 0.02); bright = torch.full((4, 3, 128, 128), 0.6)
    n_dark = isp.image_through_camera(lin, cam, use_mosaic=False, noise=True, generator=G(2))
    n_bright = isp.image_through_camera(bright, cam, use_mosaic=False, noise=True, generator=G(2))
    raw_std = lambda x, ref: (isp.to_linear(x) - isp.to_linear(isp.image_through_camera(ref, cam, use_mosaic=False, noise=False))).std()
    assert raw_std(n_bright, bright) > 2 * raw_std(n_dark, lin)          # shot noise grows with signal
    a = synthesize(J, d, V, A, arm, generator=G()); b = synthesize(J, d, V, A, ARMS["R1"], generator=G())
    assert (a - b).std() > 1e-3


def test_mosaic_arm_runs():
    J, d, V, A = _batch()
    out = synthesize(J, d, V, A, ARMS["R3M"], generator=G())
    assert torch.isfinite(out).all()


def test_params_helpers_match_isp():
    import prepare_params as pp
    x = np.linspace(0.02, 0.98, 50)
    assert np.abs(pp.to_linear(x) - isp.to_linear(torch.tensor(x)).numpy()).max() < 1e-6
    assert np.abs(pp.to_display(pp.to_linear(x)) - x).max() < 1e-5
    d65, blue, warm = pp.daylight_rgb(6504), pp.daylight_rgb(10000), pp.daylight_rgb(4500)
    lum = np.array([0.2126729, 0.7151522, 0.0721750])
    assert np.abs(d65 - 1).max() < 1e-6
    assert blue[2] > 1.2 and blue[0] < 0.9 and warm[0] > 1.15 and warm[2] < 0.75
    assert abs(lum @ blue - 1) < 1e-9 and abs(lum @ warm - 1) < 1e-9
    print(f"   daylight 10000 K {blue.round(3).tolist()}  4500 K {warm.round(3).tolist()}")


def _fake_dataset(root, n=6, size=96):
    from PIL import Image
    for sub in ["GT", "hazy", "depth"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        Image.fromarray(img).save(root / "GT" / f"{i:04d}.png")
        Image.fromarray((img * 0.5 + 100).astype(np.uint8)).save(root / "hazy" / f"{i:04d}.png")
        d = np.linspace(5, 400, size * size, dtype=np.float32).reshape(size, size)
        d[: size // 4] = 1000
        np.save(root / "depth" / f"{i:04d}.npy", d.astype(np.float16))


def test_params_and_datasets():
    import subprocess
    from haze.data import CleanDepthDataset, EpochSampler, PairedDataset, pair_files
    root = ROOT / "tests" / "_data"
    shutil.rmtree(root, ignore_errors=True)
    _fake_dataset(root)
    r = subprocess.run([sys.executable, str(ROOT / "prepare_params.py"), "--clean_dir", str(root / "GT"),
                        "--depth_dir", str(root / "depth"), "--out", str(root / "p.npz")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    p = np.load(root / "p.npz")
    assert p["V"].shape == (6, 8) and p["A"].shape == (6, 8, 3)
    assert (p["tmean"] >= 0.099).all() and (p["tmean"] <= 0.851).all(), p["tmean"]
    assert p["A"].min() >= 0.35 and p["A"].max() <= 1.0 and (p["A"][..., 2] - p["A"][..., 0]).std() > 0.03
    # same parameters when recomputed (seeded by image name)
    r = subprocess.run([sys.executable, str(ROOT / "prepare_params.py"), "--clean_dir", str(root / "GT"),
                        "--depth_dir", str(root / "depth"), "--out", str(root / "p2.npz")], capture_output=True, text=True)
    assert np.array_equal(p["V"], np.load(root / "p2.npz")["V"])

    a = CleanDepthDataset(root / "GT", root / "depth", root / "p.npz", crop=64, seed=7)
    b = PairedDataset(pair_files(root / "hazy", root / "GT"), crop=64, seed=7)
    s = list(EpochSampler(len(a), 3, 7))
    assert len(s) == 18 and sorted(g % 6 for g in s[:6]) == list(range(6)) and s[:6] != [g - 6 for g in s[6:12]]
    for g in [0, 5, 6 + 2, 12 + 4]:
        xa, xb = a[g], b[g]
        assert torch.equal(xa["J"], xb["J"]), "B0 and synthesis arms must see identical crops"
        assert xa["J"].shape == (3, 64, 64) and xa["depth"].shape == (1, 64, 64) and xa["A"].shape == (3,)
    assert not torch.equal(a[2]["J"], a[6 + 2]["J"])                 # a new crop in a new epoch
    assert torch.equal(a[8]["J"], CleanDepthDataset(root / "GT", root / "depth", root / "p.npz", crop=64, seed=7)[8]["J"])
    assert len(CleanDepthDataset(root / "GT", root / "depth", root / "p.npz", crop=64, n_train=4)) == 4
    shutil.rmtree(root, ignore_errors=True)


def test_gpu_all_arms_if_available():
    if not torch.cuda.is_available():
        print("   (no GPU: skipped)"); return
    J, d, V, A = _batch(B=4, H=128, W=128, device="cuda")
    for name, arm in ARMS.items():
        a, t = synthesize_pair(J, d, V, A, arm, generator=G(3))
        b, _ = synthesize_pair(J, d, V, A, arm, generator=G(3))
        assert a.device.type == "cuda" and torch.isfinite(a).all() and torch.equal(a, b), name


if __name__ == "__main__":
    os.chdir(ROOT)
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
    print("ALL SYNTH TESTS PASSED")
