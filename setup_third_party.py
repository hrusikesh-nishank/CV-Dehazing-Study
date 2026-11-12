"""Fetch the official model code (not weights) into third_party/.

  python setup_third_party.py

Downloads GitHub source archives (no git needed) and extracts only what is used:
  third_party/DehazeFormer/  (IDKiro/DehazeFormer, MIT)   models/dehazeformer.py
  third_party/gUNet/         (IDKiro/gUNet, MIT)          models/gunet.py
  third_party/DEA-Net/       (cecret3350/DEA-Net)         code/model/**  (inference Backbone)
"""
import io
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "third_party"
REPOS = {
    "DehazeFormer": ("IDKiro/DehazeFormer", "main", ["models/", "LICENSE", "README.md"]),
    "gUNet": ("IDKiro/gUNet", "master", ["models/", "LICENSE", "README.md"]),
    "DEA-Net": ("cecret3350/DEA-Net", "main", ["code/model/", "README.md"]),
}


def fetch(name, repo, branch, keep):
    dst = ROOT / name
    if (dst / ".complete").exists():
        print("ok (already):", name); return
    url = f"https://github.com/{repo}/archive/HEAD.tar.gz"
    print("downloading", url)
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
    except Exception as e:  # fall back to git (e.g. behind a proxy that blocks codeload)
        print("  archive download failed (", e, ") -> trying git clone")
        import shutil, subprocess, tempfile
        tmp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        f"https://github.com/{repo}", str(tmp / "r")], check=True)
        for k in keep:
            src = tmp / "r" / k
            if src.is_dir():
                shutil.copytree(src, dst / k, dirs_exist_ok=True)
            elif src.exists():
                (dst / k).parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst / k)
        (dst / ".complete").write_text(url)
        print("ok (git):", name); return
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel or not any(rel.startswith(k) for k in keep) or not m.isfile():
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tf.extractfile(m).read())
    (dst / ".complete").write_text(url)
    print("ok:", name)


if __name__ == "__main__":
    for k, v in REPOS.items():
        fetch(k, *v)
