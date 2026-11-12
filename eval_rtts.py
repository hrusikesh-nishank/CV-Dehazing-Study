"""RTTS (4,322 real hazy images, no ground truth): detection + no-reference IQA.

  python eval_rtts.py --model dehazeformer-t --ckpt runs/dehazeformer-t__R3_s0/final.pth
  python eval_rtts.py --model hazy-input --name pre__hazy-input
  python eval_rtts.py --pred_dir data/sota_outputs/HazeFlow/RTTS --name pre__hazeflow-iccv25   # published outputs
  python eval_rtts.py ... --subset 500        # fixed 500-image subset (quick protocol)

Detection: COCO-pretrained YOLOv8m (ckpt/yolov8m.pt) on the dehazed images, 5 RTTS classes
(person, bicycle, car, motorbike->motorcycle, bus). mAP@0.5 computed with the PASCAL-VOC
protocol (all-point interpolated AP, objects marked `difficult` ignored), implemented here
so no extra dependency can change the numbers.

NR-IQA (pyiqa) on a fixed subset of --n_iqa images: NIQE, BRISQUE (lower better), MUSIQ, CLIPIQA
(higher better). NR metrics can reward over-enhancement: always report them together with mAP.

Images are processed at native resolution (tiled at 1024 px when the long side > 1280).
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eval_paired import default_name  # noqa: E402
from haze.models import build, infer, load_weights  # noqa: E402

RTTS2COCO = {"person": 0, "bicycle": 1, "car": 2, "motorbike": 3, "bus": 5}
CLASSES = list(RTTS2COCO)
COCO2IDX = {v: i for i, v in enumerate(RTTS2COCO.values())}


def read_voc(xml):
    objs = []
    for o in ET.parse(xml).getroot().iter("object"):
        n = o.find("name").text.strip().lower()
        if n not in RTTS2COCO:
            continue
        b = o.find("bndbox")
        d = o.find("difficult")
        objs.append((CLASSES.index(n), [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")],
                     int(d.text) if d is not None else 0))
    return objs


def iou(box, boxes):
    ix1 = np.maximum(box[0], boxes[:, 0]); iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2]); iy2 = np.minimum(box[3], boxes[:, 3])
    iw = np.maximum(ix2 - ix1 + 1, 0); ih = np.maximum(iy2 - iy1 + 1, 0)
    inter = iw * ih
    ua = (box[2] - box[0] + 1) * (box[3] - box[1] + 1) + (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) - inter
    return inter / ua


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0])); mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def voc_map(gts, dets, thr=0.5):
    """gts: {img: [(cls, box, difficult)]}; dets: list of (img, cls, score, box)."""
    aps = {}
    for c in range(len(CLASSES)):
        R = {}
        npos = 0
        for img, objs in gts.items():
            bb = np.array([o[1] for o in objs if o[0] == c]).reshape(-1, 4)
            df = np.array([o[2] for o in objs if o[0] == c], dtype=bool)
            R[img] = {"bb": bb, "df": df, "hit": np.zeros(len(bb), bool)}
            npos += int((~df).sum())
        D = sorted([d for d in dets if d[1] == c], key=lambda d: -d[2])
        tp = np.zeros(len(D)); fp = np.zeros(len(D))
        for k, (img, _, _, box) in enumerate(D):
            r = R.get(img)
            if r is None or len(r["bb"]) == 0:
                fp[k] = 1; continue
            ov = iou(np.array(box), r["bb"]); j = int(ov.argmax())
            if ov[j] >= thr:
                if r["df"][j]:
                    continue                       # difficult: neither TP nor FP
                if not r["hit"][j]:
                    tp[k] = 1; r["hit"][j] = True
                else:
                    fp[k] = 1
            else:
                fp[k] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        rec = tp / max(npos, 1); prec = tp / np.maximum(tp + fp, np.finfo(float).eps)
        aps[CLASSES[c]] = voc_ap(rec, prec) * 100
    return float(np.mean(list(aps.values()))), aps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--pred_dir", default=None, help="evaluate precomputed outputs (e.g. published SOTA results)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--res_dir", default="results")
    ap.add_argument("--rtts", default="data/test/RTTS")
    ap.add_argument("--yolo", default="ckpt/yolov8m.pt")
    ap.add_argument("--subset", type=int, default=0, help="evaluate a fixed random subset of N images (0 = all 4,322)")
    ap.add_argument("--n_iqa", type=int, default=500)
    ap.add_argument("--skip_iqa", action="store_true")
    ap.add_argument("--skip_det", action="store_true")
    ap.add_argument("--keep_outputs", action="store_true", help="keep all dehazed RTTS images (~2 GB)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_partial", action="store_true", help="skip the image-count check (debugging only)")
    a = ap.parse_args()
    assert a.pred_dir or a.model, "--model or --pred_dir"
    assert a.name or a.model, "--name is required with --pred_dir"
    assert a.pred_dir or a.model in ("hazy-input", "dcp") or a.ckpt, "--ckpt is required for learned models"
    name = a.name or default_name(a.model, a.ckpt)
    res_dir = Path(a.res_dir) / name
    res_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted((Path(a.rtts) / "JPEGImages").glob("*.png"))
    assert a.allow_partial or len(imgs) == 4322, f"RTTS: expected 4322 images, found {len(imgs)}"
    if a.subset and a.subset < len(imgs):
        pick = np.random.default_rng(0).choice(len(imgs), a.subset, replace=False)
        imgs = [imgs[i] for i in sorted(pick)]

    # ---- produce dehazed images (or point at precomputed / raw inputs) ----
    own_outputs = False
    if a.pred_dir:
        outdir = Path(a.pred_dir)
    elif a.model == "hazy-input":
        outdir = Path(a.rtts) / "JPEGImages"
    else:
        own_outputs = True
        outdir = res_dir / "RTTS_out"
        outdir.mkdir(exist_ok=True)
        model = build(a.model).to(a.device).eval()
        if a.model != "dcp":
            print("load", a.ckpt, load_weights(model, a.ckpt))
        for i, p in enumerate(imgs):
            dst = outdir / p.name
            if not dst.exists():
                x = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None].to(a.device)
                tile = 1024 if max(x.shape[-2:]) > 1280 else 0
                y = infer(model, x, tile=tile)[0].permute(1, 2, 0).cpu().numpy()
                tmp = outdir / (p.stem + ".tmp.png")
                Image.fromarray((y * 255).round().astype(np.uint8)).save(tmp)
                tmp.replace(dst)                               # never leave a half-written image
            if i % 500 == 0:
                print(f"dehazed {i}/{len(imgs)}", flush=True)
        del model
        if a.device.startswith("cuda"):
            torch.cuda.empty_cache()
    missing = [p.name for p in imgs if not (outdir / p.name).exists()]
    assert not missing, f"{len(missing)} outputs missing in {outdir}, e.g. {missing[:3]}"

    res = json.load(open(res_dir / "rtts.json")) if (res_dir / "rtts.json").exists() else {}
    res["n_images"] = len(imgs)
    # ---- detection ----
    if not a.skip_det:
        from ultralytics import YOLO
        det = YOLO(a.yolo)
        gts, dets = {}, []
        for p in imgs:
            gts[p.stem] = read_voc(Path(a.rtts) / "Annotations" / (p.stem + ".xml"))
            r = det.predict(str(outdir / p.name), conf=0.001, iou=0.6, max_det=300, verbose=False,
                            classes=list(RTTS2COCO.values()), device=0 if a.device.startswith("cuda") else "cpu")[0]
            for box, cls, sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().cpu().tolist(), r.boxes.conf.cpu().numpy()):
                if cls in COCO2IDX:
                    dets.append((p.stem, COCO2IDX[cls], float(sc), box.tolist()))
        m, aps = voc_map(gts, dets)
        res["mAP50"] = m
        res["AP50"] = aps
        print(f"{name}: RTTS mAP@0.5 (VOC, {len(imgs)} images) = {m:.2f}  " + " ".join(f"{k} {v:.1f}" for k, v in aps.items()), flush=True)
        json.dump(res, open(res_dir / "rtts.json", "w"), indent=1)     # save before the (slower) IQA
        del det

    # ---- NR-IQA ----
    if not a.skip_iqa:
        import pyiqa
        names = [p.name for p in imgs]
        sub = [names[i] for i in np.random.default_rng(1).choice(len(names), min(a.n_iqa, len(names)), replace=False)]
        for mname in ["niqe", "brisque", "musiq", "clipiqa"]:
            try:
                f = pyiqa.create_metric(mname, device=a.device)
                vals = []
                for s in sub:
                    with torch.no_grad():
                        vals.append(float(f(str(outdir / s)).item()))
                res[mname] = float(np.nanmean(vals))
                print(f"  {mname}: {res[mname]:.3f}", flush=True)
                del f
            except Exception as e:  # e.g. weight download blocked: keep the detection result
                print(f"  WARNING: {mname} failed ({type(e).__name__}: {e}); skipped", flush=True)
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
    res["_meta"] = {"model": a.model, "ckpt": a.ckpt, "pred_dir": a.pred_dir, "subset": a.subset, "n_iqa": a.n_iqa}
    json.dump(res, open(res_dir / "rtts.json", "w"), indent=1)
    if own_outputs and not a.keep_outputs:                             # keep 12 examples for figures
        keep = set(sorted(p.name for p in imgs)[:12])
        for f in outdir.iterdir():
            if f.name not in keep:
                f.unlink()


if __name__ == "__main__":
    main()
