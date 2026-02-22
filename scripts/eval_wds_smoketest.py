"""
Evaluate a WDS-trained JointSignalKmerTransformer checkpoint.

Assumes WDS shards contain:
  - signal.pth : torch float32 [400]
  - kmer.pth   : torch int64 [9]
  - y.pth      : torch float32 scalar

Checkpoint is a state_dict saved by:
  torch.save(model.state_dict(), <wds_dir>/joint_model_wds.pt)

This script:
  - Imports --model_py and calls build_model() to construct architecture
  - Loads state_dict into that model
  - Evaluates on split shards (val-*.tar by default)
  - Reports regression metrics (MSE, MAE)
  - Reports calibration diagnostics treating predictions as probabilities in [0,1]:
      ECE + reliability bins
  - Optional affine calibration (a*p + b) fitted on the eval split

Output:
  --out_json : eval_metrics.json
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds


def _torch_load_bytes(b: bytes):
    return torch.load(io.BytesIO(b), map_location="cpu")


def decode_sample(sample: dict):
    sig = _torch_load_bytes(sample["signal.pth"])
    km = _torch_load_bytes(sample["kmer.pth"])
    y = _torch_load_bytes(sample["y.pth"])
    return sig, km, y


def make_loader(shards, batch_size: int, num_workers: int):
    ds = wds.WebDataset(shards, shardshuffle=0).map(decode_sample)
    ds = ds.batched(batch_size, partial=False)
    return wds.WebLoader(ds, num_workers=num_workers, batch_size=None)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _mse(y: np.ndarray, p: np.ndarray) -> float:
    d = y - p
    return float(np.mean(d * d))


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def _ece_and_bins(y: np.ndarray, p: np.ndarray, bins: int = 15) -> Tuple[float, List[Dict]]:
    y = y.astype(np.float64)
    p = _clip01(p.astype(np.float64))

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)  # 0..bins-1

    out_bins: List[Dict] = []
    ece = 0.0
    n = len(y)

    for b in range(bins):
        m = bin_ids == b
        cnt = int(np.sum(m))
        if cnt == 0:
            out_bins.append(
                {"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "count": 0,
                 "mean_pred": None, "mean_true": None, "abs_gap": None}
            )
            continue

        mp = float(np.mean(p[m]))
        mt = float(np.mean(y[m]))
        gap = abs(mp - mt)
        ece += (cnt / max(n, 1)) * gap
        out_bins.append(
            {"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "count": cnt,
             "mean_pred": mp, "mean_true": mt, "abs_gap": float(gap)}
        )

    return float(ece), out_bins


def _fit_affine_calibration(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    A = np.vstack([p, np.ones_like(p)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _load_model_from_state_dict(ckpt_path: str, model_py: str, device: torch.device) -> nn.Module:
    import importlib.util

    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected state_dict (dict/OrderedDict) at {ckpt_path}, got {type(obj)}")

    spec = importlib.util.spec_from_file_location("eval_model_def", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model_py={model_py}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore

    build = getattr(mod, "build_model", None)
    if not callable(build):
        raise RuntimeError(f"{model_py} must define build_model() -> torch.nn.Module")

    model = build()
    if not isinstance(model, nn.Module):
        raise RuntimeError("build_model() did not return a torch.nn.Module")

    missing, unexpected = model.load_state_dict(obj, strict=False)
    if missing:
        print(f"WARN: missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"WARN: unexpected keys (first 10): {unexpected[:10]}")

    model = model.to(device)
    model.eval()
    return model


@dataclass
class EvalReport:
    split: str
    n: int
    mse: float
    mae: float
    ece: float
    bins: List[Dict]
    calibrated: bool
    calib_method: str | None
    calib_params: Dict | None
    mse_cal: float | None
    mae_cal: float | None
    ece_cal: float | None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_py", required=True, help="Training script containing build_model()")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calib_bins", type=int, default=15)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    shards = sorted(glob(str(Path(args.wds_dir) / f"{args.split}-*.tar")))
    if not shards:
        raise SystemExit(f"Could not find shards for split={args.split} under {args.wds_dir}")

    loader = make_loader(shards, args.batch_size, args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_state_dict(args.ckpt, args.model_py, device)

    preds: List[float] = []
    trues: List[float] = []

    for sig, km, y in loader:
        sig = sig.to(device).float()      # (B,400)
        km = km.to(device).long()         # (B,9)
        y = y.to(device).float().view(-1) # (B,)

        p = model(sig, km)                # sigmoid output (B,)
        preds.append(p.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    p = _clip01(np.concatenate(preds).astype(np.float64))
    y = _clip01(np.concatenate(trues).astype(np.float64))

    mse = _mse(y, p)
    mae = _mae(y, p)
    ece, bins = _ece_and_bins(y, p, bins=args.calib_bins)

    mse_cal = mae_cal = ece_cal = None
    calib_params = None
    calib_method = None

    if args.calibrate:
        a, b = _fit_affine_calibration(y, p)
        p2 = _clip01(a * p + b)
        mse_cal = _mse(y, p2)
        mae_cal = _mae(y, p2)
        ece_cal, _ = _ece_and_bins(y, p2, bins=args.calib_bins)
        calib_params = {"a": a, "b": b}
        calib_method = "affine_ls"

    rep = EvalReport(
        split=args.split,
        n=int(len(y)),
        mse=float(mse),
        mae=float(mae),
        ece=float(ece),
        bins=bins,
        calibrated=bool(args.calibrate),
        calib_method=calib_method,
        calib_params=calib_params,
        mse_cal=mse_cal,
        mae_cal=mae_cal,
        ece_cal=ece_cal,
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(rep), f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"split={rep.split} n={rep.n} mse={rep.mse:.6f} mae={rep.mae:.6f} ece={rep.ece:.6f}")
    if rep.calibrated:
        print(f"calib={rep.calib_method} a={rep.calib_params['a']:.6f} b={rep.calib_params['b']:.6f}")
        print(f"mse_cal={rep.mse_cal:.6f} mae_cal={rep.mae_cal:.6f} ece_cal={rep.ece_cal:.6f}")


if __name__ == "__main__":
    main()
