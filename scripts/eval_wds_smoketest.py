#!/usr/bin/env python3
"""
Evaluate a WDS-trained JointSignalKmerTransformer checkpoint.

Shards contain:
  - signal.pth : torch float32 [400]
  - kmer.pth   : torch int64 [9]
  - y.pth      : torch float32 scalar (treated as probability in [0,1])

This script:
  - imports --model_py and calls build_model()
  - loads state_dict from --ckpt
  - evaluates on --split shards (val-*.tar by default)
  - reports MSE/MAE + calibration bins/ECE
  - optional calibration (fit on --calib_fit_split, apply on --split):
      * affine least squares: p_cal = clip01(a*p + b)
      * temperature scaling: p_cal = sigmoid(logit(p)/T)
      * isotonic regression: monotone step function p_cal = f(p) via PAV

NEW (AWS streaming friendly):
  - You can pass shard URLs directly via --shards_txt (one URL per line).
    This supports WebDataset "pipe:" URLs, e.g.:
      pipe:aws s3 cp s3://bucket/key/train-000000.tar -
  - For leakage-safe calibration in streaming mode, pass a separate list via
    --calib_shards_txt (one URL per line) for the calib-fit split.

Back-compat:
  - --wds_dir mode still works (globs {split}-*.tar from a directory).

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


def make_loader(shards: List[str], batch_size: int, num_workers: int):
    """WebDataset loader with guardrails (worker<=shards, empty_check disabled)."""
    nw = int(num_workers)
    if nw > 0:
        nw = min(nw, max(1, len(shards)))
    ds = wds.WebDataset(shards, shardshuffle=0, empty_check=False).map(decode_sample)
    ds = ds.batched(batch_size, partial=True)
    return wds.WebLoader(ds, num_workers=nw, batch_size=None)


def read_shard_list(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            urls.append(u)
    return urls


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


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
                {
                    "bin": b,
                    "lo": float(edges[b]),
                    "hi": float(edges[b + 1]),
                    "count": 0,
                    "mean_pred": None,
                    "mean_true": None,
                    "abs_gap": None,
                }
            )
            continue

        mp = float(np.mean(p[m]))
        mt = float(np.mean(y[m]))
        gap = abs(mp - mt)
        ece += (cnt / max(n, 1)) * gap
        out_bins.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": cnt,
                "mean_pred": mp,
                "mean_true": mt,
                "abs_gap": float(gap),
            }
        )

    return float(ece), out_bins


def _fit_affine_calibration(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Fit affine calibration p_cal = a*p + b via least squares to approximate y."""
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    A = np.vstack([p, np.ones_like(p)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _fit_temperature(y: np.ndarray, p: np.ndarray, max_iter: int = 200) -> float:
    """Fit scalar temperature T>0 on logits to minimize MSE between sigmoid(logit(p)/T) and y."""
    z = _logit(p).astype(np.float32)
    y = y.astype(np.float32)

    z_t = torch.tensor(z)
    y_t = torch.tensor(y)

    logT = torch.tensor([0.0], requires_grad=True)
    opt = torch.optim.LBFGS([logT], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        T = torch.exp(logT)
        p_cal = torch.sigmoid(z_t / T)
        loss = torch.mean((p_cal - y_t) ** 2)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).detach().cpu().item())


# -------------------------
# Isotonic regression (PAV)
# -------------------------

def _isotonic_fit_pav(p: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit isotonic regression y ~= f(p) with f non-decreasing, minimizing squared error.
    Returns (x_knots, y_hat_knots) describing a right-continuous step function.

    Implementation: Pool Adjacent Violators (PAV) on points sorted by p.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order = np.argsort(p, kind="mergesort")
    x = p[order]
    t = y[order]

    w = np.ones_like(t)
    avg = t.copy()

    starts: List[int] = []
    ends: List[int] = []
    weights: List[float] = []
    avgs: List[float] = []

    for i in range(len(x)):
        starts.append(i)
        ends.append(i)
        weights.append(float(w[i]))
        avgs.append(float(avg[i]))

        while len(avgs) >= 2 and avgs[-2] > avgs[-1]:
            s1, e1, w1, a1 = starts[-2], ends[-2], weights[-2], avgs[-2]
            s2, e2, w2, a2 = starts[-1], ends[-1], weights[-1], avgs[-1]
            w_new = w1 + w2
            a_new = (w1 * a1 + w2 * a2) / w_new

            starts[-2] = s1
            ends[-2] = e2
            weights[-2] = w_new
            avgs[-2] = a_new

            starts.pop()
            ends.pop()
            weights.pop()
            avgs.pop()

    x_knots = []
    y_knots = []
    for s, e, a in zip(starts, ends, avgs):
        x_knots.append(float(x[e]))
        y_knots.append(float(a))

    xk = np.asarray(x_knots, dtype=np.float64)
    yk = _clip01(np.asarray(y_knots, dtype=np.float64))
    return xk, yk


def _isotonic_predict(p: np.ndarray, xk: np.ndarray, yk: np.ndarray) -> np.ndarray:
    """
    Predict using right-continuous step function defined by knots (xk, yk).
    For each p:
      find first index i where p <= xk[i], return yk[i]
      if p > xk[-1], return yk[-1]
    """
    p = np.asarray(p, dtype=np.float64)
    idx = np.searchsorted(xk, p, side="left")
    idx = np.clip(idx, 0, len(yk) - 1)
    return yk[idx]


# -------------------------
# Model loading + prediction
# -------------------------

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


def _collect_preds_trues(
    *,
    model: nn.Module,
    wds_dir: str | None,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shards: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if shards is None:
        if not wds_dir:
            raise SystemExit("Must provide either --shards_txt (streaming) or --wds_dir (directory mode).")
        shards = sorted(glob(str(Path(wds_dir) / f"{split}-*.tar")))

    if not shards:
        raise SystemExit(f"Could not find shards for split={split}")

    loader = make_loader(shards, batch_size, num_workers)

    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    for sig, km, y in loader:
        sig = sig.to(device).float()
        km = km.to(device).long()
        y = y.to(device).float().view(-1)

        p = model(sig, km)  # sigmoid output
        preds.append(p.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    p = _clip01(np.concatenate(preds).astype(np.float64))
    y = _clip01(np.concatenate(trues).astype(np.float64))
    return p, y


@dataclass
class EvalReport:
    split: str
    n: int
    mse: float
    mae: float
    ece: float
    bins: List[Dict]
    bins_cal: List[Dict] | None
    calibrated: bool
    calib_method: str | None
    calib_fit_split: str | None
    calib_params: Dict | None
    mse_cal: float | None
    mae_cal: float | None
    ece_cal: float | None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds_dir", required=False, help="Directory containing {split}-*.tar shards (legacy mode)")
    ap.add_argument("--shards_txt", default=None, help="Text file: one shard URL per line (supports pipe:...)")
    ap.add_argument(
        "--calib_shards_txt",
        default=None,
        help="Optional shard list for calib-fit split (pipe: supported). If absent, uses --wds_dir + --calib_fit_split.",
    )

    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_py", required=True, help="Training script containing build_model()")

    ap.add_argument("--split", default="val")
    ap.add_argument("--calib_fit_split", default=None)

    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--calibrate", action="store_true", help="Back-compat: equivalent to --calib_method affine_ls")

    ap.add_argument(
        "--calib_method",
        default="none",
        choices=["none", "affine_ls", "temp", "isotonic"],
        help="Calibration method to apply (fit on calib_fit_split, apply on split)",
    )
    ap.add_argument("--calib_bins", type=int, default=15)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    if not args.shards_txt and not args.wds_dir:
        raise SystemExit("Must provide --shards_txt (streaming) or --wds_dir (directory mode).")

    if args.calib_fit_split is None:
        args.calib_fit_split = args.split

    if args.calibrate and args.calib_method == "none":
        args.calib_method = "affine_ls"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_state_dict(args.ckpt, args.model_py, device)

    # Resolve eval shards
    eval_shards = read_shard_list(args.shards_txt) if args.shards_txt else None

    # Eval split (raw)
    p_eval, y_eval = _collect_preds_trues(
        model=model,
        wds_dir=args.wds_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shards=eval_shards,
    )
    mse = _mse(y_eval, p_eval)
    mae = _mae(y_eval, p_eval)
    ece, bins = _ece_and_bins(y_eval, p_eval, bins=args.calib_bins)

    # Calibration fit/apply
    mse_cal = mae_cal = ece_cal = None
    bins_cal: List[Dict] | None = None
    calib_params = None
    calib_method = None
    calibrated = args.calib_method != "none"

    if calibrated:
        # Resolve fit shards:
        # - if calib_shards_txt provided, use it
        # - else if shards_txt provided but calib_shards_txt missing: fall back to wds_dir + calib_fit_split
        #   (this is intentional: leakage-safe calibration via streaming should pass calib_shards_txt)
        fit_shards = read_shard_list(args.calib_shards_txt) if args.calib_shards_txt else None

        p_fit, y_fit = _collect_preds_trues(
            model=model,
            wds_dir=args.wds_dir,
            split=args.calib_fit_split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            shards=fit_shards,
        )

        if args.calib_method == "affine_ls":
            a, b = _fit_affine_calibration(y_fit, p_fit)
            p2 = _clip01(a * p_eval + b)
            calib_params = {"a": a, "b": b}
            calib_method = "affine_ls"

        elif args.calib_method == "temp":
            T = _fit_temperature(y_fit, p_fit)
            z = _logit(p_eval)
            p2 = _clip01(_sigmoid(z / T))
            calib_params = {"T": T}
            calib_method = "temp"

        elif args.calib_method == "isotonic":
            xk, yk = _isotonic_fit_pav(p_fit, y_fit)
            p2 = _clip01(_isotonic_predict(p_eval, xk, yk))
            calib_params = {"x_knots": xk.tolist(), "y_knots": yk.tolist()}
            calib_method = "isotonic"

        else:
            raise RuntimeError(f"Unhandled calib_method={args.calib_method}")

        mse_cal = _mse(y_eval, p2)
        mae_cal = _mae(y_eval, p2)
        ece_cal, bins_cal = _ece_and_bins(y_eval, p2, bins=args.calib_bins)

    rep = EvalReport(
        split=str(args.split),
        n=int(len(y_eval)),
        mse=float(mse),
        mae=float(mae),
        ece=float(ece),
        bins=bins,
        bins_cal=bins_cal,
        calibrated=bool(calibrated),
        calib_method=calib_method,
        calib_fit_split=str(args.calib_fit_split) if calibrated else None,
        calib_params=calib_params,
        mse_cal=mse_cal,
        mae_cal=mae_cal,
        ece_cal=ece_cal,
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(rep), f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"eval split={rep.split} n={rep.n} mse={rep.mse:.6f} mae={rep.mae:.6f} ece={rep.ece:.6f}")
    if rep.calibrated:
        print(f"calib_fit_split={rep.calib_fit_split} method={rep.calib_method}")
        if rep.calib_method == "affine_ls":
            print(f"  a={rep.calib_params['a']:.6f} b={rep.calib_params['b']:.6f}")
        elif rep.calib_method == "temp":
            print(f"  T={rep.calib_params['T']:.6f}")
        elif rep.calib_method == "isotonic":
            print(f"  knots={len(rep.calib_params['x_knots'])} (stored in JSON)")
        print(f"mse_cal={rep.mse_cal:.6f} mae_cal={rep.mae_cal:.6f} ece_cal={rep.ece_cal:.6f}")


if __name__ == "__main__":
    main()