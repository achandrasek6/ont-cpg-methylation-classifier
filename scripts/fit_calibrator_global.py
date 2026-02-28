#!/usr/bin/env python3
"""
Fit a global calibrator (affine_ls / temp / isotonic) from WebDataset shards.

Input:
  --shards_txt : text file, one shard URL per line (supports WebDataset "pipe:" URLs)
  --ckpt       : model state_dict
  --model_py   : python file that defines build_model()

Output:
  --out_json   : calibration params JSON to be applied later.
    Format:
      {
        "calib_method": "affine_ls"|"temp"|"isotonic",
        "params": {...}
      }
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
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


def read_shard_list(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if not u or u.startswith("#"):
                continue
            urls.append(u)
    return urls


def make_loader(shards: List[str], batch_size: int, num_workers: int):
    nw = int(num_workers)
    if nw > 0:
        nw = min(nw, max(1, len(shards)))
    ds = wds.WebDataset(shards, shardshuffle=0, empty_check=False).map(decode_sample)
    ds = ds.batched(batch_size, partial=True)
    return wds.WebLoader(ds, num_workers=nw, batch_size=None)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _fit_affine_calibration(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    A = np.vstack([p, np.ones_like(p)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _fit_temperature(y: np.ndarray, p: np.ndarray, max_iter: int = 200) -> float:
    """
    Fit scalar temperature T>0 on logits to minimize MSE between sigmoid(logit(p)/T) and y.
    Runs on CPU for stability.
    """
    z = _logit(p).astype(np.float32)
    y = y.astype(np.float32)

    z_t = torch.tensor(z, device="cpu")
    y_t = torch.tensor(y, device="cpu")

    logT = torch.tensor([0.0], requires_grad=True, device="cpu")
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


def _isotonic_fit_pav(p: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit isotonic regression via PAV; returns (x_knots, y_knots) describing a non-decreasing step function.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    order = np.argsort(p, kind="mergesort")
    x = p[order]
    t = y[order]

    w = np.ones_like(t)

    starts: List[int] = []
    ends: List[int] = []
    weights: List[float] = []
    avgs: List[float] = []

    for i in range(len(x)):
        starts.append(i)
        ends.append(i)
        weights.append(float(w[i]))
        avgs.append(float(t[i]))

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

    xk = np.asarray([float(x[e]) for e in ends], dtype=np.float64)
    yk = _clip01(np.asarray([float(a) for a in avgs], dtype=np.float64))

    # Make representation slightly more compact (optional): drop duplicate x knots
    if len(xk) > 1:
        keep = np.ones_like(xk, dtype=bool)
        keep[1:] = xk[1:] != xk[:-1]
        xk = xk[keep]
        yk = yk[keep]

    return xk, yk


def _load_model_from_state_dict(ckpt_path: str, model_py: str, device: torch.device) -> nn.Module:
    import importlib.util

    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected state_dict at {ckpt_path}, got {type(obj)}")

    spec = importlib.util.spec_from_file_location("calib_model_def", model_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model_py={model_py}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore

    build = getattr(mod, "build_model", None)
    if not callable(build):
        raise RuntimeError(f"{model_py} must define build_model()")

    model = build()
    if not isinstance(model, nn.Module):
        raise RuntimeError("build_model() did not return nn.Module")

    model.load_state_dict(obj, strict=False)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_preds_trues(
    model: nn.Module,
    shards: List[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = make_loader(shards, batch_size, num_workers)
    preds = []
    trues = []

    for sig, km, y in loader:
        sig = sig.to(device).float()
        km = km.to(device).long()
        y = y.to(device).float().view(-1)
        p = model(sig, km)
        preds.append(p.detach().cpu().numpy())
        trues.append(y.detach().cpu().numpy())

    if not preds:
        raise SystemExit("No samples were read from shards (preds empty). Check shard URLs / permissions / pipe failures.")

    p = _clip01(np.concatenate(preds).astype(np.float64))
    y = _clip01(np.concatenate(trues).astype(np.float64))
    return p, y


@dataclass
class CalibParams:
    calib_method: str
    params: Dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_txt", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_py", required=True)
    ap.add_argument("--calib_method", required=True, choices=["affine_ls", "temp", "isotonic"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    shards = read_shard_list(args.shards_txt)
    if not shards:
        raise SystemExit(f"No shards found in {args.shards_txt}")

    # Use GPU if available for forward pass; calibration math itself is CPU anyway.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_state_dict(args.ckpt, args.model_py, device)
    p_fit, y_fit = collect_preds_trues(model, shards, args.batch_size, args.num_workers, device)

    if args.calib_method == "affine_ls":
        a, b = _fit_affine_calibration(y_fit, p_fit)
        out = CalibParams(calib_method="affine_ls", params={"a": a, "b": b})
    elif args.calib_method == "temp":
        T = _fit_temperature(y_fit, p_fit)
        out = CalibParams(calib_method="temp", params={"T": T})
    else:
        xk, yk = _isotonic_fit_pav(p_fit, y_fit)
        out = CalibParams(calib_method="isotonic", params={"x_knots": xk.tolist(), "y_knots": yk.tolist()})

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(out), f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"method={out.calib_method} fit_n={len(p_fit)}")


if __name__ == "__main__":
    main()