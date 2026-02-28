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
        "params": {
          ...,
          "tail_clip": {
            "enabled": true|false,
            "mode": "always"|"if_sparse",
            "q_hi": 0.995,
            "q_lo": 0.005,
            "p_hi": <float>,
            "p_lo": <float>,
            "tail_count_hi": <int>,
            "tail_count_lo": <int>,
            "min_n": <int>
          }
        }
      }

Notes:
  - Tail clip is only relevant for isotonic.
  - We compute clip bounds from the *fit* distribution of raw predictions.
  - Default behavior is "if_sparse": only enable clipping if tail support is low.
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple, Optional

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


# -------------------------
# Isotonic regression (PAV)
# -------------------------

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

    # drop duplicate x knots (compact)
    if len(xk) > 1:
        keep = np.ones_like(xk, dtype=bool)
        keep[1:] = xk[1:] != xk[:-1]
        xk = xk[keep]
        yk = yk[keep]

    return xk, yk


def _isotonic_predict(p: np.ndarray, xk: np.ndarray, yk: np.ndarray) -> np.ndarray:
    """
    Right-continuous step function:
      idx = first i where p <= xk[i]; return yk[i]; if beyond last, return yk[-1]
    """
    p = np.asarray(p, dtype=np.float64)
    idx = np.searchsorted(xk, p, side="left")
    idx = np.clip(idx, 0, len(yk) - 1)
    return yk[idx]


def _compute_iso_tail_clip(
    p_fit: np.ndarray,
    xk: np.ndarray,
    yk: np.ndarray,
    q_hi: float,
    mode: str,
    min_n: int,
    enable_lower: bool,
) -> Dict:
    """
    Compute tail clip bounds from raw preds distribution on the *fit* set,
    mapped through isotonic.
    """
    p_fit = np.asarray(p_fit, dtype=np.float64)
    q_hi = float(q_hi)
    q_hi = min(max(q_hi, 0.5), 0.999999)  # guardrails

    q_lo = 1.0 - q_hi
    # upper tail
    thr_hi = float(np.quantile(p_fit, q_hi))
    n_hi = int(np.sum(p_fit >= thr_hi))
    p_hi = float(_isotonic_predict(np.array([thr_hi], dtype=np.float64), xk, yk)[0])

    # lower tail (optional)
    if enable_lower:
        thr_lo = float(np.quantile(p_fit, q_lo))
        n_lo = int(np.sum(p_fit <= thr_lo))
        p_lo = float(_isotonic_predict(np.array([thr_lo], dtype=np.float64), xk, yk)[0])
    else:
        thr_lo = None
        n_lo = None
        p_lo = None

    if mode not in ("always", "if_sparse"):
        raise ValueError(f"iso_tail_clip_mode must be 'always' or 'if_sparse', got {mode}")

    if mode == "always":
        enabled = True
    else:
        # enable only when tail support is low
        enabled = n_hi < int(min_n)

    out: Dict = {
        "enabled": bool(enabled),
        "mode": mode,
        "q_hi": float(q_hi),
        "p_hi": float(_clip01(np.array([p_hi]))[0]),
        "tail_count_hi": int(n_hi),
        "min_n": int(min_n),
    }

    if enable_lower:
        out.update(
            {
                "q_lo": float(q_lo),
                "p_lo": float(_clip01(np.array([p_lo]))[0]) if p_lo is not None else None,
                "tail_count_lo": int(n_lo) if n_lo is not None else None,
            }
        )

    return out


# -------------------------
# Model loading
# -------------------------

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

    missing, unexpected = model.load_state_dict(obj, strict=False)
    if missing:
        print(f"WARN: missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"WARN: unexpected keys (first 10): {unexpected[:10]}")

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

    # --- NEW: isotonic tail clip options ---
    ap.add_argument(
        "--iso_tail_clip",
        action="store_true",
        help="If set and calib_method=isotonic, compute tail clip bounds and store them in JSON.",
    )
    ap.add_argument(
        "--iso_tail_clip_mode",
        default="if_sparse",
        choices=["always", "if_sparse"],
        help="Tail clip enable policy: always, or only if tail support is low.",
    )
    ap.add_argument(
        "--iso_tail_q",
        type=float,
        default=0.995,
        help="Quantile defining the upper tail for clip bound (e.g., 0.995). Lower tail uses 1-q if enabled.",
    )
    ap.add_argument(
        "--iso_tail_min_n",
        type=int,
        default=200,
        help="If mode=if_sparse, enable clipping only if tail_count_hi < this value.",
    )
    ap.add_argument(
        "--iso_tail_lower",
        action="store_true",
        help="Also compute/apply lower-tail clip (usually not needed).",
    )

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
        params: Dict = {"x_knots": xk.tolist(), "y_knots": yk.tolist()}

        if args.iso_tail_clip:
            tc = _compute_iso_tail_clip(
                p_fit=p_fit,
                xk=xk,
                yk=yk,
                q_hi=args.iso_tail_q,
                mode=args.iso_tail_clip_mode,
                min_n=args.iso_tail_min_n,
                enable_lower=bool(args.iso_tail_lower),
            )
            params["tail_clip"] = tc

            print(
                f"[TAIL_CLIP] mode={tc['mode']} enabled={tc['enabled']} "
                f"q_hi={tc['q_hi']} tail_n_hi={tc['tail_count_hi']} p_hi={tc['p_hi']:.6f}"
            )
            if args.iso_tail_lower:
                print(
                    f"[TAIL_CLIP] q_lo={tc.get('q_lo')} tail_n_lo={tc.get('tail_count_lo')} p_lo={tc.get('p_lo')}"
                )

        out = CalibParams(calib_method="isotonic", params=params)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(asdict(out), f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"method={out.calib_method} fit_n={len(p_fit)}")
    if out.calib_method == "isotonic":
        print(f"isotonic_knots={len(out.params.get('x_knots', []))}")
        if "tail_clip" in out.params:
            tc = out.params["tail_clip"]
            print(
                f"tail_clip: enabled={tc.get('enabled')} mode={tc.get('mode')} "
                f"q_hi={tc.get('q_hi')} p_hi={tc.get('p_hi')} n_hi={tc.get('tail_count_hi')}"
            )


if __name__ == "__main__":
    main()