#!/usr/bin/env python3
"""
Neutronics Surrogate Builder (Refactored)
----------------------------------------
- Loads RAW and TEST matrices (whitespace or CSV)
- Coerces to numeric and drops rows with NaNs
- Optionally infers lattice geometry (for diagnostics)
- Builds polynomial feature map up to degree N (default 2)
- Fits ridge-regularized least squares (closed-form)
- Exports LIMPS-ready payload and a coefficients NPZ
- (Optional) Generates a minimal Python client for your LIMPS server
"""

import json, math, itertools, argparse
from pathlib import Path
import numpy as np
import pandas as pd
np.seterr(all="ignore")

def _poly_feature_count(d, degree):
    """Calculate number of polynomial features for d dimensions and given degree."""
    from math import comb
    return sum(comb(d + k - 1, k) for k in range(0, degree + 1))

def detect_delimiter(sample_lines):
    """
    Heuristics:
    - If >=60% of preview lines contain ',' -> comma
    - Else fallback to whitespace (\s+)
    """
    if not sample_lines:
        return r"\s+"
    comma_hits = sum(1 for line in sample_lines if ',' in line)
    return ',' if comma_hits / max(1, len(sample_lines)) >= 0.6 else r"\s+"

def load_matrix(path, sep_override=None, max_preview_lines=5):
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = list(itertools.islice(f, max_preview_lines))
    if not sample:
        raise ValueError(f"File {path} is empty or could not be read")

    delim = sep_override if sep_override else detect_delimiter(sample)
    try:
        if delim == ',':
            df = pd.read_csv(path, header=None)
        else:
            df = pd.read_csv(path, header=None, sep=delim, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to parse {path} with delimiter '{delim}': {e}")

    df = df.apply(pd.to_numeric, errors='coerce')
    return df, delim

def coerce_dropna_pair(X_df, Y_df):
    n = min(len(X_df), len(Y_df))
    X = X_df.iloc[:n, :].copy()
    Y = Y_df.iloc[:n, :].copy()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    valid_count = mask.sum()
    total_count = n
    Xc = X[mask].to_numpy(dtype=float)
    Yc = Y[mask].to_numpy(dtype=float)
    return Xc, Yc, valid_count, total_count

def infer_square_dim(n_cols):
    r = int(math.isqrt(n_cols))
    return r if r*r == n_cols else None

def poly_feature_names(d, degree):
    names = ["1"]
    for deg in range(1, degree+1):
        for comb in itertools.combinations_with_replacement(range(d), deg):
            term = "*".join(f"x{i+1}" for i in comb)
            names.append(term)
    return names

def poly_features(X, degree=2):
    n, d = X.shape
    feats = [np.ones((n,1))]
    names = ["1"]
    for deg in range(1, degree+1):
        for comb in itertools.combinations_with_replacement(range(d), deg):
            col = np.prod([X[:, i] for i in comb], axis=0).reshape(n,1)
            feats.append(col)
            names.append("*".join(f"x{i+1}" for i in comb))
    Phi = np.hstack(feats)
    return Phi, names

def ridge_closed_form(Phi, Y, lam=1e-6):
    PtP = Phi.T @ Phi
    PtY = Phi.T @ Y
    K = PtP + lam * np.eye(PtP.shape[0], dtype=PtP.dtype)
    try:
        return np.linalg.solve(K, PtY)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(K) @ PtY

def rmse_columns(pred, Y):
    return np.sqrt(np.mean((pred - Y)**2, axis=0))

def build_payload(X_used, variables, degree_limit, min_rank, structure, coeff_threshold, chebyshev, rmse_first10, n_targets):
    return {
        "matrix": X_used.tolist(),
        "variables": variables,
        "degree_limit": degree_limit,
        "min_rank": min_rank,
        "structure": structure,
        "coeff_threshold": coeff_threshold,
        "chebyshev": chebyshev,
        "targets_preview": {
            "n_targets_used": int(n_targets),
            "rmse_first10": [float(x) for x in rmse_first10]
        }
    }

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raw", required=True, help="Path to RAW matrix file (whitespace or CSV)")
    p.add_argument("--test", required=True, help="Path to TEST matrix file (whitespace or CSV)")
    p.add_argument("--degree", type=int, default=2, help="Polynomial degree (1..N)")
    p.add_argument("--max-input-cols", type=int, default=8, help="Cap number of input columns from RAW")
    p.add_argument("--max-target-cols", type=int, default=12, help="Cap number of target columns from TEST")
    p.add_argument("--max-rows", type=int, default=5000, help="Cap number of rows used for fitting")
    p.add_argument("--lambda", dest="lam", type=float, default=1e-6, help="Ridge regularization lambda")
    p.add_argument("--outdir", default="./out", help="Output directory")
    p.add_argument("--emit-client", action="store_true", help="Also emit a minimal Python client for LIMPS")
    p.add_argument("--host", default="localhost", help="Host for emitted client")
    p.add_argument("--port", type=int, default=8081, help="Port for emitted client")
    p.add_argument("--sep", help="Override delimiter detection (e.g., ',', '\\t', ';')")

    args = p.parse_args()

    if args.degree < 1:
        raise ValueError("--degree must be >= 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        raw_df, raw_delim = load_matrix(args.raw, sep_override=args.sep)
        test_df, test_delim = load_matrix(args.test, sep_override=args.sep)
    except Exception as e:
        raise RuntimeError(f"Failed to load matrices: {e}")

    info = {
        "raw_shape": raw_df.shape,
        "test_shape": test_df.shape,
        "raw_delimiter": raw_delim,
        "test_delimiter": test_delim,
        "raw_square_dim": infer_square_dim(raw_df.shape[1]),
        "test_square_dim": infer_square_dim(test_df.shape[1]),
        "degree": int(args.degree),
        "max_input_cols": int(args.max_input_cols),
        "max_target_cols": int(args.max_target_cols),
        "max_rows": int(args.max_rows),
        "lambda": float(args.lam),
    }

    n_rows = min(args.max_rows, raw_df.shape[0], test_df.shape[0])
    X_df = raw_df.iloc[:n_rows, :args.max_input_cols]
    Y_df = test_df.iloc[:n_rows, :args.max_target_cols]

    d = X_df.shape[1]
    feat_budget = _poly_feature_count(d, args.degree)
    max_feat = 100_000
    if feat_budget > max_feat:
        raise RuntimeError(
            f"Polynomial feature count {feat_budget} exceeds budget {max_feat} "
            f"(inputs d={d}, degree={args.degree}). "
            f"Try lowering --degree or --max-input-cols."
        )

    X_used, Y_used, valid_count, total_count = coerce_dropna_pair(X_df, Y_df)
    if X_used.size == 0 or Y_used.size == 0:
        raise RuntimeError(f"No valid rows after cleaning (valid {valid_count} / total {total_count}). "
                           f"Check delimiters or increase caps.")

    Phi, feat_names = poly_features(X_used, degree=args.degree)
    B = ridge_closed_form(Phi, Y_used, lam=args.lam)
    pred = Phi @ B
    rmse = rmse_columns(pred, Y_used).tolist()

    coef_path = outdir / "polynomial_surrogate_coefficients.npz"
    np.savez(
        coef_path,
        B=B,
        feat_names=np.array(feat_names, dtype=object),
        meta=np.array([{"degree": args.degree, "d": X_used.shape[1]}], dtype=object),
    )

    variables = [f"x{i+1}" for i in range(X_used.shape[1])]
    payload = build_payload(
        X_used=X_used,
        variables=variables,
        degree_limit=args.degree,
        min_rank=None,
        structure="dense",
        coeff_threshold=0.15,
        chebyshev=False,
        rmse_first10=rmse[:10],
        n_targets=Y_used.shape[1],
    )
    payload_path = outdir / "limps_payload.json"
    with payload_path.open("w") as f:
        json.dump(payload, f, indent=2)

    if args.emit_client:
        client_code = f'''import requests, json
import pathlib

class PolyOptimizerClient:
    def __init__(self, host="{args.host}", port={args.port}):
        self.url = f"http://{{host}}:{{port}}/optimize"

    def optimize_polynomials(self, matrix, variables, degree_limit=None, min_rank=None,
                             structure=None, coeff_threshold=0.15, chebyshev=False, timeout=30):
        payload = {{
            "matrix": matrix,
            "variables": variables,
            "coeff_threshold": coeff_threshold,
            "chebyshev": chebyshev,
        }}
        if degree_limit is not None:
            payload["degree_limit"] = degree_limit
        if min_rank is not None:
            payload["min_rank"] = min_rank
        if structure is not None:
            payload["structure"] = structure

        resp = requests.post(self.url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

if __name__ == "__main__":
    here = pathlib.Path(__file__).resolve().parent
    with open(here / "limps_payload.json", "r") as f:
        payload = json.load(f)
    client = PolyOptimizerClient()
    out = client.optimize_polynomials(
        matrix=payload["matrix"],
        variables=payload["variables"],
        degree_limit=payload.get("degree_limit"),
        min_rank=payload.get("min_rank"),
        structure=payload.get("structure"),
        coeff_threshold=payload.get("coeff_threshold", 0.15),
        chebyshev=payload.get("chebyshev", False),
    )
    print(json.dumps(out, indent=2))
'''
        client_path = outdir / "limps_client.py"
        with client_path.open("w") as f:
            f.write(client_code)

    report = {
        "info": info,
        "rmse_first10": rmse[:10],
        "rmse_all_targets": {f"y{j+1}": float(rmse[j]) for j in range(len(rmse))},
        "n_samples_fit": int(Phi.shape[0]),
        "n_features": int(Phi.shape[1]),
        "n_targets_fit": int(Y_used.shape[1]),
        "feature_names_count": len(feat_names),
        "feature_budget": int(feat_budget),
        "valid_rows": valid_count,
        "total_rows_considered": total_count,
        "coef_path": str(coef_path),
        "payload_path": str(payload_path),
    }
    report_path = outdir / "fit_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
