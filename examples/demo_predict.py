#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Minimal end-to-end demo: load the released checkpoint and predict the
wall-pressure field for one operating condition in the demo dataset, then
report the per-wall relative-L2 error against the CFD ground truth.

Run from anywhere:
    python examples/demo_predict.py
    python examples/demo_predict.py --case O80_C05
"""
import os
import sys
import argparse
import numpy as np
import torch
import h5py

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from _paths import DATA_PATH, CKPT_DIR                       # noqa: E402
from train_face_pressure import (FusionDeepONet, BRANCH_LAYERS,  # noqa: E402
                                 TRUNK_LAYERS, OUTPUT_DIM)

KEY_WALLS = ["wall_plug", "wall_valve_body", "wall_expand", "wall_bottom"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="O50_C01", help="CaseID present in the dataset")
    ap.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "fusion_load_consistency.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    nrm = ck["norms"]
    pmin, pmax = np.array(nrm["params_min"]), np.array(nrm["params_max"])
    cmin, cmax = np.array(nrm["coords_min"]), np.array(nrm["coords_max"])
    pr = np.where(pmax - pmin == 0, 1, pmax - pmin)
    cr = np.where(cmax - cmin == 0, 1, cmax - cmin)

    cid = args.case
    op = int(cid.split("_")[0][1:])
    with h5py.File(DATA_PATH, "r") as f:
        if cid not in f["walls"]:
            raise SystemExit(f"case {cid} not in {DATA_PATH}. "
                             f"Available: {sorted(f['walls'].keys())}")
        a = f[f"integrals/{cid}"].attrs
        params = np.array([a["P_in_kPa"], a["P_out_kPa"], op], np.float32)
        pt = torch.tensor(((params - pmin) / pr)[None], dtype=torch.float32, device=device)

        print(f"Case {cid}: P_in={a['P_in_kPa']:.0f} kPa  P_out={a['P_out_kPa']:.0f} kPa  "
              f"opening={op}%  r={a['r']:.3f}")
        print(f"{'wall':20s} {'n_faces':>8s} {'rel-L2':>8s}  pred-range (kPa)")
        for z in KEY_WALLS:
            cen = f[f"geometry/O{op:03d}/{z}/centroid"][()]
            c = torch.tensor(((cen - cmin) / cr)[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(pt, c).squeeze(0).cpu().numpy() * nrm["p_std"] + nrm["p_mean"]
            ptrue = f[f"walls/{cid}/{z}/pressure"][()]
            rl2 = np.linalg.norm(pred - ptrue) / np.linalg.norm(ptrue)
            print(f"{z:20s} {len(ptrue):8d} {rl2:8.4f}  "
                  f"{pred.min()/1e3:6.0f} - {pred.max()/1e3:<6.0f}")


if __name__ == "__main__":
    main()
