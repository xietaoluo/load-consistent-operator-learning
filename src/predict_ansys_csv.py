"""
Predict wall pressure from a trained Fusion-DeepONet checkpoint and export
an ANSYS External Data CSV (mm / MPa, ABSOLUTE pressure) -- same format as
export_ansys_csv.py but the pressure comes from the SURROGATE, not CFD truth.

The DeepONet needs query coordinates, so we reuse the face-center mesh of the
requested opening (stored in dataset_face.h5 geometry group). Operating
conditions (P_in, P_out, theta) can be given directly or pulled from a case.

Usage:
  # predict a case that exists in the dataset (also prints truth comparison)
  python predict_ansys_csv.py --case O50_C01

  # predict an arbitrary operating point at an existing opening's mesh
  python predict_ansys_csv.py --pin 800 --pout 550 --opening 50

  # pick a specific checkpoint / export all key walls
  python predict_ansys_csv.py --case O50_C01 --ckpt experiments/face_abl_full_.../best_model.pt --all
"""

import os
import sys
import glob
import argparse
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_face_pressure import FusionDeepONet, BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM

BASE = os.path.dirname(os.path.abspath(__file__))
from _paths import DATA_PATH  # resolves VALVE_DATA env / ../data/*.h5
OUT_DIR = os.path.join(BASE, "ansys_csv")
P_ATM = 101325.0
# Full 7-wall FSI surface set (matches train_7walls.py)
KEY_WALLS = ["wall_valve_body", "wall_out_sleeve", "wall_in_sleeve", "wall_valve_seat",
             "wall_plug", "wall_expand", "wall_bottom"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pick_default_ckpt():
    """Prefer the 7-wall physics model; then any physics-full; else newest."""
    cands = sorted(glob.glob(os.path.join(BASE, "experiments", "*", "best_model.pt")))
    if not cands:
        raise FileNotFoundError("No best_model.pt found under experiments/")
    for key in ("7walls", "abl_full", "_phys_"):
        matches = [c for c in cands if key in c]
        if matches:
            return matches[-1]   # newest (cands sorted ascending)
    return cands[-1]


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(ck['model'])
    model.eval()
    n = ck['norms']
    norms = {k: np.asarray(v, dtype=np.float64) if isinstance(v, list) else float(v)
             for k, v in n.items()}
    return model, norms, ck.get('val_rel_l2', None)


def predict_zone(model, norms, coords_m, params):
    """coords_m (N,3) meters; params (3,)=(P_in,P_out,theta). Returns gauge Pa (N,)."""
    pmin, pmax = norms['params_min'], norms['params_max']
    prange = np.where(pmax - pmin == 0, 1.0, pmax - pmin)
    cmin, cmax = norms['coords_min'], norms['coords_max']
    crange = np.where(cmax - cmin == 0, 1.0, cmax - cmin)

    pn = torch.tensor(((params - pmin) / prange), dtype=torch.float32,
                      device=DEVICE).unsqueeze(0)          # (1,3)
    preds = []
    with torch.no_grad():
        for s in range(0, len(coords_m), 8192):
            c = (coords_m[s:s+8192] - cmin) / crange
            ct = torch.tensor(c, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            preds.append(model(pn, ct).squeeze(0).cpu().numpy())
    p_norm = np.concatenate(preds)
    return p_norm * norms['p_std'] + norms['p_mean']       # gauge Pa


def write_csv(cid_tag, zone, coords_m, p_gauge, gauge=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    xyz_mm = coords_m * 1000.0
    # gauge=True -> export gauge (matches Fluent Static Pressure / no operating-pressure added)
    # gauge=False -> export absolute = gauge + P_ATM
    p_out_MPa = (p_gauge if gauge else p_gauge + P_ATM) / 1.0e6
    suffix = "gauge" if gauge else "pred"
    out = os.path.join(OUT_DIR, f"{cid_tag}_{zone}_{suffix}.csv")
    with open(out, "w", newline="") as f:
        f.write("X [mm],Y [mm],Z [mm],Pressure [MPa]\n")
        for i in range(len(p_gauge)):
            f.write(f"{xyz_mm[i,0]:.6f},{xyz_mm[i,1]:.6f},{xyz_mm[i,2]:.6f},{p_out_MPa[i]:.9f}\n")
    return out, p_out_MPa


def resultant_force(p_gauge, area, normal, gauge=False):
    p = p_gauge if gauge else p_gauge + P_ATM
    dF = -(p * area)[:, None] * normal
    return dF.sum(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="dataset CaseID e.g. O50_C01 (uses its P_in/P_out/theta + truth compare)")
    ap.add_argument("--pin", type=float, default=None, help="P_in [kPa]")
    ap.add_argument("--pout", type=float, default=None, help="P_out [kPa]")
    ap.add_argument("--opening", type=int, default=None, help="opening (mesh to query), must exist in dataset geometry")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default: physics-full)")
    ap.add_argument("--all", action="store_true", help="export all key walls (default: wall_plug only)")
    ap.add_argument("--gauge", action="store_true",
                    help="export GAUGE pressure (no +101325 Pa) to match Fluent-direct External Data")
    args = ap.parse_args()

    ckpt = args.ckpt or pick_default_ckpt()
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(BASE, ckpt)
    model, norms, val_rel = load_model(ckpt)
    print(f"checkpoint : {ckpt}")
    print(f"  trained val rel-L2 = {val_rel}")

    h5 = h5py.File(DATA_PATH, "r")

    # resolve operating point + opening
    have_truth = False
    if args.case:
        cid = args.case
        ig = h5[f"integrals/{cid}"]
        opening = int(ig.attrs["opening"])
        params = np.array([ig.attrs["P_in_kPa"], ig.attrs["P_out_kPa"], opening], np.float64)
        tag = cid
        have_truth = True
    else:
        if None in (args.pin, args.pout, args.opening):
            sys.exit("Provide --case OR all of --pin --pout --opening")
        opening = args.opening
        params = np.array([args.pin, args.pout, opening], np.float64)
        tag = f"P{int(args.pin)}_{int(args.pout)}_O{opening:03d}"

    zones = KEY_WALLS if args.all else ["wall_plug"]
    ptype = "GAUGE" if args.gauge else "ABSOLUTE"
    print(f"case/point : {tag}  P_in={params[0]:.1f} P_out={params[1]:.1f} theta={params[2]:.0f}")
    print(f"units      : mm / MPa, {ptype} pressure\n" + "=" * 72)

    for z in zones:
        gkey = f"geometry/O{opening:03d}/{z}"
        if gkey not in h5:
            print(f"{z:16s} SKIP (no mesh for opening {opening})"); continue
        cen = h5[f"{gkey}/centroid"][()].astype(np.float64)
        area = h5[f"{gkey}/area"][()].astype(np.float64)
        normal = h5[f"{gkey}/normal"][()].astype(np.float64)

        p_pred = predict_zone(model, norms, cen, params)
        out, p_out = write_csv(tag, z, cen, p_pred, gauge=args.gauge)
        F = resultant_force(p_pred, area, normal, gauge=args.gauge)

        plabel = "p_g " if args.gauge else "p_abs"
        line = (f"{z:16s} N={len(p_pred):6d}  {plabel}=[{p_out.min():.4f},{p_out.max():.4f}] MPa"
                f"  |F|={np.linalg.norm(F):8.1f} N")
        if have_truth:
            p_true = np.concatenate([h5[f"walls/{cid}/{z}/pressure"][()].astype(np.float64)]) \
                if f"walls/{cid}/{z}/pressure" in h5 else None
            if p_true is not None:
                rel = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)
                F_true = resultant_force(p_true, area, normal, gauge=args.gauge)
                fe = np.linalg.norm(F - F_true) / (np.linalg.norm(F_true) + 1e-9)
                line += f"  | rel-L2={rel:.4f}  F_err={fe:.4f}"
        print(line)
        print(f"{'':16s} F=({F[0]:.1f},{F[1]:.1f},{F[2]:.1f}) N  -> {out}")

    h5.close()
    print("=" * 72)
    csv_tag = "*_gauge.csv (GAUGE)" if args.gauge else "*_pred.csv (ABSOLUTE)"
    print(f"Import the {csv_tag} into ANSYS External Data (mm/MPa) exactly like the")
    print("CFD-truth CSV. Reaction force should match |F| above within model error.")
    if args.gauge:
        print("GAUGE mode: matches Fluent-direct coupling (Static Pressure, no operating pressure).")


if __name__ == "__main__":
    main()
