"""
Standalone evaluator for a trained face-pressure checkpoint.

Rebuilds the SAME train/val split as training (deterministic in `seed`) and
reports the metrics the paper's ablation needs:
  - total rel-L2               (pointwise field accuracy, all walls concatenated)
  - per-wall rel-L2            (which surfaces the model gets right/wrong)
  - per-opening rel-L2         (generalization across openings)
  - total FSI force error      (|F_hat - F_cfd| / |F_cfd|, all walls summed)
  - total FSI moment error     (|M_hat - M_cfd| / |M_cfd|, about MOMENT_REF)

Force/moment use the SAME sign convention + P_ATM as physics_losses (absolute
pressure), and CFD truth is the stored integral (integrals/<cid>/zone/...).

Usage (standalone):
  python eval_faces.py <path/to/best_model.pt> [--seed 42]
"""

import os
import sys
import argparse
import numpy as np
import h5py
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_face_physics as T
import physics_losses as pl
from train_face_pressure import FusionDeepONet, BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEVEN_WALLS = ["wall_valve_body", "wall_out_sleeve", "wall_in_sleeve",
               "wall_valve_seat", "wall_plug", "wall_expand", "wall_bottom"]
MOMENT_REF = np.array([0.0, 0.0, 0.0], np.float32)
CHUNK = 8192


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck['norms']


def _npar(params, norms):
    lo = np.asarray(norms['params_min'], np.float64)
    hi = np.asarray(norms['params_max'], np.float64)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (params - lo) / rng


def _ncoord(c, norms):
    lo = np.asarray(norms['coords_min'], np.float64)
    hi = np.asarray(norms['coords_max'], np.float64)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (c - lo) / rng


def _predict_zone(model, norms, cen, params):
    """Return predicted gauge pressure (N,) in Pa for one zone."""
    pn = torch.tensor(_npar(params, norms), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    out = []
    with torch.no_grad():
        for s in range(0, len(cen), CHUNK):
            c = torch.tensor(_ncoord(cen[s:s + CHUNK], norms), dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
            out.append(model(pn, c).squeeze(0).cpu().numpy())
    p_norm = np.concatenate(out)
    return p_norm * norms['p_std'] + norms['p_mean']


def _val_cases(h5, walls, val_per_opening=2, seed=42):
    """Reproduce PhysicsFaceDataset's deterministic val split."""
    rng = np.random.default_rng(seed)
    cases = sorted(h5['walls'].keys())
    case_info = [(c, int(c.split('_')[0][1:])) for c in cases]
    openings = sorted(set(o for _, o in case_info))
    val = []
    for op in openings:
        oc = [c for c, o in case_info if o == op]
        rng.shuffle(oc)
        val += oc[:val_per_opening]
    return val


def evaluate(ckpt_path, walls=SEVEN_WALLS, seed=42, verbose=True):
    """Return a metrics dict for one checkpoint on the deterministic val split."""
    model, norms = load_model(ckpt_path)
    h5 = h5py.File(T.DATA_PATH, 'r')
    val_cases = _val_cases(h5, walls, seed=seed)

    # accumulators
    per_wall_num = {z: 0.0 for z in walls}   # sum ||pred-true||^2 per wall
    per_wall_den = {z: 0.0 for z in walls}
    per_open = {}                            # op -> [num, den]
    all_num = all_den = 0.0
    F_errs, M_errs = [], []

    ref = torch.tensor(MOMENT_REF, dtype=torch.float32)
    for cid in val_cases:
        ig = h5[f'integrals/{cid}']
        op = int(ig.attrs['opening'])
        params = np.array([ig.attrs['P_in_kPa'], ig.attrs['P_out_kPa'], op], np.float64)
        F_hat = np.zeros(3); F_cfd = np.zeros(3)
        M_hat = np.zeros(3); M_cfd = np.zeros(3)
        per_open.setdefault(op, [0.0, 0.0])
        for z in walls:
            g = h5[f'geometry/O{op:03d}/{z}']
            cen = g['centroid'][()].astype(np.float64)
            area = g['area'][()].astype(np.float64)
            normal = g['normal'][()].astype(np.float64)
            p_true = h5[f'walls/{cid}/{z}/pressure'][()].astype(np.float64)
            p_pred = _predict_zone(model, norms, cen, params)

            num = float(np.sum((p_pred - p_true) ** 2))
            den = float(np.sum(p_true ** 2))
            per_wall_num[z] += num; per_wall_den[z] += den
            per_open[op][0] += num; per_open[op][1] += den
            all_num += num; all_den += den

            # FSI resultants (absolute pressure, matches physics_losses)
            pt = torch.tensor(p_pred, dtype=torch.float32)
            at = torch.tensor(area, dtype=torch.float32)
            nt = torch.tensor(normal, dtype=torch.float32)
            ct = torch.tensor(cen, dtype=torch.float32)
            F_hat += pl.integrate_force(pt, at, nt).numpy()
            M_hat += pl.integrate_moment(pt, at, nt, ct, ref).numpy()
            F_cfd += ig[f'zone/{z}/F_pressure'][()]
            M_cfd += ig[f'zone/{z}/M_pressure'][()]

        F_errs.append(np.linalg.norm(F_hat - F_cfd) / (np.linalg.norm(F_cfd) + 1e-9))
        M_errs.append(np.linalg.norm(M_hat - M_cfd) / (np.linalg.norm(M_cfd) + 1e-9))
    h5.close()

    metrics = {
        'ckpt': ckpt_path,
        'n_val': len(val_cases),
        'rel_l2': float(np.sqrt(all_num / all_den)),
        'force_err': float(np.mean(F_errs)),
        'moment_err': float(np.mean(M_errs)),
        'per_wall': {z: float(np.sqrt(per_wall_num[z] / (per_wall_den[z] + 1e-30)))
                     for z in walls},
        'per_opening': {op: float(np.sqrt(v[0] / (v[1] + 1e-30)))
                        for op, v in sorted(per_open.items())},
    }
    if verbose:
        _print_metrics(metrics, walls)
    return metrics


def _print_metrics(m, walls):
    print(f"\ncheckpoint : {m['ckpt']}")
    print(f"val cases  : {m['n_val']}")
    print(f"rel-L2     : {m['rel_l2']:.4f}   (pointwise field, all walls)")
    print(f"force err  : {m['force_err']:.4f}   moment err : {m['moment_err']:.4f}")
    print("per-wall rel-L2:")
    for z in walls:
        print(f"   {z:18s} {m['per_wall'][z]:.4f}")
    print("per-opening rel-L2:")
    for op, v in m['per_opening'].items():
        print(f"   O{op:03d}  {v:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to best_model.pt")
    ap.add_argument("--seed", type=int, default=42, help="val-split seed (must match training)")
    args = ap.parse_args()
    ckpt = args.ckpt
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt)
    evaluate(ckpt, seed=args.seed)


if __name__ == '__main__':
    main()
