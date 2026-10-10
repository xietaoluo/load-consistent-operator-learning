"""
Per-model key-wall prediction grids for the baseline methods in Section 3.2
Table 3.1 (everything except the Fusion backbone, which already has
fig_3_2e_grid7x6).  One figure per model:

    5 rows  = Outer sleeve / Valve seat / Plug / Expander / Bottom
    3 cols  = Truth / Pred / Error

Predictions come from compare_methods.py on the SAME 7-wall data / split /
seed as Table 3.1; each method returns the concatenated 7-wall pressure
vector, which we slice per zone with D['geom'][op]['zone_slices'] and render
with the same PyVista + matplotlib pipeline as fig_3_2e_grid7x6 (absolute
pressure in kPa, inferno for field / magma for |error|, per-panel colorbars).

Usage:
    python fig_baseline_walls.py                 # default case O20_C12
    python fig_baseline_walls.py --case O80_C05
    python fig_baseline_walls.py --steps 30000   # Vanilla-DeepONet budget
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import h5py
import pyvista as pv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

pv.OFF_SCREEN = True
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import glob  # noqa: E402
import torch  # noqa: E402
import compare_methods as CM  # noqa: E402
from train_face_pressure import (FusionDeepONet, BRANCH_LAYERS, TRUNK_LAYERS,  # noqa: E402
                                 OUTPUT_DIM, DEVICE)

# match Table 3.1: baselines are evaluated on the 7-wall set
SEVEN_WALLS = ["wall_valve_body", "wall_out_sleeve", "wall_in_sleeve",
               "wall_valve_seat", "wall_plug", "wall_expand", "wall_bottom"]
CM.KEY_WALLS = SEVEN_WALLS

from _paths import DATA_PATH  # resolves VALVE_DATA env / ../data/*.h5
PAPER = os.path.abspath(os.path.join(BASE, "..", "examples", "outputs"))
os.makedirs(PAPER, exist_ok=True)
P_ATM = 101325.0

# rows requested (subset + order of the 7 walls)
ROWS = [("wall_out_sleeve", "Outer sleeve"),
        ("wall_valve_seat", "Valve seat"),
        ("wall_plug",       "Plug"),
        ("wall_expand",     "Expander"),
        ("wall_bottom",     "Bottom")]

CMAP_P = "inferno"     # pressure field
CMAP_E = "magma"       # absolute error
PANEL_PX = 620

def _newest(pattern):
    # Prefer a released checkpoint shipped in ../checkpoints (e.g.
    # fusion_load_consistency.pt); fall back to a locally trained run under
    # ./experiments/<run>/best_model.pt if present.
    from _paths import CKPT_DIR
    released = os.path.join(CKPT_DIR, 'fusion_load_consistency.pt')
    if os.path.exists(released):
        return released
    m = sorted(glob.glob(os.path.join(BASE, 'experiments', pattern)))
    return os.path.join(m[-1], 'best_model.pt') if m else '__missing__'


def _fusion_predict(ckpt_path, D):
    """Load a specific 7-wall Fusion checkpoint and predict val cases."""
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    nrm = ck['norms']
    pmin = np.array(nrm['params_min']); pmax = np.array(nrm['params_max'])
    cmin = np.array(nrm['coords_min']); cmax = np.array(nrm['coords_max'])
    prange = np.where(pmax - pmin == 0, 1, pmax - pmin)
    crange = np.where(cmax - cmin == 0, 1, cmax - cmin)
    data = D['data']; pred = {}
    for cid in D['val']:
        d = data[cid]; M = len(d['p'])
        pt = torch.tensor(((d['params'] - pmin) / prange)[None],
                          dtype=torch.float32, device=DEVICE)
        outs = []
        for s in range(0, M, 8192):
            c = torch.tensor(((d['coords'][s:s+8192] - cmin) / crange)[None],
                             dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                outs.append(model(pt, c).squeeze(0).cpu().numpy())
        pred[cid] = np.concatenate(outs) * nrm['p_std'] + nrm['p_mean']
    return pred


# model key -> (nice name, predictor factory).  The 5th is our proposed method
# (Fusion-DeepONet + load-consistency constraint = force_moment checkpoint).
MODELS = [
    ("nn",      "Nearest-Neighbor", lambda D, steps: CM.method_nn(D)),
    ("pod_krr", "POD-KRR",          lambda D, steps: CM.method_pod_krr(D)),
    ("pod_mlp", "POD-MLP",          lambda D, steps: CM.method_pod_mlp(D)),
    ("vanilla", "Vanilla DeepONet", lambda D, steps: CM.method_vanilla_deeponet(D, steps=steps)),
    ("ours",    "Ours (Fusion + load-consistency)",
     lambda D, steps: _fusion_predict(_newest('face_abl7_force_moment_s42_*'), D)),
]


def build_polydata(coords, face_conn, face_nnodes):
    faces, off = [], 0
    for nf in face_nnodes:
        faces.append(nf)
        faces.extend(face_conn[off:off + nf].tolist())
        off += nf
    return pv.PolyData(coords, faces=np.array(faces))


def compute_scales(D, preds_by_model, cid):
    """Cross-model unified colour windows per wall.

    Truth/Pred share the TRUTH field's own [min,max] (identical across all
    models, since the truth is the same) so the same real field renders
    identically in every figure and pred over/under-shoot is directly
    readable.  Error uses one absolute-kPa upper bound per wall = the max
    |err| over ALL models, so "who is worse" is comparable across figures.
    """
    op = D['data'][cid]['opening']
    zslc = D['geom'][op]['zone_slices']
    p_true_full = D['data'][cid]['p']
    scales = {}
    for z, _ in ROWS:
        s, e = zslc[z]
        vt = (p_true_full[s:e].astype(np.float64) + P_ATM) / 1000.0
        lo, hi = float(vt.min()), float(vt.max())
        mid = 0.5 * (lo + hi)
        min_span = 0.08 * abs(mid)          # keep near-flat faces from over-amplifying
        if (hi - lo) < min_span:
            lo, hi = mid - min_span / 2, mid + min_span / 2
        emax = 0.0
        for pred in preds_by_model.values():
            ae = np.abs(pred[cid][s:e] - p_true_full[s:e]).max() / 1000.0
            emax = max(emax, float(ae))
        scales[z] = {'lo': lo, 'hi': hi, 'emax': emax}
    return scales


def gather(D, pred_by_cid, h5, cid, scales):
    """Per-wall render payload for one model using the SHARED colour scales."""
    op = D['data'][cid]['opening']
    zslc = D['geom'][op]['zone_slices']
    p_pred_full = pred_by_cid[cid]
    p_true_full = D['data'][cid]['p']
    walls = {}
    for z, _ in ROWS:
        s, e = zslc[z]
        p_true = p_true_full[s:e].astype(np.float64)
        p_pred = p_pred_full[s:e].astype(np.float64)
        g = h5[f'geometry/O{op:03d}/{z}']
        coords = g['coords'][()]; fc = g['face_conn'][()]; fn = g['face_nnodes'][()]
        walls[z] = {
            'poly': build_polydata(coords, fc, fn),
            'true': (p_true + P_ATM) / 1000.0,
            'pred': (p_pred + P_ATM) / 1000.0,
            'err': np.abs(p_pred - p_true) / 1000.0,
            'lo': scales[z]['lo'], 'hi': scales[z]['hi'],
            'emax': scales[z]['emax'],
            'rel': float(np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)),
        }
    return walls


def render_panel(w, mode, tmp):
    pl = pv.Plotter(off_screen=True, window_size=(PANEL_PX, PANEL_PX))
    m = w['poly'].copy()
    if mode == 'err':
        m.cell_data['s'] = w['err']
        pl.add_mesh(m, scalars='s', cmap=CMAP_E, clim=(0, w['emax'] or 1e-9),
                    show_scalar_bar=False)
    else:
        m.cell_data['s'] = w[mode]
        pl.add_mesh(m, scalars='s', cmap=CMAP_P, clim=(w['lo'], w['hi']),
                    show_scalar_bar=False)
    pl.view_isometric()
    pl.camera.zoom(1.5)
    pl.screenshot(tmp, transparent_background=True)
    pl.close()
    return tmp


def make_figure(model_key, nice, walls, cid, disp_label):
    """Landscape layout: 3 rows (Truth / Pred / Error) x len(ROWS) columns
    (one wall per column).  A slim colorbar row sits under each image row."""
    modes = ["true", "pred", "err"]
    row_titles = ["Truth", "Pred", "Error"]
    ncol = len(ROWS)
    nrow = len(modes)
    fig = plt.figure(figsize=(ncol * 2.15, nrow * 2.25), dpi=200)
    # each image row is followed by a slim colorbar row
    hr = []
    for _ in range(nrow):
        hr += [1.0, 0.06]
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=hr,
                          wspace=0.06, hspace=0.10)
    tmpdir = tempfile.mkdtemp(prefix=f"bl_{model_key}_")

    for ci, (z, clabel) in enumerate(ROWS):
        w = walls[z]
        for mi, mode in enumerate(modes):
            ax = fig.add_subplot(gs[mi * 2, ci])
            png = render_panel(w, mode, os.path.join(tmpdir, f"{ci}_{mi}.png"))
            ax.imshow(plt.imread(png))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            # wall name as column header on the top row
            if mi == 0:
                ax.set_title(clabel, fontsize=12, pad=6, fontweight='bold')
            # Truth/Pred/Error label on the far-left column
            if ci == 0:
                ax.set_ylabel(row_titles[mi], fontsize=12, rotation=90,
                              labelpad=10, fontweight='bold')
            # horizontal colorbar beneath each panel
            cax = fig.add_subplot(gs[mi * 2 + 1, ci])
            if mode == 'err':
                norm = Normalize(0, w['emax'] or 1e-9); cmap = CMAP_E
            else:
                norm = Normalize(w['lo'], w['hi']); cmap = CMAP_P
            cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                              orientation='horizontal')
            cb.ax.tick_params(labelsize=6, length=2, pad=1)
            cb.outline.set_visible(False)

    fig.text(0.5, 0.995, f"{nice}  —  {disp_label}", ha='center', va='top',
             fontsize=13, fontweight='bold')
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.02)
    out = os.path.join(PAPER, f"fig_walls_{model_key}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="O20_C12")
    ap.add_argument("--steps", type=int, default=30000)
    args = ap.parse_args()

    D = CM.load_data()
    cid = args.case
    if cid not in D['data']:
        raise SystemExit(f"case {cid} not in dataset (val cases: {D['val']})")
    op = D['data'][cid]['opening']
    dp = (D['data'][cid]['params'][0] - D['data'][cid]['params'][1])
    disp = fr"$\theta$={op}%, $\Delta p$={dp:.0f} kPa"

    # 1) run every model first so colour scales can be unified across figures
    preds = {}
    for key, nice, factory in MODELS:
        print(f"[{key}] predicting ...", flush=True)
        preds[key] = factory(D, args.steps)

    # 2) shared per-wall windows (truth window + cross-model error ceiling)
    scales = compute_scales(D, preds, cid)
    print("shared error ceilings (kPa): " +
          "  ".join(f"{rl}:{scales[z]['emax']:.1f}" for z, rl in ROWS), flush=True)

    # 3) render every figure against the SAME scales
    for key, nice, _ in MODELS:
        h5 = h5py.File(DATA_PATH, 'r')
        walls = gather(D, preds[key], h5, cid, scales)
        h5.close()
        out = make_figure(key, nice, walls, cid, disp)
        rels = "  ".join(f"{rl}:{walls[z]['rel']:.3f}" for z, rl in ROWS)
        print(f"  -> {out}\n     relL2  {rels}", flush=True)


if __name__ == '__main__':
    main()
