"""
Compare surrogate methods for key-wall pressure prediction.

All methods share:
  - same dataset (data/dataset_face.h5), key walls only
  - same 100/20 train/val split (seed=42, 2 val per opening) as train_face_pressure.py
  - same rel-L2 metric (per-case, then averaged)

Methods:
  1. Nearest-neighbor (parameter space)          -- baseline floor
  2. POD + KRR (per-opening)                      -- classic ROM
  3. POD + MLP (per-opening)                      -- ROM + neural regressor
  4. Vanilla DeepONet (branch . trunk)            -- plain neural operator
  5. Fusion-DeepONet                              -- load existing best model

Cross-opening note: face count differs per opening, so POD is done PER OPENING
(10 train / 2 val each). NN and DeepONet variants work globally via coords.
"""

import os
import sys
import json
import glob
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_face_pressure import (FusionDeepONet, BRANCH_LAYERS, TRUNK_LAYERS,
                                 OUTPUT_DIM, KEY_WALLS, DEVICE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from _paths import DATA_PATH  # resolves VALVE_DATA env / ../data/*.h5
SEED = 42


# ============ Shared data loading ============

def load_data():
    """Load key-wall pressure + coords per case, grouped by opening.
    Returns dict: opening -> {'train':[...], 'val':[...], 'cases':{cid:{coords,p,params}}}
    plus global train/val case lists.
    """
    h5 = h5py.File(DATA_PATH, 'r')
    cases = sorted(h5['walls'].keys())
    # Optional held-out extra cases via env var VALVE_HOLDOUT (comma-separated
    # CaseIDs): excluded from the seeded shuffle so they never perturb the
    # split or POD basis, then force-appended to val as pure test points.
    HOLDOUT_EXTRA = {c.strip() for c in
                     os.environ.get("VALVE_HOLDOUT", "").split(",") if c.strip()}
    cases = [c for c in cases if c not in HOLDOUT_EXTRA]
    extra_cases = [c for c in sorted(h5['walls'].keys()) if c in HOLDOUT_EXTRA]
    case_info = [(c, int(c.split('_')[0][1:])) for c in cases]
    openings = sorted(set(o for _, o in case_info))

    rng = np.random.default_rng(SEED)
    per_opening = {}
    train_cases, val_cases = [], []
    for op in openings:
        op_cases = [c for c, o in case_info if o == op]
        rng.shuffle(op_cases)
        val = op_cases[:2]
        train = op_cases[2:]
        per_opening[op] = {'train': train, 'val': val}
        train_cases += train
        val_cases += val

    # Append held-out extras to their opening's val list (never train).
    for c in extra_cases:
        op = int(c.split('_')[0][1:])
        per_opening.setdefault(op, {'train': [], 'val': []})['val'].append(c)
        val_cases.append(c)

    # per-opening geometry (normal, area) for FSI force integration + zone slices
    geom = {}
    for op in openings:
        zone_slices = {}
        normals, areas = [], []
        off = 0
        for zone in KEY_WALLS:
            n = h5[f'geometry/O{op:03d}/{zone}/normal'][()]
            a = h5[f'geometry/O{op:03d}/{zone}/area'][()]
            zone_slices[zone] = (off, off + len(a))
            off += len(a)
            normals.append(n.astype(np.float32))
            areas.append(a.astype(np.float32))
        geom[op] = {'normal': np.concatenate(normals, 0),
                    'area': np.concatenate(areas, 0),
                    'zone_slices': zone_slices}

    data = {}
    for cid in train_cases + val_cases:
        op = int(cid.split('_')[0][1:])
        ig = h5[f'integrals/{cid}']
        params = np.array([ig.attrs['P_in_kPa'], ig.attrs['P_out_kPa'], op], np.float32)
        coords_list, p_list = [], []
        for zone in KEY_WALLS:
            centroid = h5[f'geometry/O{op:03d}/{zone}/centroid'][()]
            p = h5[f'walls/{cid}/{zone}/pressure'][()]
            coords_list.append(centroid.astype(np.float32))
            p_list.append(p.astype(np.float32))
        data[cid] = {
            'coords': np.concatenate(coords_list, 0),
            'p': np.concatenate(p_list, 0),
            'params': params,
            'opening': op,
        }
    h5.close()

    # Interpolation vs extrapolation tag: a val case is "extrapolation" if its
    # (P_in, P_out) lies outside the per-opening train bounding box.
    interp, extrap = [], []
    for op, split in per_opening.items():
        tp = np.stack([data[c]['params'][:2] for c in split['train']])
        lo, hi = tp.min(0), tp.max(0)
        for vc in split['val']:
            q = data[vc]['params'][:2]
            if np.all(q >= lo) and np.all(q <= hi):
                interp.append(vc)
            else:
                extrap.append(vc)

    return {'per_opening': per_opening, 'data': data,
            'train': train_cases, 'val': val_cases, 'openings': openings,
            'geom': geom, 'interp': interp, 'extrap': extrap}


P_ATM = 101325.0


def rel_l2(pred, true):
    return np.linalg.norm(pred - true) / np.linalg.norm(true)


def r2_score(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-30)


def fsi_force(p_gauge, normal, area):
    """Integrate pressure -> resultant force vector (N). F = sum -p_abs * n * A."""
    p_abs = p_gauge + P_ATM
    return (-(p_abs * area)[:, None] * normal).sum(0)


def eval_predictions(pred_by_cid, D, val_cases):
    """Compute full metric suite. Returns dict of aggregated metrics + per-case."""
    data = D['data']; geom = D['geom']
    per_case = {}
    for cid in val_cases:
        true = data[cid]['p']; pred = pred_by_cid[cid]
        op = data[cid]['opening']
        diff = pred - true
        # FSI force error (over all key walls combined)
        g = geom[op]
        F_true = fsi_force(true, g['normal'], g['area'])
        F_pred = fsi_force(pred, g['normal'], g['area'])
        force_err = np.linalg.norm(F_pred - F_true) / (np.linalg.norm(F_true) + 1e-9)
        per_case[cid] = {
            'rel_l2': rel_l2(pred, true),
            'rel_l1': np.sum(np.abs(diff)) / (np.sum(np.abs(true)) + 1e-30),
            'mae': np.mean(np.abs(diff)),
            'rmse': np.sqrt(np.mean(diff ** 2)),
            'max_ae': np.max(np.abs(diff)),
            'r2': r2_score(pred, true),
            'force_err': force_err,
        }

    def agg(keys, metric):
        return float(np.mean([per_case[c][metric] for c in keys]))

    metrics = {}
    for m in ['rel_l2', 'rel_l1', 'mae', 'rmse', 'max_ae', 'r2', 'force_err']:
        metrics[m] = agg(val_cases, m)
        if D['interp']:
            metrics[m + '_interp'] = agg([c for c in val_cases if c in D['interp']], m)
        if D['extrap']:
            metrics[m + '_extrap'] = agg([c for c in val_cases if c in D['extrap']], m)
    metrics['_per_case'] = per_case
    return metrics


# ============ Method 1: Nearest neighbor ============

def method_nn(D):
    """For each val case, find nearest train case IN SAME OPENING by (P_in,P_out)."""
    data = D['data']
    pred = {}
    for op, split in D['per_opening'].items():
        train, val = split['train'], split['val']
        tp = np.stack([data[c]['params'][:2] for c in train])  # (Ntr,2)
        # normalize params
        mu, sd = tp.mean(0), tp.std(0) + 1e-9
        tpn = (tp - mu) / sd
        for vc in val:
            q = (data[vc]['params'][:2] - mu) / sd
            j = int(np.argmin(((tpn - q) ** 2).sum(1)))
            pred[vc] = data[train[j]]['p'].copy()
    return pred


# ============ POD helpers (per-opening) ============

def build_pod(train_snaps, energy=0.9999, max_modes=None):
    """train_snaps: (Ntr, Nface). Returns mean, modes(Nface,r), coeffs(Ntr,r)."""
    mean = train_snaps.mean(0)
    X = train_snaps - mean
    U, S, Vt = np.linalg.svd(X.T, full_matrices=False)  # U:(Nface,Ntr)
    ev = S ** 2
    cum = np.cumsum(ev) / ev.sum()
    r = np.searchsorted(cum, energy) + 1
    if max_modes:
        r = min(r, max_modes)
    r = min(r, X.shape[0])
    modes = U[:, :r]                      # (Nface, r)
    coeffs = X @ modes                    # (Ntr, r)
    return mean, modes, coeffs


def method_pod_krr(D):
    data = D['data']
    pred = {}
    for op, split in D['per_opening'].items():
        train, val = split['train'], split['val']
        snaps = np.stack([data[c]['p'] for c in train])
        mean, modes, coeffs = build_pod(snaps, max_modes=len(train) - 1)
        Xtr = np.stack([data[c]['params'] for c in train])
        xs = StandardScaler().fit(Xtr)
        Xtr_n = xs.transform(Xtr)
        # one KRR per mode
        models = []
        for k in range(coeffs.shape[1]):
            m = KernelRidge(kernel='rbf', alpha=1e-3,
                            gamma=1.0 / Xtr_n.shape[1])
            m.fit(Xtr_n, coeffs[:, k])
            models.append(m)
        for vc in val:
            xq = xs.transform(data[vc]['params'][None])
            c = np.array([m.predict(xq)[0] for m in models])
            pred[vc] = mean + modes @ c
    return pred


def method_pod_mlp(D):
    data = D['data']
    pred = {}
    for op, split in D['per_opening'].items():
        train, val = split['train'], split['val']
        snaps = np.stack([data[c]['p'] for c in train])
        mean, modes, coeffs = build_pod(snaps, max_modes=len(train) - 1)
        Xtr = np.stack([data[c]['params'] for c in train])
        xs = StandardScaler().fit(Xtr)
        cs = StandardScaler().fit(coeffs)
        Xtr_n = torch.tensor(xs.transform(Xtr), dtype=torch.float32)
        Ctr_n = torch.tensor(cs.transform(coeffs), dtype=torch.float32)
        r = coeffs.shape[1]
        net = nn.Sequential(nn.Linear(3, 64), nn.GELU(),
                            nn.Linear(64, 64), nn.GELU(),
                            nn.Linear(64, r))
        opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
        for ep in range(3000):
            opt.zero_grad()
            loss = ((net(Xtr_n) - Ctr_n) ** 2).mean()
            loss.backward(); opt.step()
        net.eval()
        for vc in val:
            xq = torch.tensor(xs.transform(data[vc]['params'][None]), dtype=torch.float32)
            with torch.no_grad():
                cn = net(xq).numpy()[0]
            c = cs.inverse_transform(cn[None])[0]
            pred[vc] = mean + modes @ c
    return pred


# ============ Method 4: Vanilla DeepONet ============

class VanillaDeepONet(nn.Module):
    def __init__(self, p_dim=3, x_dim=3, width=128, out=128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(p_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, out))
        self.trunk = nn.Sequential(
            nn.Linear(x_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, out), nn.GELU())
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, params, coords):
        if params.dim() == 2:
            params = params.unsqueeze(1)
        b = self.branch(params.squeeze(1))          # (B,out)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(b.shape[0], -1, -1)
        t = self.trunk(coords)                       # (B,N,out)
        return (b.unsqueeze(1) * t).sum(-1) + self.bias


def _make_norms(D):
    data = D['data']
    tr = D['train']
    P = np.stack([data[c]['params'] for c in tr])
    C = np.concatenate([data[c]['coords'] for c in tr + D['val']])
    allp = np.concatenate([data[c]['p'] for c in tr])
    return {
        'pmin': P.min(0), 'pmax': P.max(0),
        'cmin': C.min(0), 'cmax': C.max(0),
        'p_mean': allp.mean(), 'p_std': allp.std(),
    }


def method_vanilla_deeponet(D, steps=30000, n_pts=4096, batch=4):
    data = D['data']; nrm = _make_norms(D)
    prange = np.where(nrm['pmax'] - nrm['pmin'] == 0, 1, nrm['pmax'] - nrm['pmin'])
    crange = np.where(nrm['cmax'] - nrm['cmin'] == 0, 1, nrm['cmax'] - nrm['cmin'])

    def npar(p): return (p - nrm['pmin']) / prange
    def ncoord(c): return (c - nrm['cmin']) / crange
    def np_(p): return (p - nrm['p_mean']) / nrm['p_std']

    model = VanillaDeepONet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    rng = np.random.default_rng(SEED)
    tr = D['train']
    for step in range(steps):
        model.train()
        ch = rng.choice(tr, batch)
        pb, cb, yb = [], [], []
        for cid in ch:
            d = data[cid]; M = len(d['p'])
            idx = rng.choice(M, min(n_pts, M), replace=False)
            pb.append(npar(d['params'])); cb.append(ncoord(d['coords'][idx])); yb.append(np_(d['p'][idx]))
        pb = torch.tensor(np.stack(pb), dtype=torch.float32, device=DEVICE)
        cb = torch.tensor(np.stack(cb), dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(np.stack(yb), dtype=torch.float32, device=DEVICE)
        pred = model(pb, cb)
        loss = ((pred - yb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    model.eval()
    pred = {}
    for cid in D['val']:
        d = data[cid]; M = len(d['p'])
        pt = torch.tensor(npar(d['params'])[None], dtype=torch.float32, device=DEVICE)
        outs = []
        for s in range(0, M, 8192):
            c = torch.tensor(ncoord(d['coords'][s:s+8192])[None], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                outs.append(model(pt, c).squeeze(0).cpu().numpy())
        pr = np.concatenate(outs) * nrm['p_std'] + nrm['p_mean']
        pred[cid] = pr
    return pred


# ============ Method 5: load Fusion-DeepONet ============

def method_fusion_load(D):
    exps = sorted(glob.glob(os.path.join(BASE_DIR, 'experiments', 'face_fusion_*')))
    if not exps:
        return None
    ckpt = torch.load(os.path.join(exps[-1], 'best_model.pt'), map_location=DEVICE)
    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(ckpt['model']); model.eval()
    nrm = ckpt['norms']
    pmin = np.array(nrm['params_min']); pmax = np.array(nrm['params_max'])
    cmin = np.array(nrm['coords_min']); cmax = np.array(nrm['coords_max'])
    prange = np.where(pmax - pmin == 0, 1, pmax - pmin)
    crange = np.where(cmax - cmin == 0, 1, cmax - cmin)
    data = D['data']; pred = {}
    for cid in D['val']:
        d = data[cid]; M = len(d['p'])
        pt = torch.tensor(((d['params'] - pmin) / prange)[None], dtype=torch.float32, device=DEVICE)
        outs = []
        for s in range(0, M, 8192):
            c = torch.tensor(((d['coords'][s:s+8192] - cmin) / crange)[None], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                outs.append(model(pt, c).squeeze(0).cpu().numpy())
        pred[cid] = np.concatenate(outs) * nrm['p_std'] + nrm['p_mean']
    return pred


# ============ Main ============

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("Loading data...")
    D = load_data()
    print(f"  {len(D['train'])} train / {len(D['val'])} val, key walls={KEY_WALLS}")

    methods = [
        ('Nearest-Neighbor', method_nn),
        ('POD+KRR', method_pod_krr),
        ('POD+MLP', method_pod_mlp),
        ('Vanilla-DeepONet', method_vanilla_deeponet),
        ('Fusion-DeepONet', method_fusion_load),
    ]

    print(f"  interp={len(D['interp'])} extrap={len(D['extrap'])} val cases")

    results = {}
    for name, fn in methods:
        t0 = time.time()
        print(f"\n>>> {name} ...")
        try:
            pred = fn(D)
            if pred is None:
                print(f"    (skipped)"); continue
            met = eval_predictions(pred, D, D['val'])
            met['time'] = time.time() - t0
            results[name] = met
            print(f"    rel-L2={met['rel_l2']:.4f}  MAE={met['mae']/1000:.2f}kPa  "
                  f"RMSE={met['rmse']/1000:.2f}kPa  Linf={met['max_ae']/1000:.1f}kPa  "
                  f"R2={met['r2']:.4f}  Ferr={met['force_err']:.4f}  ({met['time']:.0f}s)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    FAILED: {e}")

    # ---- Main metric table ----
    order = sorted(results.items(), key=lambda x: x[1]['rel_l2'])
    print("\n" + "=" * 92)
    print("MULTI-METRIC COMPARISON (key walls, val set)")
    print("=" * 92)
    hdr = f"{'Method':<20}{'relL2':>8}{'MAE[kPa]':>10}{'RMSE[kPa]':>11}{'Linf[kPa]':>11}{'R2':>9}{'F_err':>8}{'t[s]':>7}"
    print(hdr); print("-" * 92)
    for name, r in order:
        print(f"{name:<20}{r['rel_l2']:>8.4f}{r['mae']/1000:>10.2f}{r['rmse']/1000:>11.2f}"
              f"{r['max_ae']/1000:>11.1f}{r['r2']:>9.4f}{r['force_err']:>8.4f}{r['time']:>7.0f}")

    # ---- Interpolation vs extrapolation table ----
    print("\n" + "=" * 60)
    print("INTERPOLATION vs EXTRAPOLATION (rel-L2)")
    print("=" * 60)
    print(f"{'Method':<20}{'interp':>12}{'extrap':>12}")
    print("-" * 60)
    for name, r in order:
        vi = r.get('rel_l2_interp', float('nan'))
        ve = r.get('rel_l2_extrap', float('nan'))
        print(f"{name:<20}{vi:>12.4f}{ve:>12.4f}")

    # ---- Save JSON + plots ----
    out = os.path.join(BASE_DIR, 'experiments', 'method_comparison.json')
    save = {}
    for k, v in results.items():
        save[k] = {m: float(v[m]) for m in v if m not in ('_per_case',)}
    with open(out, 'w') as f:
        json.dump(save, f, indent=2)
    print(f"\nSaved -> {out}")

    make_plots(results, order)


def make_plots(results, order):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [n for n, _ in order]
    metrics_plot = [
        ('rel_l2', 'Relative L2', 1.0),
        ('rmse', 'RMSE [kPa]', 1e-3),
        ('max_ae', 'Max abs error [kPa]', 1e-3),
        ('r2', 'R2 (higher better)', 1.0),
        ('force_err', 'FSI force rel-error', 1.0),
        ('mae', 'MAE [kPa]', 1e-3),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = ['#2a9d8f', '#8ab17d', '#e9c46a', '#f4a261', '#e76f51']
    for ax, (mkey, title, scale) in zip(axes.flat, metrics_plot):
        vals = [results[n][mkey] * scale for n in names]
        bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.set_title(title, fontsize=11)
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f' {v:.3g}', va='center', fontsize=9)
        ax.tick_params(labelsize=9)
    fig.suptitle('Key-wall pressure prediction: multi-metric comparison (120 cases, 100/20)',
                 fontsize=13)
    plt.tight_layout()
    p = os.path.join(BASE_DIR, 'experiments', 'method_comparison_multi.png')
    plt.savefig(p, dpi=130); plt.close()
    print(f"Saved -> {p}")


if __name__ == '__main__':
    main()
