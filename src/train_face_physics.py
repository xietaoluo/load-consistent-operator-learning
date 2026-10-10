"""
Train PHYSICS-INFORMED Fusion-DeepONet for FSI-consistent wall-pressure.

Extends train_face_pressure.py with three physics losses (see
PLAN_physics_informed_operator.md):
  C1  force + moment consistency  (integrated resultant matches CFD truth)
  C2  wall-pressure spatial smoothness  (KNN tangential-gradient penalty)
  C3  inlet/outlet hard BC          (optional; not enabled by default)

Data loss (L_data) uses SAMPLED points (fast); physics losses use the
FULL face set of a case (accurate integration). Physics terms are ramped
in after a warmup so the net first learns the coarse field.

Config flags at top let run_ablation.py toggle each term.
"""

import os
import sys
import json
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_face_pressure import FusionDeepONet, BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM
import physics_losses as pl

# ============ Config ============
from _paths import DATA_PATH  # resolves VALVE_DATA env / ../data/*.h5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KEY_WALLS = ["wall_plug", "wall_valve_body", "wall_expand", "wall_bottom"]

# Training
N_PTS = 4096
BATCH_CASES = 4
LR = 1e-3
EPOCHS = 40000
VAL_EVERY = 500
PATIENCE = 6000
SEED = 42

# ---- Physics config (overridable by run_ablation) ----
USE_FORCE = True
USE_MOMENT = True
USE_SMOOTH = True
LAMBDA_F = 0.1
LAMBDA_M = 0.05
LAMBDA_S = 0.01
WARMUP = 5000          # steps of pure data loss
RAMP = 5000            # steps to linearly ramp physics weight to full
KNN_K = 8
SMOOTH_GATE = None     # e.g. set to a Pa/m scale to gate real high-gradient regions
PHYS_CASES = 2         # #cases per step used for physics losses (full-face fwd)
MOMENT_REF = np.array([0.0, 0.0, 0.0], np.float32)  # moment reference point
TAG = "phys"           # experiment name suffix
TRAIN_PER_OPENING = None  # None=all; int caps train cases/opening (sample-size sweep)


# ============ Dataset (extends baseline with geometry + true loads + KNN) ============

class PhysicsFaceDataset:
    def __init__(self, h5_path, key_walls, val_per_opening=2, seed=42, knn_k=8,
                 train_per_opening=None):
        """train_per_opening: if set, keep only the first N train cases per opening
        (after the deterministic shuffle). The val split is untouched, so subsets
        at different N are nested and share one fixed validation set -- required for
        a clean accuracy-vs-sample-size curve (paper Section 3.2)."""
        self.key_walls = key_walls
        self.rng = np.random.default_rng(seed)
        h5 = h5py.File(h5_path, 'r')

        cases = sorted(h5['walls'].keys())
        # Optional held-out extra cases via env var VALVE_HOLDOUT (comma-
        # separated CaseIDs): excluded from the shuffle, appended to val only.
        HOLDOUT_EXTRA = {c.strip() for c in
                         os.environ.get("VALVE_HOLDOUT", "").split(",") if c.strip()}
        extra_cases = [c for c in cases if c in HOLDOUT_EXTRA]
        cases = [c for c in cases if c not in HOLDOUT_EXTRA]
        case_info = [(c, int(c.split('_')[0][1:])) for c in cases]
        openings = sorted(set(o for _, o in case_info))

        self.train_cases, self.val_cases = [], []
        for op in openings:
            oc = [c for c, o in case_info if o == op]
            self.rng.shuffle(oc)
            self.val_cases += oc[:val_per_opening]
            op_train = oc[val_per_opening:]
            if train_per_opening is not None:
                op_train = op_train[:train_per_opening]
            self.train_cases += op_train

        # Held-out extras -> val only (never train).
        self.val_cases += extra_cases

        # per-opening geometry (shared mesh) + KNN
        self.geom = {}       # op -> {normal, area, centroid, knn_idx, knn_d2}
        for op in openings:
            normals, areas, cens = [], [], []
            for z in key_walls:
                g = h5[f'geometry/O{op:03d}/{z}']
                cens.append(g['centroid'][()].astype(np.float32))
                normals.append(g['normal'][()].astype(np.float32))
                areas.append(g['area'][()].astype(np.float32))
            cen = np.concatenate(cens); nrm = np.concatenate(normals); ar = np.concatenate(areas)
            idx, d2 = pl.precompute_knn(cen, k=knn_k)
            self.geom[op] = {'normal': nrm, 'area': ar, 'centroid': cen,
                             'knn_idx': idx, 'knn_d2': d2}

        # per-case pressure, params, true resultant loads
        self.data = {}
        for cid in self.train_cases + self.val_cases:
            ig = h5[f'integrals/{cid}']
            op = int(ig.attrs['opening'])
            params = np.array([ig.attrs['P_in_kPa'], ig.attrs['P_out_kPa'], op], np.float32)
            p_list = []
            F = np.zeros(3); M = np.zeros(3)
            for z in key_walls:
                p_list.append(h5[f'walls/{cid}/{z}/pressure'][()].astype(np.float32))
                F += h5[f'integrals/{cid}/zone/{z}/F_pressure'][()]
                M += h5[f'integrals/{cid}/zone/{z}/M_pressure'][()]
            self.data[cid] = {'pressure': np.concatenate(p_list),
                              'params': params, 'opening': op,
                              'F_cfd': F.astype(np.float32), 'M_cfd': M.astype(np.float32)}
        h5.close()

        self._compute_norms()
        print(f"Dataset: {len(self.train_cases)} train, {len(self.val_cases)} val")

    def _compute_norms(self):
        allp = np.stack([self.data[c]['params'] for c in self.train_cases])
        self.params_min = allp.min(0); self.params_max = allp.max(0)
        self.params_range = np.where(self.params_max - self.params_min == 0, 1,
                                     self.params_max - self.params_min)
        pr = np.concatenate([self.data[c]['pressure'] for c in self.train_cases])
        self.p_mean = float(pr.mean()); self.p_std = float(pr.std())
        allc = np.concatenate([self.geom[op]['centroid'] for op in self.geom])
        self.coords_min = allc.min(0); self.coords_max = allc.max(0)
        self.coords_range = np.where(self.coords_max - self.coords_min == 0, 1,
                                     self.coords_max - self.coords_min)
        print(f"  pressure mean={self.p_mean:.0f} std={self.p_std:.0f} Pa")

    def npar(self, p): return (p - self.params_min) / self.params_range
    def ncoord(self, c): return (c - self.coords_min) / self.coords_range
    def denorm_p(self, pn): return pn * self.p_std + self.p_mean

    def sample_data_batch(self, batch_size, n_pts):
        """Sampled points for L_data (batched, regular tensors)."""
        chosen = self.rng.choice(self.train_cases, batch_size, replace=True)
        pb, cb, yb = [], [], []
        for cid in chosen:
            d = self.data[cid]; op = d['opening']
            cen = self.geom[op]['centroid']; M = len(d['pressure'])
            idx = self.rng.choice(M, min(n_pts, M), replace=False)
            pb.append(self.npar(d['params']))
            cb.append(self.ncoord(cen[idx]))
            yb.append((d['pressure'][idx] - self.p_mean) / self.p_std)
        return (torch.tensor(np.stack(pb), dtype=torch.float32, device=DEVICE),
                torch.tensor(np.stack(cb), dtype=torch.float32, device=DEVICE),
                torch.tensor(np.stack(yb), dtype=torch.float32, device=DEVICE))

    def get_phys_case(self):
        """Return one random train case's FULL-face tensors for physics losses."""
        cid = self.rng.choice(self.train_cases)
        d = self.data[cid]; op = d['opening']; g = self.geom[op]
        T = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
        return {
            'params': T(self.npar(d['params'])).unsqueeze(0),         # (1,3)
            'coords': T(self.ncoord(g['centroid'])).unsqueeze(0),     # (1,N,3)
            'area': T(g['area']), 'normal': T(g['normal']),
            'centroid': T(g['centroid']),
            'p_true': T(d['pressure']),
            'F_cfd': T(d['F_cfd']), 'M_cfd': T(d['M_cfd']),
            'knn_idx': torch.tensor(g['knn_idx'], device=DEVICE),
            'knn_d2': torch.tensor(g['knn_d2'], dtype=torch.float32, device=DEVICE),
        }

    def full_val(self, model, n_pts=8192):
        model.eval()
        rel_l2s, force_errs = [], []
        with torch.no_grad():
            for cid in self.val_cases:
                d = self.data[cid]; op = d['opening']; g = self.geom[op]
                cen = g['centroid']; M = len(d['pressure'])
                params = torch.tensor(self.npar(d['params']), dtype=torch.float32,
                                      device=DEVICE).unsqueeze(0)
                preds = []
                for s in range(0, M, n_pts):
                    c = torch.tensor(self.ncoord(cen[s:s+n_pts]), dtype=torch.float32,
                                     device=DEVICE).unsqueeze(0)
                    preds.append(model(params, c).squeeze(0).cpu().numpy())
                pred_p = np.concatenate(preds) * self.p_std + self.p_mean
                rel_l2s.append(np.linalg.norm(pred_p - d['pressure']) /
                               np.linalg.norm(d['pressure']))
                # FSI force error
                pt = torch.tensor(pred_p, dtype=torch.float32)
                Fh = pl.integrate_force(pt, torch.tensor(g['area']),
                                        torch.tensor(g['normal'])).numpy()
                force_errs.append(np.linalg.norm(Fh - d['F_cfd']) /
                                  (np.linalg.norm(d['F_cfd']) + 1e-9))
        return float(np.mean(rel_l2s)), float(np.mean(force_errs))


# ============ Training ============

def train(cfg=None):
    """cfg: optional dict overriding physics flags/weights (for ablation)."""
    g = globals()
    if cfg:
        g.update(cfg)
    torch.manual_seed(SEED); np.random.seed(SEED)

    print(f"Device: {DEVICE}  | force={USE_FORCE} moment={USE_MOMENT} smooth={USE_SMOOTH}"
          f" | lF={LAMBDA_F} lM={LAMBDA_M} lS={LAMBDA_S}")
    tpo = g.get('TRAIN_PER_OPENING', None)
    ds = PhysicsFaceDataset(DATA_PATH, KEY_WALLS, val_per_opening=2, seed=SEED,
                            knn_k=KNN_K, train_per_opening=tpo)

    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    ref = torch.tensor(MOMENT_REF, dtype=torch.float32, device=DEVICE)

    exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments',
                           f'face_{TAG}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Exp: {exp_dir}\n" + "=" * 60)

    best_val, best_fe, best_step = float('inf'), None, 0
    hist = []
    t0 = time.time()

    for step in range(1, EPOCHS + 1):
        model.train()
        # --- data loss (sampled, batched) ---
        params, coords, p_true = ds.sample_data_batch(BATCH_CASES, N_PTS)
        pred = model(params, coords)
        loss_data = ((pred - p_true) ** 2).mean()
        loss = loss_data

        # --- physics losses (full-face, ramped) ---
        lf = lm = ls_ = 0.0
        if step > WARMUP and (USE_FORCE or USE_MOMENT or USE_SMOOTH):
            ramp = min(1.0, (step - WARMUP) / max(RAMP, 1))
            for _ in range(PHYS_CASES):
                pc = ds.get_phys_case()
                p_hat_n = model(pc['params'], pc['coords']).squeeze(0)   # (N,) norm
                p_hat = p_hat_n * ds.p_std + ds.p_mean                    # Pa
                if USE_FORCE:
                    lfi = pl.loss_force(p_hat, pc['area'], pc['normal'], pc['F_cfd'])
                    loss = loss + ramp * LAMBDA_F * lfi / PHYS_CASES; lf += float(lfi) / PHYS_CASES
                if USE_MOMENT:
                    lmi = pl.loss_moment(p_hat, pc['area'], pc['normal'], pc['centroid'],
                                         pc['M_cfd'], ref)
                    loss = loss + ramp * LAMBDA_M * lmi / PHYS_CASES; lm += float(lmi) / PHYS_CASES
                if USE_SMOOTH:
                    lsi = pl.loss_smooth(p_hat, pc['knn_idx'], pc['knn_d2'],
                                         p_true_pa=pc['p_true'], gate_scale=SMOOTH_GATE)
                    loss = loss + ramp * LAMBDA_S * lsi / PHYS_CASES; ls_ += float(lsi) / PHYS_CASES

        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append(loss_data.item())

        if step % VAL_EVERY == 0:
            val_rel, val_fe = ds.full_val(model)
            print(f"  step {step:6d} | data_mse={np.mean(hist[-VAL_EVERY:]):.5f} "
                  f"| val_relL2={val_rel:.4f} | val_Ferr={val_fe:.4f} "
                  f"| Lf={lf:.3e} Lm={lm:.3e} Ls={ls_:.3e} | {time.time()-t0:.0f}s")
            if val_rel < best_val:
                best_val, best_fe, best_step = val_rel, val_fe, step
                torch.save({'model': model.state_dict(), 'step': step,
                            'val_rel_l2': val_rel, 'val_force_err': val_fe,
                            'norms': {'params_min': ds.params_min.tolist(),
                                      'params_max': ds.params_max.tolist(),
                                      'p_mean': ds.p_mean, 'p_std': ds.p_std,
                                      'coords_min': ds.coords_min.tolist(),
                                      'coords_max': ds.coords_max.tolist()}},
                           os.path.join(exp_dir, 'best_model.pt'))
                print(f"    -> new best relL2={val_rel:.4f} Ferr={val_fe:.4f}")
            if step - best_step > PATIENCE:
                print(f"  early stop @ {step}"); break

    # Fallback: if training was too short to ever hit VAL_EVERY, no checkpoint
    # was saved. Run one validation + save so downstream eval never sees a
    # missing ckpt / None metrics (guards short smoke runs and pathological runs).
    if best_step == 0:
        val_rel, val_fe = ds.full_val(model)
        best_val, best_fe, best_step = val_rel, val_fe, EPOCHS
        torch.save({'model': model.state_dict(), 'step': best_step,
                    'val_rel_l2': val_rel, 'val_force_err': val_fe,
                    'norms': {'params_min': ds.params_min.tolist(),
                              'params_max': ds.params_max.tolist(),
                              'p_mean': ds.p_mean, 'p_std': ds.p_std,
                              'coords_min': ds.coords_min.tolist(),
                              'coords_max': ds.coords_max.tolist()}},
                   os.path.join(exp_dir, 'best_model.pt'))
        print(f"  (fallback save) relL2={val_rel:.4f} Ferr={val_fe:.4f}")

    print("=" * 60)
    print(f"Best: relL2={best_val:.4f}  Ferr={best_fe:.4f}  @step {best_step}")
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump({'USE_FORCE': USE_FORCE, 'USE_MOMENT': USE_MOMENT, 'USE_SMOOTH': USE_SMOOTH,
                   'LAMBDA_F': LAMBDA_F, 'LAMBDA_M': LAMBDA_M, 'LAMBDA_S': LAMBDA_S,
                   'WARMUP': WARMUP, 'RAMP': RAMP, 'best_val_rel_l2': best_val,
                   'best_force_err': best_fe, 'best_step': best_step, 'tag': TAG}, f, indent=2)
    return {'rel_l2': best_val, 'force_err': best_fe, 'step': best_step, 'exp_dir': exp_dir}


if __name__ == '__main__':
    train()

