"""
Train Fusion-DeepONet for wall-pressure prediction on key faces.

Input:  (P_in, P_out, theta) -> face-centered pressure on key walls
Data:   data/dataset_face.h5
Architecture: Fusion-DeepONet (branch modulates trunk layer-by-layer)
  - Branch: MLP on (P_in_norm, P_out_norm, theta_norm) -> hidden states
  - Trunk: MLP on (x, y, z) face-center coords -> spatial basis
  - Output: scalar pressure per query point

Training strategy:
  - Per-step: pick a random case, sample N_PTS face centers from key walls
  - Min-max normalize inputs; pressure normalized per-dataset
  - Validate on held-out cases (2 per opening = 20 total)
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

# ============ Config ============
from _paths import DATA_PATH  # resolves VALVE_DATA env / ../data/*.h5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
KEY_WALLS = ["wall_plug", "wall_valve_body", "wall_expand", "wall_bottom"]

# Architecture
BRANCH_LAYERS = [3, 128, 128, 128]   # input=3 (P_in, P_out, theta)
TRUNK_LAYERS = [3, 128, 128, 128]    # input=3 (x, y, z)
OUTPUT_DIM = 128                       # latent dim (branch & trunk output width)

# Training
N_PTS = 4096        # face points sampled per step
BATCH_CASES = 4     # cases per gradient step
LR = 1e-3
EPOCHS = 50000
VAL_EVERY = 500
PATIENCE = 5000     # early stopping patience (steps without improvement)
SEED = 42

# ============ Model ============

class RowdyActivation(nn.Module):
    """Trainable activation: a*tanh(x+c) + a1*sin(F1*x+c1)"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.c = nn.Parameter(torch.zeros(1))
        self.a1 = nn.Parameter(torch.tensor(0.1))
        self.F1 = nn.Parameter(torch.ones(1))
        self.c1 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.a * torch.tanh(x + self.c) + self.a1 * torch.sin(self.F1 * x + self.c1)


class FusionDeepONet(nn.Module):
    """
    Branch modulates Trunk at every hidden layer (Peyvan 2024 style).
    Branch: MLP -> sequence of hidden vectors S^0, S^1, ...
    Trunk: at layer l, Y^l = activation(W @ Y^{l-1} + b) * S^{l-1}
    Final: dot(branch_out, trunk_out) + bias -> scalar
    """
    def __init__(self, branch_layers, trunk_layers, output_dim):
        super().__init__()
        # Branch layers
        self.branch_layers = nn.ModuleList()
        self.branch_acts = nn.ModuleList()
        for i in range(len(branch_layers) - 1):
            self.branch_layers.append(nn.Linear(branch_layers[i], branch_layers[i+1]))
            self.branch_acts.append(RowdyActivation())
        self.branch_out = nn.Linear(branch_layers[-1], output_dim)

        # Trunk layers (same hidden widths for fusion)
        self.trunk_layers = nn.ModuleList()
        self.trunk_acts = nn.ModuleList()
        for i in range(len(trunk_layers) - 1):
            self.trunk_layers.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            self.trunk_acts.append(RowdyActivation())
        self.trunk_out = nn.Linear(trunk_layers[-1], output_dim)

        # Modulation projections: branch hidden -> trunk hidden width
        n_mod = min(len(branch_layers) - 1, len(trunk_layers) - 1)
        self.mod_projs = nn.ModuleList()
        for i in range(n_mod):
            self.mod_projs.append(nn.Linear(branch_layers[min(i+1, len(branch_layers)-1)],
                                           trunk_layers[min(i+1, len(trunk_layers)-1)]))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        """
        branch_input: (B, 3) or (B, 1, 3)  operating conditions
        trunk_input: (B, N, 3) or (N, 3)   spatial coords
        Returns: (B, N) pressure predictions
        """
        if branch_input.dim() == 2:
            branch_input = branch_input.unsqueeze(1)  # (B,1,3)
        B = branch_input.shape[0]

        # Branch forward: collect hidden states
        h = branch_input.squeeze(1)  # (B, 3)
        branch_hiddens = []
        for i, (layer, act) in enumerate(zip(self.branch_layers, self.branch_acts)):
            h = act(layer(h))
            branch_hiddens.append(h)
        b_out = self.branch_out(h)  # (B, output_dim)

        # Trunk forward with modulation
        if trunk_input.dim() == 2:
            trunk_input = trunk_input.unsqueeze(0).expand(B, -1, -1)
        N = trunk_input.shape[1]
        t = trunk_input  # (B, N, 3)
        for i, (layer, act) in enumerate(zip(self.trunk_layers, self.trunk_acts)):
            t = act(layer(t))  # (B, N, hidden)
            if i < len(self.mod_projs):
                s = self.mod_projs[i](branch_hiddens[min(i, len(branch_hiddens)-1)])  # (B, hidden)
                s = torch.sigmoid(s).unsqueeze(1)  # (B, 1, hidden)
                t = t * s  # modulate
        t_out = self.trunk_out(t)  # (B, N, output_dim)

        # Dot product
        out = torch.sum(b_out.unsqueeze(1) * t_out, dim=-1) + self.bias  # (B, N)
        return out


# ============ Dataset ============

class FaceDataset:
    """Load dataset, handle normalization, train/val split."""

    def __init__(self, h5_path, key_walls, val_per_opening=2, seed=42):
        self.h5 = h5py.File(h5_path, 'r')
        self.key_walls = key_walls
        self.rng = np.random.default_rng(seed)

        # Load case list and split
        cases = list(self.h5['walls'].keys())
        cases.sort()
        # Optional held-out extra cases: any CaseIDs listed in the env var
        # VALVE_HOLDOUT (comma-separated) are excluded from the seeded shuffle
        # and appended to val only -- useful for adding pure model-unseen test
        # points without perturbing the published split. Default: none.
        HOLDOUT_EXTRA = {c.strip() for c in
                         os.environ.get("VALVE_HOLDOUT", "").split(",") if c.strip()}
        extra_cases = [c for c in cases if c in HOLDOUT_EXTRA]
        cases = [c for c in cases if c not in HOLDOUT_EXTRA]
        # Parse opening from CaseID (e.g. O50_C01 -> 50)
        case_info = [(c, int(c.split('_')[0][1:])) for c in cases]
        openings = sorted(set(o for _, o in case_info))

        self.train_cases = []
        self.val_cases = []
        for op in openings:
            op_cases = [c for c, o in case_info if o == op]
            self.rng.shuffle(op_cases)
            self.val_cases.extend(op_cases[:val_per_opening])
            self.train_cases.extend(op_cases[val_per_opening:])

        # Held-out extras -> val only (never train).
        self.val_cases.extend(extra_cases)

        print(f"Dataset: {len(self.train_cases)} train, {len(self.val_cases)} val")
        print(f"Key walls: {key_walls}")

        # Pre-load all data into memory for speed
        self._load_all()
        self._compute_norms()

    def _load_all(self):
        """Load all case data into numpy arrays."""
        self.data = {}  # cid -> {'coords': (M,3), 'pressure': (M,), 'params': (3,)}
        for cid in self.train_cases + self.val_cases:
            ig = self.h5[f'integrals/{cid}']
            p_in = ig.attrs['P_in_kPa']
            p_out = ig.attrs['P_out_kPa']
            opening = ig.attrs['opening']
            params = np.array([p_in, p_out, opening], dtype=np.float32)

            # Concatenate key wall face data
            coords_list = []
            p_list = []
            for zone in self.key_walls:
                geom = self.h5[f'geometry/O{int(opening):03d}/{zone}']
                centroid = geom['centroid'][()]
                p = self.h5[f'walls/{cid}/{zone}/pressure'][()]
                coords_list.append(centroid.astype(np.float32))
                p_list.append(p.astype(np.float32))
            coords = np.concatenate(coords_list, axis=0)
            pressure = np.concatenate(p_list, axis=0)
            self.data[cid] = {'coords': coords, 'pressure': pressure, 'params': params}

        self.h5.close()

    def _compute_norms(self):
        """Compute normalization stats from training set only."""
        all_params = np.stack([self.data[c]['params'] for c in self.train_cases])
        self.params_min = all_params.min(axis=0)
        self.params_max = all_params.max(axis=0)
        self.params_range = self.params_max - self.params_min
        self.params_range[self.params_range == 0] = 1.0

        all_p = np.concatenate([self.data[c]['pressure'] for c in self.train_cases])
        self.p_mean = all_p.mean()
        self.p_std = all_p.std()

        # Coords: global min/max across all openings
        all_coords = np.concatenate([self.data[c]['coords']
                                     for c in self.train_cases + self.val_cases])
        self.coords_min = all_coords.min(axis=0)
        self.coords_max = all_coords.max(axis=0)
        self.coords_range = self.coords_max - self.coords_min
        self.coords_range[self.coords_range == 0] = 1.0

        print(f"  params range: {self.params_min} -> {self.params_max}")
        print(f"  pressure: mean={self.p_mean:.0f}, std={self.p_std:.0f} Pa")
        print(f"  coords range: {self.coords_min} -> {self.coords_max}")

    def normalize_params(self, params):
        return (params - self.params_min) / self.params_range

    def normalize_coords(self, coords):
        return (coords - self.coords_min) / self.coords_range

    def normalize_p(self, p):
        return (p - self.p_mean) / self.p_std

    def denormalize_p(self, p_norm):
        return p_norm * self.p_std + self.p_mean

    def sample_batch(self, batch_size, n_pts, train=True):
        """Sample batch_size cases, n_pts points each. Returns tensors."""
        cases = self.train_cases if train else self.val_cases
        chosen = self.rng.choice(cases, size=batch_size, replace=True)

        params_b = []
        coords_b = []
        p_b = []
        for cid in chosen:
            d = self.data[cid]
            M = len(d['pressure'])
            idx = self.rng.choice(M, size=min(n_pts, M), replace=False)
            params_b.append(self.normalize_params(d['params']))
            coords_b.append(self.normalize_coords(d['coords'][idx]))
            p_b.append(self.normalize_p(d['pressure'][idx]))

        params_t = torch.tensor(np.stack(params_b), dtype=torch.float32)
        coords_t = torch.tensor(np.stack(coords_b), dtype=torch.float32)
        p_t = torch.tensor(np.stack(p_b), dtype=torch.float32)
        return params_t.to(DEVICE), coords_t.to(DEVICE), p_t.to(DEVICE)

    def full_val(self, model, n_pts=8192):
        """Evaluate on full val set. Returns mean rel-L2."""
        model.eval()
        rel_l2s = []
        with torch.no_grad():
            for cid in self.val_cases:
                d = self.data[cid]
                M = len(d['pressure'])
                params = torch.tensor(self.normalize_params(d['params']),
                                      dtype=torch.float32).unsqueeze(0).to(DEVICE)
                # Process in chunks to avoid OOM
                preds = []
                for start in range(0, M, n_pts):
                    end = min(start + n_pts, M)
                    coords = torch.tensor(self.normalize_coords(d['coords'][start:end]),
                                          dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    pred = model(params, coords).squeeze(0)
                    preds.append(pred.cpu().numpy())
                pred_all = np.concatenate(preds)
                # Denormalize
                pred_p = pred_all * self.p_std + self.p_mean
                true_p = d['pressure']
                rel_l2 = np.linalg.norm(pred_p - true_p) / np.linalg.norm(true_p)
                rel_l2s.append(rel_l2)
        return np.mean(rel_l2s), rel_l2s


# ============ Training ============

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Loading dataset...")
    ds = FaceDataset(DATA_PATH, KEY_WALLS, val_per_opening=2, seed=SEED)

    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val = float('inf')
    best_epoch = 0
    train_losses = []

    exp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'experiments',
                           f'face_fusion_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\nTraining for {EPOCHS} steps...")
    print(f"Experiment dir: {exp_dir}")
    print("=" * 60)

    t0 = time.time()
    for step in range(1, EPOCHS + 1):
        model.train()
        params, coords, p_true = ds.sample_batch(BATCH_CASES, N_PTS, train=True)
        pred = model(params, coords)
        loss = ((pred - p_true) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if step % VAL_EVERY == 0:
            val_rel, val_per = ds.full_val(model)
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]
            print(f"  step {step:6d} | train_mse={np.mean(train_losses[-VAL_EVERY:]):.6f} "
                  f"| val_rel_L2={val_rel:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s")

            if val_rel < best_val:
                best_val = val_rel
                best_epoch = step
                torch.save({'model': model.state_dict(),
                            'step': step, 'val_rel_l2': val_rel,
                            'norms': {
                                'params_min': ds.params_min.tolist(),
                                'params_max': ds.params_max.tolist(),
                                'p_mean': float(ds.p_mean),
                                'p_std': float(ds.p_std),
                                'coords_min': ds.coords_min.tolist(),
                                'coords_max': ds.coords_max.tolist(),
                            }},
                           os.path.join(exp_dir, 'best_model.pt'))
                print(f"    -> new best! rel_L2={val_rel:.4f}")

            if step - best_epoch > PATIENCE:
                print(f"  Early stopping at step {step} (no improve for {PATIENCE} steps)")
                break

    # Final summary
    print("=" * 60)
    print(f"Best val rel-L2: {best_val:.4f} at step {best_epoch}")
    print(f"Saved to: {exp_dir}/best_model.pt")

    # Save training log
    np.save(os.path.join(exp_dir, 'train_losses.npy'), np.array(train_losses))
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump({
            'branch_layers': BRANCH_LAYERS, 'trunk_layers': TRUNK_LAYERS,
            'output_dim': OUTPUT_DIM, 'n_pts': N_PTS, 'batch_cases': BATCH_CASES,
            'lr': LR, 'epochs': EPOCHS, 'best_step': best_epoch,
            'best_val_rel_l2': float(best_val), 'n_params': n_params,
            'key_walls': KEY_WALLS, 'train_cases': len(ds.train_cases),
            'val_cases': len(ds.val_cases),
        }, f, indent=2)


if __name__ == '__main__':
    train()
