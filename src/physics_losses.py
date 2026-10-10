"""
Physics-informed loss terms for FSI-consistent wall-pressure operator.

All losses operate on PREDICTED PRESSURE IN PASCALS (gauge), i.e. after the
model output has been de-normalized. They are dimensionless (relative-error
form) so weights are easy to balance.

C1  loss_force / loss_moment : integrated resultant force / moment must match
     the CFD ground truth. Gradient flows through p_pred; area/normal/centroid
     are constant geometry buffers.
C2  loss_smooth : tangential pressure gradient penalty via precomputed KNN,
     optionally gated by the ground-truth gradient so real high-gradient
     regions (valve throat) are not over-smoothed.

Sign convention (matches pack_face_dataset.integrate_loads):
     dF = -(p_gauge + P_ATM) * A * n     (n outward from fluid)
"""

import numpy as np
import torch

P_ATM = 101325.0


# ---------- C2 precompute ----------

def precompute_knn(coords, k=8):
    """coords: (M,3) np array. Returns (idx (M,k) int, dist2 (M,k) float).
    Excludes self. Uses scipy cKDTree."""
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    # query k+1 (first neighbor is self), drop self column
    dist, idx = tree.query(coords, k=k + 1)
    idx = idx[:, 1:]                       # (M,k)
    dist = dist[:, 1:]                     # (M,k)
    return idx.astype(np.int64), (dist ** 2).astype(np.float32)


# ---------- C1 force / moment ----------

def integrate_force(p_pred_pa, area, normal):
    """p_pred_pa (N,) gauge pressure [Pa]; area (N,); normal (N,3).
    Returns force vector (3,). Differentiable w.r.t. p_pred_pa."""
    p_abs = p_pred_pa + P_ATM
    dF = -(p_abs * area).unsqueeze(-1) * normal      # (N,3)
    return dF.sum(dim=0)                              # (3,)


def integrate_moment(p_pred_pa, area, normal, centroid, ref_point):
    """Moment about ref_point (3,). M = sum r x dF."""
    p_abs = p_pred_pa + P_ATM
    dF = -(p_abs * area).unsqueeze(-1) * normal      # (N,3)
    r = centroid - ref_point                          # (N,3)
    return torch.cross(r, dF, dim=-1).sum(dim=0)      # (3,)


def loss_force(p_pred_pa, area, normal, F_cfd, eps=1e-8):
    """Relative-L2 force error (scalar)."""
    F_hat = integrate_force(p_pred_pa, area, normal)
    return ((F_hat - F_cfd) ** 2).sum() / ((F_cfd ** 2).sum() + eps)


def loss_moment(p_pred_pa, area, normal, centroid, M_cfd, ref_point, eps=1e-8):
    """Relative-L2 moment error (scalar)."""
    M_hat = integrate_moment(p_pred_pa, area, normal, centroid, ref_point)
    return ((M_hat - M_cfd) ** 2).sum() / ((M_cfd ** 2).sum() + eps)


# ---------- C2 smoothness ----------

def loss_smooth(p_pred_pa, knn_idx, knn_dist2, p_true_pa=None,
                gate_scale=None, eps=1e-6):
    """
    Tangential-gradient penalty.
      mean_i mean_j (p_i - p_j)^2 / (|x_i-x_j|^2 + eps)   [normalized by p_std^2 outside]
    knn_idx   : (N,k) long
    knn_dist2 : (N,k) float, squared distances
    p_true_pa : (N,) optional ground-truth pressure for gradient gating
    gate_scale: float; if set, weight_ij = exp(-|grad_true|/gate_scale) so that
                regions with large TRUE gradient are penalized less.
    Returns scalar. Non-dimensionalized by dividing by mean p magnitude^2.
    """
    pi = p_pred_pa.unsqueeze(1)                       # (N,1)
    pj = p_pred_pa[knn_idx]                           # (N,k)
    diff2 = (pi - pj) ** 2 / (knn_dist2 + eps)        # (N,k)

    if p_true_pa is not None and gate_scale is not None:
        ti = p_true_pa.unsqueeze(1)
        tj = p_true_pa[knn_idx]
        true_grad = torch.sqrt((ti - tj) ** 2 / (knn_dist2 + eps) + eps)
        w = torch.exp(-true_grad / gate_scale)
        val = (w * diff2).sum() / (w.sum() + eps)
    else:
        val = diff2.mean()

    # non-dimensionalize by pressure scale (variance of predicted field)
    scale = p_pred_pa.var() + eps
    return val / scale * knn_dist2.mean()   # multiply back a length^2 to keep O(1)


if __name__ == '__main__':
    # ---- self-test on real geometry + true pressure ----
    import h5py, os
    from _paths import DATA_PATH
    h5 = h5py.File(DATA_PATH, 'r')
    KEY = ["wall_plug", "wall_valve_body", "wall_expand", "wall_bottom"]
    cid, op = 'O50_C01', 50

    coords, p, normal, area, centroid = [], [], [], [], []
    F_cfd = np.zeros(3); M_cfd = np.zeros(3)
    for z in KEY:
        g = h5[f'geometry/O{op:03d}/{z}']
        coords.append(g['centroid'][()]); centroid.append(g['centroid'][()])
        normal.append(g['normal'][()]); area.append(g['area'][()])
        p.append(h5[f'walls/{cid}/{z}/pressure'][()])
        F_cfd += h5[f'integrals/{cid}/zone/{z}/F_pressure'][()]
        M_cfd += h5[f'integrals/{cid}/zone/{z}/M_pressure'][()]
    coords = np.concatenate(coords); centroid = np.concatenate(centroid)
    normal = np.concatenate(normal); area = np.concatenate(area)
    p = np.concatenate(p)
    h5.close()

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    p_t = t(p); area_t = t(area); normal_t = t(normal); cen_t = t(centroid)
    F_t = t(F_cfd); M_t = t(M_cfd); ref = t([0., 0., 0.])

    print(f"case {cid}: N={len(p)} faces")
    print(f"CFD stored  F={F_cfd},  |F|={np.linalg.norm(F_cfd):.1f} N")
    F_hat = integrate_force(p_t, area_t, normal_t)
    print(f"re-integrated F={F_hat.numpy()},  |F|={F_hat.norm():.1f} N")
    print(f"  -> loss_force(true)   = {loss_force(p_t, area_t, normal_t, F_t):.3e}  (should be ~0)")
    print(f"CFD stored  M={M_cfd}")
    M_hat = integrate_moment(p_t, area_t, normal_t, cen_t, ref)
    print(f"re-integrated M={M_hat.numpy()}")
    print(f"  -> loss_moment(true)  = {loss_moment(p_t, area_t, normal_t, cen_t, M_t, ref):.3e} (should be ~0)")

    idx, d2 = precompute_knn(coords, k=8)
    ls = loss_smooth(p_t, torch.tensor(idx), torch.tensor(d2))
    print(f"  -> loss_smooth(true)  = {ls:.4f}  (finite, O(1))")

    # perturb prediction -> losses should grow
    p_bad = p_t + 5000.0 * torch.randn_like(p_t)
    print(f"\nperturbed (+5kPa noise):")
    print(f"  loss_force  = {loss_force(p_bad, area_t, normal_t, F_t):.3e}")
    print(f"  loss_smooth = {loss_smooth(p_bad, torch.tensor(idx), torch.tensor(d2)):.4f}")

    # gradient check
    p_var = p_t.clone().requires_grad_(True)
    L = loss_force(p_var, area_t, normal_t, F_t)
    L.backward()
    print(f"\ngradient flows: dL_force/dp norm = {p_var.grad.norm():.3e} (nonzero=OK)")
