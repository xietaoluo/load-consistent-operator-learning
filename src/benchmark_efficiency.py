#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""§3.6 计算效率对比：参数量 / 训练时间 / 推理时间 / 加速比。

对 5 种方法在同一硬件、同一 7 面数据集上统一测量：
  - Parameter count : 可训练参数量（NN 无参、POD 记模态+回归器、神经算子记网络参数）
  - Training time   : 在 100 训练工况上拟合的墙钟时间（神经算子按实测吞吐×总步数）
  - Inference time  : 单工况全 7 面场（~5.4 万面心）一次前向，excl. 载入，均值
  - Speedup         : CFD 单工况墙钟 / 推理时间

用法：
  python benchmark_efficiency.py
"""
import os
import sys
import json
import glob
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_methods as CM
from compare_methods import VanillaDeepONet, _make_norms
from train_face_pressure import (FusionDeepONet, BRANCH_LAYERS, TRUNK_LAYERS,
                                 OUTPUT_DIM, DEVICE)

SEVEN_WALLS = ["wall_valve_body", "wall_out_sleeve", "wall_in_sleeve",
               "wall_valve_seat", "wall_plug", "wall_expand", "wall_bottom"]
CM.KEY_WALLS = SEVEN_WALLS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CFD_WALLCLOCK_S = 3600.0        # 单工况 CFD 墙钟（约 1 h）
FUSION_STEPS = 50000            # train_face_pressure EPOCHS
VANILLA_STEPS = 30000           # compare_methods 默认
POD_MLP_EPOCHS = 3000           # method_pod_mlp 内部
TIME_STEPS = 500                # 神经算子吞吐测量步数
INFER_REPEATS = 30
CHUNK = 8192


def _newest(pattern):
    m = sorted(glob.glob(os.path.join(BASE_DIR, 'experiments', pattern)))
    return os.path.join(m[-1], 'best_model.pt') if m else None


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))


# ---------- inference timing helpers ----------

def _time_torch_infer(fwd, repeats=INFER_REPEATS):
    for _ in range(3):
        fwd()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeats):
        fwd()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / repeats * 1000.0


def _rep_case(D):
    """Representative mid-opening val case + its point count."""
    cid = D['val'][len(D['val']) // 2]
    return cid, len(D['data'][cid]['p'])


# ---------- per-method benchmark ----------

def bench_fusion(D, ckpt):
    """params, training-time (throughput x steps), inference ms."""
    ck = torch.load(ckpt, map_location=DEVICE)
    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    model.load_state_dict(ck['model']); model.eval()
    nrm = ck['norms']
    pmin = np.array(nrm['params_min']); pmax = np.array(nrm['params_max'])
    cmin = np.array(nrm['coords_min']); cmax = np.array(nrm['coords_max'])
    prange = np.where(pmax - pmin == 0, 1, pmax - pmin)
    crange = np.where(cmax - cmin == 0, 1, cmax - cmin)
    params = count_params(model)

    cid, M = _rep_case(D)
    d = D['data'][cid]
    pt = torch.tensor(((d['params'] - pmin) / prange)[None], dtype=torch.float32, device=DEVICE)
    ct = torch.tensor(((d['coords'] - cmin) / crange)[None], dtype=torch.float32, device=DEVICE)

    def fwd():
        with torch.no_grad():
            out = [model(pt, ct[:, s:s+CHUNK]).squeeze(0) for s in range(0, M, CHUNK)]
            return torch.cat(out)
    infer_ms = _time_torch_infer(fwd)

    train_s = _time_fusion_training(D, nrm, prange, crange, pmin, cmin)
    return dict(params=params, train_s=train_s, infer_ms=infer_ms, npts=M)


def _time_fusion_training(D, nrm, prange, crange, pmin, cmin):
    """Measure Fusion training throughput over TIME_STEPS, scale to FUSION_STEPS."""
    model = FusionDeepONet(BRANCH_LAYERS, TRUNK_LAYERS, OUTPUT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = D['data']; tr = D['train']
    rng = np.random.default_rng(0)
    n_pts, batch = 4096, 4

    def np_(p): return (p - nrm['p_mean']) / nrm['p_std']

    def one_step():
        ch = rng.choice(tr, batch)
        pb, cb, yb = [], [], []
        for cid in ch:
            dd = data[cid]; M = len(dd['p'])
            idx = rng.choice(M, min(n_pts, M), replace=False)
            pb.append((dd['params'] - pmin) / prange)
            cb.append((dd['coords'][idx] - cmin) / crange)
            yb.append(np_(dd['p'][idx]))
        pb = torch.tensor(np.stack(pb), dtype=torch.float32, device=DEVICE)
        cb = torch.tensor(np.stack(cb), dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(np.stack(yb), dtype=torch.float32, device=DEVICE)
        out = model(pb, cb)
        loss = ((out - yb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(10):          # warmup
        one_step()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(TIME_STEPS):
        one_step()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    per_step = (time.time() - t0) / TIME_STEPS
    return per_step * FUSION_STEPS


def bench_vanilla(D):
    nrm = _make_norms(D)
    prange = np.where(nrm['pmax'] - nrm['pmin'] == 0, 1, nrm['pmax'] - nrm['pmin'])
    crange = np.where(nrm['cmax'] - nrm['cmin'] == 0, 1, nrm['cmax'] - nrm['cmin'])
    model = VanillaDeepONet().to(DEVICE)
    params = count_params(model)

    # inference on representative case
    cid, M = _rep_case(D)
    d = D['data'][cid]
    pt = torch.tensor(((d['params'] - nrm['pmin']) / prange)[None], dtype=torch.float32, device=DEVICE)
    ct = torch.tensor(((d['coords'] - nrm['cmin']) / crange)[None], dtype=torch.float32, device=DEVICE)

    def fwd():
        with torch.no_grad():
            out = [model(pt, ct[:, s:s+CHUNK]).squeeze(0) for s in range(0, M, CHUNK)]
            return torch.cat(out)
    infer_ms = _time_torch_infer(fwd)

    # training throughput
    data = D['data']; tr = D['train']; rng = np.random.default_rng(0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_pts, batch = 4096, 4

    def np_(p): return (p - nrm['p_mean']) / nrm['p_std']

    def one_step():
        ch = rng.choice(tr, batch)
        pb, cb, yb = [], [], []
        for cid2 in ch:
            dd = data[cid2]; MM = len(dd['p'])
            idx = rng.choice(MM, min(n_pts, MM), replace=False)
            pb.append((dd['params'] - nrm['pmin']) / prange)
            cb.append((dd['coords'][idx] - nrm['cmin']) / crange)
            yb.append(np_(dd['p'][idx]))
        pb = torch.tensor(np.stack(pb), dtype=torch.float32, device=DEVICE)
        cb = torch.tensor(np.stack(cb), dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(np.stack(yb), dtype=torch.float32, device=DEVICE)
        loss = ((model(pb, cb) - yb) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    for _ in range(10):
        one_step()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(TIME_STEPS):
        one_step()
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    per_step = (time.time() - t0) / TIME_STEPS
    return dict(params=params, train_s=per_step * VANILLA_STEPS, infer_ms=infer_ms, npts=M)


def bench_pod(D, kind):
    """POD-KRR / POD-MLP: params = modes*faces (+regressor); time full fit+predict."""
    data = D['data']
    # count POD "parameters": sum over openings of (modes * faces) reconstruction basis
    total_basis = 0
    for op, split in D['per_opening'].items():
        train = split['train']
        snaps = np.stack([data[c]['p'] for c in train])
        mean, modes, coeffs = CM.build_pod(snaps, max_modes=len(train) - 1)
        total_basis += modes.size + mean.size      # basis + mean field
    # regressor params
    if kind == 'krr':
        # KRR stores dual coeffs: per opening, per mode, Ntrain duals
        reg = 0
        for op, split in D['per_opening'].items():
            train = split['train']
            snaps = np.stack([data[c]['p'] for c in train])
            _, _, coeffs = CM.build_pod(snaps, max_modes=len(train) - 1)
            reg += coeffs.shape[1] * len(train)
        params = int(total_basis + reg)
    else:  # mlp: one net (3->64->64->r) per opening; r varies
        reg = 0
        for op, split in D['per_opening'].items():
            train = split['train']
            snaps = np.stack([data[c]['p'] for c in train])
            _, _, coeffs = CM.build_pod(snaps, max_modes=len(train) - 1)
            r = coeffs.shape[1]
            reg += 3*64 + 64 + 64*64 + 64 + 64*r + r
        params = int(total_basis + reg)

    fn = CM.method_pod_krr if kind == 'krr' else CM.method_pod_mlp
    t0 = time.time()
    pred = fn(D)
    fit_predict_s = time.time() - t0

    # inference-only: time predicting one case's field. POD reconstruct is cheap;
    # measure per-opening predict cost by re-timing predict on all val, divide.
    # Simpler: reconstruction is mean + modes@c -> time it for one representative op.
    op = D['data'][_rep_case(D)[0]]['opening']
    split = D['per_opening'][op]
    snaps = np.stack([data[c]['p'] for c in split['train']])
    mean, modes, coeffs = CM.build_pod(snaps, max_modes=len(split['train']) - 1)
    c = coeffs[0]
    reps = 200
    t0 = time.time()
    for _ in range(reps):
        _ = mean + modes @ c
    recon_ms = (time.time() - t0) / reps * 1000.0
    _, M = _rep_case(D)
    return dict(params=params, train_s=fit_predict_s, infer_ms=recon_ms, npts=M)


def bench_nn(D):
    """Nearest-neighbor: no trainable params; 'inference' = param-space search + copy."""
    data = D['data']
    # stored snapshots count as memory, not params -> report 0 trainable params
    cid, M = _rep_case(D)
    op = data[cid]['opening']
    split = D['per_opening'][op]
    train = split['train']
    tp = np.stack([data[c]['params'][:2] for c in train])
    mu, sd = tp.mean(0), tp.std(0) + 1e-9
    tpn = (tp - mu) / sd
    q = (data[cid]['params'][:2] - mu) / sd
    reps = 500
    t0 = time.time()
    for _ in range(reps):
        j = int(np.argmin(((tpn - q) ** 2).sum(1)))
        _ = data[train[j]]['p']
    infer_ms = (time.time() - t0) / reps * 1000.0
    # "training" = store snapshots: time the trivial fit
    t0 = time.time()
    _ = CM.method_nn(D)
    train_s = time.time() - t0
    return dict(params=0, train_s=train_s, infer_ms=infer_ms, npts=M)


def main():
    torch.manual_seed(CM.SEED); np.random.seed(CM.SEED)
    print(f"Device: {DEVICE}")
    D = CM.load_data()
    print(f"{len(D['train'])} train / {len(D['val'])} val | walls={len(SEVEN_WALLS)}")

    ckpt_ours = _newest('face_abl7_force_moment_s42_*')

    rows = {}
    print("\n>>> Nearest-Neighbor"); rows['Nearest-Neighbor'] = bench_nn(D)
    print(">>> POD-KRR");           rows['POD-KRR'] = bench_pod(D, 'krr')
    print(">>> POD-MLP");           rows['POD-MLP'] = bench_pod(D, 'mlp')
    print(">>> Vanilla-DeepONet");  rows['Vanilla DeepONet'] = bench_vanilla(D)
    print(">>> Fusion (ours)");     rows['Fusion + load (ours)'] = bench_fusion(D, ckpt_ours)

    # report
    print("\n" + "=" * 96)
    hdr = f"{'Method':<24}{'Params':>12}{'Train[s]':>12}{'Infer[ms]':>12}{'Speedup':>14}"
    print(hdr); print("-" * 96)
    for name, r in rows.items():
        sp = CFD_WALLCLOCK_S / (r['infer_ms'] / 1000.0)
        r['speedup'] = sp
        print(f"{name:<24}{r['params']:>12,}{r['train_s']:>12.1f}{r['infer_ms']:>12.2f}{sp:>14,.0f}")

    out = os.path.join(BASE_DIR, 'results', 'efficiency_3_6.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'device': DEVICE, 'cfd_wallclock_s': CFD_WALLCLOCK_S,
                   'fusion_steps': FUSION_STEPS, 'vanilla_steps': VANILLA_STEPS,
                   'rows': rows}, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == '__main__':
    main()
