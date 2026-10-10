"""
Physics-constraint ABLATION on the 7-wall FSI surface set (paper Result A).

Isolates each physics term so the paper can state exactly what it buys:
  baseline     pure data loss (= plain Fusion-DeepONet)
  force        + C1 force consistency only
  moment       + C1 moment consistency only
  force_moment + C1 force + moment
  smooth       + C2 KNN wall-pressure smoothness only
  full         + C1 force + moment + C2 smooth

Everything else (architecture, seed, step budget, dataset split) is IDENTICAL
across configs, so differences are attributable to the physics terms alone.

After each run, eval_faces.evaluate() reloads the best checkpoint and reports
the metrics that matter for the novelty claim:
  total rel-L2, per-wall rel-L2, per-opening rel-L2, FSI force err, moment err.

Multi-seed: pass --seeds 42 1 2 to repeat every config and get mean +/- std,
which is what a reviewer expects before believing a small rel-L2 gap.

Usage:
  python run_ablation_7walls.py                      # all configs, seed 42
  python run_ablation_7walls.py baseline full        # subset
  python run_ablation_7walls.py --seeds 42 1 2       # all configs x3 seeds
  python run_ablation_7walls.py --epochs 40000       # override step budget
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_face_physics as T
import eval_faces as E

SEVEN_WALLS = ["wall_valve_body", "wall_out_sleeve", "wall_in_sleeve",
               "wall_valve_seat", "wall_plug", "wall_expand", "wall_bottom"]

# physics-term toggles per config (weights/schedule shared via COMMON)
CONFIGS = {
    'baseline':     dict(USE_FORCE=False, USE_MOMENT=False, USE_SMOOTH=False),
    'force':        dict(USE_FORCE=True,  USE_MOMENT=False, USE_SMOOTH=False),
    'moment':       dict(USE_FORCE=False, USE_MOMENT=True,  USE_SMOOTH=False),
    'force_moment': dict(USE_FORCE=True,  USE_MOMENT=True,  USE_SMOOTH=False),
    'smooth':       dict(USE_FORCE=False, USE_MOMENT=False, USE_SMOOTH=True),
    'full':         dict(USE_FORCE=True,  USE_MOMENT=True,  USE_SMOOTH=True),
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments')


def run_one(name, seed, common):
    """Train one config at one seed, then evaluate its best checkpoint."""
    cfg = dict(common)
    cfg.update(CONFIGS[name])
    cfg['KEY_WALLS'] = SEVEN_WALLS
    cfg['SEED'] = seed
    cfg['TAG'] = f"abl7_{name}_s{seed}"

    # keep the module's KEY_WALLS in sync (PhysicsFaceDataset reads the global)
    T.KEY_WALLS = SEVEN_WALLS

    print("\n" + "#" * 72)
    print(f"# ABLATION 7-wall | config={name}  seed={seed}")
    print(f"#   force={cfg['USE_FORCE']} moment={cfg['USE_MOMENT']} smooth={cfg['USE_SMOOTH']}")
    print("#" * 72)

    r = T.train(cfg)
    ckpt = os.path.join(r['exp_dir'], 'best_model.pt')

    # reset physics flags so globals don't bleed into the next run
    T.USE_FORCE = T.USE_MOMENT = T.USE_SMOOTH = True

    m = E.evaluate(ckpt, walls=SEVEN_WALLS, seed=seed, verbose=True)
    m['config'] = name
    m['seed'] = seed
    m['train_best_step'] = r['step']
    m['exp_dir'] = r['exp_dir']
    return m


def aggregate(runs):
    """Group runs by config -> mean/std of scalar metrics across seeds."""
    by_cfg = {}
    for m in runs:
        by_cfg.setdefault(m['config'], []).append(m)
    agg = {}
    for name, ms in by_cfg.items():
        def ms_stat(key):
            vals = np.array([mm[key] for mm in ms], float)
            return float(vals.mean()), float(vals.std())
        # per-wall mean across seeds
        walls = SEVEN_WALLS
        pw = {z: float(np.mean([mm['per_wall'][z] for mm in ms])) for z in walls}
        agg[name] = {
            'n_seeds': len(ms),
            'rel_l2': ms_stat('rel_l2'),
            'force_err': ms_stat('force_err'),
            'moment_err': ms_stat('moment_err'),
            'per_wall': pw,
            'seeds': [mm['seed'] for mm in ms],
        }
    return agg


def write_reports(runs, agg, order):
    os.makedirs(OUT_DIR, exist_ok=True)
    # raw per-run JSON
    with open(os.path.join(OUT_DIR, 'ablation7_runs.json'), 'w') as f:
        json.dump(runs, f, indent=2)
    with open(os.path.join(OUT_DIR, 'ablation7_summary.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    order = [c for c in order if c in agg]

    # CSV: one row per config, scalar summary (mean+/-std)
    csv_path = os.path.join(OUT_DIR, 'ablation7_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        f.write("config,n_seeds,rel_l2_mean,rel_l2_std,"
                "force_err_mean,force_err_std,moment_err_mean,moment_err_std\n")
        for name in order:
            a = agg[name]
            f.write(f"{name},{a['n_seeds']},"
                    f"{a['rel_l2'][0]:.5f},{a['rel_l2'][1]:.5f},"
                    f"{a['force_err'][0]:.5f},{a['force_err'][1]:.5f},"
                    f"{a['moment_err'][0]:.5f},{a['moment_err'][1]:.5f}\n")

    # Markdown: scalar table + per-wall table (paper-ready)
    md_path = os.path.join(OUT_DIR, 'ablation7_report.md')
    with open(md_path, 'w') as f:
        f.write("# Physics-constraint ablation (7-wall FSI surface set)\n\n")
        f.write("Identical architecture / seed(s) / step budget / split; only physics "
                "terms toggled. rel-L2 is pointwise field error (all walls); "
                "force/moment err are FSI resultant errors on the val split.\n\n")
        multi = any(a['n_seeds'] > 1 for a in agg.values())
        if multi:
            f.write("| config | rel-L2 | FSI force err | FSI moment err |\n")
            f.write("|---|---|---|---|\n")
            for name in order:
                a = agg[name]
                f.write(f"| {name} | {a['rel_l2'][0]:.4f} ± {a['rel_l2'][1]:.4f} "
                        f"| {a['force_err'][0]:.4f} ± {a['force_err'][1]:.4f} "
                        f"| {a['moment_err'][0]:.4f} ± {a['moment_err'][1]:.4f} |\n")
        else:
            f.write("| config | rel-L2 | FSI force err | FSI moment err |\n")
            f.write("|---|---|---|---|\n")
            for name in order:
                a = agg[name]
                f.write(f"| {name} | {a['rel_l2'][0]:.4f} "
                        f"| {a['force_err'][0]:.4f} | {a['moment_err'][0]:.4f} |\n")

        f.write("\n## Per-wall rel-L2 (mean over seeds)\n\n")
        f.write("| wall | " + " | ".join(order) + " |\n")
        f.write("|" + "---|" * (len(order) + 1) + "\n")
        for z in SEVEN_WALLS:
            row = " | ".join(f"{agg[name]['per_wall'][z]:.4f}" for name in order)
            f.write(f"| {z} | {row} |\n")
    return md_path, csv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=None,
                    help=f"subset of {list(CONFIGS)} (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="one or more seeds; >1 gives mean +/- std")
    ap.add_argument("--epochs", type=int, default=40000)
    ap.add_argument("--warmup", type=int, default=5000)
    ap.add_argument("--ramp", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=6000)
    ap.add_argument("--phys-cases", type=int, default=1,
                    help="physics cases/step (1 keeps 7-wall memory bounded on 8GB)")
    args = ap.parse_args()

    which = args.configs if args.configs else list(CONFIGS.keys())
    bad = [c for c in which if c not in CONFIGS]
    if bad:
        sys.exit(f"unknown config(s): {bad}\nvalid: {list(CONFIGS)}")

    common = dict(EPOCHS=args.epochs, WARMUP=args.warmup, RAMP=args.ramp,
                  VAL_EVERY=500, PATIENCE=args.patience, PHYS_CASES=args.phys_cases,
                  LAMBDA_F=0.1, LAMBDA_M=0.05, LAMBDA_S=0.01, KNN_K=8)

    runs = []
    for seed in args.seeds:
        for name in which:
            runs.append(run_one(name, seed, common))
            # checkpoint the raw results after every run (crash-safe)
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, 'ablation7_runs.json'), 'w') as f:
                json.dump(runs, f, indent=2)

    agg = aggregate(runs)
    md_path, csv_path = write_reports(runs, agg, order=list(CONFIGS.keys()))

    # console summary
    print("\n" + "=" * 68)
    print(f"{'config':<14}{'rel-L2':>12}{'force_err':>12}{'moment_err':>12}")
    print("=" * 68)
    for name in [c for c in CONFIGS if c in agg]:
        a = agg[name]
        print(f"{name:<14}{a['rel_l2'][0]:>12.4f}{a['force_err'][0]:>12.4f}"
              f"{a['moment_err'][0]:>12.4f}")
    print("=" * 68)
    print(f"report -> {md_path}")
    print(f"csv    -> {csv_path}")


if __name__ == '__main__':
    main()
