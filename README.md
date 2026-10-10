# Load-Consistency-Constrained Operator Learning for Control-Valve FSI

Code and a demo dataset for the paper **"Load-consistency-constrained operator
learning for fast fluid–structure interaction analysis of control valves."**

A single coordinate-based neural operator maps operating conditions
`μ = (P_in, P_out, θ)` directly to the 3-D pressure field on the seven
pressure-bearing walls of a sleeve-type control valve. The predicted field is
used as the surface load that drives a structural solve, replacing the most
expensive step (CFD) of a one-way FSI loop with a millisecond-scale query.

## Key ideas

1. **Integral-load-consistency constraint.** Instead of penalising PDE
   residuals at scattered points (as in physics-informed operator learning),
   the physical constraint acts on the quantities FSI actually transfers — the
   resultant **force and moment** obtained by integrating pressure over each
   coupling surface. This reduces the FSI resultant-force error by ~32% while
   leaving point-wise accuracy essentially unchanged (*load–accuracy
   decoupling*).
2. **Mixed-resolution training.** Point-wise data loss is evaluated on a sparse
   random sample of face points (cheap, unbiased, broad case coverage), while
   the integral-load loss is evaluated on the *full* wall mesh (quadrature is
   biased under sub-sampling). The two losses are decoupled in resolution,
   enabling the integral constraint to scale to meshes with 10^5 faces.
3. **Coordinate-based operator over variable geometry.** The Fusion-DeepONet
   backbone (layer-wise gated fusion + Rowdy adaptive activation) takes the
   query coordinate on the trunk, so a single operator covers all openings and
   all seven heterogeneous walls without per-mesh resampling.

> Note: the Fusion-DeepONet backbone is adopted from Peyvan et al. (2025); the
> contribution of this work is the integral-load constraint, the
> mixed-resolution training scheme, and the surrogate-driven FSI application —
> not the backbone itself.

## Repository layout

```
.
├── src/                        core library (importable modules + runnable scripts)
│   ├── _paths.py               dataset / checkpoint path resolution
│   ├── train_face_pressure.py  Fusion-DeepONet backbone + data-only training
│   ├── train_face_physics.py   training with the load-consistency constraint
│   ├── physics_losses.py       force / moment integral-load losses
│   ├── compare_methods.py      baselines: nearest-neighbour, POD-KRR/MLP, Vanilla DeepONet
│   ├── eval_faces.py           per-wall evaluation of a checkpoint
│   ├── run_ablation_7walls.py  constraint ablation driver (multi-seed)
│   ├── predict_ansys_csv.py    export predicted pressure as CSV for FEA (External Data)
│   ├── benchmark_efficiency.py parameter count / inference time / speedup
│   ├── fig_baseline_walls.py   per-model prediction/error figure
│   ├── pack_face_dataset.py    build the HDF5 dataset from Fluent exports
│   └── generate_face_journals.py  generate Fluent journals for the case matrix
├── data/
│   ├── dataset_demo.h5         small demo subset (openings 50% & 80%, 6 cases each)
│   └── case_matrix.csv         the 120-case operating-condition matrix
├── checkpoints/
│   ├── fusion_load_consistency.pt   released model (with load constraint)
│   └── fusion_data_only.pt          data-only backbone (ablation baseline)
├── examples/
│   └── demo_predict.py         minimal end-to-end inference demo
├── requirements.txt
├── LICENSE                     MIT
└── CITATION.cff
```

## Installation

```bash
git clone https://github.com/xietaoluo/load-consistent-operator-learning.git
cd load-consistent-operator-learning
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Python ≥ 3.9 and a PyTorch build matching your CUDA (CPU also works for the
demo). The released checkpoint has ~1.5×10^5 parameters and runs on CPU.

## Quick start

Run inference with the released model on the demo dataset:

```bash
python examples/demo_predict.py                # case O50_C01
python examples/demo_predict.py --case O80_C05
```

Expected output (per wall: face count and relative-L2 error vs. CFD):

```
Case O50_C01: P_in=870 kPa  P_out=468 kPa  opening=50%  r=0.586
wall                  n_faces   rel-L2  pred-range (kPa)
wall_plug               15801   0.0255      98 - 874
wall_valve_body          6710   0.0036     843 - 882
wall_expand              3651   0.0049     455 - 470
wall_bottom              3502   0.0039     460 - 473
```

## Dataset

The repository ships a **demo subset** (`data/dataset_demo.h5`, ~20 MB:
openings 50% and 80%, six cases each) so that inference and the figures run out
of the box. The **full 120-case dataset** (10 openings × 12 conditions,
~144 MB) is available from the authors on request; place it at
`data/dataset_face.h5` or point the environment variable at it:

```bash
export VALVE_DATA=/path/to/dataset_face.h5
```

HDF5 layout:

```
/geometry/O{opening:03d}/{wall}/{centroid,normal,area,coords,face_conn,face_nnodes}
/walls/{CaseID}/{wall}/pressure          # face-centred gauge pressure (Pa)
/integrals/{CaseID}.attrs                # P_in_kPa, P_out_kPa, opening, r
/meta/case_matrix
```

## Reproducing the paper

All scripts read the dataset through `src/_paths.py` (env `VALVE_DATA` →
`data/dataset_face.h5` → `data/dataset_demo.h5`).

```bash
cd src
python train_face_pressure.py          # data-only backbone
python train_face_physics.py           # + load-consistency constraint
python run_ablation_7walls.py          # constraint ablation (seeds 42/1/2)
python eval_faces.py ../checkpoints/fusion_load_consistency.pt
python compare_methods.py              # baselines vs. ours (needs full dataset)
python benchmark_efficiency.py         # parameters / inference time / speedup
python fig_baseline_walls.py --case O40_C07   # per-model prediction/error figure
```

The train/val split is seeded (`seed=42`, 2 cases per opening held out). To add
extra held-out test cases without perturbing the published split, list their
CaseIDs in `VALVE_HOLDOUT` (comma-separated); they are excluded from the
shuffle and appended to validation only.

### Building the dataset from CFD (optional)

`generate_face_journals.py` writes ANSYS Fluent journals for the operating
conditions in `case_matrix.csv`; after running them, `pack_face_dataset.py`
assembles the wall exports into the HDF5 file. The raw Fluent case/result files
are not distributed — set `VALVE_CAS_DIR` and `VALVE_CFD_RESULTS` to your own.

## FEA coupling

`predict_ansys_csv.py` exports the predicted wall pressure as point-cloud CSV
that commercial FEA packages import through an External Data / point-cloud
load, driving the structural solve on an independent mesh.

## Citation

If you use this code or data, please cite the paper (see `CITATION.cff`).
Journal reference and DOI will be added upon publication.

## License

MIT — see [LICENSE](LICENSE).
