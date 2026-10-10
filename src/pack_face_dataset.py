"""
Pack data wall-pressure exports into a layered HDF5 dataset.

Adapted to actual Fluent output format:
  - Face-centered export (cellnumber, x/y/z, pressure) -- NOT nodal
  - All wall zones in one file, ordered by face zone minId
  - No _field files (skipped)
  - Only pressure available (no wall-shear, y-plus, density, temp)

Output: data/dataset_face.h5
  /geometry/O{ooo}/{zone}/  coords, face_conn, face_nnodes, centroid, normal, area
  /walls/{CaseID}/{zone}/   pressure (aligned to topo face order)
  /integrals/{CaseID}/      F_pressure per zone, massflow, BC params
  /meta                     case_matrix, key_walls, etc.
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import h5py
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_wall_topology import list_wall_zones, extract_zone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
# Fluent ASCII exports (per opening, per case: <CaseID>_wall, <CaseID>_massflow.txt)
# and the matching case files are NOT shipped with the repo (users regenerate
# them with the Fluent journals from generate_face_journals.py). Point
# VALVE_CFD_RESULTS / VALVE_CAS_DIR at your own CFD output to repack.
RESULTS_DIR = os.environ.get("VALVE_CFD_RESULTS", os.path.join(ROOT_DIR, "cfd_results"))
CAS_TEMPLATE = os.path.join(
    os.environ.get("VALVE_CAS_DIR", os.path.join(ROOT_DIR, "cfd_cases")),
    "{opening}_opening", "FFF.cas.h5")
CASE_MATRIX = os.path.join(DATA_DIR, "case_matrix.csv")
OUT_H5 = os.path.join(DATA_DIR, "dataset_face.h5")

KEY_WALLS = ["wall_plug", "wall_valve_body", "wall_expand", "wall_bottom"]
P_ATM = 101325.0  # Pa


def read_wall_export(path):
    """Read face-centered wall ASCII export. Returns (xyz(N,3), pressure(N,))."""
    df = pd.read_csv(path, skipinitialspace=True)
    cols_low = {c.strip().lower(): c for c in df.columns}
    xyz = df[[cols_low['x-coordinate'], cols_low['y-coordinate'],
              cols_low['z-coordinate']]].values
    p = df[cols_low['pressure']].values
    return xyz, p


def split_by_zone(xyz, pressure, zone_nfaces):
    """
    Split flat arrays into per-zone chunks.
    zone_nfaces: list of (zone_name, n_faces) in minId order.
    Returns dict zone_name -> (xyz_chunk, p_chunk).
    """
    offset = 0
    out = {}
    for name, nf in zone_nfaces:
        out[name] = (xyz[offset:offset + nf], pressure[offset:offset + nf])
        offset += nf
    return out


def align_to_topo(export_xyz, topo_centroid):
    """
    Build permutation: export row i -> topo face idx[i].
    Returns idx array such that export_pressure[i] belongs to topo face idx[i].
    """
    tree = cKDTree(topo_centroid)
    dist, idx = tree.query(export_xyz)
    if dist.max() > 1e-4:
        print(f"    WARNING: max align dist = {dist.max():.2e} (>0.1mm)")
    return idx


def integrate_face_pressure(topo, p_face_gauge):
    """
    Integrate face-centered gauge pressure over a wall zone.
    p_face_gauge: pressure at each face center, aligned to topo face order.
    Returns dict with F_pressure(3), M_pressure(3), area.
    """
    normal = topo['normal']
    area = topo['area']
    centroid = topo['centroid']
    p_abs = p_face_gauge + P_ATM
    dF = -(p_abs * area)[:, None] * normal
    F_pressure = dF.sum(axis=0)
    M_pressure = np.cross(centroid, dF).sum(axis=0)
    return {'F_pressure': F_pressure, 'M_pressure': M_pressure,
            'area': float(area.sum())}


def parse_massflow(path):
    """Parse Fluent flux report."""
    res = {}
    if not os.path.exists(path):
        return res
    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = re.match(r'\s*([A-Za-z0-9_\-\:]+)\s+(-?\d[\d\.eE\+\-]*)\s*$', line)
            if m:
                name, val = m.group(1), float(m.group(2))
                res[name] = val
    return res


def main():
    df = pd.read_csv(CASE_MATRIX, encoding='utf-8-sig')
    openings = sorted(int(o) for o in df['开度(%)'].unique())

    print("Packing valve wall-pressure dataset ->", OUT_H5)
    print("=" * 60)

    with h5py.File(OUT_H5, 'w') as h5:
        # Meta
        gmeta = h5.create_group('meta')
        gmeta.attrs['key_walls'] = json.dumps(KEY_WALLS)
        gmeta.attrs['p_atm'] = P_ATM
        gmeta.attrs['data_type'] = 'face-centered pressure'
        # Store case matrix
        cm = h5.create_group('meta/case_matrix')
        for col in df.columns:
            try:
                cm.create_dataset(col, data=df[col].values.astype(np.float64))
            except (ValueError, TypeError):
                cm.create_dataset(col, data=df[col].values.astype('S'))

        # Per-opening: extract topology + build alignment index
        topo_cache = {}   # opening -> {zone: topo_dict}
        align_cache = {}  # opening -> {zone: idx_array}
        zone_order_cache = {}  # opening -> [(name, nf)]

        for opening in openings:
            cas = CAS_TEMPLATE.format(opening=opening)
            if not os.path.exists(cas):
                print(f"  O{opening:03d}: no cas.h5, skip")
                continue

            # Get wall zones sorted by minId (= export row order)
            all_zones = list_wall_zones(cas)
            walls = [(name, nf, mn) for zid, name, nf, mn, mx in all_zones
                     if name.lower().startswith('wall')]
            walls.sort(key=lambda x: x[2])  # sort by minId
            zone_nfaces = [(name, nf) for name, nf, _ in walls]
            zone_order_cache[opening] = zone_nfaces

            # Extract topology for each wall zone
            topo_by_zone = {}
            for name, nf, _ in walls:
                topo_by_zone[name] = extract_zone(cas, name)
            topo_cache[opening] = topo_by_zone

            # Write geometry
            gg = h5.create_group(f'geometry/O{opening:03d}')
            for name, topo in topo_by_zone.items():
                zg = gg.create_group(name)
                zg.create_dataset('centroid', data=topo['centroid'], compression='gzip')
                zg.create_dataset('normal', data=topo['normal'], compression='gzip')
                zg.create_dataset('area', data=topo['area'], compression='gzip')
                zg.create_dataset('coords', data=topo['coords'], compression='gzip')
                zg.create_dataset('face_conn', data=topo['face_conn'], compression='gzip')
                zg.create_dataset('face_nnodes', data=topo['face_nnodes'], compression='gzip')
                zg.attrs['n_faces'] = len(topo['face_nnodes'])
                zg.attrs['n_vertices'] = len(topo['coords'])

            # Build alignment index from first available case
            first_case = df[df['开度(%)'] == opening].iloc[0]['CaseID']
            wall_f = os.path.join(RESULTS_DIR, f"O{opening:03d}", f"{first_case}_wall")
            if os.path.exists(wall_f):
                xyz, _ = read_wall_export(wall_f)
                per_zone_data = split_by_zone(xyz, np.zeros(len(xyz)), zone_nfaces)
                align_idx = {}
                for name, (ez, _) in per_zone_data.items():
                    align_idx[name] = align_to_topo(ez, topo_by_zone[name]['centroid'])
                align_cache[opening] = align_idx

            print(f"  O{opening:03d}: geometry OK ({sum(nf for _,nf in zone_nfaces)} faces, {len(zone_nfaces)} zones)")

        # Pack cases
        n_ok = 0
        for _, row in df.iterrows():
            cid = row['CaseID']
            opening = int(row['开度(%)'])
            if opening not in topo_cache:
                continue
            rdir = os.path.join(RESULTS_DIR, f"O{opening:03d}")
            wall_f = os.path.join(rdir, f"{cid}_wall")
            mass_f = os.path.join(rdir, f"{cid}_massflow.txt")
            if not os.path.exists(wall_f):
                continue

            zone_nfaces = zone_order_cache[opening]
            topo_by_zone = topo_cache[opening]
            align_idx = align_cache.get(opening, {})

            xyz, pressure = read_wall_export(wall_f)
            per_zone = split_by_zone(xyz, pressure, zone_nfaces)

            # Write wall pressure per zone (aligned to topo face order)
            wg = h5.create_group(f'walls/{cid}')
            ig = h5.create_group(f'integrals/{cid}')

            for name, (ez, pz) in per_zone.items():
                topo = topo_by_zone[name]
                # Align export order -> topo face order
                if name in align_idx:
                    idx = align_idx[name]
                    p_aligned = np.empty(len(topo['centroid']), dtype=np.float32)
                    p_aligned[idx] = pz.astype(np.float32)
                else:
                    p_aligned = pz.astype(np.float32)

                zg = wg.create_group(name)
                zg.create_dataset('pressure', data=p_aligned, compression='gzip')
                zg.attrs['n_faces'] = len(p_aligned)

                # FSI load integration
                loads = integrate_face_pressure(topo, p_aligned.astype(np.float64))
                zig = ig.require_group(f'zone/{name}')
                zig.create_dataset('F_pressure', data=loads['F_pressure'])
                zig.create_dataset('M_pressure', data=loads['M_pressure'])
                zig.attrs['area'] = loads['area']

            # Massflow
            mf = parse_massflow(mass_f)
            if mf:
                mg = ig.create_group('massflow')
                for zn, val in mf.items():
                    mg.attrs[zn] = val

            # Store BC params
            ig.attrs['CaseID'] = cid
            ig.attrs['opening'] = opening
            ig.attrs['P_in_kPa'] = float(row['P_in(kPa)'])
            ig.attrs['P_out_kPa'] = float(row['P_out(kPa)'])
            ig.attrs['r'] = float(row['r(P_out/P_in_abs)'])

            n_ok += 1
            if n_ok % 20 == 0:
                print(f"    packed {n_ok} cases...")

        print(f"  Done. {n_ok}/{len(df)} cases packed.")
    sz = os.path.getsize(OUT_H5) / 1e9
    print(f"Output: {OUT_H5}  ({sz:.3f} GB)")


if __name__ == '__main__':
    main()
