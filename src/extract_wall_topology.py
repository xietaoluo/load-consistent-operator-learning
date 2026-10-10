"""
Extract wall-zone topology (coords + polygon connectivity + face normals /
areas / centroids) directly from a Fluent FFF.cas.h5 file.

This is the geometric foundation for:
  - pressure-surface visualization (pyvista/ParaView)
  - FSI load integration (F = sum p * n * A, viscous = sum tau * A)

The .cas.h5 layout (Fluent):
  meshes/1/nodes/coords/<tag>          (Nnode, 3) float64  node coordinates
  meshes/1/faces/nodes/1/nnodes        (Nface,)   int      #nodes per face
  meshes/1/faces/nodes/1/nodes         (sum,)     uint32   flattened face->node (1-based)
  meshes/1/faces/zoneTopology/{id,minId,maxId,name}        zone face-id ranges (1-based)

Faces are laid out contiguously by zone in global face-id order, so zone k
owns flat face rows [minId-1 : maxId].
"""

import h5py
import numpy as np


def _zone_names(zt):
    raw = zt['name'][()]
    if isinstance(raw, np.ndarray) and raw.dtype.kind == 'S' and raw.shape == (1,):
        s = raw[0].decode()
    elif isinstance(raw, bytes):
        s = raw.decode()
    else:
        s = raw.tobytes().decode()
    return s.split(';')


def list_wall_zones(cas_path):
    """Return list of (zone_id, name, n_faces, minId, maxId) for all face zones."""
    with h5py.File(cas_path, 'r') as f:
        zt = f['meshes/1/faces/zoneTopology']
        ids = zt['id'][()]
        mn = zt['minId'][()].astype(np.int64)
        mx = zt['maxId'][()].astype(np.int64)
        names = _zone_names(zt)
    out = []
    for i in range(len(ids)):
        out.append((int(ids[i]), names[i], int(mx[i] - mn[i] + 1),
                    int(mn[i]), int(mx[i])))
    return out


def _face_offsets(nnodes):
    """Prefix-sum offsets into the flat nodes array. offsets[i]:offsets[i+1] = face i."""
    off = np.zeros(len(nnodes) + 1, dtype=np.int64)
    np.cumsum(nnodes.astype(np.int64), out=off[1:])
    return off


def extract_zone(cas_path, zone_name):
    """
    Extract topology for one wall zone.

    Returns dict:
      coords      (Nv, 3)    unique node coords used by this zone
      faces       list[array] polygon connectivity (local 0-based vertex idx)
      face_nnodes (Nf,)      #verts per face
      face_conn   (sum,)     flattened local connectivity
      centroid    (Nf, 3)    face centroids
      normal      (Nf, 3)    outward-ish unit normals (Newell)
      area        (Nf,)      face areas
      node_ids    (Nv,)      original global (1-based) node ids -> maps ASCII nodenumber
    """
    with h5py.File(cas_path, 'r') as f:
        zt = f['meshes/1/faces/zoneTopology']
        ids = zt['id'][()]
        mn = zt['minId'][()].astype(np.int64)
        mx = zt['maxId'][()].astype(np.int64)
        names = _zone_names(zt)
        if zone_name not in names:
            raise KeyError(f"zone '{zone_name}' not in {names}")
        zi = names.index(zone_name)
        f0, f1 = mn[zi] - 1, mx[zi]  # 0-based [f0, f1)

        nnodes = f['meshes/1/faces/nodes/1/nnodes'][()]
        offsets = _face_offsets(nnodes)
        start, end = int(offsets[f0]), int(offsets[f1])
        flat = f['meshes/1/faces/nodes/1/nodes'][start:end].astype(np.int64)
        zn_nnodes = nnodes[f0:f1].astype(np.int64)

        coords_key = list(f['meshes/1/nodes/coords'].keys())[0]
        all_coords = f['meshes/1/nodes/coords/' + coords_key]
        node_base = int(all_coords.attrs.get('minId', 1)) if 'minId' in all_coords.attrs else 1

    # Fluent node indices are 1-based; remap used nodes to local 0-based.
    used = np.unique(flat)
    remap = np.full(int(used.max()) + 1, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    with h5py.File(cas_path, 'r') as f:
        coords_key = list(f['meshes/1/nodes/coords'].keys())[0]
        ac = f['meshes/1/nodes/coords/' + coords_key][()]
    coords = ac[used - 1]  # 1-based -> 0-based row
    local_conn = remap[flat]

    # per-face geometry (Newell's method, handles arbitrary polygons)
    zn_off = _face_offsets(zn_nnodes)
    Nf = len(zn_nnodes)
    centroid = np.zeros((Nf, 3))
    normal = np.zeros((Nf, 3))
    area = np.zeros(Nf)
    for i in range(Nf):
        vs = coords[local_conn[zn_off[i]:zn_off[i + 1]]]
        centroid[i] = vs.mean(axis=0)
        n = np.zeros(3)
        m = len(vs)
        for j in range(m):
            a, b = vs[j], vs[(j + 1) % m]
            n[0] += (a[1] - b[1]) * (a[2] + b[2])
            n[1] += (a[2] - b[2]) * (a[0] + b[0])
            n[2] += (a[0] - b[0]) * (a[1] + b[1])
        area[i] = 0.5 * np.linalg.norm(n)
        if area[i] > 0:
            normal[i] = n / (2 * area[i])

    return {
        'coords': coords,
        'face_nnodes': zn_nnodes.astype(np.int32),
        'face_conn': local_conn.astype(np.int32),
        'centroid': centroid,
        'normal': normal,
        'area': area,
        'node_ids': used.astype(np.int64),  # global 1-based node ids
    }


if __name__ == '__main__':
    import sys
    cas = sys.argv[1] if len(sys.argv) > 1 else 'cfd_cases/50_opening/FFF.cas.h5'
    print(f"cas: {cas}\n")
    print("=== wall/face zones ===")
    for zid, name, nf, a, b in list_wall_zones(cas):
        print(f"  id={zid:4d}  n_faces={nf:>9}  {name}")

    test_zone = 'wall_plug'
    print(f"\n=== extract '{test_zone}' ===")
    z = extract_zone(cas, test_zone)
    print(f"  vertices   : {z['coords'].shape}")
    print(f"  faces      : {len(z['face_nnodes'])}")
    print(f"  poly sizes : {np.unique(z['face_nnodes'])}")
    print(f"  total area : {z['area'].sum():.6e} m^2")
    print(f"  bbox min   : {z['coords'].min(axis=0)}")
    print(f"  bbox max   : {z['coords'].max(axis=0)}")
    print(f"  |normal| mean (should be ~1): {np.linalg.norm(z['normal'],axis=1).mean():.4f}")
    print(f"  zero-area faces: {(z['area']==0).sum()}")
