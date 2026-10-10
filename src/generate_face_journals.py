"""
Generate Fluent journals for the valve wall-pressure dataset.

Per opening it reads that opening's FFF.cas.h5 to discover the *actual* zone
names (interior name carries the opening suffix and changes per mesh), then
writes a journal that, for each (P_in, P_out) case:
  - sets pressure-inlet / pressure-outlet BC
  - hybrid-initializes and iterates to convergence
  - exports WALL data (all wall zones)   -> pressure, wall-shear (x/y/z),
       y-plus, density, temperature
  - exports FULL FIELD (interior)         -> velocity(x/y/z), pressure, density,
       temperature, mach, k, omega, turb-viscosity   (downsampled at pack time)
  - writes mass-flow report

FSI loads (plug force/moment) and the 9 experimental tap pressures are NOT
computed here -- they are integrated from the wall data + cas.h5 topology at
pack time, which is more flexible (per-zone, per-direction).

NOTE on variable tokens: Fluent's ASCII-export accepts the quantity *names*
shown in the export GUI list. Tokens below match Fluent 2021R1+/2022+/2023.
If a token is rejected, run one case interactively, open File > Export > Solution
Data, and copy the exact name into the WALL_VARS / FIELD_VARS lists.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_wall_topology import list_wall_zones

# ============ Config ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CASE_MATRIX_PATH = os.path.join(DATA_DIR, "case_matrix.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "cfd_journals")

# User-supplied Fluent case files / output dirs (not shipped with the repo).
CASE_FILE_TEMPLATE = os.path.join(
    os.environ.get("VALVE_CAS_DIR", os.path.join(ROOT_DIR, "cfd_cases")),
    "{opening}_opening", "FFF.cas.h5")
RESULT_DIR_TEMPLATE = os.path.join(
    os.environ.get("VALVE_CFD_RESULTS", os.path.join(ROOT_DIR, "cfd_results")),
    "O{opening:03d}")

MAX_ITER = 1000
CONV = 1e-4          # continuity / momentum / energy
CONV_TURB = 1e-3     # k / omega

# Boundary-condition zone names (constant across meshes)
INLET_ZONE = "inlet"
OUTLET_ZONE = "outlet"

# Turbulent BC at inlet/outlet: TI(%) and hydraulic-diameter or viscosity ratio.
# Matches the reference journals (TI=5, 80). Adjust if your case differs.
TI = 5
LEN = 80

# Wall quantities (per-face) -- FSI load + near-wall diagnostics
WALL_VARS = [
    "pressure",
    "x-wall-shear-stress", "y-wall-shear-stress", "z-wall-shear-stress",
    "y-plus",
    "density", "temperature",
]

# Full-field quantities (interior) -- downsampled at pack time
FIELD_VARS = [
    "x-velocity", "y-velocity", "z-velocity",
    "pressure", "density", "temperature",
    "mach-number",
    "turb-kinetic-energy", "specific-diss-rate", "viscosity-turb",
]

WALL_SURFACE_PREFIX = ""  # wall zones are exportable by name directly
FIELD_SURFACE = "fluid_interior"


def discover_zones(opening):
    """Return (interior_name, [wall_zone_names]) for this opening's mesh."""
    cas = CASE_FILE_TEMPLATE.format(opening=opening)
    zones = list_wall_zones(cas)  # (id, name, nfaces, minId, maxId)
    interior = None
    walls = []
    for zid, name, nf, a, b in zones:
        low = name.lower()
        if low.startswith("interior"):
            interior = name
        elif low.startswith("wall"):
            walls.append(name)
    return interior, walls


def generate_journal(opening, cases_df):
    lines = []
    case_file = CASE_FILE_TEMPLATE.format(opening=opening)
    result_dir = RESULT_DIR_TEMPLATE.format(opening=opening)
    os.makedirs(result_dir, exist_ok=True)

    interior, walls = discover_zones(opening)
    if interior is None:
        raise RuntimeError(f"opening {opening}: no interior zone found")

    wall_list = " ".join(f'"{w}"' for w in walls)
    wall_vars = " ".join(WALL_VARS)
    field_vars = " ".join(FIELD_VARS)

    lines.append(f"; ===== journal - Opening {opening}% - {len(cases_df)} cases =====")
    lines.append(f"; interior zone: {interior}")
    lines.append(f"; wall zones ({len(walls)}): {', '.join(walls)}")
    lines.append(f'/file/read-case "{case_file}"')
    lines.append("")
    # Build an interior surface for full-field export (boundary walls export by name)
    lines.append(f'/surface/zone-surface {FIELD_SURFACE} "{interior}"')
    lines.append("")
    lines.append(f'/solve/monitors/residual/convergence-criteria {CONV} {CONV} {CONV} {CONV} {CONV} {CONV_TURB} {CONV_TURB}')
    lines.append("")

    for _, row in cases_df.iterrows():
        cid = row['CaseID']
        p_in = row['P_in(kPa)'] * 1000
        p_out = row['P_out(kPa)'] * 1000
        lines.append(f"; --- {cid}  Pin={row['P_in(kPa)']:.0f}kPa Pout={row['P_out(kPa)']:.0f}kPa r={row['r(P_out/P_in_abs)']:.3f} ---")
        lines.append(f'/define/boundary-conditions/pressure-inlet {INLET_ZONE} yes no {p_in:.1f} no 0 no 300 no yes no no no yes {TI} {LEN}')
        lines.append(f'/define/boundary-conditions/pressure-outlet {OUTLET_ZONE} yes no {p_out:.1f} no 300 no yes no no no yes {TI} {LEN} yes no yes no')
        lines.append('/solve/initialize/hyb-initialization yes')
        lines.append(f'/solve/iterate {MAX_ITER}')
        lines.append('yes')  # accept "create new file?" from monitors if prompted
        # Wall export: all wall zones -> one file per case
        lines.append(f'/file/export/ascii "{result_dir}/{cid}_wall" {wall_list} () yes {wall_vars} () no')
        # Full-field export: interior surface
        lines.append(f'/file/export/ascii "{result_dir}/{cid}_field" {FIELD_SURFACE} () yes {field_vars} () no')
        # Mass-flow sanity check
        lines.append(f'/report/fluxes/mass-flow yes {INLET_ZONE} {OUTLET_ZONE} () yes "{result_dir}/{cid}_massflow.txt"')
        lines.append("")

    lines.append('/exit yes')
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CASE_MATRIX_PATH, encoding='utf-8-sig')
    openings = sorted(df['开度(%)'].unique())

    print("Generating Fluent journals")
    print("=" * 60)
    for opening in openings:
        opening = int(opening)
        cas = CASE_FILE_TEMPLATE.format(opening=opening)
        if not os.path.exists(cas):
            print(f"  Opening {opening:3d}%: SKIP (no {cas})")
            continue
        cases_df = df[df['开度(%)'] == opening].reset_index(drop=True)
        content = generate_journal(opening, cases_df)
        jpath = os.path.join(OUTPUT_DIR, f"run_opening_{opening:03d}.jou")
        with open(jpath, 'w', encoding='ascii', errors='replace') as f:
            f.write(content)
        interior, walls = discover_zones(opening)
        print(f"  Opening {opening:3d}%: {len(cases_df)} cases, {len(walls)} walls -> {os.path.basename(jpath)}")
    print("=" * 60)
    print(f"Journals in: {OUTPUT_DIR}")
    print("Run e.g.:  fluent 3ddp -t24 -i journals/run_opening_050.jou")


if __name__ == '__main__':
    main()
