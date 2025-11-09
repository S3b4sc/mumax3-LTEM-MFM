import numpy as np, pathlib

# --- parameters ---
nx, ny = 64, 64            # grid size (small test)
dx, dy, dz = 4e-9, 4e-9, 1e-9

# Simple gradient: Ku increases along x
x = np.linspace(2.0e5, 3.0e5, nx)
Ku = np.tile(x, (ny, 1))

# --- write OVF2 scalar file ---
out = pathlib.Path("Ku_map.ovf")
header = (
    "# OOMMF: rectangular mesh v2.0\n"
    "# Title: Ku test map\n"
    "# Desc: simple gradient for MuMax3 load() test\n"
    f"# xnodes: {nx}\n# ynodes: {ny}\n# znodes: 1\n"
    f"# xstepsize: {dx}\n# ystepsize: {dy}\n# zstepsize: {dz}\n"
    "Begin: Data Text\n"
)
data = "\n".join(" ".join(f"{v:.6e}" for v in row) for row in Ku)
footer = "\nEnd: Data Text\n"
out.write_text(header + data + footer)
print("Ku_map.ovf written.")
