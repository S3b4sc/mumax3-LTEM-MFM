import numpy as np

# ============================
# User parameters
# ============================
gridX = 512
gridY = 512
gridZ = 1

meanKu = 1.40e6      # J/m^3
sigma  = 0.15         # 15% dispersion

# Choose block grid size (e.g., 64x64 blocks)
blocksX = 64
blocksY = 64

# ============================
# Generate blocky Ku map
# ============================

# 1. Random values per block
block_values = np.random.normal(
    loc=meanKu,
    scale=sigma * meanKu,
    size=(blocksX, blocksY)
)

# 2. Upscale into full 512×512 map
Ku_map = np.kron(
    block_values,
    np.ones((gridX // blocksX, gridY // blocksY))
)

Ku_map = Ku_map[:gridX, :gridY]  # Ensure exact size

# Flatten into vector (OVF stores column-major)
data = Ku_map.reshape(-1)

# ============================
# Write OVF2 ASCII file
# ============================

with open("Ku_map.ovf", "w") as f:
    f.write("# OOMMF OVF 2.0\n")
    f.write("# Segment count: 1\n")
    f.write("# Begin: Segment\n")
    f.write("# Begin: Header\n")
    f.write("Title: Ku map\n")
    f.write("meshunit: m\n")
    f.write(f"xnodes: {gridX}\n")
    f.write(f"ynodes: {gridY}\n")
    f.write(f"znodes: {gridZ}\n")
    f.write("valuedim: 1\n")
    f.write("valueunits: J/m^3\n")
    f.write("valuelabels: Ku\n")
    f.write("# End: Header\n")
    f.write("# Begin: Data Text\n")

    # Write each Ku value on a new line
    for v in data:
        f.write(f"{v:.6e}\n")

    f.write("# End: Data Text\n")
    f.write("# End: Segment\n")

print("Saved Ku_map.ovf successfully.")
