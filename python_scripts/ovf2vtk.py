import numpy as np

def ovf2vtk(ovf_file, vtk_file):
    with open(ovf_file, "rb") as f:
        lines = []
        while True:
            pos = f.tell()
            line = f.readline().decode("utf-8", errors="ignore").strip()
            lines.append(line)
            if line.startswith("# Begin: Data Binary"):
                data_start = f.tell()
                break

        # extract mesh size from header
        header = {l.split(":")[0].strip("# ").strip(): l.split(":", 1)[1].strip() 
                  for l in lines if ":" in l}
        nx = int(header["xnodes"])
        ny = int(header["ynodes"])
        nz = int(header["znodes"])
        dx = float(header["xstepsize"])
        dy = float(header["ystepsize"])
        dz = float(header["zstepsize"])

        # read float32 binary data (3 components per node)
        n_values = nx * ny * nz * 3
        data = np.fromfile(f, dtype=np.float32, count=n_values)
        data = data.reshape((nz, ny, nx, 3))

    # Write VTK (structured points)
    with open(vtk_file, "w") as out:
        out.write("# vtk DataFile Version 3.0\n")
        out.write("OVF converted to VTK\n")
        out.write("ASCII\n")
        out.write("DATASET STRUCTURED_POINTS\n")
        out.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        out.write(f"ORIGIN {header['xmin']} {header['ymin']} {header['zmin']}\n")
        out.write(f"SPACING {dx} {dy} {dz}\n")
        out.write(f"POINT_DATA {nx*ny*nz}\n")
        out.write("VECTORS m float\n")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    mx, my, mz = data[k, j, i]
                    out.write(f"{mx} {my} {mz}\n")

    print(f"✅ Wrote VTK file: {vtk_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python ovf2vtk.py input.ovf output.vtk")
    else:
        ovf2vtk(sys.argv[1], sys.argv[2])
