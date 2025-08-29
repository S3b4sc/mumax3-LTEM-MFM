import numpy as np
import pyvista as pv
from typing import List, Optional

def read_plot(in_route:str, out_route_html:str = './html_plots/plot.html') -> None:
    """
    This function reads and plots an npy  file, containing the final relaxed state
    from Mumax3 simulation

    args

    returns
        None

    """
    # Load Mumax3-converted numpy file
    spins = np.load(in_route)  # shape = (3, nz, ny, nx)
    print("Shape:", spins.shape)

    _, nz, ny, nx = spins.shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Flatten coordinates
    positions = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    positions = np.array(positions, dtype=np.float32)

    # Move spin components to last axis → (nz, ny, nx, 3), then flatten
    vectors = np.moveaxis(spins, 0, -1).reshape(-1, 3)

    # Magnitude of spins
    magnitudes = np.linalg.norm(vectors, axis=1)

    # Z-component for coloring
    sz = vectors[:, 2]

    # Create point cloud with vector field
    mesh = pv.PolyData(positions)
    mesh["spins"] = vectors
    mesh["S_z"] = sz
    mesh.set_active_vectors("spins")
    mesh.set_active_scalars("S_z")

    # Glyph arrows
    arrow = pv.Arrow(tip_length=0.4, tip_radius=0.5, shaft_radius=0.3)
    glyphs = mesh.glyph(orient="spins", scale="spins", factor=1.0, geom=arrow)  

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(glyphs, scalars="S_z", cmap="inferno", show_scalar_bar=True)
    plotter.set_background("c2c2c2")
    plotter.export_html(out_route_html)
    plotter.show()
    plotter.close()