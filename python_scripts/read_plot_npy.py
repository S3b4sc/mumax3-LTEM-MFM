import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Optional

def read_plot(in_route: str,
              out_route_html: Optional[str] = "./html_plots/plot.html",
              cell_size=(1.0, 1.0, 1.0),
              glyph_factor: float = 0.8,
              lift_mz: float = 0.0) -> None:
    """
    Read a mumax3-converted .npy file (shape (3, nz, ny, nx)) and plot
    3D arrows oriented with the full (mx,my,mz) vectors. Color = mz.

    Parameters
    ----------
    in_route : str
        Path to the .npy file (mumax3-convert result).
    out_route_html : str, optional
        Path to export HTML (requires pyvista[jupyter]/trame). If None, skip export.
    cell_size : tuple(float,float,float)
        Physical spacing (dx,dy,dz) used to scale positions (e.g. in meters).
    glyph_factor : float
        Arrow size multiplier.
    downsample : int
        Plot only every `downsample` point (1 = all, 2 = every other, ...).
    lift_mz : float
        If > 0, shift each arrow base in z by (mz * lift_mz) to make arrows non-coplanar
        when the grid is a single layer (nz==1). Units are same as cell_size (e.g. meters).
    """
    # Load numpy
    spins = np.load(in_route)  # expected shape: (3, nz, ny, nx)
    if spins.ndim != 4 or spins.shape[0] != 3:
        raise ValueError("Expected spins shape (3, nz, ny, nx). Got: " + str(spins.shape))

    _, nz, ny, nx = spins.shape
    dx, dy, dz = cell_size

    # Rearrange so we have (nx, ny, nz, 3) — x fastest
    # input: (3, nz, ny, nx)
    arr = np.moveaxis(spins, 0, -1)   # (nz, ny, nx, 3)
    arr = arr.transpose(2, 1, 0, 3)   # (nx, ny, nz, 3)

    # Flatten vectors and build positions with same ordering (x fastest)
    vectors = arr.reshape(-1, 3).astype(np.float32)    # (N,3)
    ix, iy, iz = np.indices((nx, ny, nz))
    coords = np.stack((ix, iy, iz), axis=-1).reshape(-1, 3).astype(np.float32)
    # to physical coordinates
    positions = coords * np.array([dx, dy, dz], dtype=np.float32)


    # Scalars
    magnitudes = np.linalg.norm(vectors, axis=1).astype(np.float32)
    mz_scalar = vectors[:, 2].astype(np.float32)
    #print(mz_scalar)

    # NEW: compute in-plane azimuthal angle
    angles = np.arctan2(vectors[:,1], vectors[:,0])   # atan2(my, mx)
    angles = (angles + 2*np.pi) % (2*np.pi)           # wrap to [0, 2π]

    # Build PyVista
    mesh = pv.PolyData(positions)
    mesh["spins"] = vectors        # full vector for orientation
    mesh["magnitude"] = magnitudes # scalar for scaling
    mesh["mz"] = mz_scalar         # scalar for coloring
    mesh["angles"] = angles     # <-- for coloring

    # Activate arrays
    mesh.set_active_vectors("spins")
    #mesh.set_active_scalars("mz")
    mesh.set_active_scalars("angles")

    # Create glyphs
    arrow = pv.Arrow(tip_length=0.7, tip_radius=0.6, shaft_radius=0.35)
    glyphs = mesh.glyph(orient="spins", scale="spins", factor=glyph_factor, geom=arrow)

    # Plot
    plotter = pv.Plotter()
    #plotter.add_mesh(glyphs, scalars="mz", cmap="inferno", show_scalar_bar=True)
    plotter.add_mesh(glyphs, scalars="angles", cmap="hsv", show_scalar_bar=True)
    plotter.add_axes(line_width=2,
                 labels_off=False,
                 xlabel="x",
                 ylabel="y",
                 zlabel="z")
    
    plotter.set_background("c2c2c2")

    # Use an oblique/isometric camera so out-of-plane tilt is visible
    plotter.camera_position = 'iso'   # good default (not strictly top view)

    # Try export if requested
    if out_route_html:
        try:
            out_path = Path(out_route_html)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.export_html(str(out_path))
        except Exception as e:
            print("Warning: could not export HTML (optional dependency).", e)

    # Show interactive window
    plotter.show()
    plotter.close()



    ## Load numpy file converted from OVF
    #spins = np.load('./mumax_files/demo.out/final.npy')  # shape (3, nz, ny, nx)
    #print("Shape:", spins.shape)
#
    ## Extract one layer (z=0 since nz=1 in your case)
    #mx, my, mz = spins[:, 0, :, :]
#
    #ny, nx = mx.shape
    #x = np.arange(nx)
    #y = np.arange(ny)
    #X, Y = np.meshgrid(x, y)
#
    ## Plot: color = mz, arrows = (mx,my)
    #plt.figure(figsize=(6, 5))
    ##plt.pcolormesh(X, Y, mz, shading="auto", cmap="jet")  # background color
    ##plt.quiver(X, Y, mx, my, color="k", scale=30)        # arrows
    ##plt.colorbar(label="m_z")
    ##plt.title("Magnetization field")
    #plt.xlabel("x")
    #plt.ylabel("y")
    ##plt.gca().set_aspect("equal")
    ##plt.savefig('try2.png')
#
    #m_inplane = np.sqrt(mx**2 + my**2)
    #plt.imshow(m_inplane, cmap="inferno")
    #plt.colorbar(label="|m_in-plane|")
    #plt.title("In-plane magnetization magnitude")
    #plt.savefig('try2.png')
