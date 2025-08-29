from pathlib import Path
import matplotlib.pyplot as plt
from PyLorentz.sim.sim import SimLTEM
from PyLorentz.sim import comp_phase
from PyLorentz.sim import sim  # for types like Microscope if needed
from PyLorentz.sim.sim import Microscope  # adjust import if required
# Or: from PyLorentz.sim.sim import SimLTEM


def generate_ltem_image(ovf_path:str = "./mumax_files/demo.out/final.ovf", save_path:str = './LTEM_images/') -> None:
    """
    This function takes the Mumax3 .ovf output file, and generate the Lorentz Tranmission Microscopy image

    args


    returns
        None
    """

    sim = SimLTEM.load_ovf(str(ovf_path), verbose=1)

    sim.set_sample_params({
        "B0": 1.0,             # Tesla
        "sample_V0": None,     # leave None if you don’t use them
        "sample_xip0": None,
        "mem_V0": None,
        "mem_xip0": None,
        "mem_thickness": 50.0  # nm
    })


    # ---- 1. Compute the electron phase shift ----
    # Choose 'mansuripur' (default) or 'linsup' method
    sim.compute_phase(
        method='mansuripur',
        tilt_x=0.0,        # tilt angles (if needed)
        tilt_y=0.0,
        beam_energy=300e3,  # in eV or keV? Leave None to use defaults
        device='cpu',       # or 'gpu'
        multiproc=True
    )

    # ---- 2. Simulate LTEM image(s) ----
    # For one defocus value (e.g., 1 µm = 1e3 nm)
    

    # Define a microscope; placeholder values—update as needed
    scope = Microscope()  # typical parameters to define: aperture, Cs, etc.

    defocus_nm = 1000.0  # 1000 nm = 1 µm
    dataset = sim.sim_images(
        defocus_values=defocus_nm,
        scope=scope,
        flip=False,
        filter_sigma=1.0,
        amorphous_bkg=None,
        padded_shape=None
    )

    # ---- 3. Extract image array and display ----
    # Assuming dataset contains defocused images, possibly as numpy arrays
    # This depends on the dataset structure; adjust name accordingly
    img = dataset.images[0] if hasattr(dataset, 'images') else dataset[0]

    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap='gray')
    plt.colorbar(label='Intensity (a.u.)')
    plt.title(f"Simulated LTEM (defocus = {defocus_nm} nm) /n")
    plt.savefig(save_path + 'LTEM1.png')

