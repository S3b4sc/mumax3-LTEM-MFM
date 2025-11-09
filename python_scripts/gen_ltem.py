from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PyLorentz.sim.sim import SimLTEM
from PyLorentz.sim import comp_phase
from PyLorentz.sim import sim  # for types like Microscope if needed
from PyLorentz.sim.sim import Microscope  # adjust import if required
# Or: from PyLorentz.sim.sim import SimLTEM
import re

def generate_ltem_image(ovf_path:str = "./mumax_files/demo.out/final.ovf", save_path:str = './images/LTEM_images/') -> None:
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
        "mem_thickness": 10.0  # nm
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
    

    # Define a microscope; 
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
    # Assuming dataset contains defocused images as numpy arrays
    # This depends on the dataset structure
    img = dataset.images[0] if hasattr(dataset, 'images') else dataset[0]
    # Nomalize image to [0, 1]
    img = (img - np.min(img)) / (np.max(img) - np.min(img))     #-------------------------------------------- Optional normalization to [0, 1] to enhance contrast
    #print(img.shape)
    #print(img)

    plt.figure(figsize=(6, 5))
    plt.imshow(np.exp(img), cmap='gray')
    #plt.imshow(img, cmap='gray')
    plt.colorbar(label='Intensity (a.u.)')
    plt.title(rf"$Simulated\ LTEM$" + "\n" + rf"$(\ defocus = {defocus_nm}\ nm)$",loc='left')
    np.save("./images/original_npy_images/" + '512_trial11.npy', img)
    plt.savefig(save_path + '512_trial.png')      # To save the image as a cmap

def gen_ltem_dataset(ovf_path:str = "./mumax_files/logs/", save_path:str = './images/LTEM_images/') -> None:
    '''
    '''

    # find all .ovf files
    ovf_files = sorted(Path(ovf_path).glob("**/final_*.ovf"),
    key=lambda f: int(re.search(r"final_(\d+)\.ovf", f.name).group(1)))

    print(ovf_files)
    #print(ovf_files)

    # count them
    #print(f"Number of .ovf files: {len(ovf_files)}")
    #print(ovf_files)

    for index,ovf in enumerate(ovf_files,start=1):
        #print(index)
        #print(ovf)

        sim = SimLTEM.load_ovf(str(ovf), verbose=1)
    
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

        img = dataset.images[0] if hasattr(dataset, 'images') else dataset[0]
        image_name = f'LTEM_{index}'

        np.save( "./images/original_npy_images/" + image_name + '.npy', img)   # Sabe the origin al npy image

    
        plt.figure(figsize=(6, 5))
        plt.imshow(img, cmap='gray')
        plt.colorbar(label='Intensity (a.u.)')
        plt.title(rf"$Simulated\ LTEM$" + "\n" + rf"$(\ defocus = {defocus_nm}\ nm)$",loc='left')
        plt.savefig(save_path + image_name + '.png')
        plt.close()