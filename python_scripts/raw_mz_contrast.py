import numpy as np
from PIL import Image
import cv2
import subprocess
from pathlib import Path
import re

def load_mz_from_npy(path):
    spins = np.load(path)  # shape: (nx, ny, nz, 3)
    arr = np.moveaxis(spins, 0, -1)   # (nz, ny, nx, 3)
    arr = arr.transpose(2, 1, 0, 3)   # (nx, ny, nz, 3)
    mz = arr[:, :, 0, 2]  # take mz component
    binary = (mz > 0).astype(np.uint8)   # 1 = white, 0 = black
    binary_img = binary * 255     # Convert to gray scale
    #small = cv2.resize(binary_img, (128, 128), interpolation=cv2.INTER_NEAREST)      # Donwnsample

    return binary_img

def mz_to_image(mz):
    # Normalize to 0–255
    im = (mz - mz.min()) / (mz.max() - mz.min())
    im8 = (im * 255).astype(np.uint8)
    return im8

def save_image(img, outpath):
    Image.fromarray(img).save(outpath)


# main function
def run_mz_binary_dataset(ovf_path:str='./', save_path:str='.'):
     # find all .ovf files
    ovf_files = sorted(Path(ovf_path).glob("**/final_*.ovf"),
    key=lambda f: int(re.search(r"final_(\d+)\.ovf", f.name).group(1)))
    
    for index,ovf in enumerate(ovf_files,start=1):
        # Convert to numpy array
        subprocess.run(['mumax3-convert', '-numpy',str(ovf)])

        # Name of the new created numpy array
        npy_file = './' + str(ovf).split('.')[0] + '.npy'
        #print(npy_file)
        
        mz = load_mz_from_npy(npy_file)
        img = mz_to_image(mz)
        image_name = save_path + f"mz_binary_{index}.png"
        save_image(img, image_name)


