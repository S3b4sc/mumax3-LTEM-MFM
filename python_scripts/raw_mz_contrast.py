import numpy as np
from PIL import Image
import subprocess
from pathlib import Path
import re

# ------------------------------------------------------------
# 1) Load mz using YOUR ORIGINAL shape logic
# ------------------------------------------------------------
def load_mz_from_npy(path, binary: bool = False):
    spins = np.load(path)  # whatever mumax3-convert writes

    # KEEP ORIGINAL LOGIC (no asserts, no fancy branching)
    arr = np.moveaxis(spins, 0, -1)      # (nz, ny, nx, 3) in your convention
    arr = arr.transpose(2, 1, 0, 3)      # (nx, ny, nz, 3)
    mz = arr[:, :, 0, 2]                 # mz component (nz=1 assumed)

    if binary:
        b = (mz > 0).astype(np.uint8)    # 1 white, 0 black
        img = b * 255
        return img.astype(np.uint8)

    return mz.astype(np.float32)


# ------------------------------------------------------------
# 2) Convert mz -> image (choose physical vs per-image)
# ------------------------------------------------------------
def mz_to_image(mz, mode="physical", out_bits=16):
    """
    mode:
      - "physical": assumes mz ~ [-1,1], maps to [0,1] (NO per-image scaling)
      - "per_image": your original normalization (looks good but loses absolute contrast)
    out_bits:
      - 8 or 16
    """
    if mode == "per_image":
        denom = (mz.max() - mz.min())
        if denom < 1e-12:
            im = np.zeros_like(mz, dtype=np.float32)
        else:
            im = (mz - mz.min()) / denom  # [0,1]
    elif mode == "physical":
        mz = np.clip(mz, -1.0, 1.0)
        im = (mz + 1.0) * 0.5             # [-1,1] -> [0,1]
    else:
        raise ValueError("mode must be 'physical' or 'per_image'")

    im = np.clip(im, 0.0, 1.0).astype(np.float32)

    if out_bits == 16:
        return (im * 65535).astype(np.uint16)
    elif out_bits == 8:
        return (im * 255).astype(np.uint8)
    else:
        raise ValueError("out_bits must be 8 or 16")


# ------------------------------------------------------------
# 3) Save image (8-bit or 16-bit)
# ------------------------------------------------------------
def save_image(img, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if img.dtype == np.uint16:
        Image.fromarray(img, mode="I;16").save(outpath)
    elif img.dtype == np.uint8:
        Image.fromarray(img).save(outpath)
    else:
        raise ValueError(f"Unsupported dtype for saving: {img.dtype}")


# ------------------------------------------------------------
# 4) Main dataset generation (ALIGNED with run_id)
# ------------------------------------------------------------
def run_mz_dataset(
    ovf_path: str = "./",
    save_path: str = "./",
    binary: bool = False,
    norm_mode: str = "physical",   # "physical" or "per_image"
    out_bits: int = 16             # 16 recommended
):
    ovf_path = Path(ovf_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # find all .ovf files
    ovf_files = sorted(
        ovf_path.glob("**/final_*.ovf"),
        key=lambda f: int(re.search(r"final_(\d+)\.ovf", f.name).group(1))
    )

    for ovf in ovf_files:
        # IMPORTANT: align using run_id from filename, NOT enumerate()
        run_id = int(re.search(r"final_(\d+)\.ovf", ovf.name).group(1))

        # Convert to numpy array (write .npy next to ovf)
        subprocess.run(["mumax3-convert", "-numpy", str(ovf)], check=True)

        # Use Path logic (robust) instead of string split
        npy_file = ovf.with_suffix(".npy")

        mz = load_mz_from_npy(npy_file, binary=binary)

        if binary:
            # already uint8 image
            img = mz
            out_name = save_path / f"mz_binary_{run_id}.png"
        else:
            img = mz_to_image(mz, mode=norm_mode, out_bits=out_bits)
            out_name = save_path / f"mz_binary_{run_id}.png"

        save_image(img, out_name)
        print(f"[OK] run_id={run_id:05d} -> {out_name}")


# Example
if __name__ == "__main__":
    run_mz_dataset(
        ovf_path="./mumax_dataset_ku_by_block_disorder_phy_corrected",
        save_path="./mz_images",
        binary=False,
        norm_mode="physical",   # set to "per_image" if you want EXACT old look
        out_bits=16
    )
