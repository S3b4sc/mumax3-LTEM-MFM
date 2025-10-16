from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent
# use a path relative to this script to avoid depending on current working dir
run_dir = repo_root / 'mumax_files' / 'logs' / 'run_00005'

print(f"Looking for .npy files in: {run_dir}")

if not run_dir.exists():
	print(f"Directory not found: {run_dir}")
	logs_dir = repo_root / 'mumax_files' / 'logs'
	if logs_dir.exists():
		print("Available directories under mumax_files/logs:")
		for p in sorted(logs_dir.iterdir()):
			if p.is_dir():
				print(' -', p.name)
	else:
		print(f"Also couldn't find parent logs directory: {logs_dir}")
	sys.exit(0)

# Path.glob returns a generator (iterator). Convert to list to inspect results.
npy_files = sorted(run_dir.glob('*.npy'))
print('Found .npy files:', npy_files)


def load_npy_file(path: Path, mmap_mode: str | None = None):
	"""Load a single .npy file and return the numpy array.

	Raises FileNotFoundError if path doesn't exist and RuntimeError if numpy is missing.
	"""
	try:
		import numpy as _np
	except Exception:
		raise RuntimeError('numpy is not installed')

	if not path.exists():
		raise FileNotFoundError(path)

	return _np.load(path, mmap_mode=mmap_mode)


def load_all_npy(directory: Path, mmap_mode: str | None = None):
	"""Load all .npy files in `directory`.

	Returns list of tuples (Path, ndarray).
	"""
	files = sorted(directory.glob('*.npy'))
	results = []
	for f in files:
		try:
			arr = load_npy_file(f, mmap_mode=mmap_mode)
			results.append((f, arr))
		except Exception as e:
			print(f'Warning: failed to load {f.name}: {e}')
	return results


def save_preview_image(array, out_path: Path):
	"""Save a quick preview image of a 2D array using matplotlib.

	Returns True on success, False if matplotlib is missing or array unsuitable.
	"""
	try:
		import matplotlib.pyplot as plt
		import numpy as _np
	except Exception:
		print('matplotlib not available; skipping preview save')
		return False

	arr = _np.asarray(array)
	if arr.ndim < 2:
		print('Array is not 2D; cannot create image preview')
		return False

	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(4, 4))
	plt.imshow(arr, cmap='viridis')
	plt.colorbar()
	plt.axis('off')
	plt.savefig(out_path, bbox_inches='tight', dpi=150)
	plt.close()
	print('Saved preview to', out_path)
	return True


if __name__ == '__main__':
	# Load all numpy files (non-mmap by default) and print a summary.
	try:
		data = load_all_npy(run_dir)
	except RuntimeError as e:
		print(e)
		print('Install numpy (pip install numpy) to load .npy files')
		sys.exit(0)

	print(f'Loaded {len(data)} arrays from {run_dir.name}')
	for p, arr in data:
		try:
			shape = getattr(arr, 'shape', 'unknown')
			dtype = getattr(arr, 'dtype', 'unknown')
		except Exception:
			shape = 'unknown'
			dtype = 'unknown'
		print('-', p.name, 'shape=', shape, 'dtype=', dtype)

	# Save a preview image for the first array if possible
	if data:
		first_path, first_arr = data[0]
		preview = repo_root / 'images' / f'preview_{first_path.stem}.png'
		save_preview_image(first_arr, preview)
