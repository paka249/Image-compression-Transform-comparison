import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import psutil
import sys
import argparse
from scipy.fftpack import dct, idct

# Determine image path. If user passes a CLI argument it is used,
# otherwise list images in ../pictures and prompt the user to choose.
script_dir = os.path.dirname(os.path.abspath(__file__))
pictures_dir = os.path.normpath(os.path.join(script_dir, '..', 'pictures'))
allowed_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')

def choose_image():
    if not os.path.isdir(pictures_dir):
        raise FileNotFoundError(f"Pictures directory not found: {pictures_dir}")

    files = [f for f in os.listdir(pictures_dir) if f.lower().endswith(allowed_ext)]
    if not files:
        raise FileNotFoundError(f"No images found in: {pictures_dir}")

    print("Available images in pictures/:")
    for i, f in enumerate(files):
        print(f"  {i}: {f}")

    choice = input("Enter index or filename to use (press Enter to pick 0): ").strip()
    if choice == "":
        choice = "0"

    # numeric index
    if choice.isdigit():
        idx = int(choice)
        if idx < 0 or idx >= len(files):
            raise ValueError("Index out of range")
        return os.path.join(pictures_dir, files[idx])

    # filename
    candidate = os.path.join(pictures_dir, choice)
    if os.path.exists(candidate):
        return candidate
    # try case-insensitive match
    for f in files:
        if f.lower() == choice.lower():
            return os.path.join(pictures_dir, f)

    raise FileNotFoundError(f"Image not found: {choice}")

# Parse CLI args: optional --image and --percent
parser = argparse.ArgumentParser(description='DCT compression on grayscale image')
parser.add_argument('-i', '--image', help='Path to input image (optional)')
parser.add_argument('-p', '--percent', type=float, help='Percent of DCT coefficients to keep (0-100)')
args = parser.parse_args()

img_path = args.image if args.image else choose_image()
img = Image.open(img_path).convert("L")
A = np.array(img).astype(np.float64)

# Track memory before processing
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 ** 2)
print(f"Memory before processing: {mem_before:.2f} MB")

# Define 2D DCT and inverse
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# Perform DCT
Adct = dct2(A)

# Sort by magnitude
flat = np.abs(Adct).flatten()
sorted_indices = np.argsort(flat)[::-1]

# Determine percentage of coefficients to keep
if args.percent is not None:
    percent = args.percent
else:
    # prompt interactively
    default_pct = 10.0
    resp = input(f"Enter percent of coefficients to keep (0-100) [default {default_pct}%]: ").strip()
    percent = float(resp) if resp != "" else default_pct

if not (0.0 < percent <= 100.0):
    raise ValueError("Percent must be > 0 and <= 100")

percentage = percent / 100.0
keep = int(percentage * flat.size)

# Create mask and compress
mask = np.zeros_like(Adct).flatten()
mask[sorted_indices[:keep]] = 1
mask = mask.reshape(Adct.shape)
Adct_compressed = Adct * mask

# Reconstruct image
A_recon = idct2(Adct_compressed)
A_recon = np.clip(A_recon, 0, 255).astype(np.uint8)

# Save result (name derived from input image)
base = os.path.splitext(os.path.basename(img_path))[0]
# Include chosen percent in the filename (use integer percent when appropriate)
pct_str = f"{percent:.0f}" if percent.is_integer() else f"{percent:g}"
out_name = f"dct_compressed_{base}_{pct_str}pct.jpg"
Image.fromarray(A_recon).save(out_name)

# Display original vs compressed
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(A, cmap='gray')
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(A_recon, cmap='gray')
plt.title(f"DCT Compressed ({pct_str}%)")
plt.axis('off')
plt.tight_layout()
plt.show()

# Track memory after
mem_after = process.memory_info().rss / (1024 ** 2)
print(f"Memory after processing: {mem_after:.2f} MB")

# Difference
mem_diff = mem_after - mem_before
print(f"Estimated increase in memory usage: {mem_diff:.2f} MB")
