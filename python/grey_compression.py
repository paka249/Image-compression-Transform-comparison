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
parser = argparse.ArgumentParser(description='Transform compression on grayscale image')
parser.add_argument('-i', '--image', help='Path to input image (optional)')
parser.add_argument('-p', '--percent', type=float, help='Percent of transform coefficients to keep (0-100)')
parser.add_argument('-t', '--transform', choices=('dct', 'fft', 'dwt', 'all'), default='dct',
                    help='Transform to use: dct (default), fft, dwt, or all')
parser.add_argument('--mode', choices=('grid', 'single'), default='grid',
                    help='Display mode: grid (2x2 all transforms) or single (original + chosen transform)')
parser.add_argument('--wavelet', default='db1', help='Wavelet name for DWT (only used when --transform dwt)')
parser.add_argument('--level', type=int, default=None, help='DWT decomposition level (None = max)')
args = parser.parse_args()

# Brief program description shown at startup
print("Image-compression Transform Comparison:\n"
    "This program computes sparse reconstructions of a grayscale image using three transforms (DCT, FFT, DWT). "
    "It keeps the top-X% of transform coefficients (controlled with -p/--percent), reconstructs and saves each result, "
    "and produces a 2×2 comparison figure that shows file sizes for each output by default. Use -i/--image to select an image, "
    "--mode single to display a single transform next to the original, and --wavelet/--level to control the DWT settings.")

img_path = args.image if args.image else choose_image()
img = Image.open(img_path).convert("L")
A = np.array(img).astype(np.float64)



# Define 2D DCT and inverse
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# Always compute and display all transforms in a 2x2 grid (Original, DCT, FFT, DWT).

# Decide which transforms to run based on display mode
if args.mode == 'grid':
    transforms_to_run = ['dct', 'fft', 'dwt']
else:
    # single mode: respect the chosen transform (default 'dct'). If user passed 'all', default to 'dct'.
    chosen = args.transform if args.transform != 'all' else 'dct'
    transforms_to_run = [chosen]

# store reconstructions per transform
recons = {}
saved_files = []


# Determine percentage of coefficients to keep
if args.percent is not None:
    percent = args.percent
else:
    # prompt interactively
    default_pct = 10.0
    resp = input(f"Enter percent of coefficients to keep (0-100) [default {default_pct}%]: ").strip()
    percent = float(resp) if resp != "" else default_pct

# Validate percent
if not (0.0 < percent <= 100.0):
    raise ValueError("Percent must be > 0 and <= 100")

percentage = percent / 100.0

# Helper to compute, mask, reconstruct for a single transform
def process_transform(transform_name):
    if transform_name == 'dct':
        Acoef_local = dct2(A)

    elif transform_name == 'fft':
        import numpy.fft as fft
        Acoef_local = fft.fft2(A)

    elif transform_name == 'dwt':
        try:
            import pywt
        except Exception as e:
            raise ImportError("PyWavelets is required for DWT transform. Install with 'pip install PyWavelets'.") from e
        coeffs_local = pywt.wavedec2(A, wavelet=args.wavelet, level=args.level)
        Acoef_local, coeff_slices_local = pywt.coeffs_to_array(coeffs_local)
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    flat_local = np.abs(Acoef_local).flatten()
    sorted_idx_local = np.argsort(flat_local)[::-1]
    keep_local = int(percentage * flat_local.size)
    mask_local = np.zeros_like(Acoef_local).flatten()
    mask_local[sorted_idx_local[:keep_local]] = 1
    mask_local = mask_local.reshape(Acoef_local.shape)
    Acoef_comp_local = Acoef_local * mask_local

    # Reconstruct
    if transform_name == 'dct':
        recon = idct2(Acoef_comp_local)
    elif transform_name == 'fft':
        import numpy.fft as fft
        recon = fft.ifft2(Acoef_comp_local)
        recon = np.real(recon)
    elif transform_name == 'dwt':
        coeffs_masked_local = pywt.array_to_coeffs(Acoef_comp_local, coeff_slices_local, output_format='wavedec2')
        recon = pywt.waverec2(coeffs_masked_local, wavelet=args.wavelet)
        recon = recon[:A.shape[0], :A.shape[1]]
    else:
        raise ValueError(f"Unknown transform: {transform_name}")

    recon = np.clip(recon, 0, 255).astype(np.uint8)
    return recon

# Process each requested transform
for t in transforms_to_run:
    try:
        recon_img = process_transform(t)
        recons[t] = recon_img
        # save individual files
        base = os.path.splitext(os.path.basename(img_path))[0]
        pct_str = f"{percent:.0f}" if float(percent).is_integer() else f"{percent:g}"
        out_name = f"{t}_compressed_{base}_{pct_str}pct.jpg"
        Image.fromarray(recon_img).save(out_name)
        saved_files.append(out_name)
    except ImportError as ie:
        print(f"Skipping transform {t}: {ie}")
    except Exception as e:
        print(f"Error processing transform {t}: {e}")

# Plotting: if multiple transforms run, create a 2x2 grid: original, then the transforms
if len(transforms_to_run) > 1:
    plt.figure(figsize=(10, 8))
    # top-left: original
    plt.subplot(2, 2, 1)
    plt.imshow(A, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # fill remaining slots with available transforms in order dct, fft, dwt
    slots = [('dct', 2), ('fft', 3), ('dwt', 4)]
    for name, pos in slots:
        plt.subplot(2, 2, pos)
        if name in recons:
            plt.imshow(recons[name], cmap='gray')
            plt.title(f"{name.upper()} Compressed ({pct_str}%)")
        else:
            plt.text(0.5, 0.5, f"{name.upper()} not available", ha='center', va='center')
            plt.axis('off')
        plt.axis('off')

    # save composite image too and show sizes in titles
    comp_name = f"all_compressed_{base}_{pct_str}pct.jpg"
    plt.tight_layout()
    plt.savefig(comp_name)

    # compute sizes
    def size_kb(path):
        try:
            return os.path.getsize(path) / 1024.0
        except Exception:
            return None

    orig_size = size_kb(img_path)
    sizes = { }
    for f in saved_files:
        sizes[os.path.splitext(os.path.basename(f))[0].split('_')[0]] = size_kb(f)

    # update subplot titles to include sizes
    # top-left: original
    plt.subplot(2, 2, 1)
    title = 'Original'
    if orig_size is not None:
        title += f" — {orig_size:.1f} KB"
    plt.title(title)

    slots = [('dct', 2), ('fft', 3), ('dwt', 4)]
    for name, pos in slots:
        plt.subplot(2, 2, pos)
        if name in recons:
            plt.imshow(recons[name], cmap='gray')
            s = sizes.get(name)
            size_str = f" — {s:.1f} KB" if s is not None else ""
            plt.title(f"{name.upper()} Compressed ({pct_str}%)" + size_str)
        else:
            plt.text(0.5, 0.5, f"{name.upper()} not available", ha='center', va='center')
            plt.title(f"{name.upper()} not available")
        plt.axis('off')

    # also show composite filename and summary
    plt.savefig(comp_name)
    plt.show()
    print("Saved:", ", ".join(saved_files + [comp_name]))

else:
    # single transform: show original + compressed side-by-side
    single_t = transforms_to_run[0]
    recon_single = recons.get(single_t)
    pct_str = f"{percent:.0f}" if float(percent).is_integer() else f"{percent:g}"
    out_name = f"{single_t}_compressed_{base}_{pct_str}pct.jpg"
    print(f"Saved: {out_name}")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(A, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(recon_single, cmap='gray')
    plt.title(f"{single_t.upper()} Compressed ({pct_str}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Print saved files and sizes for single-transform case (sizes already shown for composite)
def format_size(path):
    try:
        return os.path.getsize(path) / 1024.0
    except Exception:
        return None

if len(transforms_to_run) == 1:
    # report sizes
    single_t = transforms_to_run[0]
    saved = f"{single_t}_compressed_{base}_{pct_str}pct.jpg"
    orig_s = format_size(img_path)
    saved_s = format_size(saved)
    print(f"Original size: {orig_s:.1f} KB" if orig_s is not None else "Original size: n/a")
    print(f"{single_t.upper()} compressed size: {saved_s:.1f} KB" if saved_s is not None else f"{single_t.upper()} compressed size: n/a")
