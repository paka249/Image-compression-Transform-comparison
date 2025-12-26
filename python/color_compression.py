import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import argparse
from scipy.fftpack import dct, idct

# Parse CLI args
parser = argparse.ArgumentParser(description='Transform compression on color (RGB) images')
parser.add_argument('-i', '--image', help='Path to input image (optional)')
parser.add_argument('-p', '--percent', type=float, help='Percent of transform coefficients to keep (0-100)')
parser.add_argument('-t', '--transform', choices=('dct', 'fft', 'dwt', 'all'), default='dct',
                    help='Transform to use: dct (default), fft, dwt, or all')
parser.add_argument('--mode', choices=('grid', 'single'), default='grid',
                    help='Display mode: grid (2x2 all transforms) or single (original + chosen transform)')
parser.add_argument('--wavelet', default='db1', help='Wavelet name for DWT (only used when --transform dwt)')
parser.add_argument('--level', type=int, default=None, help='DWT decomposition level (None = max)')
args = parser.parse_args()

# Brief description
print("Color Image-compression Transform Comparison:\n"
      "This program applies DCT/FFT/DWT per RGB channel, keeps top-X% coefficients per channel, "
      "reconstructs and saves each result, and shows a 2×2 comparison figure with sizes.")

# Helper to pick image
script_dir = os.path.dirname(os.path.abspath(__file__))
def choose_image():
    pictures_dir = os.path.normpath(os.path.join(script_dir, '..', 'pictures'))
    allowed_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
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
    if choice.isdigit():
        idx = int(choice)
        if idx < 0 or idx >= len(files):
            raise ValueError("Index out of range")
        return os.path.join(pictures_dir, files[idx])
    candidate = os.path.join(pictures_dir, choice)
    if os.path.exists(candidate):
        return candidate
    for f in files:
        if f.lower() == choice.lower():
            return os.path.join(pictures_dir, f)
    raise FileNotFoundError(f"Image not found: {choice}")

img_path = args.image if args.image else choose_image()
img = Image.open(img_path).convert('RGB')
A_color = np.array(img).astype(np.float64)  # H x W x 3

# percent
if args.percent is not None:
    percent = args.percent
else:
    default_pct = 10.0
    resp = input(f"Enter percent of coefficients to keep (0-100) [default {default_pct}%]: ").strip()
    percent = float(resp) if resp != "" else default_pct
if not (0.0 < percent <= 100.0):
    raise ValueError("Percent must be > 0 and <= 100")
percentage = percent / 100.0

# transforms to run
if args.mode == 'grid':
    transforms_to_run = ['dct', 'fft', 'dwt']
else:
    chosen = args.transform if args.transform != 'all' else 'dct'
    transforms_to_run = [chosen]

# 2D DCT helpers
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

# Process per channel
recons = {}
saved_files = []
base = os.path.splitext(os.path.basename(img_path))[0]

for t in transforms_to_run:
    try:
        H, W, C = A_color.shape
        recon_color = np.zeros_like(A_color, dtype=np.uint8)
        for ch in range(3):
            channel = A_color[:, :, ch]
            # forward
            if t == 'dct':
                coef = dct2(channel)
            elif t == 'fft':
                import numpy.fft as fft
                coef = fft.fft2(channel)
            elif t == 'dwt':
                try:
                    import pywt
                except Exception as e:
                    raise ImportError("PyWavelets is required for DWT transform. Install with 'pip install PyWavelets'.") from e
                coeffs = pywt.wavedec2(channel, wavelet=args.wavelet, level=args.level)
                coef, coeff_slices = pywt.coeffs_to_array(coeffs)
            else:
                raise ValueError(f"Unknown transform: {t}")

            flat = np.abs(coef).flatten()
            sorted_idx = np.argsort(flat)[::-1]
            keep = int(percentage * flat.size)
            mask = np.zeros_like(coef).flatten()
            mask[sorted_idx[:keep]] = 1
            mask = mask.reshape(coef.shape)
            coef_comp = coef * mask

            # inverse
            if t == 'dct':
                recon = idct2(coef_comp)
            elif t == 'fft':
                import numpy.fft as fft
                recon = fft.ifft2(coef_comp)
                recon = np.real(recon)
            elif t == 'dwt':
                coeffs_masked = pywt.array_to_coeffs(coef_comp, coeff_slices, output_format='wavedec2')
                recon = pywt.waverec2(coeffs_masked, wavelet=args.wavelet)
                recon = recon[:H, :W]
            recon_color[:, :, ch] = np.clip(recon, 0, 255).astype(np.uint8)

        # save
        pct_str = f"{percent:.0f}" if float(percent).is_integer() else f"{percent:g}"
        out_name = f"{t}_color_compressed_{base}_{pct_str}pct.jpg"
        Image.fromarray(recon_color).save(out_name)
        recons[t] = recon_color
        saved_files.append(out_name)
    except ImportError as ie:
        print(f"Skipping transform {t}: {ie}")
    except Exception as e:
        print(f"Error processing transform {t}: {e}")

# Plotting
if len(transforms_to_run) > 1:
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(np.clip(A_color.astype(np.uint8), 0, 255))
    plt.title('Original')
    plt.axis('off')

    slots = [('dct', 2), ('fft', 3), ('dwt', 4)]
    for name, pos in slots:
        plt.subplot(2, 2, pos)
        if name in recons:
            plt.imshow(recons[name])
            plt.title(f"{name.upper()} Compressed ({pct_str}%)")
        else:
            plt.text(0.5, 0.5, f"{name.upper()} not available", ha='center', va='center')
            plt.axis('off')
        plt.axis('off')

    comp_name = f"all_color_compressed_{base}_{pct_str}pct.jpg"
    plt.tight_layout()
    plt.savefig(comp_name)
    plt.show()
    print("Saved:", ", ".join(saved_files + [comp_name]))
else:
    single_t = transforms_to_run[0]
    recon_single = recons.get(single_t)
    out_name = f"{single_t}_color_compressed_{base}_{pct_str}pct.jpg"
    print(f"Saved: {out_name}")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.clip(A_color.astype(np.uint8), 0, 255))
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(recon_single)
    plt.title(f"{single_t.upper()} Compressed ({pct_str}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# show sizes for single case
if len(transforms_to_run) == 1:
    def size_kb(path):
        try:
            return os.path.getsize(path) / 1024.0
        except Exception:
            return None
    orig = size_kb(img_path)
    saved = size_kb(out_name)
    print(f"Original size: {orig:.1f} KB" if orig is not None else "Original size: n/a")
    print(f"{single_t.upper()} compressed size: {saved:.1f} KB" if saved is not None else f"{single_t.upper()} compressed size: n/a")
