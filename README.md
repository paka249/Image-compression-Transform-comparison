# Image-compression Transform Comparison

This repository contains small Python tools for experimenting with image compression using mathematical transforms (DCT, FFT, and DWT). There are separate scripts for grayscale and color workflows and convenience examples to compare outputs.

## Contents

- `python/DCT_grey.py` — original grayscale tool (2D DCT/FFT/DWT, interactive or CLI, shows 2×2 comparisons and saves results).
- `python/DCT_color.py` — color tool that applies the transforms per RGB channel and saves color reconstructions.
- `python/grey_compression.py` and `python/color_compression.py` — additional drivers created during development (see script headers for usage).

> Tip: use `--mode grid` (default) to see a 2×2 comparison (Original, DCT, FFT, DWT) or `--mode single` with `-t` to show a single transform next to the original.

## Requirements

- Python 3.8+
- pip packages:
	- numpy
	- pillow (PIL)
	- matplotlib
	- psutil
	- scipy
	- (optional) PyWavelets for DWT: `pip install PyWavelets`

Install quick:

```powershell
pip install numpy pillow matplotlib psutil scipy
pip install PyWavelets  # optional, for DWT
```

## Examples

Run grayscale (interactive pick):

```powershell
python .\python\DCT_grey.py
# or specify image, transform and percent non-interactively
python .\python\DCT_grey.py -i .\pictures\AmeSprite.webp -p 10 -t dct --mode grid
```

Run color (per-channel):

```powershell
python .\python\DCT_color.py -i .\pictures\AmeSprite.webp -p 10 --mode grid
# single transform color example
python .\python\DCT_color.py -i .\pictures\AmeSprite.webp -p 5 --mode single -t fft
```

## Output

- Scripts save reconstructed images in the working directory with names like `dct_compressed_<basename>_10pct.jpg` or `dct_color_compressed_<basename>_10pct.jpg`.
- The composite `all` grid is saved as `all_compressed_<basename>_10pct.jpg` (or `all_color_compressed_...`).
- Use `--save-coeffs` (if available in script) to save compressed coefficient arrays to `.npz` for a realistic storage size comparison.

## Planned work: MATLAB port

This project will be extended with a MATLAB port of the same transform experiments (DCT, FFT, DWT). Planned items include:

- Implementing MATLAB equivalents for the transforms and per-channel color support
- Adding export/import of coefficient files for fair file-size comparison (decide on `.mat` vs interoperable `.npz`)
- Implementing PSNR/SSIM parity tests across Python and MATLAB implementations
- Adding example MATLAB scripts under a `matlab/` folder and short usage notes

Developer-focused commands or workflow notes have been intentionally left out of this public README to keep it user-facing and concise. If you'd like, I can add a short contributor guide or a separate developer document on request.

