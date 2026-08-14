# DISpy

DISpy is a PyQt5 desktop application for ambient-noise cross-correlation, surface-wave dispersion imaging, automatic dispersion picking, and shear-wave velocity inversion. The software is designed for DAS / dense-array surface-wave processing workflows and supports GPU acceleration for several computation steps.

## Features

- Cross-correlation data loading and visualization
- Dispersion imaging with PhaseShift, FK, and CCFJ workflows
- Automatic dispersion energy-band picking using a UNet model
- Centerline extraction from picked energy bands to generate dispersion curves
- Interactive manual editing of picked dispersion points
- Surface-wave inversion and velocity profile visualization
- Import / export of picked points, dispersion images, and inversion results
- PyInstaller packaging support for Windows executable builds

## Project Structure

```text
main.py                     Main PyQt5 application entry
main_window.py              Main UI module
pick_window.py              Automatic picking window UI module
inv_window.py               Inversion window UI module
utillis_new_gpu.py          GPU correlation and data-processing routines
disp_cal.py                 Dispersion calculation routines
SDI.py / ls_inv.py          Inversion-related algorithms
unet.py                     UNet model definition
dispersion_centerline.py    Centerline extraction for picked energy bands
dispersion_params.py        Parameter precedence and metadata handling
h5_auto.py                  HDF5 structure and parameter discovery
models/                     Local automatic-picking model directory
example.yaml                Example parameter file
requirements.txt            Direct Python dependencies
DISpy.spec                  PyInstaller build spec
```

## Installation

Python 3.10 is recommended. A CUDA-capable environment is recommended for GPU workflows. The GPU smoke tests for v1.0.0 used CuPy 13.4.1 with CUDA 12.8 and PyTorch 2.7.1+cu128.

```bash
conda create -n dispy python=3.10
conda activate dispy
pip install -r requirements.txt
```

If you use a different CUDA version, adjust the CuPy / PyTorch packages in `requirements.txt` accordingly.

## Run

```bash
python main.py
```

The included `example.yaml` uses project-relative `./data` and `./output`
directories. Update it for your data, load it in DISpy, then follow the GUI
workflow:

1. Load or calculate cross-correlation data.
2. Generate / load dispersion images.
3. Use automatic picking or manually edit dispersion points.
4. Open the inversion window and run inversion.
5. Save picked curves and inversion results.

## Parameter File

The YAML file contains three main sections:

- `cc`: cross-correlation and data-loading parameters
- `disp`: dispersion imaging parameters
- `inv_para`: inversion parameters

Example:

```yaml
disp:
  fmin: 0.01
  fmax: 8
  vmin: 100
  vmax: 1000
  method: Phaseshift

inv_para:
  optimizer: Adam
  iter_max: 100
  inv_object: vs
  device: cuda
```

## Automatic Picking

The automatic picker uses a trained UNet model to segment dispersion energy
bands. Model weights are not stored in Git. Place a compatible checkpoint at:

```text
models/unet1.pth
```

Alternatively, select a `.pth` or `.pt` file in the picking window. The
checkpoint must be a PyTorch `state_dict` compatible with `UNet(6)` in
`unet.py`. The tested local checkpoint is approximately 151 MB and has this
SHA-256 checksum:

```text
DEFC4349BF5374331CAB3ECC069DFE2CD1617B61FBC85290EB35A93B4A787EBE
```

After segmentation, `dispersion_centerline.py` extracts the centerline of each
connected energy band and converts it into dispersion points:

```text
frequency, phase velocity
```

These points are passed back to the main workflow and can be saved, edited, or used directly for inversion.

## Build EXE

On Windows, use PyInstaller:

```bash
pyinstaller DISpy.spec --distpath pyi_dist --workpath pyi_build
```

The executable will be generated under:

```text
pyi_dist/DISpy/
```

## Notes

- Large datasets, model weights, and generated results are not recommended for Git tracking.
- TDMS files are read through the `nptdms` dependency; no separate TDMS reader source is bundled.
- Some workflows require CUDA, CuPy, and a compatible NVIDIA driver.
- ObsPy warnings about version detection usually do not affect normal use if the software runs correctly.

## License

DISpy source code is available under the MIT License, except for third-party
material identified in source headers and `THIRD_PARTY_NOTICES.md`. See
`LICENSE` for details.
