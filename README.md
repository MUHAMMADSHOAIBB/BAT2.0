# BAT 2.0 Brain Annotation Toolbox

BAT 2.0 is an open-source Python toolbox for single-cell-informed cell-type annotation of neuroimaging result maps. It integrates Allen Human Brain Atlas (AHBA) transcriptomic data, deconvolved with single-nucleus RNA-sequencing-derived cell-type signatures (CIBERSORTx), to generate abundance estimates for 16 neuronal and non-neuronal cell classes. BAT 2.0 accepts volumetric NIfTI maps, surface-based CIFTI maps, user-defined parcellated maps, and functional connectivity (FC) matrices, and evaluates each input against a null model matched to its statistical structure (a spatially coherent 3D Euclidean sphere null for cluster-level/volumetric maps, and a degree-preserving topological edge-swap null for FC matrices).

This repository contains the source code and a small example dataset sufficient to run every module of the toolbox end-to-end. The complete reference and result data needed to reproduce every analysis reported in the manuscript (all parcellation resolutions, all 11 conditions) is archived separately on Zenodo, where it has a permanent, citable DOI see [Data Availability](#data-availability) below.

## Repository Structure

| Path | Contents |
|---|---|
| `scripts/Master_Toolbox.py` | Main PyQt5 GUI application (entry point for interactive use). |
| `scripts/spin_test_utils.py` | Spin-test / spatial permutation utilities used by the GUI. |
| `run_all_fc_celltype_enrichment.py` | Standalone batch script for FC network cell-type enrichment (no GUI required). |
| `data/atlases/` | Reference brain atlases (Harvard-Oxford, Desikan-Killiany, Schaefer-200) used for parcellated and cluster-level analyses. |
| `data/test_data/` | Example inputs for the Cluster & Parcellation pipeline (a cortical surface map and the multiple sclerosis demyelination cluster mask used in the manuscript). |
| `data/results/` | An example pre-computed enrichment results table. |
| `cell_type_csv/` | Parcel-wise cell-type abundance tables (Schaefer-Yeo, 100–1,000 parcels), used by the FC enrichment pipeline. |
| `cell_types/` | Surface (CIFTI) cell-type abundance maps, 100-parcel resolution. |
| `cell_types_3D/` | Volumetric (NIfTI) cell-type abundance maps, one per cell class. |
| `Functional_connectivity/` | Group-level FC matrices for all 11 neurological/psychiatric conditions at the 100-parcel Schaefer-Yeo resolution, plus the aparc-based depression dataset, including pre-computed `*_celltype_enrichment.csv` outputs for reference. This is a representative subset every condition is included, at one resolution — so the toolbox and batch script can be exercised immediately without any external download. **The remaining resolutions (200–1,000 parcels) are included in the full Zenodo data archive**, not in this repository. |
| `requirements.txt` | Python package dependencies. |

**Note on Connectome Workbench:** this repository previously bundled Windows binaries for Connectome Workbench (used internally for surface rendering). These have been removed to keep the repository lightweight. If you need them, download the appropriate build for your OS directly from the official Connectome Workbench site: https://www.humanconnectome.org/software/connectome-workbench.

## Requirements & Setup

This software is built in Python 3.8+. We recommend creating a fresh virtual environment (e.g. using `conda` or `venv`) before installing dependencies:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Core dependencies: NumPy, SciPy, pandas, nibabel, nilearn, statsmodels, Numba (JIT-accelerated permutations), and PyQt5 / PyQtWebEngine (graphical interface). All are available via PyPI.

**Note on the GUI:** `PyQtWebEngine`, used for the interactive 3D connectome and cluster viewers, requires a small number of system-level libraries that are typically already present on a desktop Linux, macOS, or Windows installation (e.g. `libxdamage1`, `libxcomposite1`, `libxrandr2` on Linux). If the GUI fails to launch with an error mentioning a missing shared library, install your distribution's standard X11/Qt runtime packages (e.g. on Debian/Ubuntu: `sudo apt-get install libxdamage1 libxcomposite1 libxrandr2 libxkbcommon-x11-0`).

## Running the GUI Toolbox

Launch the main interactive toolbox from the repository root:

```bash
python scripts/Master_Toolbox.py
```

This opens a graphical interface with four modules:
- **FC Network Enrichment Analysis** — load an FC adjacency matrix, threshold/binarize edges, and compute cell-type co-expression enrichment with topological null testing.
- **FC Network Viewer (3D)** — interactive connectome visualization with cell-type abundance overlays.
- **Cluster & Parcellation Analysis** — annotate volumetric (NIfTI) or surface (CIFTI) cluster maps and parcellated maps, with spatial null model testing (3D Euclidean sphere null or random/contiguous-block spin tests).
- **Cluster & Parcellation Viewer (3D)** — interactive volumetric/surface visualization of results.

Example datasets are provided in `data/test_data/`, `data/results/`, and `Functional_connectivity/` (one resolution, all 11 conditions) so you can exercise every tab without downloading any external data. To reproduce the full set of results across every parcellation resolution reported in the manuscript, download the complete data archive from Zenodo (see [Data Availability](#data-availability)) and merge its folders into the repository root.

## Running the Batch Script (FC Cell-Type Enrichment)

To compute FC cell-type enrichment across all binarized network files in batch, from the repository root run:

```bash
python run_all_fc_celltype_enrichment.py
```

By default, this script:
- Searches for binarized FC matrices using the relative pattern `Functional_connectivity/**/*_binarized_*.csv`.
- Matches each one to its corresponding cell-type expression table in `cell_type_csv/`, based on the Schaefer-Yeo resolution encoded in the file path.
- Runs 1,000 degree-preserving network permutations to build an empirical null distribution.
- Writes a `*_celltype_enrichment.csv` file alongside each input matrix.

This repository already ships with the `*_celltype_enrichment.csv` outputs included for reference, so a default run will report `Finished processing 0 files` — that's expected; the script skips inputs whose output already exists. To regenerate results from scratch, pass `--force`, which deletes existing outputs first:

```bash
python run_all_fc_celltype_enrichment.py --permutations 5000 --force
```

You can also point it at a different set of input files:

```bash
python run_all_fc_celltype_enrichment.py --pattern "/path/to/your/data/**/*_binarized_*.csv" --permutations 1000
```

### Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--permutations` | `1000` | Number of degree-preserving edge-swap permutations used to build the null distribution. |
| `--pattern` | `Functional_connectivity/**/*_binarized_*.csv` | Glob pattern (relative to the repository root, or an absolute path) for locating binarized FC input matrices. |
| `--force` | off | Deletes existing `*_celltype_enrichment.csv` outputs and reprocesses all matched inputs. |

## Data Availability

The full reference and result dataset backing the manuscript — all parcellation resolutions (100–1,000 parcels) for all 11 conditions, plus the reference atlases, cell-type maps, and example data also included here in miniature is permanently archived on Zenodo:

**https://doi.org/10.5281/zenodo.20839415**

The small subset of this data shipped directly in this repository (one parcellation resolution, all conditions) is sufficient to run and verify every feature of the toolbox; the Zenodo archive is needed only to reproduce every individual result reported in the paper.

## License

This project is distributed under the MIT License.

## Citation

If you use BAT 2.0 in your research, please cite our manuscript (citation details to be added upon publication) and the Zenodo data archive: https://doi.org/10.5281/zenodo.20839415
