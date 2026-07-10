import sys
import os
import pandas as pd
import numpy as np
import nibabel as nib
import tempfile
import subprocess
import glob
import re
from nilearn import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QComboBox, QLabel, QPushButton, 
                             QMessageBox, QGroupBox, QListWidget, QTabWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QFormLayout, QSpinBox, QDoubleSpinBox, QFileDialog, QAbstractItemView,
                             QCheckBox, QStackedWidget, QSplitter)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QFont, QColor, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numba import njit
from statsmodels.stats.multitest import fdrcorrection

# IMPORT FC ENRICHMENT FUNCTIONS FROM BULK ANALYSIS SCRIPT
import sys as _sys

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_fc_enrich_dir = BASE_DIR
if _fc_enrich_dir not in _sys.path:
    _sys.path.insert(0, _fc_enrich_dir)
try:
    from run_all_fc_celltype_enrichment import (
        calculate_coexpression_fast,
        fast_double_edge_swap,
        load_data as _fc_load_data,
    )
    _FC_ENRICH_AVAILABLE = True
except Exception as _e:
    _FC_ENRICH_AVAILABLE = False
    _FC_ENRICH_ERROR = str(_e)

# IMPORT SPIN TEST FUNCTION (same scripts/ directory)
_spin_dir = os.path.dirname(os.path.abspath(__file__))
if _spin_dir not in _sys.path:
    _sys.path.insert(0, _spin_dir)
try:
    from spin_test_utils import generate_spin_permutations
    _SPIN_AVAILABLE = True
except Exception as _se:
    _SPIN_AVAILABLE = False
    _SPIN_ERROR = str(_se)

def get_schaefer_coords(resolution):
    """Load Schaefer centroid RAS coordinates and ROI names.
    Downloads from ThomasYeoLab/CBIG GitHub if not cached locally.
    Returns (coords_array, roi_names_array) or (None, None) on failure.
    """
    coord_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schaefer_coords")
    os.makedirs(coord_dir, exist_ok=True)
    coord_file = os.path.join(
        coord_dir,
        f"Schaefer2018_{resolution}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    )
    if os.path.exists(coord_file):
        df = pd.read_csv(coord_file)
        return df[['R', 'A', 'S']].values, df['ROI Name'].values
    url = (
        f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
        f"brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/"
        f"Schaefer2018_{resolution}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    )
    try:
        import requests, io
        r = requests.get(url, verify=False)
        if r.status_code == 200:
            with open(coord_file, 'wb') as f:
                f.write(r.content)
            df = pd.read_csv(io.StringIO(r.text))
            return df[['R', 'A', 'S']].values, df['ROI Name'].values
    except Exception:
        pass
    return None, None


def get_node_colors(roi_names):
    """Map Yeo-7 network labels in ROI names to hex display colours."""
    color_map = {
        'Vis': '#781286', 'SomMot': '#4682b4', 'DorsAttn': '#00760e',
        'SalVentAttn': '#c43afa', 'Limbic': '#dcf8a4', 'Cont': '#e69422',
        'Default': '#cd3e4e'
    }
    colors = []
    for name in roi_names:
        matched = False
        for net, color in color_map.items():
            if (f"7Networks_{net}" in name or
                    f"7Networks_LH_{net}" in name or
                    f"7Networks_RH_{net}" in name):
                colors.append(color)
                matched = True
                break
        if not matched:
            colors.append('black')
    return colors

class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, cluster_path, cell_files, wb_command_path, n_permutations, is_cifti=True, mask_path=None, perm_method="Random", apply_fdr=True):
        super().__init__()
        self.cluster_path = cluster_path
        self.cell_files = cell_files
        self.wb_command_path = wb_command_path
        self.n_permutations = n_permutations
        self.is_cifti = is_cifti
        self.mask_path = mask_path
        self.perm_method = perm_method
        self.apply_fdr = apply_fdr

    def extract_cifti_data(self, cifti_path):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_txt:
            temp_txt_path = temp_txt.name
            
        cmd = [self.wb_command_path, '-cifti-convert', '-to-text', cifti_path, temp_txt_path]
        
        try:
            if os.name == 'nt':
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
            data = np.loadtxt(temp_txt_path)
            data = data.flatten().copy()
            return data
        except subprocess.CalledProcessError as e:
            pass
        finally:
            if os.path.exists(temp_txt_path):
                try:
                    os.remove(temp_txt_path)
                except Exception:
                    pass
                
        try:
            img = nib.nifti2.Nifti2Image.from_filename(cifti_path)
            data = np.array(img.dataobj).flatten().copy()
            del img
            import gc
            gc.collect()
            return data
        except Exception as e:
            raise ValueError(f"Failed to load {cifti_path}: {e}")

    def run(self):
        try:
            self.log.emit(f"Loading cluster map: {os.path.basename(self.cluster_path)}...")
            self.progress.emit(5)
            
            if self.is_cifti:
                try:
                    cluster_data = self.extract_cifti_data(self.cluster_path)
                except Exception as e:
                    raise Exception(f"Failed to load CIFTI map '{os.path.basename(self.cluster_path)}'.\n\nEnsure this is a true surface file (e.g. .dscalar.nii).\nError details: {e}")
                brain_indices = np.where(cluster_data > 0)[0]
                brain_clusters = cluster_data[brain_indices]
            else:
                cluster_img = nib.load(self.cluster_path)
                cluster_data = cluster_img.get_fdata()
                mask_data = cluster_data > 1e-6
                brain_indices = np.where(mask_data)
                brain_clusters = cluster_data[brain_indices]

            unique_clusters = np.unique(brain_clusters)
            unique_clusters = unique_clusters[(unique_clusters > 1e-6) & (~np.isnan(unique_clusters))]
            
            if self.mask_path:
                self.log.emit(f"Loading permutation mask: {os.path.basename(self.mask_path)}...")
                try:
                    if self.is_cifti:
                        mask_data = self.extract_cifti_data(self.mask_path)
                        self.valid_perm_indices = np.where(mask_data > 0)[0]
                    else:
                        mask_img = nib.load(self.mask_path)
                        mask_data = mask_img.get_fdata()
                        if mask_data.shape != cluster_data.shape:
                            if mask_data.size == cluster_data.size:
                                mask_data = mask_data.reshape(cluster_data.shape)
                            else:
                                from nilearn.image import resample_to_img
                                cluster_img = nib.load(self.cluster_path)
                                resampled_mask = resample_to_img(self.mask_path, cluster_img, interpolation='nearest')
                                mask_data = resampled_mask.get_fdata()
                        self.valid_perm_indices = np.where(mask_data > 1e-6)
                except Exception as e:
                    self.log.emit(f"Error loading mask: {e}. Falling back to whole brain.")
                    self.valid_perm_indices = brain_indices
            else:
                self.valid_perm_indices = brain_indices
            
            self.log.emit(f"Found {len(unique_clusters)} unique cluster(s): {unique_clusters[:5]}...")
            self.progress.emit(10)

            _sphere_tree = None
            _sphere_coords = None
            if not self.is_cifti:
                from scipy.spatial import cKDTree as _cKDTree
                _sphere_coords = np.vstack(self.valid_perm_indices).T
                _sphere_tree = _cKDTree(_sphere_coords)
                self.log.emit(f"KDTree built on {len(_sphere_coords)} universe voxels (used for 3D sphere permutations).")

            results = []
            total_cells = len(self.cell_files)
            
            for idx, (cell_name, cell_path) in enumerate(self.cell_files.items()):
                self.log.emit(f"Processing {cell_name}...")
                
                try:
                    if self.is_cifti:
                        import gc
                        gc.collect()
                        cell_data = self.extract_cifti_data(cell_path)
                        if cell_data.size != cluster_data.size:
                            continue
                        brain_cell_exp = cell_data[brain_indices].copy()
                        if self.mask_path:
                            perm_universe_exp = cell_data[self.valid_perm_indices].copy()
                        else:
                            perm_universe_exp = brain_cell_exp.copy()
                    else:
                        import gc
                        gc.collect()
                        cell_img = nib.load(cell_path)
                        cell_data = cell_img.get_fdata().copy()
                        del cell_img
                        
                        if cell_data.shape != cluster_data.shape:
                            if cell_data.size == cluster_data.size:
                                cell_data = cell_data.reshape(cluster_data.shape)
                            else:
                                try:
                                    from nilearn.image import resample_to_img
                                    cluster_img = nib.load(self.cluster_path)
                                    resampled_img = resample_to_img(cell_path, cluster_img, interpolation='continuous')
                                    cell_data = resampled_img.get_fdata().copy()
                                    del resampled_img
                                    del cluster_img
                                except Exception as e:
                                    continue
                        brain_cell_exp = cell_data[brain_indices].copy()
                        if self.mask_path:
                            perm_universe_exp = cell_data[self.valid_perm_indices].copy()
                        else:
                            perm_universe_exp = brain_cell_exp.copy()
                except Exception as e:
                    continue

                for cluster_id in unique_clusters:
                    cluster_mask = (brain_clusters == cluster_id)
                    cluster_size = np.sum(cluster_mask)
                    
                    if cluster_size == 0: continue
                    
                    # Log cluster to terminal so user can see progress
                    print(f"Processing {cell_name} - Cluster ID: {cluster_id} (Size: {cluster_size} voxels)", flush=True)
                    self.log.emit(f"Processing {cell_name} - Cluster ID: {cluster_id} (Size: {cluster_size} voxels)")
                        
                    current_brain_exp = brain_cell_exp.copy()
                    
                    actual_mean = np.mean(current_brain_exp[cluster_mask])
                    
                    universe_len = len(perm_universe_exp)
                    
                    # Pre-allocate array for permutation means
                    perm_means = np.zeros(self.n_permutations)
                    
                    # ── Random Permutation (memory-safe, vectorized where possible) ──
                    if self.perm_method in ("Random", "Random (Default)"):

                        matrix_bytes = self.n_permutations * universe_len * 8  
                        if matrix_bytes < 256 * 1024 * 1024: 
                            rand_mat = np.random.rand(self.n_permutations, universe_len)
                            rand_indices = np.argpartition(rand_mat, cluster_size, axis=1)[:, :cluster_size]
                            perm_means = np.mean(perm_universe_exp[rand_indices], axis=1)
                        else:
                            rng = np.random.default_rng()
                            rand_mat = np.empty((self.n_permutations, cluster_size), dtype=np.int64)
                            for i in range(self.n_permutations):
                                rand_mat[i] = rng.choice(universe_len, size=cluster_size, replace=False)
                            perm_means = np.mean(perm_universe_exp[rand_mat], axis=1)
                    else:
                        is_contiguous = self.perm_method in (
                            "Contiguous Block (partial spatial control)",
                            "Nearby Spatial (Contiguous)"
                        )
                        
                        if is_contiguous and not self.is_cifti:
                            start_indices = np.random.randint(0, universe_len, size=self.n_permutations)
                            start_coords = _sphere_coords[start_indices]
                            k_neighbors = min(cluster_size, len(_sphere_coords))
                            _, nearest_indices = _sphere_tree.query(start_coords, k=k_neighbors, workers=-1)
                            
                            if k_neighbors == 1:
                                nearest_indices = np.expand_dims(nearest_indices, axis=1)
                            perm_means = np.mean(perm_universe_exp[nearest_indices], axis=1)
                            
                        else:
                            for i in range(self.n_permutations):
                                if is_contiguous:
                                    start_idx = np.random.randint(0, universe_len)
                                    end_idx = start_idx + cluster_size
                                    if end_idx <= universe_len:
                                        perm_indices = np.arange(start_idx, end_idx)
                                    else:
                                        perm_indices = np.concatenate([
                                            np.arange(start_idx, universe_len),
                                            np.arange(0, end_idx - universe_len)
                                        ])
                                else:
                                    perm_indices = np.random.choice(universe_len, size=cluster_size, replace=False)
                                    
                                perm_means[i] = np.mean(perm_universe_exp[perm_indices])
                        
                    null_mean = np.mean(perm_means)
                    actual_diff = abs(actual_mean - null_mean)
                    perm_diffs = np.abs(perm_means - null_mean)

                    p_val = np.sum(perm_diffs >= actual_diff) / self.n_permutations

                    p_val_display = "< 0.001" if p_val == 0.0 else f"{p_val:.4f}"

                    results.append({
                        'Cluster_ID': cluster_id,
                        'Cell_Type': cell_name,
                        'Mean_Expression': actual_mean,
                        'P_Value_Raw': p_val,
                        'P_Value': p_val_display,
                        'Significant (p<0.05)': p_val < 0.05
                    })
                
                progress_val = 10 + int(90 * ((idx + 1) / total_cells))
                self.progress.emit(progress_val)
                
            df_results = pd.DataFrame(results)
            if self.apply_fdr and 'P_Value_Raw' in df_results.columns and len(df_results) > 0:
                try:
                    from statsmodels.stats.multitest import fdrcorrection
                    _, fdr_adj = fdrcorrection(df_results['P_Value_Raw'].values, alpha=0.05)
                    df_results['FDR_Adjusted_P'] = fdr_adj
                    df_results['FDR_P_Display'] = [
                        '< 0.001' if q <= 0.0 else f"{q:.4f}" for q in fdr_adj
                    ]
                    df_results['Significant_FDR (q<0.05)'] = fdr_adj < 0.05
                    self.log.emit(
                        f"FDR correction applied across {len(df_results)} tests "
                        f"({int((fdr_adj < 0.05).sum())} survive q<0.05)."
                    )
                except Exception as e:
                    self.log.emit(f"FDR correction skipped: {e}")

            self.progress.emit(100)
            self.log.emit("Analysis Complete!")
            self.finished.emit(df_results)
            
        except Exception as e:
            self.error.emit(str(e))


# TAB 1: CELL TYPE TOOLBOX WIDGET
class CellTypeWidget(QWidget):
    def __init__(self, parent_window=None):
        super().__init__()
        self.parent_window = parent_window
        self.cluster_file = ""
        self.cell_files = {}
        self.wb_path = f"{BASE_DIR}/workbench_win/bin_windows64/wb_command.exe"
        self.results_df = None
        self.initUI()
        
    def initUI(self):
        self.cell_type_data = None
        main_layout = QHBoxLayout(self)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(420)
        
        # Add back button
        btn_back = QPushButton("← Back to Main Menu")
        btn_back.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        if self.parent_window:
            btn_back.clicked.connect(self.parent_window.show_main_menu)
        left_layout.addWidget(btn_back)
        
        cluster_group = QGroupBox("1. Target Cluster/Region Map")
        cluster_layout = QVBoxLayout()
        self.combo_atlas = QComboBox()
        self.combo_atlas.addItems(["Custom File (Select Below)", "Auto: Harvard-Oxford 1mm", "Auto: Harvard-Oxford 2mm"])
        self.combo_atlas.currentIndexChanged.connect(self.on_atlas_changed)
        cluster_layout.addWidget(self.combo_atlas)
        
        self.lbl_cluster = QLabel("No file selected")
        self.lbl_cluster.setWordWrap(True)
        self.btn_cluster = QPushButton("Select Custom Cluster Map\n(.nii / .dscalar.nii)")
        self.btn_cluster.setMinimumHeight(40) # Ensure it has enough vertical space
        self.btn_cluster.clicked.connect(self.load_cluster)
        cluster_layout.addWidget(self.btn_cluster)
        cluster_layout.addWidget(self.lbl_cluster)
        cluster_group.setLayout(cluster_layout)
        left_layout.addWidget(cluster_group)
        
        cell_group = QGroupBox("2. Cell Type Maps")
        cell_layout = QVBoxLayout()
        btn_add_cells = QPushButton("Add Custom Cell Type Maps")
        btn_add_cells.clicked.connect(self.add_cell_types)
        
        btn_load_default_3d = QPushButton("Load Default 3D Cell Types\n(NIFTI)")
        btn_load_default_3d.setMinimumHeight(40)
        btn_load_default_3d.setStyleSheet("background-color: #e6f7ff;")
        btn_load_default_3d.clicked.connect(self.load_default_cells_3d)
        
        btn_load_default = QPushButton("Load Default Surface Cell Types\n(CIFTI)")
        btn_load_default.setMinimumHeight(40)
        btn_load_default.clicked.connect(self.load_default_cells)
        
        btn_clear_cells = QPushButton("Clear List")
        btn_clear_cells.clicked.connect(self.clear_cell_types)
        
        self.list_cells = QListWidget()
        self.list_cells.setSelectionMode(QAbstractItemView.ExtendedSelection)
        
        cell_layout.addWidget(btn_add_cells)
        cell_layout.addWidget(btn_load_default_3d)
        cell_layout.addWidget(btn_load_default)
        cell_layout.addWidget(self.list_cells)
        cell_layout.addWidget(btn_clear_cells)
        cell_group.setLayout(cell_layout)
        left_layout.addWidget(cell_group)
        
        settings_group = QGroupBox("3. Analysis Settings")
        settings_layout = QVBoxLayout()
        
        self.combo_format = QComboBox()
        self.combo_format.addItems(["CIFTI (Surface .dscalar.nii)", "NIFTI (Volume 3D .nii)"])
        
        self.combo_perm_method = QComboBox()
        self.combo_perm_method.addItems([
            "Random (Default)",
            "Contiguous Block (partial spatial control)",
        ])
        self.combo_perm_method.setToolTip(
            "Random (Default): randomly sample cluster_size voxels/parcels from the\n"
            "universe. Memory-safe and fastest — no spatial overhead.\n\n"
            "Contiguous Block (partial spatial control):\n"
            "  • NIfTI / volume data → BAT 1.0 method: picks a random seed voxel and\n"
            "    selects its cluster_size nearest neighbours by 3D Euclidean distance\n"
            "    (KDTree sphere). Preserves realistic spatial autocorrelation.\n"
            "  • CIFTI / surface data → 1D contiguous index slicing (surface topology\n"
            "    does not have simple 3D XYZ coordinates in this context).\n\n"
            "Note: for true spin-test nulls, pre-compute rotations with neuromaps or\n"
            "brainsmash and supply the rotated maps directly."
        )

        self.spin_perms = QSpinBox()
        self.spin_perms.setRange(100, 100000)
        self.spin_perms.setValue(1000)
        self.spin_perms.setSingleStep(1000)

        self.chk_fdr = QCheckBox("Apply Benjamini-Hochberg FDR across cell types")
        self.chk_fdr.setChecked(True)
        self.chk_fdr.setToolTip(
            "Recommended. Controls the family-wise false-discovery rate across the\n"
            "16-18 cell types tested against each cluster."
        )

        self.mask_file = None
        self.combo_mask = QComboBox()
        self.combo_mask.addItems(["Whole Brain (Atlas Default)", "Auto: MNI152 Gray Matter Mask", "Custom Mask File"])
        self.combo_mask.currentIndexChanged.connect(self.on_mask_changed)
        
        self.btn_mask = QPushButton("Select Mask")
        self.btn_mask.setEnabled(False)
        self.btn_mask.clicked.connect(self.load_mask)
        self.lbl_mask = QLabel("No mask selected")
        
        settings_layout.addWidget(QLabel("Data Format:"))
        settings_layout.addWidget(self.combo_format)
        
        settings_layout.addWidget(QLabel("Permutation Method:"))
        settings_layout.addWidget(self.combo_perm_method)
        
        settings_layout.addWidget(QLabel("Permutations:"))
        settings_layout.addWidget(self.spin_perms)

        settings_layout.addWidget(self.chk_fdr)

        settings_layout.addWidget(QLabel("Permutation Mask:"))
        mask_layout = QHBoxLayout()
        mask_layout.addWidget(self.combo_mask)
        mask_layout.addWidget(self.btn_mask)
        mask_layout.addWidget(self.lbl_mask)
        settings_layout.addLayout(mask_layout)
        
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        
        self.btn_run = QPushButton("Run Enrichment Analysis")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_run.clicked.connect(self.run_analysis)
        left_layout.addWidget(self.btn_run)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        
        self.lbl_log = QLabel("Ready.")
        self.lbl_log.setWordWrap(True)
        left_layout.addWidget(self.lbl_log)
        left_layout.addStretch()
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Cluster ID", "Cell Type", "Mean Expression",
            "P-Value", "Significant (p<0.05)",
            "FDR q-value", "Significant FDR (q<0.05)",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.table)
        
        self.figure = plt.figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)
        
        self.btn_export = QPushButton("Export Results to CSV & Plot")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results)
        right_layout.addWidget(self.btn_export)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def on_atlas_changed(self):
        idx = self.combo_atlas.currentIndex()
        if idx == 0:
            self.btn_cluster.setEnabled(True)
            self.cluster_file = ""
            self.lbl_cluster.setText("No file selected")
        elif idx == 1:
            self.btn_cluster.setEnabled(False)
            try:
                from nilearn import datasets
                self.update_log("Fetching Harvard-Oxford 1mm atlas...")
                ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
                atlas_dir = os.path.join(BASE_DIR, "data", "atlases")
                os.makedirs(atlas_dir, exist_ok=True)
                atlas_path = os.path.join(atlas_dir, "HarvardOxford_1mm.nii.gz")
                if isinstance(ho.maps, str):
                    import shutil
                    shutil.copy(ho.maps, atlas_path)
                else:
                    nib.save(ho.maps, atlas_path)
                self.cluster_file = atlas_path
                self.lbl_cluster.setText(f"Auto-loaded: HarvardOxford_1mm.nii.gz")
                self.combo_format.setCurrentIndex(1)
                self.update_log("Ready.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to fetch 1mm atlas: {e}")
        elif idx == 2:
            self.btn_cluster.setEnabled(False)
            try:
                from nilearn import datasets
                self.update_log("Fetching Harvard-Oxford 2mm atlas...")
                ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
                atlas_dir = os.path.join(BASE_DIR, "data", "atlases")
                os.makedirs(atlas_dir, exist_ok=True)
                atlas_path = os.path.join(atlas_dir, "HarvardOxford_2mm.nii.gz")
                if isinstance(ho.maps, str):
                    import shutil
                    shutil.copy(ho.maps, atlas_path)
                else:
                    nib.save(ho.maps, atlas_path)
                self.cluster_file = atlas_path
                self.lbl_cluster.setText(f"Auto-loaded: HarvardOxford_2mm.nii.gz")
                self.combo_format.setCurrentIndex(1)
                self.update_log("Ready.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to fetch 2mm atlas: {e}")

    def load_cluster(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Cluster Map", BASE_DIR, 
                                                   "Brain Maps (*.nii *.dscalar.nii *.gii);;All Files (*)", options=options)
        if file_name:
            self.cluster_file = file_name
            self.lbl_cluster.setText(os.path.basename(file_name))
            if ".dscalar" in file_name.lower():
                self.combo_format.setCurrentIndex(0)
            else:
                self.combo_format.setCurrentIndex(1)
                
    def on_mask_changed(self):
        idx = self.combo_mask.currentIndex()
        if idx == 0:
            self.btn_mask.setEnabled(False)
            self.mask_file = None
            self.lbl_mask.setText("No mask selected")
        elif idx == 1:
            self.btn_mask.setEnabled(False)
            try:
                from nilearn import datasets
                self.update_log("Fetching MNI152 Gray Matter mask...")
                mask_img = datasets.load_mni152_gm_mask()
                atlas_dir = os.path.join(BASE_DIR, "data", "atlases")
                os.makedirs(atlas_dir, exist_ok=True)
                mask_path = os.path.join(atlas_dir, "MNI152_GM_mask.nii.gz")
                nib.save(mask_img, mask_path)
                self.mask_file = mask_path
                self.lbl_mask.setText("Auto-loaded: MNI152_GM_mask.nii.gz")
                self.combo_format.setCurrentIndex(1)
                self.update_log("Ready.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to fetch Gray Matter mask: {e}")
        elif idx == 2:
            self.btn_mask.setEnabled(True)
            self.mask_file = None
            self.lbl_mask.setText("No mask selected")

    def load_mask(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Permutation Mask", BASE_DIR, 
                                                   "Brain Maps (*.nii *.dscalar.nii);;All Files (*)", options=options)
        if file_name:
            self.mask_file = file_name
            self.lbl_mask.setText(os.path.basename(file_name))
                
    def add_cell_types(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select Cell Type Maps", f"{BASE_DIR}/cell_types", 
                                                "Brain Maps (*.nii *.dscalar.nii);;All Files (*)", options=options)
        if files:
            for f in files:
                name = os.path.basename(f).split('_')[0].split('.')[0].capitalize()
                self.cell_files[name] = f
                self.list_cells.addItem(f"{name} ({os.path.basename(f)})")

    def load_default_cells_3d(self):
        default_dir = f"{BASE_DIR}/cell_types_3D"
        if not os.path.exists(default_dir):
            QMessageBox.warning(self, "Error", f"3D cell types directory not found: {default_dir}")
            return
        self.clear_cell_types()
        loaded = 0
        for file in os.listdir(default_dir):
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                if "_L" in file or "_R" in file: continue
                name = file.split('.')[0].capitalize()
                full_path = os.path.join(default_dir, file)
                self.cell_files[name] = full_path
                self.list_cells.addItem(f"{name} ({file})")
                loaded += 1
        if loaded > 0:
            self.combo_format.setCurrentIndex(1)

    def load_default_cells(self):
        default_dir = f"{BASE_DIR}/cell_types/atlas_100_7net"
        if not os.path.exists(default_dir): return
        default_map = {
            'Astrocyte': 'astro_sig_schaef100_net7.dscalar.nii',
            'Endothelial': 'endothelial_sig_schaef100_net7.dscalar.nii',
            'Microglia': 'microglia_sig_schaef100_net7.dscalar.nii',
            'Oligo': 'oligo_sig_schaef100_net7.dscalar.nii',
            'OPC': 'opc_sig_schaef100_net7.dscalar.nii',
            'PVALB': 'pvalb_sig_schaef100_net7.dscalar.nii',
            'SST': 'sst_sig_schaef100_net7.dscalar.nii',
            'VIP': 'vip_sig_schaef100_net7.dscalar.nii'
        }
        self.clear_cell_types()
        for name, filename in default_map.items():
            full_path = os.path.join(default_dir, filename)
            if os.path.exists(full_path):
                self.cell_files[name] = full_path
                self.list_cells.addItem(f"{name} ({filename})")
                
    def clear_cell_types(self):
        self.cell_files.clear()
        self.list_cells.clear()
        
    def update_log(self, msg):
        self.lbl_log.setText(msg)
        
    def update_progress(self, val):
        self.progress_bar.setValue(val)
        
    def handle_error(self, err_msg):
        QMessageBox.critical(self, "Analysis Error", err_msg)
        self.btn_run.setEnabled(True)
        self.lbl_log.setText("Error occurred.")
        self.progress_bar.setValue(0)
        
    def display_results(self, df):
        self.results_df = df
        self.btn_run.setEnabled(True)
        self.btn_export.setEnabled(True)
        if df.empty:
            self.lbl_log.setText("Analysis finished, but no results were generated.")
            return
        self.table.setRowCount(0)
        self.table.setRowCount(len(df))
        has_fdr = 'FDR_P_Display' in df.columns
        for i, row in df.iterrows():
            self.table.setItem(i, 0, QTableWidgetItem(str(row['Cluster_ID'])))
            self.table.setItem(i, 1, QTableWidgetItem(str(row['Cell_Type'])))
            self.table.setItem(i, 2, QTableWidgetItem(f"{row['Mean_Expression']:.4f}"))
          
            self.table.setItem(i, 3, QTableWidgetItem(str(row['P_Value'])))

            sig_item = QTableWidgetItem(str(row['Significant (p<0.05)']))
            if row['Significant (p<0.05)']:
                sig_item.setBackground(Qt.green)
            self.table.setItem(i, 4, sig_item)

            if has_fdr:
                self.table.setItem(i, 5, QTableWidgetItem(str(row['FDR_P_Display'])))
                fdr_sig_item = QTableWidgetItem(str(row['Significant_FDR (q<0.05)']))
                if bool(row['Significant_FDR (q<0.05)']):
                    fdr_sig_item.setBackground(Qt.green)
                self.table.setItem(i, 6, fdr_sig_item)
            else:
                self.table.setItem(i, 5, QTableWidgetItem("—"))
                self.table.setItem(i, 6, QTableWidgetItem("—"))
        self.plot_results(df)
        
    def plot_results(self, df):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        cluster_id = df['Cluster_ID'].iloc[0]
        plot_df = df[df['Cluster_ID'] == cluster_id].sort_values(by="Mean_Expression", ascending=False)
        colors = ["#d62728" if sig else "#999999" for sig in plot_df["Significant (p<0.05)"]]
        sns.barplot(x="Mean_Expression", y="Cell_Type", data=plot_df, palette=colors, ax=ax)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_title(f"Cell Type Enrichment (Cluster {cluster_id})", fontsize=12)
        ax.set_xlabel("Mean Expression (Z-score)")
        ax.set_ylabel("")
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', label='Significant (p < 0.05)'),
            Patch(facecolor='#999999', label='Not Significant')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def export_results(self):
        if self.results_df is None: return
        results_dir = f"{BASE_DIR}/data/results"
        os.makedirs(results_dir, exist_ok=True)
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Results", os.path.join(results_dir, "enrichment_results.csv"), 
                                                   "CSV Files (*.csv)", options=options)
        if file_name:
            self.results_df.to_csv(file_name, index=False)
            plot_file = file_name.replace('.csv', '_plot.png')
            self.figure.savefig(plot_file, dpi=300)
            QMessageBox.information(self, "Success", f"Results and plot saved to:\n{file_name}")

    def run_analysis(self):
        if not self.cluster_file:
            QMessageBox.warning(self, "Missing Input", "Please select a Cluster Map.")
            return
        if not self.cell_files:
            QMessageBox.warning(self, "Missing Input", "Please add at least one Cell Type Map.")
            return
            
        self.btn_run.setEnabled(False)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)
        self.figure.clear()
        self.canvas.draw()
        
        is_cifti = self.combo_format.currentIndex() == 0
        perms = self.spin_perms.value()
        mask_path = self.mask_file if self.combo_mask.currentIndex() > 0 else None
        perm_method = self.combo_perm_method.currentText()
        apply_fdr = self.chk_fdr.isChecked()

        self.worker = AnalysisWorker(self.cluster_file, self.cell_files, self.wb_path, perms, is_cifti, mask_path, perm_method, apply_fdr)
        self.worker.progress.connect(self.update_progress)
        self.worker.log.connect(self.update_log)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.handle_error)
        self.worker.start()

# TAB 2: INTERACTIVE BRAIN VIEWER WIDGET
class InteractiveViewerWidget(QWidget):
    def __init__(self, parent_window=None):
        super().__init__()
        self.parent_window = parent_window
        self.csv_path = f"{BASE_DIR}/data/results/cleaned_over_expressed_results.csv"
        self.atlas_path = f"{BASE_DIR}/data/atlases/HarvardOxford_2mm.nii.gz"
        if not os.path.exists(self.atlas_path):
            self.atlas_path = f"{BASE_DIR}/data/atlases/HarvardOxford_1mm.nii.gz"
            
        self.df = None
        self.atlas_img = None
        self.atlas_data = None
        self.cell_types = []
        
        self.load_data()
        self.initUI()
        
    def load_data(self):
        try:
            if os.path.exists(self.csv_path):
                self.df = pd.read_csv(self.csv_path)
                if 'Anatomical_Region' not in self.df.columns:
                    self.df['Anatomical_Region'] = "Cluster " + self.df['Cluster_ID'].astype(str)
                
                # Convert string booleans to actual booleans just in case
                for col in ['Significant_FDR (q<0.05)', 'Significant_FDR', 'Significant (p<0.05)', 'Significant_Raw']:
                    if col in self.df.columns and self.df[col].dtype == object:
                        self.df[col] = self.df[col].astype(str).str.lower() == 'true'
                
                if 'Significant_FDR (q<0.05)' in self.df.columns and self.df['Significant_FDR (q<0.05)'].any():
                    self.df = self.df[self.df['Significant_FDR (q<0.05)'] == True]
                elif 'Significant_FDR' in self.df.columns and self.df['Significant_FDR'].any():
                    self.df = self.df[self.df['Significant_FDR'] == True]
                elif 'Significant (p<0.05)' in self.df.columns:
                    self.df = self.df[self.df['Significant (p<0.05)'] == True]
                elif 'Significant_Raw' in self.df.columns:
                    self.df = self.df[self.df['Significant_Raw'] == True]
                    
                self.cell_types = sorted(self.df['Cell_Type'].unique().tolist())
            if os.path.exists(self.atlas_path):
                self.atlas_img = nib.load(self.atlas_path)
                self.atlas_data = self.atlas_img.get_fdata()
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def load_custom_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Enrichment Results CSV", BASE_DIR, 
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.csv_path = file_name
            self.load_data()
            if hasattr(self, 'web_view'):
                self.web_view.setHtml("<html><body style='background-color:black;'><h2 style='color:white; text-align:center; margin-top:20%; font-family:Arial;'>Load a cell type to view 3D brain map</h2></body></html>")
            
            # Update the UI
            self.combo_cell.clear()
            if self.cell_types:
                self.combo_cell.addItems(self.cell_types)
                self.update_info()
                if hasattr(self, 'lbl_csv_status'):
                    self.lbl_csv_status.setText(f"Loaded: {os.path.basename(file_name)}")
            else:
                if hasattr(self, 'lbl_csv_status'):
                    self.lbl_csv_status.setText("No significant cell types found in CSV.")
                    self.lbl_count.setText("Regions: 0")
                if hasattr(self, 'list_regions'):
                    self.list_regions.clear()
            
    def initUI(self):
        self.cell_type_data = None
        main_layout = QHBoxLayout(self)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        
        # Add back button
        btn_back = QPushButton("← Back to Main Menu")
        btn_back.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        if self.parent_window:
            btn_back.clicked.connect(self.parent_window.show_main_menu)
        left_layout.addWidget(btn_back)
        
        data_group = QGroupBox("1. Data Source")
        data_layout = QVBoxLayout()
        btn_load_csv = QPushButton("Load Custom Results CSV")
        btn_load_csv.clicked.connect(self.load_custom_csv)
        data_layout.addWidget(btn_load_csv)
        
        self.lbl_csv_status = QLabel(f"Loaded Default Master CSV")
        self.lbl_csv_status.setWordWrap(True)
        self.lbl_csv_status.setStyleSheet("color: gray; font-style: italic;")
        data_layout.addWidget(self.lbl_csv_status)
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)
        
        cell_group = QGroupBox("2. Select Cell Type")
        cell_layout = QVBoxLayout()
        self.combo_cell = QComboBox()
        if self.cell_types:
            self.combo_cell.addItems(self.cell_types)
        self.combo_cell.currentIndexChanged.connect(self.update_info)
        cell_layout.addWidget(self.combo_cell)
        
        self.btn_plot = QPushButton("Generate 3D Brain Map")
        self.btn_plot.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 40px;")
        self.btn_plot.clicked.connect(self.plot_brain)
        cell_layout.addWidget(self.btn_plot)
        cell_group.setLayout(cell_layout)
        left_layout.addWidget(cell_group)
        
        info_group = QGroupBox("3. Significant Regions")
        info_layout = QVBoxLayout()
        self.lbl_count = QLabel("Regions: 0")
        self.lbl_count.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(self.lbl_count)
        
        self.list_regions = QListWidget()
        self.list_regions.itemClicked.connect(self.plot_specific_region)
        info_layout.addWidget(self.list_regions)
        info_group.setLayout(info_layout)
        left_layout.addWidget(info_group)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.web_view = QWebEngineView()
        right_layout.addWidget(self.web_view)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        if self.cell_types:
            self.update_info()
            
    def update_info(self):
        if self.df is None: return
        cell_type = self.combo_cell.currentText()
        df_cell = self.df[self.df['Cell_Type'] == cell_type]
        
        self.lbl_count.setText(f"Significantly Altered in: {len(df_cell)} regions")
        self.list_regions.clear()
        
        for _, row in df_cell.iterrows():
            region_name = row['Anatomical_Region']
            mean_exp = row['Mean_Expression']
            cluster_id = row['Cluster_ID']
            direction = "Over" if mean_exp > 0 else "Under"
            item = f"[{direction}] {region_name} (Z: {mean_exp:.3f})"
            from PyQt5.QtWidgets import QListWidgetItem
            list_item = QListWidgetItem(item)
            list_item.setData(Qt.UserRole, cluster_id)
            # Add some color to make it visually clear
            if mean_exp > 0:
                list_item.setForeground(Qt.red)
            else:
                list_item.setForeground(Qt.blue)
            self.list_regions.addItem(list_item)
            
    def plot_specific_region(self, item):
        if self.df is None or self.atlas_img is None: return
        cluster_id = item.data(Qt.UserRole)
        full_text = item.text()
        region_name = full_text.split("] ")[1].split(" (Z:")[0]

        if len(region_name) > 25:
            display_name = region_name[:22] + "..."
        else:
            display_name = region_name
            
        cell_type = self.combo_cell.currentText()
        
        highlight_data = np.zeros_like(self.atlas_data)
        df_cell = self.df[self.df['Cell_Type'] == cell_type]
        row = df_cell[df_cell['Cluster_ID'] == cluster_id]
        z_score = 1.0
        if not row.empty:
            z_score = row.iloc[0]['Mean_Expression']
            
        highlight_data[self.atlas_data == cluster_id] = z_score
        highlight_img = nib.Nifti1Image(highlight_data, self.atlas_img.affine, self.atlas_img.header)
        
        vmax = df_cell['Mean_Expression'].max()
        vmin = df_cell['Mean_Expression'].min()
        cmap = 'Reds' if z_score > 0 else 'Blues'
        
        html_view = plotting.view_img(
            highlight_img, 
            threshold=0.001,
            title=f"{cell_type} | {display_name} | Z={z_score:.2f}",
            cmap=cmap,
            colorbar=True,
            vmax=vmax,
            vmin=vmin,
            symmetric_cmap=False,
            black_bg=False
        )
        
        temp_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_brain.html")
        html_view.save_as_html(temp_html)
        self.web_view.load(QUrl.fromLocalFile(temp_html))
            
    def plot_brain(self):
        if self.df is None or self.atlas_img is None:
            QMessageBox.critical(self, "Error", "Data not loaded correctly.")
            return
            
        cell_type = self.combo_cell.currentText()
        df_cell = self.df[self.df['Cell_Type'] == cell_type]
        
        if len(df_cell) == 0:
            QMessageBox.information(self, "Info", "No significant regions to plot.")
            return
            
        highlight_data = np.zeros_like(self.atlas_data)
        for _, row in df_cell.iterrows():
            cluster_id = row['Cluster_ID']
            z_score = row['Mean_Expression']
            highlight_data[self.atlas_data == cluster_id] = z_score
            
        highlight_img = nib.Nifti1Image(highlight_data, self.atlas_img.affine, self.atlas_img.header)
        
        vmax = df_cell['Mean_Expression'].max()
        vmin = df_cell['Mean_Expression'].min()
        
        html_view = plotting.view_img(
            highlight_img, 
            threshold=0.001,
            title=f"{cell_type} Significantly Altered (Z-Score)",
            cmap='coolwarm',
            colorbar=True,
            vmax=vmax,
            vmin=vmin,
            symmetric_cmap=True,
            black_bg=False
        )
        
        temp_html = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_brain.html")
        html_view.save_as_html(temp_html)
        self.web_view.load(QUrl.fromLocalFile(temp_html))

# TAB 3: 3D NETWORK VIEWER WIDGET (FC)
class NetworkViewerWidget(QWidget):
    def __init__(self, parent_window=None):
        super().__init__()
        self.parent_window = parent_window
        self.base_dir = f"{BASE_DIR}/Functional_connectivity"
        self.diseases = []
        self.resolutions = [str(r) for r in range(100, 1001, 100) if r != 200]
        self.init_data()
        self.initUI()
        
    def init_data(self):
        if not os.path.exists(self.base_dir): return
        all_folders = glob.glob(os.path.join(self.base_dir, "schaefer*-yeo7", "fc_*_schaefer*-yeo7"))
        dis_set = set()
        for f in all_folders:
            match = re.search(r'fc_(.*)_schaefer', os.path.basename(f))
            if match: dis_set.add(match.group(1))
        self.diseases = sorted(list(dis_set))

    def get_schaefer_coords(self, resolution):
        return get_schaefer_coords(resolution)

    def get_node_colors(self, roi_names):
        return get_node_colors(roi_names)

    def initUI(self):
        self.cell_type_data = None
        main_layout = QHBoxLayout(self)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        
        # Add back button
        btn_back = QPushButton("← Back to Main Menu")
        btn_back.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        if self.parent_window:
            btn_back.clicked.connect(self.parent_window.show_main_menu)
        left_layout.addWidget(btn_back)
        
        group = QGroupBox("Network Selection")
        glayout = QVBoxLayout()
        
        self.combo_dis = QComboBox()
        if self.diseases: self.combo_dis.addItems(self.diseases)
        self.combo_dis.currentIndexChanged.connect(self.on_network_param_changed)
        glayout.addWidget(QLabel("1. Select Disease"))
        glayout.addWidget(self.combo_dis)
        
        self.combo_res = QComboBox()
        self.combo_res.addItems(self.resolutions)
        self.combo_res.currentIndexChanged.connect(self.on_network_param_changed)
        glayout.addWidget(QLabel("2. Select Resolution"))
        glayout.addWidget(self.combo_res)
        
        self.combo_dir = QComboBox()
        self.combo_dir.addItems(["Hyper-connected (Positive d)", "Hypo-connected (Negative d)"])
        self.combo_dir.currentIndexChanged.connect(self.on_network_param_changed)
        glayout.addWidget(QLabel("3. Select Direction"))
        glayout.addWidget(self.combo_dir)
        
        self.btn_plot = QPushButton("View 3D Network")
        self.btn_plot.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 40px;")
        self.btn_plot.clicked.connect(self.plot_network)
        glayout.addWidget(self.btn_plot)
        
        group.setLayout(glayout)
        left_layout.addWidget(group)
        
        ct_group = QGroupBox("Cell Type Overlay (Optional)")
        ct_layout = QVBoxLayout()
        self.chk_overlay = QCheckBox("Overlay Cell Type Expression")
        self.chk_overlay.toggled.connect(self.toggle_cell_overlay)
        ct_layout.addWidget(self.chk_overlay)
        
        self.combo_ct = QComboBox()
        self.combo_ct.setEnabled(False)
        ct_layout.addWidget(QLabel("Cell Type:"))
        ct_layout.addWidget(self.combo_ct)
        
        self.combo_ct.currentIndexChanged.connect(self.on_cell_type_changed)
        
        ct_group.setLayout(ct_layout)
        left_layout.addWidget(ct_group)
        
        left_layout.addStretch()
        
        right_panel = QSplitter(Qt.Vertical)
        
        self.web_view = QWebEngineView()
        right_panel.addWidget(self.web_view)
        
        self.web_view_overlay = QWebEngineView()
        self.web_view_overlay.hide() 
        right_panel.addWidget(self.web_view_overlay)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def toggle_cell_overlay(self):
        if self.chk_overlay.isChecked():
            self.web_view_overlay.show()
            res = self.combo_res.currentText()
            ct_file = rf"{BASE_DIR}/cell_type_csv/cell_types_{res}_7net.csv"
            
            self.combo_ct.blockSignals(True)
            self.combo_ct.setEnabled(True)
            self.combo_ct.clear()
            
            if os.path.exists(ct_file):
                try:
                    df_ct = pd.read_csv(ct_file)
                    drop_cols = [c for c in df_ct.columns if c.lower() in ['region', 'numdonorsineachparcel', 'maxdonorpresenceinparcel']]
                    
                    results_file = f"{BASE_DIR}/data/results/Filtered_PValue_0.05_CellType_Enrichment.csv"
                    if os.path.exists(results_file):
                        results_df = pd.read_csv(results_file)
                        dis = self.combo_dis.currentText()
                        dir_str = "Positive" if "Positive" in self.combo_dir.currentText() else "Negative"
                        
                        sig_cells = results_df[
                            (results_df['Disease'] == dis) &
                            (results_df['Resolution'] == int(res)) &
                            (results_df['Direction'] == dir_str)
                        ]['CellType'].tolist()
                        
                        cell_types = [c for c in df_ct.columns if c not in drop_cols and c in sig_cells]
                        if not cell_types:
                            cell_types = [c for c in df_ct.columns if c not in drop_cols]
                    else:
                        cell_types = [c for c in df_ct.columns if c not in drop_cols]
                        
                    if cell_types:
                        self.combo_ct.addItems(cell_types)
                        self.cell_type_data = df_ct
                    else:
                        self.combo_ct.addItem("No valid columns found")
                        self.cell_type_data = None
                except Exception as e:
                    self.combo_ct.addItem(f"Error reading CSV: {e}")
                    self.cell_type_data = None
            else:
                self.combo_ct.addItem(f"File not found: cell_types_{res}_7net.csv")
                self.cell_type_data = None
                
            self.combo_ct.blockSignals(False)
            
            if self.cell_type_data is not None and self.combo_ct.count() > 0:
                self.plot_network()
        else:
            self.web_view_overlay.hide()
            self.combo_ct.blockSignals(True)
            self.combo_ct.setEnabled(False)
            self.cell_type_data = None
            self.combo_ct.blockSignals(False)
            
            self.plot_network()

    def on_cell_type_changed(self, index):
        if self.chk_overlay.isChecked() and self.cell_type_data is not None and index >= 0:
            self.plot_network()

    def on_network_param_changed(self):
        if self.chk_overlay.isChecked():
            self.toggle_cell_overlay()
        else:
            self.plot_network()

    def plot_network(self):
        res = self.combo_res.currentText()
        dis = self.combo_dis.currentText()
        direction = "pos" if "Positive" in self.combo_dir.currentText() else "neg"
            
        parent = f"schaefer{res}-yeo7"
        folder = f"fc_{dis}_schaefer{res}-yeo7"
        pattern = os.path.join(self.base_dir, parent, folder, f"*_binarized_{direction}.csv")
        files = glob.glob(pattern)
        
        if not files:
            QMessageBox.warning(self, "Error", "No binarized network file found for these settings.")
            return
        fc_file = files[0]
            
        try:
            df = pd.read_csv(fc_file, index_col=0)
            matrix = df.values
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read FC Matrix: {e}")
            return
            
        matrix = np.nan_to_num(matrix, 0)
        matrix = np.maximum(matrix, matrix.T)
        np.fill_diagonal(matrix, 0)
        
        edge_count = int(np.sum(matrix) / 2)
        if edge_count == 0:
            QMessageBox.information(self, "Info", "No significant connections in this network.")
            return
            
        coords, roi_names = self.get_schaefer_coords(res)
        if coords is None:
            QMessageBox.warning(self, "Error", "Could not fetch coordinates.")
            return
            
        if coords.shape[0] != matrix.shape[0]:
            QMessageBox.warning(self, "Error", f"Mismatch: Coords {coords.shape[0]} vs Matrix {matrix.shape[0]}")
            return
            
        node_colors = self.get_node_colors(roi_names)
        node_sizes = 4
        edge_cmap = 'Reds' if direction == 'pos' else ('Blues' if direction == 'neg' else 'Purples')
        title = f"{dis.upper()} (Res {res}) | Edges: {edge_count}"
        
        html_view_standard = plotting.view_connectome(
            matrix, 
            coords, 
            edge_threshold="0.1%", 
            node_size=4,
            linewidth=1.5,
            node_color=node_colors,
            edge_cmap=edge_cmap,
            colorbar=False,
            title=title
        )
        temp_html_std = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_network_std.html")
        html_view_standard.save_as_html(temp_html_std)
        self.web_view.load(QUrl.fromLocalFile(temp_html_std))
        
        if self.chk_overlay.isChecked() and self.cell_type_data is not None:
            sel_ct = self.combo_ct.currentText()
            if sel_ct in self.cell_type_data.columns:
                expr = self.cell_type_data[sel_ct].values
                if len(expr) == len(coords):
                    norm_expr = (expr - expr.min()) / (expr.max() - expr.min() + 1e-8)
                    node_sizes_overlay = 2 + (norm_expr * 13)

                    import matplotlib.cm as cm
                    if direction == 'pos':
                        node_cmap = cm.get_cmap('Purples')
                    elif direction == 'neg':
                        node_cmap = cm.get_cmap('Greens')
                    else:
                        node_cmap = cm.get_cmap('Oranges')
                    
                    node_colors_overlay = [mcolors.to_hex(node_cmap(0.3 + (0.7 * val))) for val in norm_expr]
                    
                    title_overlay = f"{dis.upper()} (Res {res}) | {sel_ct} Overlay | Larger/Darker = Higher"
                else:
                    QMessageBox.warning(self, "Warning", "Cell type data length doesn't match matrix size. Using default sizes.")
                    node_sizes_overlay = 4
                    node_colors_overlay = node_colors
                    title_overlay = title
                    
                html_view_overlay = plotting.view_connectome(
                    matrix, 
                    coords, 
                    edge_threshold="0.1%", 
                    node_size=node_sizes_overlay,
                    linewidth=1.5,
                    node_color=node_colors_overlay,
                    edge_cmap=edge_cmap,
                    colorbar=False,
                    title=title_overlay
                )
                
                temp_html_overlay = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_network_overlay.html")
                html_view_overlay.save_as_html(temp_html_overlay)
                self.web_view_overlay.load(QUrl.fromLocalFile(temp_html_overlay))


# WORKER THREAD FOR CUSTOM FC ENRICHMENT ANALYSIS
class FCEnrichmentWorker(QThread):
    progress = pyqtSignal(int)
    log      = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame)
    error    = pyqtSignal(str)

    def __init__(self, fc_path, resolution, n_perms, direction_hint, apply_fdr=True, perm_method="Topological Edge-Swap", custom_parc_path=None, apply_threshold=False, threshold_method="Top % (Positive Edges)", threshold_val=10.0):
        super().__init__()
        self.fc_path        = fc_path
        self.resolution     = resolution
        self.n_perms        = n_perms
        self.direction_hint = direction_hint
        self.apply_fdr      = apply_fdr
        self.perm_method    = perm_method
        self.custom_parc_path = custom_parc_path
        self.apply_threshold = apply_threshold
        self.threshold_method = threshold_method
        self.threshold_val  = threshold_val
        self.fc_matrix      = None    
        self.ct_data        = None 
        self.resolved_direction = "pos"

    def run(self):
        if not _FC_ENRICH_AVAILABLE:
            self.error.emit(f"Cannot import FC enrichment functions:\n{_FC_ENRICH_ERROR}")
            return

        # ── Resolve direction ──────────────────────────────────────────────
        if self.direction_hint == "Auto-detect":
            fname = os.path.basename(self.fc_path).lower()
            self.resolved_direction = "neg" if "_neg" in fname else "pos"
        elif self.direction_hint == "Negative":
            self.resolved_direction = "neg"
        else:
            self.resolved_direction = "pos"

        # ── Load data ─────────────────────────────────────────────────────
        if self.custom_parc_path:
            self.log.emit("Loading custom parcellation and extracting cell-type data...")
            self.progress.emit(2)
            
            if not (self.custom_parc_path.lower().endswith('.nii') or self.custom_parc_path.lower().endswith('.nii.gz')):
                self.error.emit("Custom parcellation must be a NIfTI (.nii or .nii.gz) file to match the 3D AHBA maps.")
                return
            
            try:
                # 1. Load FC Matrix
                fc_df = pd.read_csv(self.fc_path, index_col=0)
                fc_matrix = fc_df.values
                fc_matrix = np.nan_to_num(fc_matrix, 0)
                fc_matrix = np.maximum(fc_matrix, fc_matrix.T)
                np.fill_diagonal(fc_matrix, 0)
                expected_n = fc_matrix.shape[0]
                if fc_matrix.shape[0] != fc_matrix.shape[1]:
                    self.error.emit("FC matrix is not square.")
                    return
                self.fc_matrix = fc_matrix

                # 2. Load Custom Parcellation NIfTI
                parc_img = nib.load(self.custom_parc_path)
                parc_data_3d = parc_img.get_fdata()
                parc_data = parc_data_3d.flatten()
                
                unique_clusters = np.unique(parc_data)
                unique_clusters = unique_clusters[(unique_clusters > 1e-6) & (~np.isnan(unique_clusters))]
                unique_clusters = np.sort(unique_clusters)
                
                # Attempt to map FC matrix rows to NIfTI cluster IDs
                fc_indices = fc_df.index.astype(str).tolist()
                self.fc_indices = fc_indices
                target_cluster_ids = []
                
                fs_map = {
                    'bankssts': 1, 'caudalanteriorcingulate': 2, 'caudalmiddlefrontal': 3,
                    'cuneus': 5, 'entorhinal': 6, 'fusiform': 7, 'inferiorparietal': 8,
                    'inferiortemporal': 9, 'isthmuscingulate': 10, 'lateraloccipital': 11,
                    'lateralorbitofrontal': 12, 'lingual': 13, 'medialorbitofrontal': 14,
                    'middletemporal': 15, 'parahippocampal': 16, 'paracentral': 17,
                    'parsopercularis': 18, 'parsorbitalis': 19, 'parstriangularis': 20,
                    'pericalcarine': 21, 'postcentral': 22, 'posteriorcingulate': 23,
                    'precentral': 24, 'precuneus': 25, 'rostralanteriorcingulate': 26,
                    'rostralmiddlefrontal': 27, 'superiorfrontal': 28, 'superiorparietal': 29,
                    'superiortemporal': 30, 'supramarginal': 31, 'frontalpole': 32,
                    'temporalpole': 33, 'transversetemporal': 34, 'insula': 35
                }
                
                is_freesurfer = any('ctx-' in idx.lower() or 'lh-' in idx.lower() or 'rh-' in idx.lower() for idx in fc_indices)
                
                if is_freesurfer:
                    for idx in fc_indices:
                        idx_lower = idx.lower()
                        # Determine hemisphere
                        hemi_offset = 1000 if ('lh' in idx_lower or 'left' in idx_lower) else 2000
                        # Find region name
                        matched_id = None
                        for fs_name, fs_val in fs_map.items():
                            if fs_name in idx_lower:
                                matched_id = hemi_offset + fs_val
                                break
                        if matched_id is not None:
                            target_cluster_ids.append(matched_id)
                        else:
                            target_cluster_ids.append(-1)
                    valid_targets = [cid for cid in target_cluster_ids if cid != -1]
                    missing_targets = [cid for cid in valid_targets if cid not in unique_clusters]
                    if valid_targets and len(missing_targets) == len(valid_targets):
                        self.log.emit("FreeSurfer labels detected, but standard IDs (1001+) not found in NIfTI. Falling back to 1-to-1 mapping.")
                        if len(unique_clusters) == expected_n:
                            target_cluster_ids = unique_clusters.tolist()
                        else:
                            self.error.emit(f"Custom parcellation has {len(unique_clusters)} regions, but FC matrix has {expected_n} regions. Standard FreeSurfer IDs were not found. They must match exactly.")
                            return
                else:
                    try:
                        target_cluster_ids = [int(idx) for idx in fc_indices]
                    except ValueError:
                        if len(unique_clusters) == expected_n:
                            target_cluster_ids = unique_clusters.tolist()
                        else:
                            self.error.emit(f"Custom parcellation has {len(unique_clusters)} regions, but FC matrix has {expected_n} regions. Row names do not match FreeSurfer or integer IDs. They must match exactly.")
                            return
                missing_clusters = [cid for cid in target_cluster_ids if cid not in unique_clusters and cid != -1]
                if missing_clusters:
                    self.log.emit(f"Warning: Some mapped clusters (e.g. {missing_clusters[:5]}) are missing in the NIfTI.")
                
                if len(target_cluster_ids) != expected_n:
                    self.error.emit("Failed to map FC matrix rows to parcellation IDs.")
                    return
                self.target_cluster_ids = target_cluster_ids

                # 3. Extract Cell-Type Values
                cell_types_3d_dir = f"{BASE_DIR}/cell_types_3D"
                ct_files = [f for f in os.listdir(cell_types_3d_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
                if not ct_files:
                    self.error.emit(f"No 3D cell-type maps found in {cell_types_3d_dir}.")
                    return

                ct_data_dict = {}
                for i, ctf in enumerate(ct_files):
                    self.log.emit(f"Extracting {ctf}...")
                    self.progress.emit(2 + int(3 * i / len(ct_files)))
                    ct_name = ctf.split('.')[0].capitalize()
                    ct_path = os.path.join(cell_types_3d_dir, ctf)
                    
                    try:
                        ct_img = nib.load(ct_path)
                        ct_img_data = ct_img.get_fdata()
                        if ct_img_data.shape != parc_data_3d.shape:
                            if ct_img_data.size == parc_data_3d.size:
                                ct_img_data = ct_img_data.reshape(parc_data_3d.shape)
                            else:
                                from nilearn.image import resample_to_img
                                resampled_img = resample_to_img(ct_img, parc_img, interpolation='continuous')
                                ct_img_data = resampled_img.get_fdata()
                                del resampled_img
                                
                        ct_img_data = ct_img_data.flatten()
                                
                        # Calculate mean per cluster
                        means = []
                        for cluster_id in target_cluster_ids:
                            if cluster_id == -1 or cluster_id not in unique_clusters:
                                means.append(np.nan)
                            else:
                                cluster_mask = (parc_data == cluster_id)
                                mean_val = np.mean(ct_img_data[cluster_mask])
                                means.append(mean_val)
                            
                        ct_data_dict[ct_name] = means
                        del ct_img, ct_img_data
                    except Exception as e:
                        self.log.emit(f"Failed to extract {ctf}: {e}")
                        
                if not ct_data_dict:
                    self.error.emit("Failed to extract any cell-type data.")
                    return
                    
                ct_data = pd.DataFrame(ct_data_dict)
                self.ct_data = ct_data
            except Exception as e:
                self.error.emit(f"Error processing custom parcellation:\n{e}")
                return
        else:
            ct_file = rf"{BASE_DIR}/cell_type_csv/cell_types_{self.resolution}_7net.csv"
            if not os.path.exists(ct_file):
                self.error.emit(f"Cell-type CSV not found:\n{ct_file}\nCheck that resolution {self.resolution} is available.")
                return

            self.log.emit("Loading FC matrix and cell-type data...")
            self.progress.emit(2)
            try:
                fc_matrix, ct_data = _fc_load_data(self.fc_path, ct_file)
            except Exception as e:
                self.error.emit(f"Failed to load data:\n{e}")
                return

            if np.sum(fc_matrix) == 0:
                self.error.emit("FC matrix has no edges after loading. Check the file format.")
                return

            expected_n = int(self.resolution)
            if fc_matrix.shape[0] != expected_n or fc_matrix.shape[1] != expected_n:
                self.error.emit(
                    f"FC matrix shape {fc_matrix.shape} does not match resolution {self.resolution} "
                    f"(expected {expected_n}×{expected_n}).\nPlease select the correct resolution."
                )
                return

            self.fc_matrix = fc_matrix
            self.ct_data   = ct_data
        if self.apply_threshold:
            self.log.emit(f"Thresholding matrix: {self.threshold_method} ({self.threshold_val}%)...")
            # Flatten upper triangle
            upper_tri_indices = np.triu_indices_from(self.fc_matrix, k=1)
            edge_vals = self.fc_matrix[upper_tri_indices]
            n_possible_edges = len(edge_vals)
            
            n_keep = max(1, int(np.round((self.threshold_val / 100.0) * n_possible_edges)))
            if "Positive" in self.threshold_method:
                vals_to_sort = edge_vals
            elif "Negative" in self.threshold_method:
                vals_to_sort = -edge_vals
            elif "Absolute" in self.threshold_method:
                vals_to_sort = np.abs(edge_vals)
            else:
                vals_to_sort = edge_vals
                
            if n_keep >= n_possible_edges:
                cutoff = -np.inf 
            else:
                sorted_vals = np.sort(vals_to_sort)[::-1] 
                cutoff = sorted_vals[n_keep - 1]
                
            # Apply threshold
            mask = vals_to_sort >= cutoff
            
            # Create new binarized matrix
            new_fc = np.zeros_like(self.fc_matrix)
            new_fc[upper_tri_indices[0][mask], upper_tri_indices[1][mask]] = 1
            new_fc = np.maximum(new_fc, new_fc.T) 
            self.fc_matrix = new_fc
            fc_matrix = self.fc_matrix 
            self.log.emit(f"Matrix thresholded. Retained {int(np.sum(fc_matrix)/2)} edges.")
        else:
            unique_vals = np.unique(self.fc_matrix)
            is_binary = np.all(np.isin(unique_vals, [0, 1]))
            if not is_binary:
                self.error.emit(
                    "The input FC matrix contains non-binary values (e.g., floats), "
                    "but 'Apply Thresholding' was NOT checked.\n"
                    "The permutation test requires a binarized matrix of 1s and 0s. "
                    "Please check the 'Apply Thresholding' box and try again."
                )
                return
        V = (~ct_data.isna()).astype(float).values
        X = ct_data.fillna(0.0).values
        cell_types = ct_data.columns.tolist()
        self.log.emit("Computing real co-expression scores...")
        self.progress.emit(5)
        real_scores = calculate_coexpression_fast(fc_matrix, X, V)
        np.random.seed(42)
        
        n_ct = X.shape[1]
        null_scores = np.full((self.n_perms, n_ct), np.nan)

        if self.perm_method == "Alexander-Bloch Spherical Spin Test":
            if not _SPIN_AVAILABLE:
                self.error.emit(f"Spin test functions unavailable:\n{_SPIN_ERROR}")
                return
                
            self.log.emit("Generating spherical spin permutations...")
            self.progress.emit(8)
            try:
                # We need coords for the current resolution
                coord_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schaefer_coords")
                os.makedirs(coord_dir, exist_ok=True)
                coord_file = os.path.join(coord_dir, f"Schaefer2018_{self.resolution}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv")
                
                if not os.path.exists(coord_file):
                    # Try to download it if missing
                    self.log.emit(f"Downloading coordinates for resolution {self.resolution}...")
                    url = f"https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_{self.resolution}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
                    try:
                        import requests
                        r = requests.get(url, verify=False)
                        if r.status_code == 200:
                            with open(coord_file, 'wb') as f:
                                f.write(r.content)
                        else:
                            self.error.emit(f"Failed to download coordinate file for spin test (status {r.status_code}):\n{url}")
                            return
                    except Exception as e:
                        self.error.emit(f"Coordinate file missing and download failed:\n{e}")
                        return
                    
                df_coords = pd.read_csv(coord_file)
                coords = df_coords[['R', 'A', 'S']].values
                roi_names = df_coords['ROI Name'].astype(str).values
                hemi_mask = np.array(['_LH_' in name for name in roi_names])
                
                spin_perms = generate_spin_permutations(coords, hemi_mask, self.n_perms)
                
                report_every = max(1, self.n_perms // 20)
                for i in range(self.n_perms):
                    perm_idx = spin_perms[i]
                    X_spun = X[perm_idx]
                    V_spun = V[perm_idx]
                    null_scores[i] = calculate_coexpression_fast(fc_matrix, X_spun, V_spun)
                    if i % report_every == 0:
                        pct = 8 + int(85 * i / self.n_perms)
                        self.progress.emit(pct)
                        self.log.emit(f"Spin Permutation {i}/{self.n_perms}...")
            except Exception as e:
                self.error.emit(f"Spin test failed: {e}")
                return

        elif self.perm_method == "Random Permutation (Edge Shuffle)":
            self.log.emit("Running completely random edge shuffle...")
            self.progress.emit(8)
            edges = np.argwhere(np.triu(fc_matrix, 1) > 0)
            n_edges = len(edges)
            N = fc_matrix.shape[0]
            
            report_every = max(1, self.n_perms // 20)
            for i in range(self.n_perms):
                rand_mat = np.zeros_like(fc_matrix)
                upper_indices = np.triu_indices(N, 1)
                idx_choices = np.random.choice(len(upper_indices[0]), size=n_edges, replace=False)
                rand_mat[upper_indices[0][idx_choices], upper_indices[1][idx_choices]] = 1
                rand_mat = rand_mat + rand_mat.T
                
                null_scores[i] = calculate_coexpression_fast(rand_mat, X, V)
                if i % report_every == 0:
                    pct = 8 + int(85 * i / self.n_perms)
                    self.progress.emit(pct)
                    self.log.emit(f"Random Permutation {i}/{self.n_perms}...")

        else:
            self.log.emit("Compiling JIT kernel (first run may take ~10 s)...")
            self.progress.emit(8)
            current_rand = fast_double_edge_swap(fc_matrix, nswap_multiplier=5)
            null_scores[0] = calculate_coexpression_fast(current_rand, X, V)
    
            report_every = max(1, self.n_perms // 20)
            for i in range(1, self.n_perms):
                current_rand = fast_double_edge_swap(current_rand, nswap_multiplier=1)
                null_scores[i] = calculate_coexpression_fast(current_rand, X, V)
                if i % report_every == 0:
                    pct = 8 + int(85 * i / self.n_perms)
                    self.progress.emit(pct)
                    self.log.emit(f"Edge-Swap Permutation {i}/{self.n_perms}...")

        fdr_label = "p-values + BH-FDR" if self.apply_fdr else "p-values (no FDR)"
        self.log.emit(f"Computing {fdr_label}...")
        self.progress.emit(94)

        p_values   = []
        null_means = []
        null_stds  = []
        for i in range(n_ct):
            valid = null_scores[:, i][~np.isnan(null_scores[:, i])]
            if len(valid) == 0 or np.isnan(real_scores[i]):
                p_values.append(np.nan)
                null_means.append(np.nan)
                null_stds.append(np.nan)
            else:
                p_values.append(float(np.sum(valid >= real_scores[i]) / len(valid)))
                null_means.append(float(np.mean(valid)))
                null_stds.append(float(np.std(valid)))

        p_arr = np.array(p_values)
        fdr_q   = np.full(n_ct, np.nan)
        sig_fdr = np.zeros(n_ct, dtype=bool)
        if self.apply_fdr:
            valid_m = ~np.isnan(p_arr)
            if valid_m.sum() > 0:
                _, fdr_q[valid_m] = fdrcorrection(p_arr[valid_m], alpha=0.05)
            sig_fdr = (~np.isnan(fdr_q)) & (fdr_q < 0.05)
        sig_raw = (~np.isnan(p_arr)) & (p_arr < 0.05)

        def _fmt(v):
            if np.isnan(v): return "N/A"
            return "< 0.001" if v <= 0.0 else f"{v:.4f}"

        df = pd.DataFrame({
            "CellType":        cell_types,
            "Real_Score":      real_scores,
            "Null_Mean":       null_means,
            "Null_Std":        null_stds,
            "P_Value":         p_arr,
            "P_Value_Display": [_fmt(v) for v in p_arr],
            "FDR_q":           fdr_q,
            "FDR_Display":     [_fmt(v) for v in fdr_q],
            "Significant_FDR": sig_fdr,
            "Significant_Raw": sig_raw,
            "FDR_Applied":     self.apply_fdr,
        })

        # ── Save results ──────────────────────────────────────────────────
        results_dir = f"{BASE_DIR}/data/results"
        os.makedirs(results_dir, exist_ok=True)
        out_csv = os.path.join(results_dir, "custom_fc_enrichment_results.csv")
        try:
            df.to_csv(out_csv, index=False)
        except Exception:
            pass

        self.progress.emit(100)
        if self.apply_fdr:
            self.log.emit(
                f"Done!  {int(sig_fdr.sum())} significant at FDR q<0.05  |  "
                f"{int(sig_raw.sum())} at raw p<0.05  |  saved to {out_csv}"
            )
        else:
            self.log.emit(
                f"Done!  {int(sig_raw.sum())} significant at raw p<0.05  "
                f"(FDR not applied)  |  saved to {out_csv}"
            )
        self.finished.emit(df)

# CUSTOM FC ENRICHMENT & VIEWER WIDGET
class FCEnrichmentViewerWidget(QWidget):
    def __init__(self, parent_window=None):
        super().__init__()
        self.parent_window = parent_window
        self.fc_path    = ""
        self.results_df = None
        self.fc_matrix  = None
        self.worker     = None
        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── LEFT PANEL ────────────────────────────────────────────────────
        left_panel  = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)
        left_panel.setFixedWidth(400)

        # Back button
        btn_back = QPushButton("← Back to Main Menu")
        btn_back.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        if self.parent_window:
            btn_back.clicked.connect(self.parent_window.show_main_menu)
        left_layout.addWidget(btn_back)

        # ── STEP 1: Load FC Matrix ────────────────────────────────────────
        grp_step1 = QGroupBox("STEP 1 — Load Your FC Matrix")
        grp_step1.setStyleSheet("QGroupBox { font-weight: bold; color: #1565C0; }")
        step1_lay = QVBoxLayout(grp_step1)

        lbl_step1_desc = QLabel(
            "Load a binarized functional connectivity adjacency matrix (CSV).\n"
            "Rows and columns = parcels; values = 0 or 1 (significant edges)."
        )
        lbl_step1_desc.setWordWrap(True)
        lbl_step1_desc.setStyleSheet("color: #555; font-size: 10px; font-weight: normal;")
        step1_lay.addWidget(lbl_step1_desc)

        self.btn_browse = QPushButton("Browse for FC Matrix CSV...")
        self.btn_browse.setStyleSheet("background-color: #1976D2; color: white; font-weight: bold;")
        self.btn_browse.clicked.connect(self.browse_fc_file)
        step1_lay.addWidget(self.btn_browse)

        self.lbl_fc_file = QLabel("No file selected")
        self.lbl_fc_file.setWordWrap(True)
        self.lbl_fc_file.setStyleSheet("color: #666; font-style: italic; font-weight: normal;")
        step1_lay.addWidget(self.lbl_fc_file)

        # ── Matrix Thresholding Options ──
        self.chk_threshold = QCheckBox("Threshold & Binarize Matrix (Recommended for raw FC/Cohen's d)")
        self.chk_threshold.setChecked(True)
        self.chk_threshold.setStyleSheet("font-weight: normal; color: #d32f2f;")
        step1_lay.addWidget(self.chk_threshold)
        
        self.thresh_widget = QWidget()
        thresh_lay = QFormLayout(self.thresh_widget)
        thresh_lay.setContentsMargins(15, 0, 0, 0)
        thresh_lay.setVerticalSpacing(4)
        
        self.combo_thresh_method = QComboBox()
        self.combo_thresh_method.addItems(["Top % (Positive Edges)", "Top % (Negative Edges)", "Top % (Absolute Edges)"])
        lbl_tm = QLabel("Method:")
        lbl_tm.setStyleSheet("font-weight: normal; font-size: 10px;")
        thresh_lay.addRow(lbl_tm, self.combo_thresh_method)
        
        self.spin_thresh_val = QDoubleSpinBox()
        self.spin_thresh_val.setRange(0.1, 100.0)
        self.spin_thresh_val.setValue(10.0)
        self.spin_thresh_val.setSuffix("%")
        lbl_tv = QLabel("Threshold:")
        lbl_tv.setStyleSheet("font-weight: normal; font-size: 10px;")
        thresh_lay.addRow(lbl_tv, self.spin_thresh_val)
        
        self.chk_threshold.toggled.connect(self.thresh_widget.setVisible)
        step1_lay.addWidget(self.thresh_widget)

        left_layout.addWidget(grp_step1)

        # ── STEP 2: Configure & Run Enrichment ───────────────────────────
        grp_step2 = QGroupBox("STEP 2 — Configure & Run Enrichment Analysis")
        grp_step2.setStyleSheet("QGroupBox { font-weight: bold; color: #2E7D32; }")
        step2_lay = QVBoxLayout(grp_step2)

        lbl_step2_desc = QLabel(
            "The co-expression score measures how much connected parcels share "
            "cell-type abundance. A permutation null (edge-swap) tests whether "
            "the score is higher than chance."
        )
        lbl_step2_desc.setWordWrap(True)
        lbl_step2_desc.setStyleSheet("color: #555; font-size: 10px; font-weight: normal;")
        step2_lay.addWidget(lbl_step2_desc)

        form = QFormLayout()
        form.setVerticalSpacing(4)

        self.combo_res = QComboBox()
        self.combo_res.addItems(["100","200","300","400","500","600","700","800","900","1000", "Custom Parcellation..."])
        lbl_res = QLabel("Parcellation / Resolution\n(must match FC matrix size):")
        lbl_res.setStyleSheet("font-weight: normal; font-size: 10px;")
        form.addRow(lbl_res, self.combo_res)

        self.btn_custom_parc = QPushButton("Browse Parcellation...")
        self.btn_custom_parc.setStyleSheet("background-color: #7B1FA2; color: white;")
        self.btn_custom_parc.clicked.connect(self.browse_custom_parc)
        self.lbl_custom_parc = QLabel("No parcellation selected")
        self.lbl_custom_parc.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
        self.custom_parc_path = ""
        
        self.btn_custom_parc.setVisible(False)
        self.lbl_custom_parc.setVisible(False)
        
        form.addRow(self.btn_custom_parc, self.lbl_custom_parc)
        
        self.combo_res.currentIndexChanged.connect(self.on_res_changed)

        self.combo_dir = QComboBox()
        self.combo_dir.addItems(["Positive (hyper-connected)", "Negative (hypo-connected)", "Auto-detect from filename"])
        self.combo_dir.setCurrentIndex(2)
        lbl_dir = QLabel("FC direction:")
        lbl_dir.setStyleSheet("font-weight: normal;")
        form.addRow(lbl_dir, self.combo_dir)
        
        self.combo_perm = QComboBox()
        self.combo_perm.addItems([
            "Topological Edge-Swap", 
            "Random Permutation (Edge Shuffle)",
            "Alexander-Bloch Spherical Spin Test"
        ])
        lbl_perm_method = QLabel("Permutation method:")
        lbl_perm_method.setStyleSheet("font-weight: normal; font-size: 10px;")
        form.addRow(lbl_perm_method, self.combo_perm)

        self.spin_perms = QSpinBox()
        self.spin_perms.setRange(100, 10000)
        self.spin_perms.setSingleStep(100)
        self.spin_perms.setValue(1000)
        lbl_perm = QLabel("Permutations\n(higher = slower but more precise):")
        lbl_perm.setStyleSheet("font-weight: normal; font-size: 10px;")
        form.addRow(lbl_perm, self.spin_perms)

        step2_lay.addLayout(form)

        # FDR checkbox
        self.chk_fdr = QCheckBox("Apply Benjamini-Hochberg FDR correction  (recommended)")
        self.chk_fdr.setChecked(True)
        self.chk_fdr.setStyleSheet("font-weight: normal; font-size: 10px;")
        lbl_fdr_note = QLabel(
            "FDR controls the false-discovery rate across all cell types.\n"
            "Uncheck to report only raw permutation p-values."
        )
        lbl_fdr_note.setWordWrap(True)
        lbl_fdr_note.setStyleSheet("color: #777; font-size: 9px; font-weight: normal;")
        step2_lay.addWidget(self.chk_fdr)
        step2_lay.addWidget(lbl_fdr_note)

        self.btn_run = QPushButton("▶  Run Enrichment Analysis")
        self.btn_run.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_run.setFixedHeight(46)
        self.btn_run.setStyleSheet("background-color: #388E3C; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.run_enrichment)
        step2_lay.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        step2_lay.addWidget(self.progress_bar)

        self.lbl_log = QLabel("Ready — complete Step 1 and click Run.")
        self.lbl_log.setWordWrap(True)
        self.lbl_log.setStyleSheet("color: #444; font-size: 10px; font-weight: normal;")
        step2_lay.addWidget(self.lbl_log)

        left_layout.addWidget(grp_step2)

        # ── STEP 3: View in 3D Network ────────────────────────────────────
        grp_step3 = QGroupBox("STEP 3 — View Results in 3D FC Network")
        grp_step3.setStyleSheet("QGroupBox { font-weight: bold; color: #00695C; }")
        step3_lay = QVBoxLayout(grp_step3)

        lbl_step3_desc = QLabel(
            "After the analysis finishes, view the FC network in 3D.\n"
            "Optionally overlay a cell type to colour and size nodes by "
            "their expression level — darker / larger = higher abundance."
        )
        lbl_step3_desc.setWordWrap(True)
        lbl_step3_desc.setStyleSheet("color: #555; font-size: 10px; font-weight: normal;")
        step3_lay.addWidget(lbl_step3_desc)

        self.chk_overlay = QCheckBox("Show Cell Type Overlay on nodes")
        self.chk_overlay.setStyleSheet("font-weight: normal;")
        self.chk_overlay.toggled.connect(self.on_overlay_toggled)
        step3_lay.addWidget(self.chk_overlay)

        lbl_ct = QLabel("Select cell type  (bold = FDR/p significant):")
        lbl_ct.setStyleSheet("font-weight: normal; font-size: 10px;")
        step3_lay.addWidget(lbl_ct)

        self.combo_ct = QComboBox()
        self.combo_ct.setEnabled(False)
        self.combo_ct.currentIndexChanged.connect(self.on_ct_changed)
        step3_lay.addWidget(self.combo_ct)

        self.btn_view3d = QPushButton("🧠  View in 3D Network")
        self.btn_view3d.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_view3d.setFixedHeight(46)
        self.btn_view3d.setStyleSheet("background-color: #00897B; color: white; font-weight: bold;")
        self.btn_view3d.setEnabled(False)
        self.btn_view3d.clicked.connect(self.view_in_3d)
        step3_lay.addWidget(self.btn_view3d)

        # Add Export CSV button
        self.btn_export = QPushButton("💾 Export Results to CSV")
        self.btn_export.setFont(QFont("Arial", 10, QFont.Bold))
        self.btn_export.setFixedHeight(36)
        self.btn_export.setStyleSheet("background-color: #555555; color: white;")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)
        step3_lay.addWidget(self.btn_export)

        left_layout.addWidget(grp_step3)
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        right_splitter = QSplitter(Qt.Vertical)

        # Results table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Cell Type", "Real Score", "Null Mean", "P-Value", "FDR q", "Significant"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        right_splitter.addWidget(self.table)

        # Viewer area (standard + overlay side-by-side)
        self.viewer_splitter = QSplitter(Qt.Horizontal)
        self.web_view_std     = QWebEngineView()
        self.web_view_overlay = QWebEngineView()
        self.web_view_std.setHtml("<body style='background:#1a1a2e;color:#aaa;font-family:Arial'>"
                                  "<p style='margin-top:80px;text-align:center;font-size:14px'>"
                                  "Run the analysis, then click<br><b>View in 3D Network</b></p></body>")
        self.web_view_overlay.setHtml("<body style='background:#1a1a2e;color:#aaa;font-family:Arial'>"
                                      "<p style='margin-top:80px;text-align:center;font-size:14px'>"
                                      "Enable the overlay to see cell-type coloured nodes.</p></body>")
        self.web_view_overlay.hide()
        self.viewer_splitter.addWidget(self.web_view_std)
        self.viewer_splitter.addWidget(self.web_view_overlay)
        right_splitter.addWidget(self.viewer_splitter)

        right_splitter.setSizes([250, 550])
        main_layout.addWidget(right_splitter, stretch=1)

    # ── Slots ──────────────────────────────────────────────────────────────

    def browse_fc_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select FC Matrix CSV", BASE_DIR,
            "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self.fc_path = path
            self.lbl_fc_file.setText(os.path.basename(path))

    def on_res_changed(self, index):
        is_custom = (self.combo_res.currentText() == "Custom Parcellation...")
        self.btn_custom_parc.setVisible(is_custom)
        self.lbl_custom_parc.setVisible(is_custom)
        
        if is_custom:
            for i in range(self.combo_perm.count()):
                if "Spin Test" in self.combo_perm.itemText(i):
                    self.combo_perm.model().item(i).setEnabled(False)
            if "Spin Test" in self.combo_perm.currentText():
                self.combo_perm.setCurrentIndex(0)
        else:
            for i in range(self.combo_perm.count()):
                self.combo_perm.model().item(i).setEnabled(True)

    def browse_custom_parc(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Parcellation Map", BASE_DIR,
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        if path:
            self.custom_parc_path = path
            self.lbl_custom_parc.setText(os.path.basename(path))

    def run_enrichment(self):
        if not self.fc_path:
            QMessageBox.warning(self, "No file", "Please browse and select an FC matrix CSV first.")
            return

        is_custom = (self.combo_res.currentText() == "Custom Parcellation...")
        if is_custom:
            if not self.custom_parc_path:
                QMessageBox.warning(self, "No parcellation", "Please browse and select a custom parcellation map.")
                return
        else:
            ct_file = rf"{BASE_DIR}/cell_type_csv/cell_types_{self.combo_res.currentText()}_7net.csv"
            if not os.path.exists(ct_file):
                QMessageBox.warning(self, "Missing cell-type CSV",
                                    f"Cell-type data not found for resolution {self.combo_res.currentText()}:\n{ct_file}")
                return

        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()

        self.btn_run.setEnabled(False)
        self.btn_view3d.setEnabled(False)
        self.combo_ct.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_log.setText("Starting analysis...")

        dir_text = self.combo_dir.currentText()
        if "Positive" in dir_text:
            dir_hint = "Positive"
        elif "Negative" in dir_text:
            dir_hint = "Negative"
        else:
            dir_hint = "Auto-detect"

        self.worker = FCEnrichmentWorker(
            fc_path=self.fc_path,
            resolution=self.combo_res.currentText(),
            n_perms=self.spin_perms.value(),
            direction_hint=dir_hint,
            apply_fdr=self.chk_fdr.isChecked(),
            perm_method=self.combo_perm.currentText(),
            custom_parc_path=self.custom_parc_path if is_custom else None,
            apply_threshold=self.chk_threshold.isChecked(),
            threshold_method=self.combo_thresh_method.currentText(),
            threshold_val=self.spin_thresh_val.value()
        )
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.lbl_log.setText)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_finished(self, df):
        self.results_df = df
        self.fc_matrix  = self.worker.fc_matrix
        self.btn_run.setEnabled(True)

        fdr_applied = bool(df["FDR_Applied"].iloc[0]) if "FDR_Applied" in df.columns else True

        if fdr_applied:
            self.table.setHorizontalHeaderLabels(
                ["Cell Type", "Real Score", "Null Mean", "p-value", "FDR q", "Significant (FDR)"]
            )
        else:
            self.table.setHorizontalHeaderLabels(
                ["Cell Type", "Real Score", "Null Mean", "p-value", "FDR q (N/A)", "Significant (p<0.05)"]
            )

        self.table.setRowCount(len(df))
        green_bg  = QBrush(QColor(200, 255, 200))
        yellow_bg = QBrush(QColor(255, 255, 180))

        for row_i, row in df.iterrows():
            is_sig_fdr = bool(row["Significant_FDR"])
            is_sig_raw = bool(row["Significant_Raw"])
            highlight  = is_sig_fdr if fdr_applied else is_sig_raw
            bg         = green_bg   if fdr_applied else yellow_bg

            fdr_display = str(row["FDR_Display"]) if fdr_applied else "—"
            sig_text    = ("Yes ✓" if is_sig_fdr else "No") if fdr_applied else ("Yes ✓" if is_sig_raw else "No")

            items = [
                QTableWidgetItem(str(row["CellType"])),
                QTableWidgetItem(f"{row['Real_Score']:.4f}" if not np.isnan(row['Real_Score']) else "N/A"),
                QTableWidgetItem(f"{row['Null_Mean']:.4f}"  if not np.isnan(row['Null_Mean'])  else "N/A"),
                QTableWidgetItem(str(row["P_Value_Display"])),
                QTableWidgetItem(fdr_display),
                QTableWidgetItem(sig_text),
            ]
            for col_i, item in enumerate(items):
                item.setTextAlignment(Qt.AlignCenter)
                if highlight:
                    item.setBackground(bg)
                self.table.setItem(row_i, col_i, item)

        sig_col  = "Significant_FDR" if fdr_applied else "Significant_Raw"
        sort_col = "FDR_q"           if fdr_applied else "P_Value"

        sig_df = df[df[sig_col]].sort_values(sort_col)

        self.combo_ct.blockSignals(True)
        self.combo_ct.clear()

        if not sig_df.empty:
            for _, row in sig_df.iterrows():
                self.combo_ct.addItem(row["CellType"])
            for i in range(self.combo_ct.count()):
                model_item = self.combo_ct.model().item(i)
                if model_item:
                    model_item.setFont(QFont("Arial", 9, QFont.Bold))
        else:
            fallback = df.sort_values("P_Value")
            for _, row in fallback.iterrows():
                self.combo_ct.addItem(row["CellType"])
            self.combo_ct.insertItem(0, "— no significant cell types (showing all) —")
            self.combo_ct.setCurrentIndex(0)

        self.combo_ct.blockSignals(False)
        self.combo_ct.setEnabled(True)
        self.btn_view3d.setEnabled(True)
        self.btn_export.setEnabled(True)

    def on_analysis_error(self, msg):
        QMessageBox.critical(self, "Analysis Error", msg)
        self.btn_run.setEnabled(True)
        self.lbl_log.setText("Error — see message box.")
        self.progress_bar.setValue(0)

    def on_overlay_toggled(self, checked):
        if checked:
            self.web_view_overlay.show()
            if self.results_df is not None:
                self._render_overlay_view()
        else:
            self.web_view_overlay.hide()

    def on_ct_changed(self, _index):
        if self.results_df is not None and self.chk_overlay.isChecked():
            self._render_overlay_view()

    def export_csv(self):
        if self.results_df is None:
            return
        
        default_name = "network_enrichment_results.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results as CSV", 
            os.path.join(f"{BASE_DIR}/data/results", default_name),
            "CSV Files (*.csv)"
        )
        if path:
            try:
                self.results_df.to_csv(path, index=False)
                QMessageBox.information(self, "Success", f"Results saved successfully to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save CSV:\n{e}")

    def view_in_3d(self):
        if self.results_df is None or self.fc_matrix is None:
            QMessageBox.warning(self, "No results", "Run the analysis first.")
            return

        res = self.combo_res.currentText()
        if res == "Custom Parcellation...":
            if not getattr(self.worker, "custom_parc_path", None):
                QMessageBox.warning(self, "Error", "No custom parcellation loaded.")
                return
            try:
                import nilearn.plotting as plotting_tools
                all_coords, all_labels = plotting_tools.find_parcellation_cut_coords(
                    self.worker.custom_parc_path, return_label_names=True
                )
                coords = []
                roi_names = []
                for cid, cname in zip(self.worker.target_cluster_ids, self.worker.fc_indices):
                    if cid in all_labels:
                        idx = list(all_labels).index(cid)
                        coords.append(all_coords[idx])
                    else:
                        coords.append([0, 0, 0])
                    roi_names.append(cname)
                coords = np.array(coords)
                node_colors = ["#A0A0A0"] * len(roi_names)
            except Exception as e:
                QMessageBox.critical(self, "Coordinates Error", f"Failed to compute coordinates from custom parcellation:\n{e}")
                return
        else:
            coords, roi_names = get_schaefer_coords(res)
            if coords is None:
                QMessageBox.warning(self, "Coordinates unavailable",
                                    f"Could not load Schaefer-{res} centroid coordinates.\n"
                                    "Check your internet connection or the schaefer_coords folder.")
                return
            node_colors = get_node_colors(roi_names)

        if coords.shape[0] != self.fc_matrix.shape[0]:
            QMessageBox.warning(self, "Shape mismatch",
                                f"Coordinates have {coords.shape[0]} parcels but FC matrix "
                                f"has {self.fc_matrix.shape[0]} rows. "
                                f"Did you select the correct resolution?")
            return

        direction   = self.worker.resolved_direction
        edge_cmap   = "Reds" if direction == "pos" else "Blues"
        edge_count  = int(np.sum(self.fc_matrix) / 2)
        title_std   = (f"Custom FC | Resolution {res} | "
                       f"{'Positive' if direction=='pos' else 'Negative'} | "
                       f"Edges: {edge_count}")

        try:
            html_std = plotting.view_connectome(
                self.fc_matrix, coords,
                edge_threshold="0.1%", node_size=4, linewidth=1.5,
                node_color=node_colors, edge_cmap=edge_cmap,
                colorbar=False, title=title_std
            )
            temp_std = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "temp_fc_enrich_std.html")
            html_std.save_as_html(temp_std)
            self.web_view_std.load(QUrl.fromLocalFile(temp_std))
        except Exception as e:
            QMessageBox.critical(self, "3D render error", str(e))
            return

        if self.chk_overlay.isChecked():
            self.web_view_overlay.show()
            self._render_overlay_view()

    def _render_overlay_view(self):
        if self.results_df is None or self.fc_matrix is None:
            return
        if self.worker is None:
            return

        sel_ct = self.combo_ct.currentText()
        if not sel_ct:
            return

        res = self.combo_res.currentText()
        if res == "Custom Parcellation...":
            if not getattr(self.worker, "ct_data", None) is not None:
                return
            ct_df = self.worker.ct_data
            if sel_ct not in ct_df.columns:
                return
            expr = ct_df[sel_ct].fillna(0).values
            try:
                import nilearn.plotting as plotting_tools
                all_coords, all_labels = plotting_tools.find_parcellation_cut_coords(
                    self.worker.custom_parc_path, return_label_names=True
                )
                coords = []
                for cid in self.worker.target_cluster_ids:
                    if cid in all_labels:
                        idx = list(all_labels).index(cid)
                        coords.append(all_coords[idx])
                    else:
                        coords.append([0, 0, 0])
                coords = np.array(coords)
            except Exception:
                return
        else:
            ct_file = rf"{BASE_DIR}/cell_type_csv/cell_types_{res}_7net.csv"
            try:
                ct_df = pd.read_csv(ct_file)
            except Exception:
                return
    
            if sel_ct not in ct_df.columns:
                return
    
            expr = ct_df[sel_ct].fillna(0).values
            coords, roi_names = get_schaefer_coords(res)

        if coords is None or len(expr) != len(coords):
            return

        e_min, e_max = expr.min(), expr.max()
        norm_expr = (expr - e_min) / (e_max - e_min + 1e-8)
        node_sizes = 2 + norm_expr * 13

        import matplotlib.cm as cm
        direction  = self.worker.resolved_direction
        node_cmap  = cm.get_cmap("Purples" if direction == "pos" else "Greens")
        node_colors_ov = [mcolors.to_hex(node_cmap(0.3 + 0.7 * float(v))) for v in norm_expr]

        edge_cmap   = "Reds" if direction == "pos" else "Blues"
        title_ov    = f"Custom FC | Res {res} | {sel_ct} overlay (darker = higher expression)"

        try:
            html_ov = plotting.view_connectome(
                self.fc_matrix, coords,
                edge_threshold="0.1%", node_size=node_sizes, linewidth=1.5,
                node_color=node_colors_ov, edge_cmap=edge_cmap,
                colorbar=False, title=title_ov
            )
            temp_ov = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "temp_fc_enrich_overlay.html")
            html_ov.save_as_html(temp_ov)
            self.web_view_overlay.load(QUrl.fromLocalFile(temp_ov))
            self.web_view_overlay.show()
        except Exception:
            pass


# MAIN MENU WIDGET
class MainMenuWidget(QWidget):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        layout = QVBoxLayout(self)
        
        layout.addStretch()
        
        title = QLabel("Single Cell Annotation Toolbox for Neuroimage Data")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Select an analysis module to begin:")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(50)
        
        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignCenter)
        
        btn1 = QPushButton("1. FC Network Enrichment Analysis")
        btn1.setFont(QFont("Arial", 14, QFont.Bold))
        btn1.setFixedSize(750, 60)
        btn1.setStyleSheet("background-color: #00695C; color: white; text-align: left; padding-left: 160px;")
        btn1.clicked.connect(lambda: self.parent_window.show_screen(1))
        btn_layout.addWidget(btn1)
        
        btn_layout.addSpacing(20)
        
        btn2 = QPushButton("2. FC Network Enrichment Viewer (3D)")
        btn2.setFont(QFont("Arial", 14, QFont.Bold))
        btn2.setFixedSize(750, 60)
        btn2.setStyleSheet("background-color: #2196F3; color: white; text-align: left; padding-left: 160px;")
        btn2.clicked.connect(lambda: self.parent_window.show_screen(2))
        btn_layout.addWidget(btn2)
        
        btn_layout.addSpacing(20)
        
        btn3 = QPushButton("3. Cluster && Parcellation Analysis")
        btn3.setFont(QFont("Arial", 14, QFont.Bold))
        btn3.setFixedSize(750, 60)
        btn3.setStyleSheet("background-color: #4CAF50; color: white; text-align: left; padding-left: 160px;")
        btn3.clicked.connect(lambda: self.parent_window.show_screen(3))
        btn_layout.addWidget(btn3)

        btn_layout.addSpacing(20)

        btn4 = QPushButton("4. Cluster && Parcellation Viewer (3D)")
        btn4.setFont(QFont("Arial", 14, QFont.Bold))
        btn4.setFixedSize(750, 60)
        btn4.setStyleSheet("background-color: #FF9800; color: white; text-align: left; padding-left: 160px;")
        btn4.clicked.connect(lambda: self.parent_window.show_screen(4))
        btn_layout.addWidget(btn4)

        layout.addLayout(btn_layout)
        layout.addStretch()

# MAIN MASTER WINDOW
class MasterToolbox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single Cell Annotation Toolbox for Neuroimage Data")
        self.setGeometry(50, 50, 1200, 800)
        
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        self.main_menu = MainMenuWidget(self)
        self.stacked_widget.addWidget(self.main_menu)
        
        self.tab1 = FCEnrichmentViewerWidget(self)
        self.stacked_widget.addWidget(self.tab1)
        
        self.tab2 = NetworkViewerWidget(self)
        self.stacked_widget.addWidget(self.tab2)
        
        self.tab3 = CellTypeWidget(self)
        self.stacked_widget.addWidget(self.tab3)

        self.tab4 = InteractiveViewerWidget(self)
        self.stacked_widget.addWidget(self.tab4)

    def show_screen(self, index):
        self.stacked_widget.setCurrentIndex(index)
        
    def show_main_menu(self):
        self.stacked_widget.setCurrentIndex(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MasterToolbox()
    window.show()
    sys.exit(app.exec_())
