
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def load_schaefer_centroids(csv_path: str | Path):
    """
    Return (labels, coords_ras, hemi_mask) where
        labels      : list of parcel names in order
        coords_ras  : (N, 3) array of R, A, S coordinates
        hemi_mask   : bool array, True for LH, False for RH
    """
    df = pd.read_csv(csv_path)
    names = df["ROI Name"].astype(str).tolist()
    coords = df[["R", "A", "S"]].to_numpy(dtype=float)
    is_lh = np.array(["_LH_" in n for n in names])
    return names, coords, is_lh


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random 3x3 rotation (Arvo 1992)."""
    u1, u2, u3 = rng.random(3)
    theta = 2 * np.pi * u1
    phi = 2 * np.pi * u2
    z = u3
    r = np.sqrt(z)
    v = np.array([np.cos(phi) * r, np.sin(phi) * r, np.sqrt(1 - z)])
    st = np.sin(theta); ct = np.cos(theta)
    R = np.array([[ ct,  st, 0.0],
                  [-st,  ct, 0.0],
                  [0.0, 0.0, 1.0]])
    H = np.eye(3) - 2.0 * np.outer(v, v)
    M = -H @ R
    return M


def generate_spin_permutations(coords_ras: np.ndarray,
                                hemi_mask: np.ndarray,
                                n_spins: int,
                                seed: int | None = 0) -> np.ndarray:
    """
    Generate n_spins permutation index vectors.

    For every spin, each hemisphere is rotated by an independent random rotation
    and each rotated parcel is matched to the nearest original parcel *in the
    same hemisphere* (nearest-neighbour matching, Alexander-Bloch 2018 style).

    Parameters
    ----------
    coords_ras : (N, 3) array of parcel centroids in MNI space
    hemi_mask  : (N,) boolean, True = LH
    n_spins    : number of rotations
    seed       : RNG seed

    Returns
    -------
    perms : (n_spins, N) int array, where perms[i, j] = index of the parcel
            whose value should replace parcel j in the i-th spin.
    """
    N = coords_ras.shape[0]
    rng = np.random.default_rng(seed)

    # Per-hemisphere unit-sphere projection
    lh_idx = np.where(hemi_mask)[0]
    rh_idx = np.where(~hemi_mask)[0]
    lh_pts = coords_ras[lh_idx]
    rh_pts = coords_ras[rh_idx]
    lh_unit = lh_pts / np.linalg.norm(lh_pts, axis=1, keepdims=True)
    rh_unit = rh_pts / np.linalg.norm(rh_pts, axis=1, keepdims=True)

    perms = np.empty((n_spins, N), dtype=np.int64)
    reflect = np.diag([-1.0, 1.0, 1.0])

    for s in range(n_spins):
        R_left = _random_rotation_matrix(rng)
        R_right = reflect @ R_left @ reflect

        lh_rot = lh_unit @ R_left.T
        rh_rot = rh_unit @ R_right.T

        lh_sim = lh_rot @ lh_unit.T 
        rh_sim = rh_rot @ rh_unit.T
        lh_nn = lh_idx[np.argmax(lh_sim, axis=1)]
        rh_nn = rh_idx[np.argmax(rh_sim, axis=1)]

        perm = np.empty(N, dtype=np.int64)
        perm[lh_idx] = lh_nn
        perm[rh_idx] = rh_nn
        perms[s] = perm

    return perms


if __name__ == "__main__":
    p = rf"{BASE_DIR}/schaefer_coords/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
    names, coords, is_lh = load_schaefer_centroids(p)
    perms = generate_spin_permutations(coords, is_lh, n_spins=10, seed=0)
    print("Loaded", len(names), "parcels,", is_lh.sum(), "LH /", (~is_lh).sum(), "RH")
    print("Spin permutation shape:", perms.shape)
    lh_idx = np.where(is_lh)[0]
    rh_idx = np.where(~is_lh)[0]
    for s in range(perms.shape[0]):
        assert np.all(np.isin(perms[s, lh_idx], lh_idx))
        assert np.all(np.isin(perms[s, rh_idx], rh_idx))
    print("Hemisphere containment check: OK")
