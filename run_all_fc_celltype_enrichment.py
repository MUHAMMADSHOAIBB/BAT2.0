import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
from numba import njit

def load_data(fc_file, ct_file):
    # Load binarized FC matrix
    fc_df = pd.read_csv(fc_file, index_col=0)
    fc_matrix = fc_df.values
    
    # Ensure symmetry
    fc_matrix = np.nan_to_num(fc_matrix, 0)
    fc_matrix = np.maximum(fc_matrix, fc_matrix.T)
    np.fill_diagonal(fc_matrix, 0) # No self loops
    
    # Load cell types
    ct_df = pd.read_csv(ct_file)
    cols_to_drop = [c for c in ct_df.columns if c.lower() in ['region', 'numdonorsineachparcel', 'maxdonorpresenceinparcel']]
    ct_data = ct_df.drop(columns=cols_to_drop)
    
    return fc_matrix, ct_data

def calculate_coexpression_fast(fc_matrix, X, V):
    """
    Calculates the average co-expression of connected regions using fast matrix operations.
    X: (N, C) array of cell type expression (NaNs replaced with 0)
    V: (N, C) binary array indicating valid (non-NaN) entries
    """
    # (fc_matrix @ X) * X computes the sum of X_i * X_j for all edges
    numerator = np.sum((fc_matrix @ X) * X, axis=0) / 2.0
    # (fc_matrix @ V) * V computes the number of valid edges
    denominator = np.sum((fc_matrix @ V) * V, axis=0) / 2.0
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = np.where(denominator > 0, numerator / denominator, np.nan)
    return scores

@njit(nogil=True)
def fast_double_edge_swap(fc_matrix, nswap_multiplier=5):
    """Generates a degree-preserving random network using an optimized algorithm."""
    fc_rand = fc_matrix.copy()
    
    # Get upper triangle edges
    n = fc_rand.shape[0]
    m = 0
    for i in range(n):
        for j in range(i+1, n):
            if fc_rand[i, j] == 1:
                m += 1
                
    if m < 2:
        return fc_rand
        
    edges_i = np.zeros(m, dtype=np.int32)
    edges_j = np.zeros(m, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if fc_rand[i, j] == 1:
                edges_i[idx] = i
                edges_j[idx] = j
                idx += 1
                
    nswap = int(nswap_multiplier * m)
    max_tries = nswap * 10
    
    tries = 0
    swaps = 0
    while swaps < nswap and tries < max_tries:
        tries += 1
        e1 = np.random.randint(0, m)
        e2 = np.random.randint(0, m)
        if e1 == e2:
            continue
            
        u, v = edges_i[e1], edges_j[e1]
        x, y = edges_i[e2], edges_j[e2]
        
        if u == x or u == y or v == x or v == y:
            continue
            
        if np.random.rand() > 0.5:
            if fc_rand[u, x] == 0 and fc_rand[v, y] == 0:
                fc_rand[u, v] = 0; fc_rand[v, u] = 0
                fc_rand[x, y] = 0; fc_rand[y, x] = 0
                fc_rand[u, x] = 1; fc_rand[x, u] = 1
                fc_rand[v, y] = 1; fc_rand[y, v] = 1
                edges_i[e1] = min(u, x); edges_j[e1] = max(u, x)
                edges_i[e2] = min(v, y); edges_j[e2] = max(v, y)
                swaps += 1
        else:
            if fc_rand[u, y] == 0 and fc_rand[v, x] == 0:
                fc_rand[u, v] = 0; fc_rand[v, u] = 0
                fc_rand[x, y] = 0; fc_rand[y, x] = 0
                fc_rand[u, y] = 1; fc_rand[y, u] = 1
                fc_rand[v, x] = 1; fc_rand[x, v] = 1
                edges_i[e1] = min(u, y); edges_j[e1] = max(u, y)
                edges_i[e2] = min(v, x); edges_j[e2] = max(v, x)
                swaps += 1
                
    return fc_rand

def process_single_file(fc_file, n_permutations=1000):
    try:
        # Extract resolution from file path
        match = re.search(r'schaefer(\d+)-yeo7', fc_file)
        if not match:
            return None
        res = match.group(1)
        # Cell-type CSVs are resolved relative to the repository root (run this script from there).
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ct_file = os.path.join(script_dir, "cell_type_csv", f"cell_types_{res}_7net.csv")
        
        if not os.path.exists(ct_file):
            return None
            
        out_file = fc_file.replace('.csv', '_celltype_enrichment.csv')
        if os.path.exists(out_file):
            return None # Skip if already processed
            
        fc_matrix, ct_data = load_data(fc_file, ct_file)
        if np.sum(fc_matrix) == 0:
            return None
            
        # Prepare matrices for fast coexpression calculation
        X_df = ct_data.copy()
        V_df = (~X_df.isna()).astype(float)
        X_df = X_df.fillna(0)
        
        X = X_df.values
        V = V_df.values
        
        # Calculate real scores
        real_scores = calculate_coexpression_fast(fc_matrix, X, V)
        
        # Pre-compile the njit function by running it once on a small dummy array if it's the first run
        # Numba will compile it on the first call
        null_scores = np.zeros((n_permutations, ct_data.shape[1]))
        
        # Initial long randomization
        current_rand = fast_double_edge_swap(fc_matrix, nswap_multiplier=5)
        null_scores[0, :] = calculate_coexpression_fast(current_rand, X, V)
        
        # Subsequent permutations only need 1x swaps to maintain MCMC chain
        for i in range(1, n_permutations):
            current_rand = fast_double_edge_swap(current_rand, nswap_multiplier=1)
            null_scores[i, :] = calculate_coexpression_fast(current_rand, X, V)
            
        p_values = np.zeros(ct_data.shape[1])
        for i in range(ct_data.shape[1]):
            valid_nulls = null_scores[:, i][~np.isnan(null_scores[:, i])]
            if len(valid_nulls) == 0 or np.isnan(real_scores[i]):
                p_values[i] = np.nan
            else:
                p_values[i] = np.sum(valid_nulls >= real_scores[i]) / len(valid_nulls)
                
        results = pd.DataFrame({
            'CellType': ct_data.columns,
            'Real_Score': real_scores,
            'P_Value': p_values
        }).sort_values('P_Value')
        
        results.to_csv(out_file, index=False)
        return out_file
    except Exception as e:
        print(f"Error processing {fc_file}: {e}")
        return None

def process_file_wrapper(args):
    return process_single_file(*args)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_pattern = os.path.join(script_dir, "Functional_connectivity", "**", "*_binarized_*.csv")

    parser = argparse.ArgumentParser()
    parser.add_argument('--permutations', type=int, default=1000)
    # Relative to the repository root by default; pass --pattern to point elsewhere.
    parser.add_argument('--pattern', type=str, default=default_pattern)
    parser.add_argument('--force', action='store_true', help='Force re-run even if output exists')
    args = parser.parse_args()
    
    files = glob.glob(args.pattern, recursive=True)
    # Exclude this script's own previously-generated output files: their names also
    # contain "_binarized_" as a substring, so they would otherwise be re-matched as
    # if they were fresh FC inputs (and fail, since they aren't adjacency matrices).
    files = [f for f in files if not f.endswith('_celltype_enrichment.csv')]
    print(f"Found {len(files)} binarized FC files to process.")
    
    if args.force:
        # Delete existing outputs to force re-run
        existing_outputs = glob.glob(args.pattern.replace('.csv', '_celltype_enrichment.csv'), recursive=True)
        for f in existing_outputs:
            try:
                os.remove(f)
            except:
                pass
    
    process_args = [(f, args.permutations) for f in files]
    
    processed = 0
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file_wrapper, process_args), total=len(files)))
        processed = sum(1 for r in results if r is not None)
            
    print(f"Finished processing {processed} files.")

if __name__ == "__main__":
    # Force numba compilation on a small dummy array
    dummy = np.array([[0, 1], [1, 0]])
    _ = fast_double_edge_swap(dummy, 1)
    
    main()