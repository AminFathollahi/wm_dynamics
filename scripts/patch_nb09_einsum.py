"""Fix einsum axis transposition in notebook 09 cells 7 and 8.

epochs shape is (N_trials, N_ch, N_time) but einsum 'ntc,ck->ntk' assumes (N, T, C).
Fix: use 'nct,kc->ntk' or transpose epochs before einsum.
"""
import json
from pathlib import Path

nb_path = Path("notebooks/09_cross_dataset_replication.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

fixes = {
    # Cell 7: Z_latent einsum
    "Z_latent = np.einsum('ntc,ck->ntk', epochs_win, pca_base.components_.T)":
    "Z_latent = np.einsum('nct,kc->ntk', epochs_win, pca_base.components_)",

    # Cell 8 (Boran): Z_pca einsum
    "Z_pca = np.einsum('ntc,ck->ntk', epochs[:, :, maint_t], pca_b.components_.T)":
    "Z_pca = np.einsum('nct,kc->ntk', epochs[:, :, maint_t], pca_b.components_)",
}

n_fixed = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    for old, new in fixes.items():
        if old in src:
            src = src.replace(old, new)
            n_fixed += 1
    cell['source'] = src

# Clear all outputs for re-execution
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Fixed {n_fixed} einsum axes (nct,kc->ntk)")
