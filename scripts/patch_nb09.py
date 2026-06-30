"""Patch notebook 09 cell 2 to handle variable-length Boran sessions."""
import json
from pathlib import Path

nb_path = Path("notebooks/09_cross_dataset_replication.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

new_cell2_source = (
    "# Load Boran Sternberg data (all 9 subjects x sessions)\n"
    "from sklearn.decomposition import PCA\n"
    "\n"
    "boran_subjects = {}\n"
    "\n"
    "for sub_dir in sorted(BORAN_DIR.iterdir()):\n"
    "    if not sub_dir.is_dir() or not sub_dir.name.startswith('sub-'):\n"
    "        continue\n"
    "    subj_id = sub_dir.name\n"
    "    sessions = sorted(sub_dir.glob('*.nwb'))\n"
    "\n"
    "    subj_epochs, subj_sizes, subj_correct, subj_times = [], [], [], []\n"
    "\n"
    "    for ses_path in sessions:\n"
    "        try:\n"
    "            result = load_boran_nwb(str(ses_path), signal='ieeg')\n"
    "            epochs = result['epochs']                    # (N, C, T)\n"
    "            valid = result['valid'] & ~result['artifact']\n"
    "            hgp = compute_boran_hgp(epochs[valid], srate=result['srate'])\n"
    "            hgp_z = boran_baseline_normalize(hgp, result['times'], BASELINE_WIN)\n"
    "            subj_epochs.append(hgp_z)\n"
    "            subj_sizes.append(result['set_sizes'][valid])\n"
    "            subj_correct.append(result['correct'][valid])\n"
    "            subj_times.append(result['times'])\n"
    "        except Exception as e:\n"
    "            print(f'  [{subj_id}/{ses_path.name}] ERROR: {e}')\n"
    "\n"
    "    if subj_epochs:\n"
    "        min_ch = min(e.shape[1] for e in subj_epochs)\n"
    "        min_t  = min(e.shape[2] for e in subj_epochs)\n"
    "        subj_epochs = [e[:, :min_ch, :min_t] for e in subj_epochs]\n"
    "        times_ref = subj_times[0][:min_t]\n"
    "        combined_epochs  = np.concatenate(subj_epochs, axis=0)\n"
    "        combined_sizes   = np.concatenate(subj_sizes)\n"
    "        combined_correct = np.concatenate(subj_correct)\n"
    "        boran_subjects[subj_id] = {\n"
    "            'epochs':    combined_epochs,\n"
    "            'set_sizes': combined_sizes,\n"
    "            'correct':   combined_correct,\n"
    "            'times':     times_ref,\n"
    "            'srate':     result['srate'],\n"
    "            'n_ch':      combined_epochs.shape[1],\n"
    "        }\n"
    "        n_tr = combined_epochs.shape[0]\n"
    "        n_ch = combined_epochs.shape[1]\n"
    "        n_t  = combined_epochs.shape[2]\n"
    "        acc  = combined_correct.mean()\n"
    "        sz   = sorted(np.unique(combined_sizes))\n"
    "        print(f'[{subj_id}] {n_tr} trials x {n_ch} ch x {n_t} T, correct={acc:.1%}, set_sizes={sz}')\n"
    "\n"
    "print(f'Loaded {len(boran_subjects)} subjects')\n"
)

nb['cells'][2]['source'] = new_cell2_source

# Clear all cell outputs for re-execution
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Patched notebook 09 cell 2 — no f-string newlines")
