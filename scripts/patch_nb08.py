"""Patch notebook 08 CTG cell to use coarser time step."""
import json
from pathlib import Path

nb_path = Path("notebooks/08_extended_analysis.ipynb")
with open(nb_path) as f:
    nb = json.load(f)

# Cell 6 is the CTG cell — change step from 5 to 20
cell = nb['cells'][6]
src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
src_fixed = src.replace("    step = 5\n", "    step = 20  # 2040/20 = 102 timepoints\n")
cell['source'] = src_fixed

# Clear all outputs for re-execution
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Patched: CTG step 5 -> 20")
