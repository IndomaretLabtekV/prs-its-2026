"""Smoke V9 notebook: exec semua code cell berurutan (seperti kernel),
dengan V9_SMOKE=1 (2 fold, 180 iter). Env: conda datsci."""
import json
import os

os.environ['V9_SMOKE'] = '1'
nb = json.load(open('notebooks/V9_Representation.ipynb'))
NS = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    print(f'--- exec code cell {i} ---', flush=True)
    exec(compile(''.join(c['source']), f'<v9cell-{i}>', 'exec'), NS)
print('SMOKE V9 DONE')
