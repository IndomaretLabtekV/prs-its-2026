"""Smoke V9 tahap 1: sel 1,3,5,7,9 saja (foreground, cepat)."""
import json
import os

os.environ['V9_SMOKE'] = '1'
nb = json.load(open('notebooks/V9_Representation.ipynb'))
NS = {'__name__': '__main__'}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    if i > 9:
        break
    print(f'--- exec code cell {i} ---', flush=True)
    exec(compile(''.join(c['source']), f'<v9cell-{i}>', 'exec'), NS)
print('STAGE1 DONE')
