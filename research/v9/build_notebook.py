"""Build notebooks/V9_Representation.ipynb from V8 cells + new V9 cells.

Reuse (verbatim): V8 cells 03 (metrics), 05 (features).
Adapted (string-replace, asserted): V8 cell 01 (config/paths).
New: fold authority (artifact), V8-checkpoint validator, V9 features,
4-arm roster, 5-candidate registry (8 gates), selection/submission/diagnostic.
Env: conda datsci (builder only needs stdlib+json).
"""
import json
from pathlib import Path

ROOT = Path('/home/athilla/Documents/IF_ITB/Lomba/DAC-PRS-ITS/prs-its')
V8 = json.load(open(ROOT / 'notebooks/V8_Robust_Boundary.ipynb'))
SRC = {i: ''.join(V8['cells'][i].get('source', []))
       for i, c in enumerate(V8['cells']) if c['cell_type'] == 'code'}


def md(text):
    return {'cell_type': 'markdown', 'metadata': {},
            'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


# ---- Cell 01: config adapted from V8 cell 01 ----
c01 = SRC[1]
_old_raise = ("if not LIGHTGBM_AVAILABLE and not (os.getenv('DAC_FAST_MODE', '0') == '1'):\n"
              "    raise RuntimeError('LightGBM unavailable in full mode. Install the declared notebook dependency before training.')")
assert _old_raise in c01, 'V8 cell01 raise block changed'
c01 = c01.replace(_old_raise, "pass  # V9 trains CatBoost only; XGB/LGBM imports stay optional")
for old, new in [('/kaggle/working/v8_artifacts', '/kaggle/working/v9_artifacts'),
                 ('/tmp/dac2026_v8_artifacts', '/tmp/dac2026_v9_artifacts'),
                 ('/kaggle/working/submission_v8.csv', '/kaggle/working/submission_v9.csv'),
                 ('/tmp/submission_v8.csv', '/tmp/submission_v9.csv')]:
    assert old in c01, f'V8 cell01 path changed: {old}'
    c01 = c01.replace(old, new)
c01 += ("\nV9_SMOKE = os.getenv('V9_SMOKE', '0') == '1'\n"
        "V9_FOLDS = 2 if V9_SMOKE else CFG.folds\n"
        "V9_ITERS = 180 if V9_SMOKE else CFG.cat_iterations\n"
        "RUNTIME_LABEL = ('SMOKE run, non-comparable' if V9_SMOKE else RUNTIME_LABEL)\n"
        "print(json.dumps({'v9_smoke': V9_SMOKE, 'v9_folds': V9_FOLDS, 'v9_iters': V9_ITERS}))\n")

# ---- Cell 07: local signature validation (no TE build in V9) ----
c07 = '''signature_frame = train_clean.drop(columns=[ID_COL, TARGET_COL], errors='ignore')
SIGNATURE_GROUPS = pd.util.hash_pandas_object(signature_frame, index=False).astype('uint64').astype('string')

def make_splits(groups, n_splits, seed, stratify=y):
    if pd.Series(groups).nunique() < n_splits:
        raise ValueError('not enough groups for requested grouped folds')
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), stratify, groups=groups))

LOCAL_SPLITS = make_splits(SIGNATURE_GROUPS, CFG.folds, CFG.seed)
LOCAL_FOLD_ID = np.full(len(y), -1, dtype=int)
for fold, (fit_idx, valid_idx) in enumerate(LOCAL_SPLITS):
    LOCAL_FOLD_ID[valid_idx] = fold
assert (LOCAL_FOLD_ID >= 0).all()
_local_overlap = 0
for fold, (fit_idx, valid_idx) in enumerate(LOCAL_SPLITS):
    _local_overlap += len(set(SIGNATURE_GROUPS.iloc[fit_idx]) & set(SIGNATURE_GROUPS.iloc[valid_idx]))
assert _local_overlap == 0
print('Behavior 2 local exact-signature disjointness: PASS')
'''

# ---- Cell 09: V8 checkpoint validator (artifact folds authoritative) ----
c09 = '''V9_EVAL_FILES = {'v8_scores.npz', 'primary_fold_assignments.csv', 'v8_candidate_registry.csv'}
V9_BASE_FILES = {'predictions_catboost_v3_seed_2026.npz', 'predictions_catboost_v3_seed_2127.npz'}

def ckpt_roots():
    roots = [Path('/kaggle/input'), Path.cwd(), Path.cwd().parent]
    out = []
    for root in roots:
        if root.exists():
            out.extend(list(root.rglob('results.zip')))
            out.extend([p for p in root.rglob('v8_artifacts') if p.is_dir()])
    return sorted(set(out), key=lambda p: (p.is_dir(), -p.stat().st_size if p.is_file() else 0))

def open_ckpt(path):
    if path.is_dir():
        return {'names': {p.name for p in path.iterdir() if p.is_file()}, 'read': lambda name, _p=path: (_p / name).read_bytes()}
    archive = zipfile.ZipFile(path)
    return {'names': {Path(n).name for n in archive.namelist() if not n.endswith('/')},
            'read': lambda name: archive.read(next(n for n in archive.namelist() if n.endswith('/' + name) or n == name))}

def scan_flat_dirs():
    found = []
    for root in [Path('/kaggle/input'), Path.cwd()]:
        if not root.exists():
            continue
        for p in [root] + sorted([q for q in root.rglob('*') if q.is_dir()]):
            try:
                names = {x.name for x in p.iterdir() if x.is_file()}
            except OSError:
                continue
            if V9_EVAL_FILES <= names or V9_BASE_FILES <= names or 'v8_scores.npz' in names:
                found.append((p, {'names': names, 'read': lambda name, _p=p: (_p / name).read_bytes()}))
    return found

CKPT_STORE, BASE_STORE, ARTIFACT_PROVENANCE = None, None, None
_eval_cands, _base_cands = [], []
for _cand in ckpt_roots():
    try:
        _store = open_ckpt(_cand)
    except (zipfile.BadZipFile, StopIteration, KeyError, OSError):
        continue
    if V9_EVAL_FILES <= _store['names']:
        _eval_cands.append((_cand, _store))
    if V9_BASE_FILES <= _store['names']:
        _base_cands.append((_cand, _store))
for _cand, _store in scan_flat_dirs():
    if V9_EVAL_FILES <= _store['names'] and all(str(_cand) != str(e[0]) for e in _eval_cands):
        _eval_cands.append((_cand, _store))
    if V9_BASE_FILES <= _store['names'] and all(str(_cand) != str(e[0]) for e in _base_cands):
        _base_cands.append((_cand, _store))
if not _eval_cands:
    _near = [str(p) for p, s in scan_flat_dirs() if 'v8_scores.npz' in s['names']]
    raise RuntimeError('No V8 eval store found (need v8_scores.npz, fold assignments, registry). Dirs containing v8_scores.npz: %s' % (_near or 'none under /kaggle/input'))
if not _base_cands:
    raise RuntimeError('No base-prediction store found (need predictions_catboost_v3_seed_2026/2127.npz, e.g. results/v7/results.zip)')
CKPT_STORE, ARTIFACT_PROVENANCE = _eval_cands[0][1], str(_eval_cands[0][0])
BASE_STORE = _base_cands[0][1]
print('Eval store:', ARTIFACT_PROVENANCE, '| base store:', _base_cands[0][0])

_v8npz = np.load(io.BytesIO(CKPT_STORE['read']('v8_scores.npz')), allow_pickle=True)
assert {'selected_oof', 'selected_test', 'v3_oof', 'v3_test'} <= set(_v8npz.files)
assert np.array_equal(_v8npz['selected_oof'], _v8npz['v3_oof']) and np.array_equal(_v8npz['selected_test'], _v8npz['v3_test'])
V3_OOF_SCORE = np.clip(_v8npz['v3_oof'].astype(float), 1e-7, 1 - 1e-7)
V3_TEST_SCORE = np.clip(_v8npz['v3_test'].astype(float), 1e-7, 1 - 1e-7)

def _load_base(name):
    safe = name.lower().replace(' ', '_').replace('/', '_')
    _d = np.load(io.BytesIO(BASE_STORE['read'](f'predictions_{safe}.npz')), allow_pickle=True)
    assert _d['oof'].shape == (len(y),) and _d['test'].shape == (len(test_model),)
    assert np.isfinite(_d['oof']).all() and np.isfinite(_d['test']).all()
    return {'oof': np.clip(_d['oof'].astype(float), 1e-7, 1 - 1e-7), 'test': np.clip(_d['test'].astype(float), 1e-7, 1 - 1e-7)}

BASE2127 = _load_base('CatBoost V3 seed 2127')
BASE2026 = _load_base('CatBoost V3 seed 2026')

_assign = pd.read_csv(io.BytesIO(CKPT_STORE['read']('primary_fold_assignments.csv')))
assert _assign[ID_COL].astype(str).tolist() == TIE_KEYS.tolist()
ART_FOLD_ID = _assign['fold'].to_numpy(dtype=int)
if sorted(np.unique(ART_FOLD_ID).tolist()) == [1, 2, 3, 4, 5]:
    ART_FOLD_ID = ART_FOLD_ID - 1
print('Artifact folds match local recomputation:', bool(np.array_equal(ART_FOLD_ID, LOCAL_FOLD_ID)), '(artifact is authoritative either way)')
FOLD_ID = ART_FOLD_ID
PRIMARY_SPLITS = [(np.flatnonzero(FOLD_ID != f), np.flatnonzero(FOLD_ID == f)) for f in range(CFG.folds)]
SMOKE_FOLDS = [0, 1] if V9_SMOKE else list(range(CFG.folds))
SMOKE_MASK = np.isin(FOLD_ID, SMOKE_FOLDS)
print('folds in play:', SMOKE_FOLDS, 'rows:', int(SMOKE_MASK.sum()))

_ckreg = pd.read_csv(io.BytesIO(CKPT_STORE['read']('v8_candidate_registry.csv')))
_c0 = _ckreg.loc[_ckreg.candidate.eq('V3 control')].iloc[0]
_v3check = metric_row(y, V3_OOF_SCORE, TIE_KEYS, train_clean['jkpst'], train_clean['umur'], 'V3 control')
_tp_delta = int(_v3check['fraud_captured']) - int(_c0['fraud_captured'])
_nr_delta = float(_v3check['NormalizedRecall@5%']) - float(_c0['NormalizedRecall@5%'])
if abs(_tp_delta) > 8 or abs(_nr_delta) > 0.001:
    raise RuntimeError(f'checkpoint V3 control inconsistent with rebuild: TP delta {_tp_delta:+d}, NR delta {_nr_delta:+.6f}')
if _tp_delta or _nr_delta:
    print(f'NOTE: {_tp_delta:+d} TP / {_nr_delta:+.6f} NR vs registry from cross-version tie order at the K boundary; within V8 rebuild tolerance, rebuilt scores authoritative.')
print('Behavior 4 V8 checkpoint V3-control consistency: PASS (NR %.5f, TP %d)' % (_v3check['NormalizedRecall@5%'], _v3check['fraud_captured']))
print('Checkpoint provenance:', ARTIFACT_PROVENANCE)
'''

# ---- Cell 11: V9 representation features ----
c11 = '''RISK_KDKC = {'2101', '2103'}
V9_NEW = ['risk_subgroup', 'risk_subgroup__age_band', 'risk_subgroup__los_band', 'risk_subgroup__burden_band']

def engineer_v9(train_frame, test_frame):
    # NOTE: mutates the passed frames in place (no .copy) to cap peak RAM;
    # safe because engineer_features returned fresh frames used only here.
    b_edges = [-np.inf] + sorted(pd.to_numeric(train_frame['total_coded_burden'], errors='coerce').fillna(0).quantile([1/3, 2/3]).tolist()) + [np.inf]
    out = []
    for frame in [train_frame, test_frame]:
        k = frame['kdkc'].astype('string'); d = frame['dati2'].astype('string'); t = frame['typeppk'].astype('string')
        sub = np.where(t.eq('KM'), 'KM', np.where(k.isin(RISK_KDKC), 'KDKC2101_2103', np.where(d.eq('309'), 'DATI2_309', 'OTHER')))
        frame['risk_subgroup'] = pd.Series(sub, index=frame.index).astype('string')
        frame['risk_subgroup__age_band'] = (frame['risk_subgroup'] + '__' + frame['age_band'].astype('string')).astype('string')
        los = pd.to_numeric(frame['los'], errors='coerce').fillna(0)
        los_band = pd.cut(los, [-np.inf, 0, 2, np.inf], labels=['los0', 'los1_2', 'los3p']).astype('string')
        frame['risk_subgroup__los_band'] = (frame['risk_subgroup'] + '__' + los_band).astype('string')
        bb = pd.cut(pd.to_numeric(frame['total_coded_burden'], errors='coerce').fillna(0), bins=b_edges, labels=['b0', 'b1', 'b2']).astype('string')
        frame['risk_subgroup__burden_band'] = (frame['risk_subgroup'] + '__' + bb).astype('string')
        out.append(frame)
    return out[0], out[1]

train_v9, test_v9 = engineer_v9(train_model, test_model)

audit_rows = []
for col in V9_NEW:
    tr_col = train_v9[col]
    if str(tr_col.dtype) in {'object', 'string'}:
        vc = tr_col.value_counts()
        fr = train_v9.assign(_y=y).groupby(col)['_y'].mean()
        audit_rows.append({'column': col, 'nunique_train': int(tr_col.nunique()),
                           'unseen_in_test': int((~test_v9[col].isin(tr_col.unique())).sum()),
                           'top_level': str(vc.index[0]), 'top_level_fraud_rate': float(fr.loc[vc.index[0]])})
    else:
        audit_rows.append({'column': col, 'nunique_train': int(tr_col.nunique()),
                           'unseen_in_test': 0, 'top_level': 'binary',
                           'top_level_fraud_rate': float(train_v9.assign(_y=y).groupby(col)['_y'].mean().to_dict().get(1, float('nan')))})
pd.DataFrame(audit_rows).to_csv(OUTPUT_DIR / 'v9_feature_audit.csv', index=False)
assert all(c in train_v9 and c in test_v9 for c in V9_NEW)
assert not any(('claim_id' in c or 'row' in c) for c in V9_NEW)
assert all(bool(np.isfinite(pd.to_numeric(train_v9[c], errors='coerce').fillna(0)).all()) for c in V9_NEW)
V9_FEATURES = [c for c in FULL_FEATURES] + V9_NEW
V9_CATS = [c for c in V9_FEATURES if c in CATEGORICAL_COLUMNS or str(train_v9[c].dtype) in {'object', 'string'}]
print('V9 features:', len(V9_FEATURES), 'categorical:', len(V9_CATS))
print('Behavior 5b V9 feature quarantine (allowed columns, finite): PASS')
'''

# ---- Cell 13: three-arm roster ----
c13 = '''def catboost_ready(frame, categorical):
    out = frame.copy()
    for c in categorical:
        if c in out:
            out[c] = out[c].astype('string').fillna('__MISSING__').astype(str)
    return out

def fit_cat(X_fit, target, X_valid, X_test, features, seed, iterations, sample_weight=None):
    cats = [c for c in features if c in CATEGORICAL_COLUMNS or c in V9_CATS]
    params = dict(loss_function='Logloss', allow_writing_files=False, verbose=False, random_seed=seed, task_type=CAT_TASK,
                  iterations=iterations, learning_rate=0.03, depth=8, l2_leaf_reg=8.0, random_strength=0.35,
                  bootstrap_type='Bayesian', bagging_temperature=0.5, max_ctr_complexity=2, border_count=128)
    if CAT_DEVICES:
        params['devices'] = CAT_DEVICES
    model = CatBoostClassifier(**params)
    try:
        model.fit(catboost_ready(X_fit[features], cats), target, cat_features=cats, sample_weight=sample_weight, verbose=False)
    except Exception as error:
        msg = str(error).lower()
        if CAT_TASK != 'GPU' or not any(t in msg for t in ('gpu', 'cuda', 'nccl', 'memory')):
            raise
        params.update(devices='0', max_ctr_complexity=1)
        model = CatBoostClassifier(**params)
        model.fit(catboost_ready(X_fit[features], cats), target, cat_features=cats, sample_weight=sample_weight, verbose=False)
    return model.predict_proba(catboost_ready(X_valid[features], cats))[:, 1], model.predict_proba(catboost_ready(X_test[features], cats))[:, 1]

ARMS = [{'arm': 'v9_seed3407', 'features': FULL_FEATURES, 'seed': 3407, 'weight': None, 'frame': 'base'},
        {'arm': 'v9_interact2127', 'features': V9_FEATURES, 'seed': 2127, 'weight': None, 'frame': 'v9'},
        {'arm': 'v9_interact2026', 'features': V9_FEATURES, 'seed': 2026, 'weight': None, 'frame': 'v9'}]
ARM_OOF, ARM_TEST = {}, {}
for arm in ARMS:
    oof = np.full(len(y), np.nan)
    tests = []
    frame = train_v9 if arm['frame'] == 'v9' else train_model
    test_frame = test_v9 if arm['frame'] == 'v9' else test_model
    for fold in SMOKE_FOLDS:
        fit_idx, valid_idx = PRIMARY_SPLITS[fold]
        sw = None
        vp, tp = fit_cat(frame.iloc[fit_idx], y[fit_idx], frame.iloc[valid_idx], test_frame, arm['features'], arm['seed'] + fold, V9_ITERS, sw)
        oof[valid_idx] = vp
        tests.append(tp)
    ARM_OOF[arm['arm']] = oof
    ARM_TEST[arm['arm']] = np.mean(tests, axis=0)
    safe = arm['arm']
    np.savez_compressed(OUTPUT_DIR / f'predictions_{safe}.npz', oof=oof, test=ARM_TEST[arm['arm']], test_folds=np.vstack(tests))
    _m = SMOKE_MASK & np.isfinite(oof)
    _nr = normalized_recall_at_capacity(y[_m], oof[_m], 0.05, TIE_KEYS[_m])
    print('%s pooled-masked NR: %.5f' % (arm['arm'], _nr))
print('Behavior 6 three-arm roster complete: PASS')
'''

# ---- Cell 15: four-candidate registry, 8 gates ----
c15 = '''def churn_stats(old_score, new_score, tie_keys):
    old = top_k_mask(old_score, 0.05, tie_keys)
    new = top_k_mask(new_score, 0.05, tie_keys)
    one_way = int((old & ~new).sum())
    return one_way, float(one_way / max(1, old.sum())), float((old & new).sum() / max(1, old.sum()))

SEEDMEAN_OOF = np.nanmean(np.vstack([BASE2127['oof'], BASE2026['oof'], ARM_OOF['v9_seed3407']]), axis=0)
SEEDMEAN_TEST = np.mean(np.vstack([BASE2127['test'], BASE2026['test'], ARM_TEST['v9_seed3407']]), axis=0)
CANDIDATES = [{'candidate': 'V3 control', 'family': 'control'},
              {'candidate': 'Seed-mean3', 'family': 'seed-mean'},
              {'candidate': 'Interact-2127', 'family': 'interact', 'arm': 'v9_interact2127'},
              {'candidate': 'Interact-2026', 'family': 'interact', 'arm': 'v9_interact2026'}]
assert len(CANDIDATES) == 4 and len({c['candidate'] for c in CANDIDATES}) == 4
print('Behavior 8 candidate registry predeclared uniqueness: PASS (4 candidates)')

def score_candidate(candidate, split='oof'):
    fam = candidate['family']
    if fam == 'control':
        return V3_OOF_SCORE if split == 'oof' else V3_TEST_SCORE
    if fam == 'seed-mean':
        return SEEDMEAN_OOF if split == 'oof' else SEEDMEAN_TEST
    return ARM_OOF[candidate['arm']] if split == 'oof' else ARM_TEST[candidate['arm']]

_y, _tie = y[SMOKE_MASK], TIE_KEYS[SMOKE_MASK]
_v3m = V3_OOF_SCORE[SMOKE_MASK]
base_metrics = metric_row(_y, _v3m, _tie, train_clean.loc[SMOKE_MASK, 'jkpst'], train_clean.loc[SMOKE_MASK, 'umur'], 'V3 control')
base_nr, base_ap = base_metrics['NormalizedRecall@5%'], base_metrics['AveragePrecision']
base_fold = fold_metrics(_y, _v3m, FOLD_ID[SMOKE_MASK], _tie)
registry_rows, fold_delta_rows, churn_rows = [], [], []
for candidate in CANDIDATES:
    oof_full, test_score = score_candidate(candidate, 'oof'), score_candidate(candidate, 'test')
    oof_score = oof_full[SMOKE_MASK]
    metrics = metric_row(_y, oof_score, _tie, train_clean.loc[SMOKE_MASK, 'jkpst'], train_clean.loc[SMOKE_MASK, 'umur'], candidate['candidate'])
    one_way, oof_churn, overlap = churn_stats(_v3m, oof_score, _tie)
    _, test_churn, _ = churn_stats(V3_TEST_SCORE, test_score, TEST_TIE_KEYS)
    folds = fold_metrics(_y, oof_score, FOLD_ID[SMOKE_MASK], _tie).merge(base_fold[['fold', 'tp']], on='fold', suffixes=('', '_v3'))
    folds['tp_delta'] = folds.tp - folds.tp_v3
    for _, row in folds.iterrows():
        fold_delta_rows.append({'candidate': candidate['candidate'], **row.to_dict()})
    confirmation_gain = int(folds.loc[folds.fold > min(3, len(SMOKE_FOLDS)), 'tp_delta'].sum())
    gate_values = [metrics['NormalizedRecall@5%'] - base_nr >= CFG.min_nr5_gain,
                   int((folds.tp_delta > 0).sum()) >= CFG.min_fold_wins,
                   confirmation_gain >= CFG.min_confirmation_gain,
                   float(folds.tp_delta.min()) >= CFG.max_worst_fold_loss,
                   metrics['AveragePrecision'] - base_ap >= CFG.min_ap_delta,
                   oof_churn <= CFG.max_oof_churn,
                   (metrics['fraud_captured'] - base_metrics['fraud_captured']) / max(1, one_way) * 100 >= CFG.min_gain_per_100_swaps,
                   test_churn <= min(CFG.max_test_churn, 1.25 * oof_churn + 0.015)]
    row = {**metrics, 'family': candidate['family'],
           'primary_NR5_delta': metrics['NormalizedRecall@5%'] - base_nr, 'AP_delta': metrics['AveragePrecision'] - base_ap,
           'fold_wins': int((folds.tp_delta > 0).sum()), 'confirmation_tp_gain': confirmation_gain,
           'worst_fold_tp_delta': float(folds.tp_delta.min()), 'one_way_swaps': one_way, 'oof_churn': oof_churn,
           'overlap': overlap, 'tp_gain_per_100_swaps': (metrics['fraud_captured'] - base_metrics['fraud_captured']) / max(1, one_way) * 100,
           'test_churn': test_churn}
    for i, passed in enumerate(gate_values, 1):
        row[f'gate_{i}'] = bool(passed)
    registry_rows.append(row)
    churn_rows.append({'candidate': candidate['candidate'], 'oof_one_way_swaps': one_way, 'oof_churn': oof_churn, 'test_churn': test_churn})
REGISTRY = pd.DataFrame(registry_rows)
FOLD_DELTAS = pd.DataFrame(fold_delta_rows)
TEST_CHURN = pd.DataFrame(churn_rows)
REGISTRY.to_csv(OUTPUT_DIR / 'v9_candidate_registry.csv', index=False)
FOLD_DELTAS.to_csv(OUTPUT_DIR / 'v9_primary_fold_deltas.csv', index=False)
TEST_CHURN.to_csv(OUTPUT_DIR / 'v9_test_churn.csv', index=False)
print(REGISTRY[['candidate'] + [f'gate_{i}' for i in range(1, 9)]].to_string(index=False))
'''

# ---- Cell 17: selection, calibration, submission, diagnostics ----
c17 = '''def probability_logit(values):
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

GATE_COLUMNS = [f'gate_{i}' for i in range(1, 9)]
challenger_rows = REGISTRY[REGISTRY.family.isin(['seed-mean', 'interact'])]
eligible = challenger_rows[challenger_rows[GATE_COLUMNS].all(axis=1)].copy()
complexity = {'seed-mean': 0, 'interact': 1}
if len(eligible):
    eligible = eligible.assign(_complexity=eligible.family.map(complexity))
    selected_name = eligible.sort_values(['confirmation_tp_gain', 'fraud_captured', 'tp_gain_per_100_swaps', 'test_churn', '_complexity'], ascending=[False, False, False, True, True]).iloc[0]['candidate']
    selection_reason = 'eligible challenger selected by predeclared confirmation, gain, efficiency, churn, complexity order'
else:
    selected_name = 'V3 control'
    selection_reason = 'no challenger passed all eight promotion gates; V3 control selected'
selected_row = REGISTRY.loc[REGISTRY.candidate.eq(selected_name)].iloc[0]
challenger_failed_gates = {row['candidate']: [gate for gate in GATE_COLUMNS if not bool(row[gate])] for row in challenger_rows.to_dict('records')}
failed_gate_counts = {gate: int((~challenger_rows[gate]).sum()) for gate in GATE_COLUMNS}
assert selected_name in set(REGISTRY.candidate)
selected_score = score_candidate(next(c for c in CANDIDATES if c['candidate'] == selected_name), 'oof')
selected_test_score = score_candidate(next(c for c in CANDIDATES if c['candidate'] == selected_name), 'test')

def select_candidate(registry):
    eligible = registry[(registry.family.isin(['seed-mean', 'interact'])) & registry[GATE_COLUMNS].all(axis=1)]
    if eligible.empty:
        return 'V3 control'
    eligible = eligible.assign(_complexity=eligible.family.map(complexity))
    return eligible.sort_values(['confirmation_tp_gain', 'fraud_captured', 'tp_gain_per_100_swaps', 'test_churn', '_complexity'], ascending=[False, False, False, True, True]).iloc[0].candidate

assert select_candidate(REGISTRY) == select_candidate(REGISTRY.copy())
assert select_candidate(REGISTRY.assign(**{c: False for c in GATE_COLUMNS})) == 'V3 control'
print('Behavior 9 deterministic promotion and V3 fallback: PASS')
if selected_name == 'V3 control':
    print('No eligible challenger. Challenger failed-gate counts:', failed_gate_counts)
    print('Candidate-specific failed gates:', challenger_failed_gates)

calibrator = LogisticRegression(C=1.0, max_iter=CFG.logistic_max_iter, solver='lbfgs', random_state=CFG.seed)
calibrator.fit(probability_logit(np.clip(selected_score, 1e-6, 1-1e-6)).reshape(-1, 1), y)
final_probability = np.clip(calibrator.predict_proba(probability_logit(np.clip(selected_test_score, 1e-6, 1-1e-6)).reshape(-1, 1))[:, 1], 0, 1)
assert np.array_equal(np.argsort(-selected_test_score, kind='mergesort'), np.argsort(-final_probability, kind='mergesort'))

submission = pd.DataFrame({ID_COL: sample_submission[ID_COL].astype(str), 'fraud_probability': final_probability})
assert submission[ID_COL].tolist() == sample_submission[ID_COL].astype(str).tolist()
assert not submission[ID_COL].duplicated().any() and np.isfinite(submission.fraud_probability).all() and submission.fraud_probability.between(0, 1).all()
submission.to_csv(SUBMISSION_PATH, index=False)
submission.to_csv(OUTPUT_DIR / 'submission_v9.csv', index=False)
np.savez_compressed(OUTPUT_DIR / 'v9_scores.npz', selected_oof=selected_score.astype(np.float32), selected_test=selected_test_score.astype(np.float32),
                    calibrated_test=final_probability.astype(np.float32), v3_oof=V3_OOF_SCORE.astype(np.float32), v3_test=V3_TEST_SCORE.astype(np.float32))
selected_config = {'selected_candidate': selected_name, 'selection_reason': selection_reason, 'mode': RUNTIME_LABEL, 'official_data_only': True,
                   'config': asdict(CFG), 'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'gpus': GPU_COUNT, 'catboost_devices': CAT_DEVICES or 'CPU', 'hardware': HARDWARE},
                   'primary_metrics': selected_row.to_dict(), 'promotion_gates': {gate: bool(selected_row[gate]) for gate in GATE_COLUMNS},
                   'challenger_failed_gate_counts': failed_gate_counts, 'challenger_failed_gates': challenger_failed_gates,
                   'artifact_provenance': ARTIFACT_PROVENANCE, 'claim_id_predictive_feature': False, 'row_position_predictive_feature': False,
                   'calibration': 'post-ranking monotone sigmoid', 'candidate_count': len(CANDIDATES), 'non_comparable_fast_mode': bool(V9_SMOKE),
                   'deviation_from_v8': 'gates 9-10 (transport stress) removed: V9 has no reranker whose transport needs measuring; KDKC exposure covered by fold diagnostic below'}
with open(OUTPUT_DIR / 'selected_config.json', 'w') as handle:
    json.dump(selected_config, handle, indent=2, default=str)
manifest = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'mode': RUNTIME_LABEL, 'files': sorted(p.name for p in OUTPUT_DIR.iterdir()),
            'artifact_provenance': ARTIFACT_PROVENANCE, 'official_data_only': True}
with open(OUTPUT_DIR / 'artifact_manifest.json', 'w') as handle:
    json.dump(manifest, handle, indent=2)
assert list(pd.read_csv(OUTPUT_DIR / 'submission_v9.csv').columns) == [ID_COL, 'fraud_probability']
assert len(pd.read_csv(OUTPUT_DIR / 'submission_v9.csv')) == len(sample_submission)
print('Behavior 10 submission schema/order/uniqueness/range/finiteness: PASS')
print('Fold diagnostic (report-only, no gating power):')
print(FOLD_DELTAS.pivot_table(index='fold', columns='candidate', values='tp_delta', aggfunc='first').to_string())
print(train_clean.assign(fold=FOLD_ID).groupby('fold').agg(n=('label', 'size'), fraud_rate=('label', 'mean')).to_string())
print(json.dumps({'selected_candidate': selected_name, 'mode': RUNTIME_LABEL, 'submission': str(SUBMISSION_PATH),
                  'output_dir': str(OUTPUT_DIR), 'registry_rows': len(REGISTRY)}, indent=2))
'''

cells = [
    md('# DAC 2026 V9 Representation'),
    code(c01),
    md('## 1. Metrics and neutral top-k selection'),
    code(SRC[3]),
    md('## 2. Data discovery and deterministic feature package'),
    code(SRC[5] + "\ndel train_raw, test_raw  # V9: trim peak RAM (recomputable from CSVs)\nimport gc as _gc; _gc.collect()\n"),
    md('## 3. Grouped validation (artifact folds authoritative)'),
    code(c07),
    md('## 4. Artifact-first V8 checkpoint'),
    code(c09),
    md('## 5. V9 representation features'),
    code(c11),
    md('## 6. Three-arm training roster'),
    code(c13),
    md('## 7. Four-candidate registry and gates'),
    code(c15),
    md('## 8. Selection, calibration, submission, diagnostics'),
    code(c17),
]
nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'notebooks/V9_Representation.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('wrote', out, len(cells), 'cells')
