"""Paired OOF comparison, resampling whole fields with all their seasons."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from diagnostics import write_json
from eval import evaluate_predictions


def align_predictions(a, b):
    evaluate_predictions(a)
    evaluate_predictions(b)
    if 'cv_fold' not in a or 'cv_fold' not in b:
        raise ValueError('Paired comparison requires cv_fold assignments')
    if set(a['index']) != set(b['index']):
        raise ValueError('Pixel identities differ; compare identical target coverage')
    a = a.sort_values('index').reset_index(drop=True)
    b = b.set_index('index').loc[a['index']].reset_index()
    for key in ('field_shared_name', 'year', 'cv_fold'):
        if not np.array_equal(a[key].astype(str), b[key].astype(str)):
            raise ValueError(f'Paired predictions have different {key}')
    if not np.allclose(a.target, b.target, rtol=0, atol=1e-7):
        raise ValueError('Paired predictions have different targets')
    a = a.astype({'target': 'float64', 'prediction': 'float64'})
    a['prediction_b'] = b.prediction.to_numpy(dtype=np.float64)
    return a


def sufficient_statistics(frame):
    y, a, b = (frame[key].to_numpy(dtype=np.float64) for key in ('target', 'prediction', 'prediction_b'))
    stats = pd.DataFrame(dict(n=np.ones(len(y)), sy=y, yy=y*y,
                             sa=a, aa=a*a, ya=y*a, ea=(a-y)**2,
                             sb=b, bb=b*b, yb=y*b, eb=(b-y)**2,
                             field=frame.field_shared_name.to_numpy()))
    return stats.groupby('field', sort=True).sum()


def scores(s, suffix):
    n = s['n']
    vy = max(0., s['yy'] - s['sy'] ** 2 / n)
    vp = max(0., s[suffix*2] - s['s'+suffix] ** 2 / n)
    mse = s['e'+suffix] / n
    return dict(mse=mse, rmse=np.sqrt(mse), r2=1-s['e'+suffix]/vy if vy > 1e-12 else np.nan,
                pearson=(s['y'+suffix]-s['sy']*s['s'+suffix]/n)/np.sqrt(vy*vp) if vy*vp > 1e-12 else np.nan)


def paired_bootstrap(a, b, repetitions=2000, seed=0):
    if repetitions < 1:
        raise ValueError('Bootstrap repetitions must be positive')
    paired = align_predictions(a, b)
    fields = paired.groupby(['field_shared_name', 'year'], as_index=False)[['target', 'prediction', 'prediction_b']].mean()
    report = {'difference': 'B minus A; negative MSE/RMSE favors B, positive R2/Pearson favors B',
              'seed': seed, 'repetitions': repetitions, 'cluster': 'field_shared_name (all years together)',
              'interpretation': 'Conditional on these fitted OOF predictions; does not include retraining/seed uncertainty'}
    for level, frame in (('pixel', paired), ('field_year', fields)):
        table = sufficient_statistics(frame)
        if len(table) < 2:
            raise ValueError('Field bootstrap needs at least two distinct fields')
        report['clusters'] = len(table)
        report['small_cluster_warning'] = len(table) < 10
        names = list(table.columns)
        values = table.to_numpy()
        total = dict(zip(names, values.sum(axis=0)))
        original_a, original_b = scores(total, 'a'), scores(total, 'b')
        rng = np.random.default_rng(seed)
        differences = {key: [] for key in original_a}
        for _ in range(repetitions):
            draw = rng.integers(0, len(table), size=len(table))
            s = dict(zip(names, values[draw].sum(axis=0)))
            sa, sb = scores(s, 'a'), scores(s, 'b')
            for key in differences:
                differences[key].append(sb[key] - sa[key])
        report[level] = {}
        for key, draws in differences.items():
            valid = np.asarray(draws)[np.isfinite(draws)]
            estimate = original_b[key] - original_a[key]
            report[level][key] = dict(difference=float(estimate) if np.isfinite(estimate) else None,
                ci95=np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) else None,
                defined_replicates=len(valid))
    return report


def read_run(path):
    path = Path(path)
    csv_path = path / 'cv_predictions.csv' if path.is_dir() else path
    config_path = csv_path.parent / 'config.json'
    config = json.loads(config_path.read_text()) if config_path.exists() else None
    return pd.read_csv(csv_path, dtype={'index': str, 'field_shared_name': str}), config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', type=Path, required=True, help='Run directory or OOF CSV')
    parser.add_argument('--b', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('comparison.json'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--allow-unverified-protocol', action='store_true', help='External predictions without input provenance')
    args = parser.parse_args()
    a, config_a = read_run(args.a)
    b, config_b = read_run(args.b)
    provenance = all(config and config.get('input_fingerprint') for config in (config_a, config_b))
    if not provenance and not args.allow_unverified_protocol:
        parser.error('Missing input provenance; verify external preprocessing before using --allow-unverified-protocol')
    report = paired_bootstrap(a, b, args.bootstrap, args.seed)
    report['a_metrics'] = evaluate_predictions(a)[0]
    report['b_metrics'] = evaluate_predictions(b)[0]
    keys = ('input_fingerprint', 'representation', 'actual_time_steps', 'bands', 'normalize',
            'append_mask', 'missing_value', 'cloud_mask', 'scl_exclude')
    report['input_protocol_differences'] = {key: [config_a.get(key), config_b.get(key)] for key in keys
        if provenance and config_a.get(key) != config_b.get(key)}
    report['architecture_only_input_match'] = bool(provenance and not report['input_protocol_differences'])
    report['protocol_note'] = ('Input mismatch/unverified: do not attribute differences solely to architecture'
        if not report['architecture_only_input_match'] else 'Inputs match; also inspect optimizer, stopping, seed and parameter counts')
    report['config_a'], report['config_b'] = config_a, config_b
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(report['protocol_note'])
    print(f'Saved paired comparison to {args.output}')


if __name__ == '__main__':
    main()
