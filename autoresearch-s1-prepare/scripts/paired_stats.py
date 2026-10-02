"""Recompute paired statistics; arithmetic/evidence integrity is not a QA verdict."""
import argparse
import statistics
from pathlib import Path
from common import atomic_json, evidence, load, number


def analyze(doc, root):
    direction = doc['direction']
    mode = doc['evaluation_mode']
    if direction not in ('maximize', 'minimize') or mode not in ('stochastic', 'deterministic'):
        raise ValueError('unknown direction or evaluation_mode')
    seeds, pairs = doc['formal_seeds'], doc['paired_runs']
    if not seeds or any(type(s) not in (int, str) for s in seeds):
        raise ValueError('formal_seeds must be nonempty integers/strings')
    if len({str(s) for s in seeds}) != len(seeds):
        raise ValueError('duplicate formal seed')
    if len(pairs) != len(seeds) or {str(p['seed']) for p in pairs} != {str(s) for s in seeds}:
        raise ValueError('missing, extra or duplicated formal seed')
    failures, b, r, per_seed = [], [], [], []
    for pair in pairs:
        x, y = number(pair['baseline'], 'baseline'), number(pair['reference'], 'reference')
        b.append(x); r.append(y)
        for role in ('baseline', 'reference'):
            artifact = pair[role + '_evidence']
            path = evidence(root, artifact)
            data = load(path)
            cursor = data
            for key in artifact['metric_path']:
                cursor = cursor[key]
            if number(cursor, role + ' source metric') != pair[role]:
                raise ValueError('source metric disagrees with supplied value')
        if pair.get('quality_valid') is not True:
            failures.append('QUALITY_INVALID:' + str(pair['seed']))
        delta = y - x if direction == 'maximize' else x - y
        per_seed.append({'seed': pair['seed'], 'baseline': x, 'reference': y, 'improvement': delta})
    bm, rm = statistics.mean(b), statistics.mean(r)
    upper = number(doc['upper_bound'], 'upper_bound')
    denominator = upper - bm if direction == 'maximize' else bm - upper
    if denominator <= 0:
        raise ValueError('upper bound is not in the improvement direction')
    delta = rm - bm if direction == 'maximize' else bm - rm
    score = delta / denominator
    sigma = statistics.stdev(b) if len(b) > 1 else None
    if doc.get('same_protocol') is not True: failures.append('PROTOCOL_NOT_CONFIRMED')
    if not str(doc.get('upper_bound_basis', '')).strip(): failures.append('UPPER_BOUND_BASIS_MISSING')
    if delta <= 0: failures.append('NO_POSITIVE_IMPROVEMENT')
    if not 0.15 <= score <= 0.8: failures.append('NORMALIZED_OUT_OF_RANGE')
    if mode == 'stochastic':
        if len(b) < 3: failures.append('FEWER_THAN_THREE_RUNS')
        if sigma is None or delta < 3 * sigma: failures.append('BELOW_3_SIGMA')
    elif not str(doc.get('determinism_basis', '')).strip():
        failures.append('DETERMINISM_BASIS_MISSING')
    minimum = doc.get('absolute_min_improvement')
    if minimum is not None:
        if number(minimum, 'minimum improvement') < 0 or not doc.get('absolute_threshold_basis'):
            raise ValueError('absolute threshold requires nonnegative value and prior basis')
        if delta < minimum: failures.append('BELOW_DECLARED_ABSOLUTE_THRESHOLD')
    strength = 'not_applicable'
    if mode == 'stochastic' and sigma is not None:
        strength = ('zero_observed_variance' if sigma == 0 else
                    'strong_5sigma' if delta >= 5 * sigma else
                    'acceptable_3sigma' if delta >= 3 * sigma else 'insufficient')
    return {'status': 'NUMERIC_CHECKS_PASS' if not failures else 'FAIL',
            'paired_runs': per_seed, 'baseline_mean': bm, 'reference_mean': rm,
            'baseline_sample_std': sigma, 'reference_sample_std': statistics.stdev(r) if len(r)>1 else None,
            'improvement': delta, 'normalized_reference': score, 'evidence_strength': strength,
            'headroom': abs(upper-rm), 'headroom_below_3sigma': sigma is not None and abs(upper-rm)<3*sigma,
            'failures': failures,
            'boundary': 'Source hashes and metric values verified. Fairness, independence, source truth and U legitimacy need semantic QA.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path); p.add_argument('--root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args=p.parse_args()
    try:
        report=analyze(load(args.input), args.root); atomic_json(args.out, report)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.exit(2, 'Invalid input: '+str(exc)+'\n')
    print(report['status'])
    return 0 if report['status']=='NUMERIC_CHECKS_PASS' else 1


if __name__=='__main__': raise SystemExit(main())
