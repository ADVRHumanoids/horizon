#!/usr/bin/env python3
"""Compare matching Spot benchmark runs, including physics and trajectories."""
import argparse
import csv
import json
import math
from pathlib import Path


def physical(sample):
    upper = dict(position_error_m=.003, rotation_error_rad=.006,
                 foot_drift_m=.001, foot_velocity_m_s=.003,
                 newton_euler_norm=.005, stabilized_contact_acceleration=1e-4,
                 dynamics_gap=1e-5, quaternion_error=1e-12,
                 endpoint_velocity=1e-5, peak_torque_Nm=1000)
    lower = dict(min_normal_force_N=1, min_friction_margin_N=0,
                 joint_limit_margin_rad=0, squat_depth_m=.05)
    return (sample['converged']
            and all(math.isfinite(v) for v in sample.values())
            and sample['objective'] < .02 * sample['initial_objective']
            and all(sample[k] < v for k, v in upper.items())
            and all(sample[k] > v for k, v in lower.items()))


def traces(before, after):
    with open(before) as f:
        a = list(csv.DictReader(f))
    with open(after) as f:
        b = list(csv.DictReader(f))
    if len(a) != len(b) or not a or a[0].keys() != b[0].keys():
        raise ValueError('trajectory shapes differ')
    delta = dict(state=0., acceleration=0., force=0., torque=0.)
    for x, y in zip(a, b):
        if x['time_s'] != y['time_s']:
            raise ValueError('trajectory time grids differ')
        for key in x:
            if key == 'time_s':
                continue
            if not x[key] and not y[key]:
                continue  # terminal row has no controls
            first, second = float(x[key]), float(y[key])
            if not math.isfinite(first) or not math.isfinite(second):
                raise ValueError('nonfinite trajectory')
            group = ('torque' if key.startswith('tau') else
                     'force' if key.startswith('f') else
                     'acceleration' if key.startswith('a') else 'state')
            delta[group] = max(delta[group], abs(first-second))
    return delta


def compare(before, after, trace_tolerance=1e-6):
    if not math.isfinite(trace_tolerance) or trace_tolerance < 0:
        raise ValueError('invalid trajectory tolerance')
    for key in ['cpu', 'compiler', 'build_type', 'profiling_enabled',
                'duration_s', 'integration', 'regularization', 'solver',
                'contact', 'samples', 'warmup', 'function_samples']:
        if before.get(key) != after.get(key):
            raise ValueError(f'benchmark protocols differ: {key}')
    if not before['passed'] or not after['passed']:
        raise ValueError('a benchmark failed its validation gate')
    def cases(report):
        result = {}
        for c in report['cases']:
            key = (c['nodes'], c['threads'], c['formulation'])
            if key in result or c.get('status', 'completed') != 'completed':
                raise ValueError('duplicate or incomplete benchmark case')
            if len(c['quality_samples']) != report['samples']:
                raise ValueError('incomplete measured sample series')
            result[key] = c
        return result
    old, new = cases(before), cases(after)
    if old.keys() != new.keys():
        raise ValueError('benchmark configurations differ')
    rows = []
    for key in sorted(old):
        a, b = old[key], new[key]
        if not all(physical(q) for c in [a, b] for q in c['quality_samples']):
            raise ValueError(f'physical validation failed: {key}')
        delta = traces(a['trajectory_csv'], b['trajectory_csv'])
        if max(delta.values()) > trace_tolerance:
            raise ValueError(f'trajectories changed: {key}: {delta}')
        cost_delta = max(abs(x['objective']-y['objective'])
                         for x, y in zip(a['quality_samples'], b['quality_samples']))
        if cost_delta > 1e-10:
            raise ValueError(f'objective changed: {key}: {cost_delta}')
        def lin(c):
            phase = c.get('phase_us', {}).get('linearize_quadratize')
            return phase['total']/1000/c['iterations']['total'] if phase else None
        first_lin, second_lin = lin(a), lin(b)
        rows.append(dict(N=key[0], workers=key[1], formulation=key[2],
                         before_ms=a['solve_ms']['median'], after_ms=b['solve_ms']['median'],
                         solve_speedup=a['solve_ms']['median']/b['solve_ms']['median'],
                         before_iterations=a['iterations']['median'], after_iterations=b['iterations']['median'],
                         before_linearization_ms=first_lin, after_linearization_ms=second_lin,
                         linearization_speedup=first_lin/second_lin if first_lin and second_lin else None,
                         objective_difference=cost_delta,
                         **{f'{group}_max_difference': v for group, v in delta.items()}))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    rows = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()), args.trace_tolerance)
    result = dict(passed=True, before=str(args.before), after=str(args.after), cases=rows)
    args.output.write_text(json.dumps(result, indent=2))
    with args.output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        phase = (f"linearization {row['linearization_speedup']:.2f}x"
                 if row['linearization_speedup'] else 'phase profiling disabled')
        print(f"N={row['N']} workers={row['workers']} {row['formulation']}: "
              f"solve {row['solve_speedup']:.2f}x; {phase}")
