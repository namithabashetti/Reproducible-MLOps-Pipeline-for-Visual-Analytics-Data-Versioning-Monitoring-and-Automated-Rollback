"""Validate Compose/Prometheus and exercise dashboard queries with active/idle traffic."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    docker = str(Path(os.environ['LOCALAPPDATA']) / 'Programs/DockerDesktop/resources/bin/docker.exe')
    image = yaml.safe_load((ROOT / 'compose.monitoring.yaml').read_text())['services']['prometheus']['image']
    subprocess.run([docker, 'compose', '-f', str(ROOT / 'compose.monitoring.yaml'), 'config', '--quiet'], check=True)
    panels = {p['title']: p['targets'][0]['expr'] for p in json.loads((ROOT / 'configs/monitoring/grafana/dashboard.json').read_text())['panels']}
    series = []
    for deployment, growth in [('active', 10), ('idle', 0)]:
        for outcome in ('success', 'server_error'):
            delta = growth if outcome == 'success' else 0
            labels = f'job="visual-analytics-api",model_version="1",deployment_id="{deployment}",outcome="{outcome}"'
            for metric in ('va_prediction_requests_total', 'va_prediction_duration_seconds_count'):
                series.append({'series': f'{metric}{{{labels}}}', 'values': f'0+{delta}x12'})
            for bound in ('.1', '+Inf'):
                series.append({'series': f'va_prediction_duration_seconds_bucket{{{labels},le="{bound}"}}', 'values': f'0+{delta}x12'})
    expected = '{model_version="1",deployment_id="active"}'
    fixture = {'evaluation_interval': '10s', 'tests': [{'interval': '10s', 'input_series': series,
        'promql_expr_test': [
            {'expr': panels['Server error fraction (client errors excluded)'], 'eval_time': '2m',
             'exp_samples': [{'labels': expected, 'value': 0}]},
            {'expr': panels['Approximate p95 handling latency'], 'eval_time': '2m',
             'exp_samples': [{'labels': expected, 'value': .095}]}]}]}
    with tempfile.TemporaryDirectory(prefix='visual-analytics-promql-') as directory:
        directory = Path(directory)
        (directory / 'prometheus.yaml').write_bytes((ROOT / 'configs/monitoring/prometheus.yaml').read_bytes())
        (directory / 'queries.yaml').write_text(yaml.safe_dump(fixture))
        prefix = [docker, 'run', '--rm', '--network', 'none', '--entrypoint', 'promtool',
                  '--mount', f'type=bind,source={directory},target=/checks,readonly', image]
        subprocess.run([*prefix, 'check', 'config', '/checks/prometheus.yaml'], check=True)
        subprocess.run([*prefix, 'test', 'rules', '/checks/queries.yaml'], check=True)
    print('Monitoring configuration and active/idle dashboard query regression passed.')


if __name__ == '__main__':
    main()
