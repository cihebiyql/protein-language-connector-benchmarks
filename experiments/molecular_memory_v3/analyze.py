"""Recompute all published V3 baseline metrics without models or third-party packages."""
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from src.mm_evaluate import regression, generation_metrics, numeric_metrics, c1_metrics, summarize_runs


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def main():
    manifest = load(HERE / 'publication_manifest.json')
    for item in manifest['records']:
        path = HERE / item['public_relative']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['public_sha256'], str(path)
    n_files = n_predictions = n_metrics = 0
    for family in ('numeric_baselines', 'pooled_numeric_baselines'):
        for directory in sorted((HERE / 'results' / family).iterdir()):
            metrics = load(directory / 'metrics.json')
            scenario = metrics['scenario']
            source = jsonl(REPO / 'data/a1' / scenario / 'samples.jsonl')
            assert metrics['status'] == 'complete'
            acceptance = load(directory / 'acceptance_summary.json')
            assert acceptance['canonical_valid_for_acceptance'] is True
            for method, splits in metrics['models'].items():
                for split in ('validation', 'test'):
                    rows = [row for row in source if row['split'] == split]
                    predictions = jsonl(directory / f'{method}.{split}.predictions.jsonl')
                    assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in predictions]
                    assert all(p['gold_pkd'] == r['pkd'] for r, p in zip(rows, predictions))
                    values = [p['prediction_pkd'] for p in predictions]
                    assert all(type(v) in (int, float) and math.isfinite(v) for v in values)
                    actual = regression(rows, values)
                    expected = splits[split]
                    assert expected['n_total'] == expected['n_finite'] == len(rows) == actual['n']
                    for key, stored in (('mae','micro_MAE'), ('rmse','RMSE'), ('spearman','global_Spearman'), ('target_macro_mae','exact_target_macro_MAE')):
                        a, b = actual[key], expected[stored]
                        assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)), (method, split, key, a, b)
                        n_metrics += 1
                    n_files += 1
                    n_predictions += len(rows)
    remote = load(HERE / 'results/remote_queue_status.json')
    formal = HERE / 'results/formal'
    assert remote['completed_cases'] == 32 and remote['failed_cases'] == 4
    for item in load(formal / 'publication_manifest.json')['records']:
        path = HERE / item['public_relative']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['public_sha256'], str(path)
    audit = load(formal / 'independent_audit.json')
    assert len(audit['records']) == 36
    accepted, formal_rows = [], 0
    for case in audit['records']:
        directory = formal / 'cases' / case['case_id']
        if not case['canonical_accepted']:
            assert case['queue_status'] == 'failed' and (directory / 'failure.json').is_file()
            assert not (directory / 'result.json').exists()
            continue
        saved = load(directory / 'result.json')
        assert saved == case['evaluation']
        source = jsonl(REPO / 'data/a1' / saved['scenario'] / 'samples.jsonl')
        rows = [row for row in source if row['split'] == 'test']
        assert generation_metrics(rows,jsonl(directory/'test_lm_predictions.jsonl')) == saved['LM']
        assert numeric_metrics(rows,jsonl(directory/'test_numeric_predictions.jsonl')) == saved['numeric']
        assert c1_metrics(rows,jsonl(directory/'c1_predictions.jsonl')) == saved['C1']
        assert load(directory/'acceptance_summary.json')['current_remote_revalidation'] is True
        accepted.append(saved);formal_rows += len(rows)
    assert len(accepted) == 32 and formal_rows == 9794
    assert summarize_runs(accepted,audit['summary']['protocol_sha256']) == load(formal/'formal_summary.json')
    tensors_checked = 0
    import struct
    tensor_index = load(HERE / 'checkpoints/index.json')
    assert len(tensor_index['entries']) == 32
    for entry in tensor_index['entries']:
        data = (HERE/'checkpoints'/entry['filename']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        length = struct.unpack('<Q',data[:8])[0]
        header = json.loads(data[8:8+length])
        for name, expected in entry['tensor_value_sha256'].items():
            info = header[name]
            assert info['dtype'] == 'F32'
            a,b = info['data_offsets']
            assert hashlib.sha256(data[8+length+a:8+length+b]).hexdigest() == expected
            tensors_checked += 1
    print(json.dumps(dict(status='validated', manifest_files=len(manifest['records']),
        prediction_files=n_files, predictions=n_predictions, reconstructed_metrics=n_metrics,
        formal_accepted=len(accepted),formal_failed=4,formal_TEST_rows_per_channel=formal_rows,
        tensor_values_checked=tensors_checked,
        remote_formal_status=remote['status'], model_or_GPU_used=False), indent=2))


if __name__ == '__main__':
    main()
