"""Score every archived formal unit, retaining frozen three-valued semantics."""
import argparse
import collections
import datetime
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.external_baselines.experiment import read, unit_path, verify_freeze
from scripts.external_baselines.common import file_digest
from scripts.external_baselines.resource_control import verify_batch_freeze
from scripts.run_external_baselines import make_request


def lines(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'evaluation/results/external-formal-early3-v2_2-20261003-v1/scoring_v2_2.py'
    spec = importlib.util.spec_from_file_location('offline_scoring_v2_2', source)
    scoring = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scoring)
    (out / 'scoring_v2_2.py.txt').write_bytes(source.read_bytes())
    (out / 'scoring_entry.py.txt').write_bytes(Path(__file__).read_bytes())
    freeze_path = ROOT / 'evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json'
    freeze = read(freeze_path)
    batch = ROOT / 'evaluation/results/external-formal-test150-20261002-v1'
    control = ROOT / 'evaluation/results/external-formal-control-20261002-v1/resume-independent-20261003-v1'
    exit_record = read(control / 'launcher-exit.json')
    assert exit_record['exit_code'] == 0
    verify_freeze(freeze, ROOT)
    verify_batch_freeze(freeze, freeze_path, read(batch / 'manifest.start.json'), ROOT)
    items = read(ROOT / freeze['dataset']['path'])['items']
    review_dir = ROOT / 'evaluation/results/external-formal-early3-semantic-review-20261003-v1'
    reviews = {}
    review_files = [review_dir / 'rejection_reviews.jsonl'] + sorted(review_dir.glob('execution_reviews_*.jsonl'))
    for path in review_files:
        for record in lines(path):
            assert record['review_id'] not in reviews
            reviews[record['review_id']] = record
    hashes = {}
    counts = collections.Counter()
    used = 0
    summaries = []
    protected = {str(p): file_digest(p) for p in [freeze_path, ROOT / 'paper/bioagent_main.tex',
                                               ROOT / 'paper/bioagent_supplement.tex']}
    with (out / 'scores.jsonl').open('x', encoding='utf-8') as sink, \
         (out / 'semantic_review_queue.jsonl').open('x', encoding='utf-8') as pending:
        for index, unit in enumerate(freeze['matrix']):
            item = items[unit['question_index']]
            path = unit_path(batch, unit)
            base = {**unit, 'question_id': item['id'], 'expected_behavior': item['gold']['expected_behavior'],
                    'archive_path': str(path.relative_to(ROOT))}
            if not (path / 'result.json').exists():
                assert (path / 'claim.json').exists(), 'Batch has unstarted units'
                score = {**base, 'scoring_status': 'historical_interruption',
                         'scientific_task_success': None, 'identity_correct': None,
                         'scientific_output_correct': None, 'execution_contract_correct': None,
                         'final_answer_correct': None, 'rejection_response_correct': None}
                counts['historical_interruption'] += 1
            else:
                row = read(path / 'result.json')
                events = list(lines(path / 'events.jsonl'))
                assert row['unit'] == unit
                bounds = [e for e in events if e['kind'] in {'run_start', 'run_complete'}]
                assert len(bounds) == 2 and bounds[0]['unit'] == unit and bounds[1]['summary'] == row
                _, files = make_request(item)
                score = scoring.score_run(item, row, events, files)
                rid = 'review_' + score['evidence_sha256'][:20]
                review = reviews.get(rid)
                if review:
                    score = scoring.score_run(item, row, events, files, review)
                    used += 1
                score.update(base)
                score['scoring_status'] = 'assistant_reviewed' if review else 'automatic'
                score['latency_ms'] = row['result'].get('latency_ms')
                score['llm_requests'] = sum(e['kind'] == 'llm_request' for e in events)
                score['llm_responses'] = sum(e['kind'] == 'llm_response' for e in events)
                score['input_tokens'] = row['usage'].get('input_tokens')
                score['output_tokens'] = row['usage'].get('output_tokens')
                score['usage_complete'] = (score['llm_requests'] == score['llm_responses'] and
                                           row['result'].get('usage_complete', True))
                score['http_tool_requests'] = sum(e['kind'] == 'tool_call' and e.get('http_requested') is True for e in events)
                score['native_recovery'] = row['result'].get('native_recovery')
                score['native_execution_summary'] = row['result'].get('native_execution_summary')
                hashes[str(path.relative_to(batch))] = {n: file_digest(path / n) for n in ['result.json', 'events.jsonl']}
                if score['scientific_task_success'] is None:
                    packet = scoring.review_packet(item, row, events, files, score)
                    packet.update(unit_id=unit['unit_id'], archive_path=base['archive_path'],
                                  reviewer_blinding_claimed=False)
                    pending.write(json.dumps(packet, ensure_ascii=False) + '\n')
                    counts['pending_semantic_label'] += 1
                counts['recorded'] += 1
            sink.write(json.dumps(score, ensure_ascii=False, allow_nan=False) + '\n')
            summaries.append({k: score.get(k) for k in ['condition', 'model', 'expected_behavior',
                'scientific_task_success', 'scientific_output_correct', 'final_answer_correct',
                'execution_contract_correct', 'rejection_response_correct']})
            if (index + 1) % 250 == 0:
                print('SCORED', index + 1, flush=True)
    assert used == len(reviews) == 674
    assert counts['recorded'] == 4492 and counts['historical_interruption'] == 8
    for relative, record in hashes.items():
        for name, sha in record.items():
            assert file_digest(batch / relative / name) == sha
    assert protected == {p: file_digest(Path(p)) for p in protected}
    summary = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'all_planned_units': 4500, 'counts': dict(counts), 'scorer_version': scoring.VERSION,
        'original_frozen_scorer_retained': True, 'schema_correction_preserved': True,
        'assistant_reviews_applied': used, 'human_reviews_claimed': 0,
        'raw_archive_hashes': hashes, 'protected_file_hashes': protected,
        'freeze_sha256': file_digest(freeze_path), 'schema_scorer_sha256': file_digest(source),
        'review_file_sha256': {str(p): file_digest(p) for p in review_files},
        'frozen_execution_code_files_verified': len(freeze['code_sha256']),
        'experiment_exit': exit_record, 'no_new_api_or_scientific_tool_calls': True}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', out, dict(counts), flush=True)


if __name__ == '__main__':
    main()
