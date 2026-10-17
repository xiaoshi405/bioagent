"""Analyze archived execution outcomes and paired comparisons."""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.external_baselines.experiment import read, unit_path
from scripts.external_baselines.common import file_digest, REGISTRY
from scripts.external_baselines.statistics import rates, paired_effect
from scripts.external_baselines.resource_control import verify_batch_freeze

VERSION = 'manuscript-operational-v1.4-original-safe-stop-definition'
FREEZE = ROOT / 'evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json'
BATCH = ROOT / 'evaluation/results/external-formal-test150-20261002-v1'
METHODS = ('ReAct', 'CodeAct', 'SciToolAgent', 'Biomni-A1', 'F1')
NORMAL = {'answered', 'answer_delivered', 'rejected', 'native_incomplete'}
ERROR_RE = re.compile(r'(?m)(?:^\s*(?:Error:|Execution environment error:|Traceback)|'
                      r'\b(?:SyntaxError|NameError|KeyError|ValueError|TypeError|ImportError|'
                      r'ModuleNotFoundError|FileNotFoundError|AttributeError|IndexError):)')
INFRA_CALLS = {'print', 'str', 'repr', 'dict', 'list', 'tuple', 'set', 'len', 'range',
              'enumerate', 'zip', 'sorted', 'isinstance', 'type', 'int', 'float', 'bool',
              'open', 'read_input_file', 'read', 'read_text', 'readlines', 'readline',
              'close', 'get', 'keys', 'values', 'items', 'strip', 'rstrip', 'lstrip',
              'split', 'splitlines', 'join', 'replace', 'upper', 'lower', 'startswith',
              'endswith', 'append', 'extend', 'format', 'dumps', 'loads'}
INFRA_CALLS |= {'listdir', 'getcwd', 'exists', 'walk', 'glob', 'getsize', 'dir',
                'signature', 'getattr', 'hasattr', 'getsource', 'which', 'load',
                'findall', 'sub', 'add', 'update', 'union', 'all', 'any', 'reverse'}


def intended_registered_calls(code):
    """Read names from failed code for retry matching; never execute that code."""
    cleaned = re.sub(r'^\s*```(?:python)?\s*\n?', '', code)
    cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned)
    cleaned = cleaned.replace('</｜DSML｜>', '')
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return []
    return sorted(name for n in ast.walk(tree) if isinstance(n, ast.Call)
                  for name in [getattr(n.func, 'id', getattr(n.func, 'attr', '?'))]
                  if name in REGISTRY)


def jsonl(path):
    with path.open(encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def digest_records(row, events):
    payload = json.dumps({'row': row, 'events': events}, sort_keys=True,
                         ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def cell_info(start, end):
    """Static screening is only triage: it does not certify Python equivalence."""
    code = start.get('code', '')
    observation = end.get('observation') if end else None
    error = (not end or end.get('status') != 'returned' or
             bool(ERROR_RE.search(str(observation))))
    compile_failed = False
    try:
        tree = ast.parse(code)
        calls = {getattr(n.func, 'id', getattr(n.func, 'attr', '?'))
                 for n in ast.walk(tree) if isinstance(n, ast.Call)}
        # Flag possible direct computation for review; ignore string formatting.
        display_nodes = {id(n) for j in ast.walk(tree) if isinstance(j, ast.JoinedStr)
                         for n in ast.walk(j)}
        numerical = any(isinstance(n, ast.BinOp) and not isinstance(n.op, ast.Add)
                        and id(n) not in display_nodes
                        and not any(isinstance(x, ast.Constant) and isinstance(x.value, str)
                                    for x in (n.left, n.right)) for n in ast.walk(tree))
        unknown = sorted(calls - INFRA_CALLS - set(REGISTRY))
    except SyntaxError:
        compile_failed = True
        calls, numerical, unknown = set(), False, []
    return {'start_event': start['event_index'],
            'end_event': end.get('event_index') if end else None,
            'error': error, 'compile_failed': compile_failed,
            'registered_calls_in_code': sorted(calls & set(REGISTRY)),
            'intended_registered_calls': intended_registered_calls(code),
            'possible_direct_science': bool(unknown or numerical),
            'unknown_calls': unknown}


def score_execution(row, events):
    result = row['result']
    method = row['condition']
    status = result.get('final_status')
    calls = [e for e in events if e['kind'] == 'tool_call']
    requests = [e for e in events if e['kind'] == 'tool_request']
    # Match trace counts: interface rejections do not count as executed steps.
    attempted = max(len(requests), sum(e.get('http_requested') is True for e in calls),
                    result.get('attempted_steps') or 0)
    interface_attempts = sum(e.get('http_requested') is False for e in calls)
    successful = [e for e in calls if e.get('status') == 'success']
    failed = [e for e in calls if e.get('status') != 'success']
    ends = {e['execution_index']: e for e in events if e['kind'] == 'kernel_execution_end'}
    cells = [cell_info(e, ends.get(e['execution_index'])) for e in events
             if e['kind'] == 'kernel_execution_start']
    # A successful retry closes the operation; score identity separately.
    # Keep argument changes for identity review.
    unresolved_failures = [e for e in failed if not any(
        s['event_index'] > e['event_index'] and s.get('tool_name') == e.get('tool_name')
        for s in successful)]
    argument_changed_retry_events = [e['event_index'] for e in failed if any(
        s['event_index'] > e['event_index'] and s.get('tool_name') == e.get('tool_name')
        and s.get('resolved_arguments') != e.get('resolved_arguments') for s in successful)]
    pending_requests = [e for e in requests if not any(
        c.get('call_id') == e.get('call_id') for c in calls)]
    recovered_kernel_errors = []
    kernel_errors = []
    for cell in cells:
        if not cell['error']:
            continue
        names = set(cell['intended_registered_calls'])
        later = collections.Counter(s.get('tool_name') for s in successful
                                    if s['event_index'] > cell['start_event'])
        intended = collections.Counter(cell['intended_registered_calls'])
        if names and all(later[name] >= count for name, count in intended.items()) and not cell['possible_direct_science']:
            # The intended tools succeeded later, or only display failed.
            recovered_kernel_errors.append(cell['start_event'])
        elif not names and not cell['compile_failed'] and not cell['possible_direct_science']:
            # Keep infrastructure/display errors without undoing tool success.
            recovered_kernel_errors.append(cell['start_event'])
        else:
            kernel_errors.append(cell)
    possible_direct = [c for c in cells if c['possible_direct_science'] and not c['compile_failed']]
    scientific_kernel_calls = [c for c in cells if c['registered_calls_in_code'] and not c['compile_failed']]
    normal = status in NORMAL
    reasons = []
    native = result.get('native_execution_summary')
    plans = [e for e in events if e['kind'] == 'native_plan']
    final_declared_plan = plans[-1] if plans else None
    planned_path = final_declared_plan.get('tool_path') if final_declared_plan else None
    after_plan = [e.get('tool_name') for e in successful if final_declared_plan and
                  e['event_index'] > final_declared_plan['event_index']]
    declared_workflow_closed = (method == 'SciToolAgent' and isinstance(planned_path, list)
        and bool(planned_path) and not pending_requests and not unresolved_failures
        and all(after_plan.count(name) >= planned_path.count(name) for name in set(planned_path)))
    if method == 'F1' and isinstance(native, dict) and type(native.get('completed')) is bool:
        completed = native['completed']
        reasons.append('native_executor_summary')
        # Use the executor's completion status.
        if native.get('attempted_steps') != len(calls) or native.get('successful_steps') != len(successful):
            completed = None
            reasons.append('native_summary_trace_count_conflict')
        if completed and (not successful or pending_requests or unresolved_failures):
            completed = None
            reasons.append('native_completed_without_closed_successful_execution')
    elif declared_workflow_closed:
        completed = True
        reasons.append('declared_native_workflow_successfully_executed')
    elif status in {'llm_budget_exhausted', 'output_budget_exhausted', 'tool_budget_exhausted'}:
        completed = False
        reasons.append('budget_terminated_unclosed_workflow')
    elif not normal:
        completed = None
        reasons.append('abnormal_terminal_without_executor_completion_evidence')
    elif not calls and not possible_direct:
        completed = False
        reasons.append('no_successful_scientific_execution')
    elif not successful and calls and not possible_direct:
        completed = False
        reasons.append('all_registered_operations_failed')
    elif pending_requests or unresolved_failures or kernel_errors:
        completed = None
        reasons.append('requires_execution_evidence_review')
    elif successful:
        completed = True
        reasons.append('closed_successful_execution_episode')
        if method == 'SciToolAgent':
            plans = [e for e in events if e['kind'] == 'native_plan']
            plan = plans[-1].get('tool_path') if plans else None
            actual = [e.get('tool_name') for e in successful]
            if not isinstance(plan, list) or not plan:
                completed = None
                reasons.append('missing_nonempty_native_workflow')
            elif any(actual.count(name) < plan.count(name) for name in set(plan)):
                completed = False
                reasons.append('declared_workflow_has_unexecuted_operations')
    else:
        completed = None
        reasons.append('direct_python_execution_requires_review')
    # Rejected/incomplete runs need executor evidence to count as complete.
    if status in {'rejected', 'native_incomplete'} and method != 'F1' and not declared_workflow_closed:
        completed = False
        reasons.append('explicit_workflow_noncompletion')
    zero_science = False if attempted else (None if possible_direct else True)
    return {'completed_execution': completed, 'zero_scientific_operations': zero_science,
            'attempted_steps': attempted, 'successful_tool_steps': len(successful),
            'interface_action_attempts': interface_attempts,
            'failed_tool_steps': len(failed), 'native_status': status,
            'abnormal_terminal': not normal,
            'safe_stop_completion_evidence': normal or isinstance(native, dict),
            'completion_reasons': reasons,
            'tool_evidence': [{'event_index': e['event_index'], 'tool_name': e.get('tool_name'),
                               'status': e.get('status'), 'call_id': e.get('call_id')}
                              for e in calls],
            'unresolved_failed_event_indices': [e['event_index'] for e in unresolved_failures],
            'argument_changed_retry_event_indices': argument_changed_retry_events,
            'recovered_kernel_error_event_indices': recovered_kernel_errors,
            'kernel_cells': cells, 'kernel_scientific_call_candidates': len(scientific_kernel_calls)}


def finish_endpoints(score, expected):
    completed, zero = score['completed_execution'], score['zero_scientific_operations']
    safe = None
    if expected == 'reject':
        safe = False if zero is False or completed is True else (
            True if zero is True and completed is False else None)
    score.update({'execute_task_completion': completed if expected != 'reject' else None,
                  'safe_rejection': safe,
                  'expected_behavior_outcome': safe if expected == 'reject' else completed})
    return score


def csv_write(path, rows, fields):
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reviews', help='Evidence-bound operational review JSONL, optional')
    parser.add_argument('--scientific-source', help='Completed offline scientific-score directory')
    parser.add_argument('--final-batch', action='store_true', help='Classify recorded terminal aborts; missing results stay null')
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'analysis_source.py.txt').write_bytes(Path(__file__).read_bytes())
    frozen = read(FREEZE)
    verify_batch_freeze(frozen, FREEZE, read(BATCH / 'manifest.start.json'), ROOT)
    for relative, sha in frozen['code_sha256'].items():
        assert file_digest(ROOT / relative) == sha, relative
    assert file_digest(ROOT / frozen['dataset']['path']) == frozen['dataset']['raw_sha256']
    for fixture in frozen['dataset']['fixtures']:
        assert file_digest(ROOT / fixture['path']) == fixture['raw_sha256']
    items = read(ROOT / frozen['dataset']['path'])['items']
    protected = [ROOT / 'paper/bioagent_main.tex', ROOT / 'paper/bioagent_supplement.tex', FREEZE]
    before = {str(p): file_digest(p) for p in protected}
    # Snapshot available units before analysis.
    snapshot = [(u, unit_path(BATCH, u), (unit_path(BATCH, u) / 'result.json').exists(),
                 (unit_path(BATCH, u) / 'claim.json').exists()) for u in frozen['matrix']]
    old_dir = ROOT / (args.scientific_source or 'evaluation/results/external-formal-early3-semantic-review-20261003-v1/scored-20261003T165617')
    previous = {s['unit_id']: s for s in jsonl(old_dir / 'scores.jsonl')
                if args.scientific_source or s['condition'] in {'ReAct', 'CodeAct', 'F1'}}
    previous_hashes = read(old_dir / 'summary.json')['raw_archive_hashes']
    operational_reviews = {}
    if args.reviews:
        for review in jsonl(ROOT / args.reviews):
            assert review['unit_id'] not in operational_reviews
            operational_reviews[review['unit_id']] = review
    scores, hashes, queue, diagnostics = [], {}, [], []
    states = collections.Counter()
    for index, (unit, path, recorded, claimed) in enumerate(snapshot):
        item = items[unit['question_index']]
        base = {**unit, 'question_id': item['id'], 'expected_behavior': item['gold']['expected_behavior'],
                'scorer_version': VERSION, 'archive_path': str(path.relative_to(ROOT))}
        if not recorded:
            state = 'claimed_without_result' if claimed else 'not_yet_recorded'
            states[state] += 1
            score = dict(base, scoring_status=state, completed_execution=None,
                         zero_scientific_operations=None, attempted_steps=None, abnormal_terminal=None)
            finish_endpoints(score, base['expected_behavior'])
        else:
            row = read(path / 'result.json')
            events = list(jsonl(path / 'events.jsonl'))
            assert row['unit'] == unit
            bounds = [e for e in events if e['kind'] in {'run_start', 'run_complete'}]
            assert len(bounds) == 2 and bounds[0]['unit'] == unit and bounds[1]['summary'] == row
            raw_sha = {name: file_digest(path / name) for name in ('result.json', 'events.jsonl')}
            hashes[str(path.relative_to(BATCH))] = raw_sha
            score = dict(base, **score_execution(row, events), scoring_status='recorded')
            score['execution_evidence_sha256'] = digest_records(row, events)
            old = previous.get(unit['unit_id'])
            if old:
                assert previous_hashes[str(path.relative_to(BATCH))] == raw_sha
                # Reuse positive execution judgments only, not answer scores.
                # Incomplete request coverage does not imply execution failure.
                if old.get('execution_contract_correct') is True and old.get('review_source'):
                    if score['completed_execution'] is None and score['native_status'] in NORMAL and base['expected_behavior'] != 'reject':
                        score['completed_execution'] = True
                        score['completion_reasons'].append('existing_evidence_bound_execution_review')
                    if base['expected_behavior'] == 'reject' and score['attempted_steps'] == 0 and score['native_status'] in NORMAL:
                        score['zero_scientific_operations'] = True
                        score['completed_execution'] = False
                        score['completion_reasons'].append('existing_review_verifies_no_scientific_execution')
            if args.final_batch and score['completed_execution'] is None:
                if score['abnormal_terminal']:
                    # Explicit terminal errors count as noncompletion.
                    # score_execution already checks completion evidence first.
                    score['completed_execution'] = False
                    score['completion_reasons'].append('recorded_agent_abort_without_completed_execution')
                elif old and old.get('scientific_output_correct') is True and score['successful_tool_steps']:
                    # Supporting evidence; Gold path/value matches are not required.
                    score['completed_execution'] = True
                    score['completion_reasons'].append('finished_analysis_has_verified_tool_output_witness')
                elif score['unresolved_failed_event_indices'] or any(c['error'] for c in score['kernel_cells']):
                    score['completed_execution'] = False
                    score['completion_reasons'].append('scientific_execution_has_unresolved_failed_operations')
            review = operational_reviews.get(unit['unit_id'])
            if review:
                assert review['evidence_sha256'] == score['execution_evidence_sha256']
                assert review['reviewer_kind'] in {'assistant', 'human'}
                assert review['reviewer_id'] and review['reason'] and review['evidence_locations']
                for key, value in review['labels'].items():
                    assert key in {'completed_execution', 'zero_scientific_operations'}
                    assert value is None or type(value) is bool
                    assert score[key] is None or score[key] == value
                    if value is not None:
                        score[key] = value
                score['operational_review_source'] = review['reviewer_kind'] + ':' + review['reviewer_id']
            finish_endpoints(score, base['expected_behavior'])
            states['recorded'] += 1
            score['latency_ms'] = row['result'].get('latency_ms')
            score['llm_requests'] = sum(e['kind'] == 'llm_request' for e in events)
            if old:
                # Keep the saved supplementary review checkpoint.
                score['supplementary_frozen_scientific_score'] = {
                    k: old.get(k) for k in ('scientific_task_success', 'identity_correct',
                    'scientific_output_correct', 'execution_contract_correct', 'final_answer_correct',
                    'rejection_response_correct', 'review_source', 'evidence_sha256', 'scorer_version')}
            if score['expected_behavior_outcome'] is None:
                queue.append({**base, 'execution_evidence_sha256': score['execution_evidence_sha256'],
                              'reasons': score['completion_reasons'],
                              'zero_scientific_operations': score['zero_scientific_operations'],
                              'kernel_cells': score['kernel_cells']})
            diagnostics.append({**base, 'native_status': score['native_status'],
                'attempted_steps': score['attempted_steps'], 'completed_execution': score['completed_execution'],
                'safe_rejection': score['safe_rejection'], 'abnormal_terminal': score['abnormal_terminal'],
                'successful_tool_steps': score['successful_tool_steps'], 'failed_tool_steps': score['failed_tool_steps'],
                'interface_action_attempts': score['interface_action_attempts'],
                'completion_reasons': '|'.join(score['completion_reasons'])})
        scores.append(score)
        if (index + 1) % 1000 == 0:
            print('READ', index + 1, flush=True)
    groups = []
    for model in frozen['models']:
        for method in METHODS:
            group = [s for s in scores if s['model'] == model and s['condition'] == method]
            assert len(group) == 450
            execute = [s for s in group if s['expected_behavior'] != 'reject']
            reject = [s for s in group if s['expected_behavior'] == 'reject']
            assert len(execute) == 375 and len(reject) == 75
            groups.append({'model': model, 'method': method,
                'recorded': sum(s['scoring_status'] == 'recorded' for s in group),
                'expected_behavior_outcome': rates(s['expected_behavior_outcome'] for s in group),
                'execute_task_completion': rates(s['execute_task_completion'] for s in execute),
                'safe_rejection': rates(s['safe_rejection'] for s in reject),
                'abnormal_terminal_runs': sum(s.get('abnormal_terminal') is True for s in group),
                'zero_step_abnormal_rejection_runs': sum(s.get('abnormal_terminal') is True and
                    s.get('attempted_steps') == 0 for s in reject)})
    contrasts = []
    for model in frozen['models']:
        for method in METHODS[:-1]:
            for metric in ('expected_behavior_outcome', 'execute_task_completion', 'safe_rejection'):
                selected = [s for s in scores if (metric != 'execute_task_completion' or s['expected_behavior'] != 'reject')
                            and (metric != 'safe_rejection' or s['expected_behavior'] == 'reject')]
                # Reuse the calculation with the current endpoint.
                result = paired_effect([dict(s, scientific_task_success=s[metric]) for s in selected], model, method)
                contrasts.append(dict(result, metric=metric))
    with (out / 'scores.jsonl').open('x', encoding='utf-8') as handle:
        for score in scores:
            handle.write(json.dumps(score, ensure_ascii=False, allow_nan=False) + '\n')
    with (out / 'execution_review_queue.jsonl').open('x', encoding='utf-8') as handle:
        for packet in queue:
            handle.write(json.dumps(packet, ensure_ascii=False) + '\n')
    fields = ['question_id', 'model', 'condition', 'repeat', 'unit_id', 'native_status', 'attempted_steps',
              'successful_tool_steps', 'failed_tool_steps', 'completed_execution', 'safe_rejection',
              'interface_action_attempts', 'abnormal_terminal', 'completion_reasons', 'archive_path']
    csv_write(out / 'run_execution_audit.csv', diagnostics, fields)
    table = []
    for group in groups:
        for metric in ('expected_behavior_outcome', 'execute_task_completion', 'safe_rejection'):
            table.append({'model': group['model'], 'method': group['method'], 'metric': metric, **group[metric]})
    csv_write(out / 'main_metrics.csv', table, ['model', 'method', 'metric', 'planned', 'confirmed_success',
              'confirmed_failure', 'unresolved', 'coverage', 'lower', 'upper'])
    supplementary = []
    for model in frozen['models']:
        for method in (METHODS if args.scientific_source else ('ReAct', 'CodeAct', 'F1')):
            group = [s for s in scores if s['model'] == model and s['condition'] == method]
            execute = [s for s in group if s['expected_behavior'] != 'reject']
            reject = [s for s in group if s['expected_behavior'] == 'reject']
            completed = [s for s in execute if s['completed_execution'] is True]
            for field, selected in [('scientific_task_success', group),
                    ('scientific_output_correct', execute), ('final_answer_correct', execute),
                    ('execution_contract_correct', execute), ('rejection_response_correct', reject),
                    ('completed_run_scientific_output_correct', completed)]:
                key = field.replace('completed_run_', '')
                labels = [s.get('supplementary_frozen_scientific_score', {}).get(key) for s in selected]
                supplementary.append({'model': model, 'method': method, 'field': field,
                                      **rates(labels)})
    csv_write(out / 'supplementary_correctness.csv', supplementary,
              ['model', 'method', 'field', 'planned', 'confirmed_success', 'confirmed_failure',
               'unresolved', 'coverage', 'lower', 'upper'])
    assert before == {str(p): file_digest(p) for p in protected}
    for relative, raw_sha in hashes.items():
        for name, sha in raw_sha.items():
            assert file_digest(BATCH / relative / name) == sha
    summary = {'version': VERSION, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'post_freeze_derived_analysis': True, 'original_frozen_endpoint_preserved': True,
        'freeze_sha256': file_digest(FREEZE), 'analysis_code_sha256': file_digest(Path(__file__)),
        'snapshot_states': dict(states), 'groups': groups, 'contrasts': contrasts,
        'supplementary_correctness': supplementary,
        'execution_review_queue_units': len(queue), 'raw_archive_hashes': hashes,
        'protected_file_sha256': before, 'frozen_execution_source_files_verified': len(frozen['code_sha256']),
        'raw_sources_reverified_after_analysis': True, 'no_experiment_or_api_calls': True,
        'supplementary_scientific_score_source': str(old_dir.relative_to(ROOT)),
        'operational_reviews_applied': len(operational_reviews)}
    summary['complete_batch_mode'] = args.final_batch
    summary['scientific_scores_cover_all_methods'] = bool(args.scientific_source)
    (out / 'analysis.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report(out, summary)
    print('OUTPUT', out)
    print('STATES', dict(states), 'REVIEW_QUEUE', len(queue))


def rate_cell(rate):
    yes, n, null = rate['confirmed_success'], rate['planned'], rate['unresolved']
    if not null:
        return f'{yes}/{n} ({100*rate["lower"]:.2f}%)'
    return f'{yes} success / {rate["confirmed_failure"]} failure / {null} unknown; {100*rate["lower"]:.2f}-{100*rate["upper"]:.2f}%'


def report(out, summary):
    text = ['# Framework execution analysis', '',
        f'Created (UTC): {summary["created_at"]}. Scoring version: {VERSION}.', '',
        '## Execution endpoints', '',
        'Expected-behavior outcome uses workflow completion on execute/repair tasks and safe rejection on reject tasks.',
        'Completion requires executor evidence or a closed successful execution trace. Identity, output, request coverage and answer correctness are scored separately.',
        'Safe rejection follows completed=False and attempted_steps=0. Zero-call budget or infrastructure stops satisfy this operational definition; their termination reasons remain separate.',
        'Actual direct Python computation counts as a scientific operation. Interface rejections, file inspection and display actions do not.',
        'Unresolved scientific execution requires evidence review. A later successful retry can close an earlier failed operation.', '',
        '## Results', '',
        '| Model | Method | Expected behavior | Completion | Safe rejection |',
        '|---|---|---|---|---|']
    for group in summary['groups']:
        text.append('|' + '|'.join([group['model'], group['method'],
            rate_cell(group['expected_behavior_outcome']),
            rate_cell(group['execute_task_completion']),
            rate_cell(group['safe_rejection'])]) + '|')
    text += ['', 'Rates use planned denominators. Ranges caused by unresolved labels are identification bounds, not confidence intervals.', '',
        '## Paired comparisons', '',
        'The analysis averages three repeats per question and compares F1 with each baseline. Complete-pair 95% percentile bootstrap intervals use 10,000 resamples and seed 20261001.',
        'analysis.json retains both full-set effect bounds and complete-pair estimates with coverage.', '',
        '## Evidence', '',
        f'Archive states: {summary["snapshot_states"]}. Execution review queue: {summary["execution_review_queue_units"]}.',
        'run_execution_audit.csv records calls, successful and failed steps, terminal states and scoring reasons. execution_review_queue.jsonl identifies unresolved evidence.',
        'scores.jsonl retains the original scientific labels in supplementary_frozen_scientific_score. supplementary_correctness.csv reports three-valued labels and coverage.',
        'The original scientific-task endpoint and the derived execution endpoints remain separate. Historical missing units retain unknown labels.',
        'The historical selective answer review used an assistant. Its coverage and reviewer provenance are retained; it is not an independent human evaluation.',
        'Methods retain their native interaction budgets. API counts and latency are resource diagnostics.',
        'analysis.json records input and source hashes. This script does not call a model or a scientific tool.', '']
    (out / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')


if __name__ == '__main__':
    main()
