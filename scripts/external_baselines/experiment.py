"""Frozen matrix and crash-safe ledger. Preparation reads Test metadata only."""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

from .common import Budget, digest, file_digest, now

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('ReAct', 'CodeAct', 'SciToolAgent', 'Biomni-A1', 'F1')
MODELS = ('qwen3.7-plus-2026-05-26', 'deepseek-v4-flash-0731')
SCHEMA = 'external-baselines-formal/1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n')
        handle.flush()
        os.fsync(handle.fileno())


def confined(root, relative):
    path = (Path(root)/relative).resolve()
    path.relative_to(Path(root).resolve())
    return path


def matrix(question_count=150, repeats=3):
    """Indices avoid opening sealed questions to construct the complete design."""
    result = []
    for question_index in range(question_count):
        for repeat in range(repeats):
            model_order = MODELS if (question_index+repeat) % 2 == 0 else MODELS[::-1]
            for model in model_order:
                offset = (question_index+repeat+MODELS.index(model)) % len(METHODS)
                for method in METHODS[offset:]+METHODS[:offset]:
                    result.append({'unit_id': f'q{question_index:03d}|{model}|{repeat}|{method}',
                                   'question_index': question_index, 'model': model,
                                   'repeat': repeat, 'condition': method})
    assert len({r['unit_id'] for r in result}) == question_count*repeats*10
    return result


def method_identity(method):
    if method == 'ReAct':
        from .react import ADAPTER_VERSION, run_react
        upstream = None; agent = run_react
    elif method == 'CodeAct':
        from .codeact import ADAPTER_VERSION, run_codeact
        from .codeact_upstream import upstream_identity
        upstream = upstream_identity(); agent = run_codeact
    elif method == 'SciToolAgent':
        from .scitoolagent import ADAPTER_VERSION, run_scitoolagent
        from .scitoolagent_upstream import upstream_identity
        upstream = upstream_identity(); agent = run_scitoolagent
    elif method == 'Biomni-A1':
        from .biomni import ADAPTER_VERSION, run_biomni
        from .biomni_upstream import upstream_identity
        upstream = upstream_identity(); agent = run_biomni
    elif method == 'F1':
        from .f1 import ADAPTER_VERSION, native_identity, run_f1
        upstream = native_identity(); agent = run_f1
    else:
        raise ValueError('Unknown frozen method')
    return {'adapter': ADAPTER_VERSION, 'upstream': upstream}, agent


def public_runtime_settings():
    from app.core.config import Config
    from app.llms.api_clients.openai_api import get_api_url
    from urllib.parse import urlsplit
    cfg=Config();endpoints={m:get_api_url(m) for m in MODELS}
    for endpoint in [cfg.API_BASE,*endpoints.values()]:
        parsed=urlsplit(endpoint)
        if parsed.username or parsed.password or parsed.query:raise ValueError('Unsafe endpoint metadata')
    return {'api_base':cfg.API_BASE,'native_model_endpoints':endpoints,
            'native_temperature':cfg.LLM_TEMPERATURE,'native_thinking':cfg.LLM_ENABLE_THINKING,
            'material_limits':{k:getattr(cfg,k) for k in ('MAX_PDB_UPLOAD_MB','MAX_FASTA_UPLOAD_MB',
                'MAX_MULTI_FASTA_RECORDS','TYPE_REPAIR_MAX_BRIDGE_HOPS','TYPE_REPAIR_MAX_PRUNED_STEPS')},
            'question_schema_path':str(Path(cfg.EXPERIMENT_QUESTION_SCHEMA_PATH).resolve())}


def prepare_freeze(root=ROOT):
    """No json.loads/open of Test questions, input content or Gold here."""
    from scripts.run_external_baselines import code_snapshot
    from .f1 import native_controls
    from .scoring_v2 import SCORER_VERSION
    root = Path(root)
    metadata_path = root/'evaluation/biotooleval/test_150_freeze_manifest_v5_3.json'
    metadata = read(metadata_path)
    if metadata['status'] != 'frozen_benchmark_ready' or metadata['item_count'] != 150:
        raise ValueError('Unexpected benchmark metadata')
    data = confined(root, metadata['dataset_file'])
    expected = metadata['dataset_sha256'].removeprefix('sha256:')
    if file_digest(data, canonical=True) != expected:
        raise ValueError('Frozen benchmark digest changed')
    fixtures = []
    for entry in metadata['fixtures']:
        path = confined(data.parent, entry['fixture'])
        if file_digest(path, canonical=True) != entry['sha256'].removeprefix('sha256:'):
            raise ValueError('Frozen fixture digest changed: '+entry['fixture'])
        fixtures.append({'path':str(path.relative_to(root)), 'raw_sha256':file_digest(path),
                         'canonical_sha256':file_digest(path, canonical=True)})
    sources = code_snapshot()
    for relative in ('scripts/run_external_experiment.py', 'scripts/score_external_experiment.py',
                     'scripts/analyze_external_experiment.py',
                     'docs/EXTERNAL_BASELINES_FORMAL_SCORING_V2_ZH.md',
                     'evaluation/biotooleval/question_schema_v1.json'):
        sources[relative] = file_digest(root/relative)
    versions = {m:method_identity(m)[0] for m in METHODS}
    design = matrix()
    return {'schema_version':SCHEMA, 'status':'FROZEN_BEFORE_ADDITIONAL_COMPARISON',
            'created_at':now(), 'scope':'existing_benchmark_additional_autonomous_agent_comparison',
            'benchmark_previously_used':True, 'full_dev70_waived_by_user':True,
            'dataset':{'path':str(data.relative_to(root)), 'raw_sha256':file_digest(data),
                       'canonical_sha256':expected, 'item_count':150,
                       'metadata_path':str(metadata_path.relative_to(root)),
                       'metadata_sha256':file_digest(metadata_path), 'fixtures':fixtures,
                       'execute_units_per_method_model':375, 'reject_units_per_method_model':75},
            'methods':versions, 'models':list(MODELS), 'repeats':3, 'planned_units':4500,
            'matrix':design, 'matrix_sha256':digest(design), 'code_sha256':sources,
            'external_budget':asdict(Budget()), 'f1_native_controls':native_controls(),
            'public_runtime_settings':public_runtime_settings(),
            'runtime':{'agent_concurrency':4,'per_method_concurrency':1,
                       'temperature':0.0,'thinking_enabled':False,'provider_seed_requested':False,
                       'automatic_run_retry':False,'resume':'never-started units only',
                       'interrupted_started_units':'retain incomplete; never automatically rerun',
                       'order':'question-repeat alternating model, cyclic method; FIFO shared slots'},
            'scoring':{'version':SCORER_VERSION,
                       'semantic_review':'required complete execution contract and final answer',
                       'reviewer_kinds':['human','assistant'], 'ambiguous':'null',
                       'unresolved':'all-planned lower/upper bounds, coverage',
                       'primary_endpoint':'completed requested analyses on authorized objects and correct answer; reject before analysis',
                       'gold_never_sent_to_agent':True},
            'analysis':{'unit':'question mean over three repeats', 'models_separate':True,
                        'comparisons':'F1 versus each of four baselines',
                        'bootstrap_replicates':10000,'bootstrap_seed':20261001,
                        'confidence':0.95,'p_values':False,
                        'missing':'bounds; complete shared-question pairs separately labelled'}}


def verify_freeze(frozen, root=ROOT):
    if frozen['schema_version'] != SCHEMA or frozen['planned_units'] != 4500:
        raise ValueError('Unexpected frozen design')
    if frozen['matrix'] != matrix() or digest(frozen['matrix']) != frozen['matrix_sha256']:
        raise ValueError('Frozen matrix changed')
    for relative, expected in frozen['code_sha256'].items():
        if file_digest(confined(root, relative)) != expected:
            raise ValueError('Frozen code changed: '+relative)
    data = frozen['dataset']
    for relative, expected in [(data['path'],data['raw_sha256']),
                               (data['metadata_path'],data['metadata_sha256'])]:
        if file_digest(confined(root, relative)) != expected:
            raise ValueError('Frozen dataset metadata/content changed')
    for fixture in data['fixtures']:
        if file_digest(confined(root, fixture['path'])) != fixture['raw_sha256']:
            raise ValueError('Frozen input changed: '+fixture['path'])
    for method in METHODS:
        if method_identity(method)[0] != frozen['methods'][method]:
            raise ValueError('Method identity changed')
    from .f1 import native_controls
    if native_controls() != frozen['f1_native_controls'] or asdict(Budget()) != frozen['external_budget']:
        raise ValueError('Frozen call budgets changed')
    if public_runtime_settings()!=frozen['public_runtime_settings']:
        raise ValueError('Non-sensitive runtime settings changed')


def unit_path(batch, unit):
    return Path(batch)/'units'/digest(unit['unit_id'])[:24]


def ledger_state(batch, design):
    """A started directory is never treated as a never-started run after a crash."""
    states = {}; known = {digest(u['unit_id'])[:24]:u for u in design}
    unit_root = Path(batch)/'units'
    for path in unit_root.iterdir() if unit_root.exists() else []:
        if path.name not in known:
            raise ValueError('Unknown unit directory in ledger')
    for unit in design:
        path = unit_path(batch, unit)
        if (path/'result.json').exists():
            result = read(path/'result.json')
            if result['unit'] != unit:
                raise ValueError('Wrong unit in saved result')
            if not (path/'events.jsonl').exists():
                raise ValueError('Completed result has no trace')
            states[unit['unit_id']] = 'completed'
        elif path.exists():
            states[unit['unit_id']] = 'interrupted_started'
        else:
            states[unit['unit_id']] = 'never_started'
    return states


class RunLock:
    def __init__(self, batch): self.path = Path(batch)/'active.lock'
    def __enter__(self):
        save_new(self.path, {'pid':os.getpid(),'started_at':now()})
        return self
    def __exit__(self, *exc): self.path.unlink()


async def dispatch(design, states, run_unit, concurrency=4):
    """Earliest eligible matrix unit; skip busy methods, never skip failed tasks."""
    if concurrency not in (1,2,3,4): raise ValueError('Frozen concurrency is at most four')
    pending = [u for u in design if states[u['unit_id']] == 'never_started']
    active = {}
    try:
        while pending or active:
            busy = set(active.values())
            while len(active) < concurrency:
                index = next((i for i,u in enumerate(pending) if u['condition'] not in busy),None)
                if index is None: break
                unit = pending.pop(index)
                task = asyncio.create_task(run_unit(unit))
                active[task] = unit['condition']; busy.add(unit['condition'])
            if not active: break
            completed, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                del active[task]
                await task
    finally:
        for task in active: task.cancel()
        if active: await asyncio.gather(*active, return_exceptions=True)
