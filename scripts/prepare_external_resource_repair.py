"""Freeze an evidenced resource-only repair; never parse Test or run agents."""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

from scripts.external_baselines.common import file_digest, now
from scripts.external_baselines.experiment import ROOT, METHODS, confined, read, save_new, method_identity
from scripts.external_baselines.resource_control import POLICY, RESOURCE_VERSION, validate_resource_amendment
from scripts.run_external_experiment import packages, verify_preparation


def prepare_repair(evidence_dir, freeze_path):
    base_relative = 'evaluation/biotooleval/external_baselines_freeze_v2.json'
    base_path = confined(ROOT, base_relative)
    base = read(base_path)
    evidence_dir, freeze_path = Path(evidence_dir), Path(freeze_path)
    acceptance = read(evidence_dir/'repair_acceptance.json')
    if acceptance['status'] != 'PASS' or acceptance['external_model_calls'] != 0:
        raise ValueError('Repair acceptance missing or made model calls')
    if acceptance['native_successful_tools'] != 20 or acceptance['long_run_cycles'] < 40:
        raise ValueError('Incomplete native/long-run acceptance')
    if acceptance['warm_private_growth_mb'] >= 128:
        raise ValueError('Private memory still grows during native workflow')
    amended = copy.deepcopy(base)
    amended['created_at'] = now()
    amended['status'] = 'FROZEN_RESOURCE_AMENDMENT_AFTER_INFRASTRUCTURE_INCIDENT'
    amended['packages'] = packages()
    amended['methods'] = {m: method_identity(m)[0] for m in METHODS}
    amended['code_sha256'] = {p:file_digest(confined(ROOT,p)) for p in base['code_sha256']}
    for relative in ('scripts/external_baselines/resource_control.py',
                     'scripts/prepare_external_resource_repair.py'):
        amended['code_sha256'][relative] = file_digest(confined(ROOT,relative))
    evidence = {str(p.relative_to(ROOT)):file_digest(p)
                for p in sorted(evidence_dir.iterdir()) if p.is_file()}
    diagnosis=ROOT/'evaluation/results/external-formal-control-20261002-v1/incident-20261002-v1'
    for name in ('embedding_memory_diagnosis.json','fixed_inference_thread_100.jsonl',
                 'fresh_threads_locked.jsonl','reuse_threads_locked.jsonl','windows_resource_events.json'):
        path=diagnosis/name
        evidence[str(path.relative_to(ROOT))]=file_digest(path)
    amended['resource_amendment'] = {
        'version': RESOURCE_VERSION,
        'reason': 'Windows private commit growth with repeatedly recreated CPU BGE inference threads',
        'base_freeze_path': base_relative,
        'base_freeze_sha256': file_digest(base_path),
        'base_source_snapshot': 'evaluation/results/external-formal-preparation-20261002-v2/source_snapshot',
        'memory_policy': POLICY, 'automatic_retry': False,
        'acceptance_files_sha256': evidence,
        'unchanged': 'official cores, planner prompts, vector values, scientific tools, data, Gold, budgets, matrix, scoring and statistics',
        'changed': 'fixed CPU inference thread, resource guard, explicit freeze lineage and per-session/runtime tagging',
    }
    validate_resource_amendment(base, amended)
    verify_preparation(amended)
    snapshot = evidence_dir/'source_snapshot'
    snapshot.mkdir(exist_ok=False)
    for relative in amended['code_sha256']:
        target = confined(snapshot,relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(confined(ROOT,relative),target)
    save_new(freeze_path,amended)
    save_new(evidence_dir/('freeze_acceptance_'+RESOURCE_VERSION+'.json'),{
        'status':'PASS','created_at':now(),'resource_version':RESOURCE_VERSION,
        'freeze_file':str(freeze_path.relative_to(ROOT)),'freeze_sha256':file_digest(freeze_path),
        'superseded_freeze_preserved':base_relative,'external_model_calls':0,
        'formal_experiment_restarted':False,'automatic_retry':False})
    print('RESOURCE_FREEZE_PASS',str(freeze_path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',required=True)
    parser.add_argument('--freeze',default='evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json')
    args=parser.parse_args()
    prepare_repair(confined(ROOT,args.evidence_dir),confined(ROOT,args.freeze))


if __name__=='__main__':main()
