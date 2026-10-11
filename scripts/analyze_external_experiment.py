"""Derive comparison tables and question-paired CIs from immutable scores."""
import argparse
import json
from pathlib import Path

from scripts.external_baselines.common import file_digest, now
from scripts.external_baselines.experiment import ROOT, METHODS, MODELS, confined, read, save_new
from scripts.external_baselines.statistics import rates, paired_effect


def analyze(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Do not overwrite analysis')
    summary=read(source/'summary.json')
    scores=[json.loads(s) for s in (source/'scores.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(scores)!=4500 or len({s['unit_id'] for s in scores})!=4500:raise ValueError('Incomplete/duplicate planned matrix')
    groups=[]
    for model in MODELS:
        for method in METHODS:
            select=[s for s in scores if s['condition']==method and s['model']==model]
            if len(select)!=450:raise ValueError('Wrong per-method/model denominator')
            execute=[s for s in select if s['expected_behavior']!='reject']
            reject=[s for s in select if s['expected_behavior']=='reject']
            if len(execute)!=375 or len(reject)!=75:raise ValueError('Unexpected frozen behavior counts')
            recorded=[s for s in select if 'llm_requests' in s]
            usage_complete=all(s.get('usage_complete') and s.get('input_tokens') is not None
                               and s.get('output_tokens') is not None for s in recorded) and len(recorded)==450
            groups.append({'condition':method,'model':model,
                'scientific_task_success':rates(s['scientific_task_success'] for s in select),
                'execution_task_success':rates(s['scientific_task_success'] for s in execute),
                'strict_rejection':rates(s['strict_pre_execution_rejection'] for s in reject),
                'native_completed_diagnostic':rates(s.get('native_completed') for s in execute),
                'usage_complete_units':sum(s.get('usage_complete') is True for s in recorded),
                'recorded_units':len(recorded),'usage_complete_all_planned':usage_complete,
                'observed_llm_requests':sum(s['llm_requests'] for s in recorded),
                'observed_input_tokens':sum(s.get('input_tokens') or 0 for s in recorded),
                'observed_output_tokens':sum(s.get('output_tokens') or 0 for s in recorded),
                'observed_cached_input_tokens':sum(s.get('cached_input_tokens') or 0 for s in recorded),
                'observed_latency_ms':sum(s.get('latency_ms') or 0 for s in recorded)})
    effects=[paired_effect(scores,model,baseline) for model in MODELS for baseline in METHODS if baseline!='F1']
    output.mkdir(parents=True,exist_ok=False)
    save_new(output/'analysis.json',{'created_at':now(),'scope':summary['scope'],'groups':groups,'effects':effects,
        'scoring_summary_sha256':file_digest(source/'summary.json'),
        'scores_sha256':file_digest(source/'scores.jsonl'),
        'human_and_assistant_counts':summary.get('review_kind_counts',{}),
        'frozen_seed':20261001,'question_bootstrap_replicates':10000,
        'note':'Unresolved included in full-planned bounds; complete-pair CI labelled by coverage.'})
    table=['| Method | Model | Task success lower–upper | Coverage | Strict rejection lower–upper |\n',
           '|---|---|---:|---:|---:|\n']
    for g in groups:
        r=g['scientific_task_success'];reject=g['strict_rejection']
        table.append(f"| {g['condition']} | {g['model']} | {100*r['lower']:.2f}–{100*r['upper']:.2f}% | {100*r['coverage']:.2f}% | {100*reject['lower']:.2f}–{100*reject['upper']:.2f}% |\n")
    (output/'comparison_table.md').write_text(''.join(table),encoding='utf-8')
    print('ANALYSIS',str(output))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-id',required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    analyze(confined(ROOT/'evaluation/results',args.source_run_id),confined(ROOT/'evaluation/results',args.run_id))


if __name__=='__main__':main()
