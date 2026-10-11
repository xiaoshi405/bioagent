"""Read-only frozen-benchmark scoring; incomplete units remain unresolved."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from scripts.external_baselines.common import file_digest, now
from scripts.external_baselines.experiment import ROOT, read, save_new, confined, unit_path, ledger_state
from scripts.external_baselines.scoring_v2 import score_run, review_packet, SCORER_VERSION
from scripts.external_baselines.resource_control import verify_batch_freeze


def lines(path):
    return [read_line(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def read_line(text):
    import json
    return json.loads(text)


def score_archive(freeze_path,batch_path,output,labels_file=None):
    frozen=read(freeze_path);batch=Path(batch_path);output=Path(output)
    if output.exists():raise ValueError('Never overwrite an earlier scoring decision')
    start=read(batch/'manifest.start.json')
    verify_batch_freeze(frozen, freeze_path, start, ROOT)
    # Scoring may run after execution starts, but never feeds results to agents.
    dataset=confined(ROOT,frozen['dataset']['path'])
    if file_digest(dataset)!=frozen['dataset']['raw_sha256']:raise ValueError('Frozen input changed')
    for fixture in frozen['dataset']['fixtures']:
        if file_digest(confined(ROOT,fixture['path']))!=fixture['raw_sha256']:raise ValueError('Source fixture changed')
    items=read(dataset)['items']
    labels={}
    if labels_file:
        for label in lines(labels_file):
            sha=label['evidence_sha256']
            if sha in labels:raise ValueError('Duplicate semantic review')
            labels[sha]=label
    used=set();scores=[];packets=[];mapping=[];raw_hashes={}
    states=ledger_state(batch,frozen['matrix'])
    from scripts.run_external_baselines import make_request
    for unit in frozen['matrix']:
        item=items[unit['question_index']];path=unit_path(batch,unit)
        if states[unit['unit_id']]!='completed':
            scores.append({**unit,'question_id':item['id'],'expected_behavior':item['gold']['expected_behavior'],
                           'scientific_task_success':None,'scoring_status':states[unit['unit_id']],
                           'strict_pre_execution_rejection':None,'native_completed':None,'native_status':None})
            continue
        row=read(path/'result.json');events=lines(path/'events.jsonl')
        revision=row.get('resource_runtime_version')
        if revision and revision!=frozen.get('resource_amendment',{}).get('version'):
            raise ValueError('Unknown archived resource runtime version')
        if row['question_id']!=item['id'] or row['model']!=unit['model'] or row['repeat']!=unit['repeat']:
            raise ValueError('Archived run/plan identity mismatch')
        boundaries=[e for e in events if e['kind'] in ('run_start','run_complete')]
        if len(boundaries)!=2 or boundaries[0]['kind']!='run_start' or boundaries[1]['kind']!='run_complete':
            raise ValueError('Invalid run boundary')
        if boundaries[0]['unit']!=unit or boundaries[1]['summary']!=row:raise ValueError('Trace/commit mismatch')
        for source in boundaries[0]['input_files']:
            if file_digest(Path(source['path']))!=source['sha256']:raise ValueError('Run source changed')
        _,files=make_request(item)
        score=score_run(item,row,events,files)
        review=labels.get(score['evidence_sha256'])
        if review:
            score=score_run(item,row,events,files,review);used.add(score['evidence_sha256'])
        responses=[e for e in events if e['kind']=='llm_response']
        requests=[e for e in events if e['kind']=='llm_request']
        score.update({**unit,'question_id':item['id'],'expected_behavior':item['gold']['expected_behavior'],
            'resource_runtime_version':revision,
            'scoring_status':'reviewed' if review else 'automatic',
            'llm_requests':len(requests),'llm_responses':len(responses),
            'usage_complete':len(responses)==len(requests) and bool(row['result'].get('usage_complete',True)),
            'input_tokens':row['usage'].get('input_tokens'),'output_tokens':row['usage'].get('output_tokens'),
            'cached_input_tokens':sum(e.get('provider_usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',0)
                                      for e in responses),
            'latency_ms':row['result'].get('latency_ms')})
        scores.append(score)
        packet=review_packet(item,row,events,files,score)
        if score['pending_review_fields']:packets.append(packet)
        mapping.append({'review_id':packet['review_id'],'unit_id':unit['unit_id'],
                        'unit_path':str(path),'evidence_sha256':score['evidence_sha256']})
        raw_hashes[str(path.relative_to(batch))]={name:file_digest(path/name) for name in ('events.jsonl','result.json')}
    if set(labels)!=used:raise ValueError('Unknown/stale reviews supplied')
    output.mkdir(parents=True,exist_ok=False)
    import json
    for name,rows in [('scores.jsonl',scores),('review_queue.blind.jsonl',packets),('review_map.private.jsonl',mapping)]:
        with (output/name).open('x',encoding='utf-8') as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    templates=[{'review_id':p['review_id'],'evidence_sha256':p['evidence_sha256'],
                'reviewer_id':'','reviewer_kind':'','reviewed_at':'','reason':'','evidence_locations':[],
                'labels':{k:None for k in p['automatic_score']['pending_review_fields']}} for p in packets]
    with (output/'review_template.jsonl').open('x',encoding='utf-8') as handle:
        for row in templates:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    save_new(output/'summary.json',{'created_at':now(),'scope':'frozen_benchmark_comparison',
        'planned_units':4500,'recorded_units':sum(v=='completed' for v in states.values()),
        'primary_resolved_units':sum(s['scientific_task_success'] is not None for s in scores),
        'review_queue_units':len(packets),'state_counts':dict(collections.Counter(states.values())),
        'review_kind_counts':dict(collections.Counter(l['reviewer_kind'] for l in labels.values())),
        'scorer_version':SCORER_VERSION,'freeze_sha256':file_digest(Path(freeze_path)),
        'labels_file_sha256':file_digest(Path(labels_file)) if labels_file else None,
        'scorer_sha256':file_digest(ROOT/'scripts/external_baselines/scoring_v2.py'),
        'raw_archive_hashes':raw_hashes})
    print('SCORED',4500,'RECORDED',sum(v=='completed' for v in states.values()),'REVIEW_QUEUE',len(packets))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-frozen-benchmark',action='store_true',required=True)
    parser.add_argument('--freeze',default='evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json')
    parser.add_argument('--source-run-id',required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--labels-file')
    args=parser.parse_args()
    score_archive(confined(ROOT,args.freeze),confined(ROOT/'evaluation/results',args.source_run_id),
                  confined(ROOT/'evaluation/results',args.run_id),args.labels_file)


if __name__=='__main__':main()
