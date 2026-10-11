"""General frozen-benchmark scorer; no task-ID special cases or agent feedback."""
from __future__ import annotations

import copy
from pathlib import Path
from .audit import _named_values
from .scoring import (reference_objects as base_reference_objects, call_identity as base_call_identity, matches, tri_and,
                      evidence_digest, final_response_text)
from .audit import sequence_hash

SCORER_VERSION = 'external-scientific-outcome-v2.1'
REVIEW_FIELDS = ('identity_correct','scientific_output_correct','final_answer_correct',
                 'rejection_response_correct','execution_contract_correct')


def reference_objects(item,files):
    """Independent parsing with the frozen backend's protein/chain convention."""
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.SeqUtils import seq1
    objects=base_reference_objects(item,[p for p in files if Path(p).suffix.lower() not in ('.pdb','.ent')])
    # Read the registered constant; do not execute any scientific analysis tool.
    from ToolsAgent.ToolsFuns.Biology.pdb_analysis import MODIFIED_RESIDUE_MAP
    for file in files:
        path=Path(file)
        if path.suffix.lower() not in ('.pdb','.ent'):continue
        model=next(PDBParser(QUIET=True).get_structure('input',str(path)).get_models(),None)
        for chain in model or []:
            residues=[r for r in chain if is_aa(r,standard=False)]
            sequence=''.join(seq1(r.resname.strip().upper(),custom_map=MODIFIED_RESIDUE_MAP,undef_code='X') for r in residues)
            if sequence:
                objects.append({'source':str(path),'chain_id':str(chain.id).strip() or '_',
                    'sequence':sequence,'sequence_sha256':sequence_hash(sequence),
                    'ca_residue_count':sum('CA' in r for r in residues)})
    return objects


def call_identity(item,events,objects):
    result=base_call_identity(item,events,objects)
    from .scoring import _file_same
    for x in result:
        e=x['event']
        if e.get('tool_name')!='calculate_protein_anm':continue
        args=e.get('resolved_arguments') or {};source=args.get('file_path')
        inp=item.get('input',{});chains=[o for o in objects if 'chain_id' in o and _file_same(o['source'],source)]
        checks=[bool(chains)]
        if inp.get('chain_id'):
            wanted=inp['chain_id']
            if args.get('chain_id') is not None:checks.append(args['chain_id']==wanted)
        else:
            maximum=max([o['ca_residue_count'] for o in chains],default=0)
            longest=[o for o in chains if maximum>0 and o['ca_residue_count']==maximum]
            wanted=longest[0]['chain_id'] if len(longest)==1 else None
        actual=(e.get('output') or {}).get('chain_id')
        checks.append(actual==wanted if actual and wanted else None)
        x['identity_correct']=tri_and(checks)
        x['reasons']=['supplied_file_boundary','requested_or_unique_largest_CA_ANM_chain']
    return result


def score_run(item, row, events, files, review=None):
    """Gold witnesses and full request contract are distinct, auditable labels."""
    result = row['result']; objects = reference_objects(item,files)
    identities = call_identity(item,events,objects)
    expected = copy.deepcopy(item['gold'].get('expected_output',{}))
    calls = [x for x in identities if x['event'].get('status') == 'success']
    python_events = [e for e in events if e.get('kind') == 'kernel_execution_start']
    fields=[]
    for key,value in expected.items():
        tolerance=item['gold'].get('tolerance',{})
        tol=tolerance.get(key,1e-6) if isinstance(tolerance,dict) else tolerance
        if not isinstance(tol,(int,float)):tol=1e-6
        possible=[x for x in calls if any(matches(a,value,tol) for a in _named_values(x['event'].get('output'),key))]
        good=[x for x in possible if x['identity_correct'] is True]
        label=True if good else None if python_events or any(x['identity_correct'] is None for x in possible) else False
        fields.append({'field':key,'correct_on_requested_object':label,'tolerance':tol,
                       'evidence_event_indices':[x['event_index'] for x in good],
                       'candidate_event_indices':[x['event_index'] for x in possible]})
    attempted=max(sum(e.get('kind')=='tool_request' for e in events),int(result.get('attempted_steps',0)))
    answer=final_response_text(result)
    has_answer=isinstance(answer,(str,dict,list)) and bool(answer)
    reject=item['gold']['expected_behavior']=='reject'
    labels={'identity_correct':True if fields and all(f['correct_on_requested_object'] is True for f in fields) else None,
            'scientific_output_correct':tri_and(f['correct_on_requested_object'] for f in fields),
            'final_answer_correct':None if has_answer else False,
            'rejection_response_correct':None if has_answer else False,
            # Frozen Gold may omit requested outputs. A complete semantic check
            # must verify actual execution of every requested analysis, not just
            # a lucky Gold field or a tool transcript in the answer prefix.
            'execution_contract_correct':None if has_answer else False}
    if reject:labels['identity_correct']=labels['scientific_output_correct']=None
    evidence=evidence_digest(item,row,events); reviewer=None
    if review:
        if review.get('evidence_sha256')!=evidence:raise ValueError('Stale/wrong review evidence')
        if not review.get('reviewer_id') or review.get('reviewer_kind') not in ('human','assistant'):
            raise ValueError('Missing reviewer identity/kind')
        if not all(review.get(k) for k in ('reviewed_at','reason','evidence_locations')):
            raise ValueError('Missing review justification/evidence')
        for key,value in review.get('labels',{}).items():
            if key not in REVIEW_FIELDS or (value is not None and type(value) is not bool):
                raise ValueError('Invalid review field')
            if value is not None:
                if labels[key] is not None and labels[key]!=value:raise ValueError('Automatic/review conflict')
                labels[key]=value
        reviewer=review['reviewer_kind']+':'+review['reviewer_id']
    strict=tri_and((attempted==0,labels['rejection_response_correct'],labels['execution_contract_correct'])) if reject else None
    primary=strict if reject else tri_and(labels[k] for k in ('identity_correct','scientific_output_correct',
                                                             'execution_contract_correct','final_answer_correct'))
    relevant=('rejection_response_correct','execution_contract_correct') if reject else (
        'identity_correct','scientific_output_correct','execution_contract_correct','final_answer_correct')
    return {'scorer_version':SCORER_VERSION,'scientific_task_success':primary,
            'strict_pre_execution_rejection':strict,**labels,'field_checks':fields,
            'attempted_scientific_calls':attempted,'review_source':reviewer,'evidence_sha256':evidence,
            'pending_review_fields':[k for k in relevant if labels[k] is None],
            'identity_checks':[{k:v for k,v in x.items() if k!='event'} for x in identities],
            'python_execution_event_indices':[e['event_index'] for e in python_events],
            'native_completed':(result.get('native_execution_summary') or {}).get('completed'),
            'native_status':result.get('final_status')}


def review_packet(item,row,events,files,score):
    evidence_kinds={'tool_request','tool_call','input_read','kernel_file_open',
                    'kernel_execution_start','kernel_execution_end',
                    'artifact_pool_resolve_step_inputs','artifact_pool_register_tool_outputs'}
    return {'review_id':'review_'+score['evidence_sha256'][:20],
            'evidence_sha256':score['evidence_sha256'],'question':item['question_zh'],
            'input':{k:v for k,v in item['input'].items() if k not in ('note','fixture')},
            'source_objects':reference_objects(item,files),
            'rubric':{'expected_behavior':item['gold']['expected_behavior'],
                      'expected_output':item['gold'].get('expected_output',{}),
                      'tolerance':item['gold'].get('tolerance',{}),
                      'required_claims':item['gold'].get('required_claims',[]),
                      'forbidden_claims':item['gold'].get('forbidden_claims',[]),
                      'complete_request_execution_required':True,
                      'question_id_specific_scoring_rules':False},
            'final_response':final_response_text(row['result']),
            'evidence':[{k:v for k,v in e.items() if k not in ('run_key','recorded_at')}
                        for e in events if e.get('kind') in evidence_kinds],
            'automatic_score':{k:v for k,v in score.items() if k not in ('native_status','native_completed')}}
