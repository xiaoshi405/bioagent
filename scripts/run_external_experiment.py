"""Prepare/check a frozen 4,500-unit matrix, or explicitly execute it later."""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.metadata
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from scripts.external_baselines.common import Budget, ModelClient, Recorder, ToolFacade, file_digest, now
from scripts.external_baselines.experiment import (ROOT, METHODS, MODELS, read, save_new, confined,
    matrix, prepare_freeze, verify_freeze, method_identity, unit_path, ledger_state, RunLock, dispatch)
from scripts.external_baselines.resource_control import (verify_resource_lineage, verify_batch_freeze,
    ResourcePressure, run_guarded)


def packages():
    names=('httpx','biopython','numpy','fastapi','pydantic','ipython','tqdm','llama-index-core',
           'llama-index-embeddings-huggingface','sentence-transformers','torch','transformers',
           'pandas','langgraph','langgraph-checkpoint','langchain-core','langchain-openai','psutil')
    return {name:importlib.metadata.version(name) for name in names}


def verify_preparation(frozen):
    verify_freeze(frozen)
    verify_resource_lineage(frozen, ROOT)
    if packages()!=frozen['packages']:raise ValueError('Frozen dependency versions changed')
    from scripts.external_baselines.scitoolagent import embedding_identity
    if embedding_identity()!=frozen['embedding_identity']:raise ValueError('Frozen local embedding changed')
    for relative,expected in frozen['preparation_files_sha256'].items():
        if file_digest(confined(ROOT,relative))!=expected:raise ValueError('Preparation evidence/resource changed')


async def prepare(args):
    from scripts.run_external_baselines import execute
    out=confined(ROOT/'evaluation/results',args.run_id)
    out.mkdir(parents=True,exist_ok=False)
    # Legacy preparation entry accepts fixed Dev only and never calls ModelClient
    # when preflight_only=True. Official scripted loops exercise real local tools.
    evidence_prefix=getattr(args,'reuse_preflight',None) or args.run_id
    if getattr(args,'reuse_preflight',None):
        prior=read(confined(ROOT,'evaluation/biotooleval/external_baselines_freeze_v1.json'))
        allowed={'scripts/external_baselines/scoring_v2.py','scripts/external_baselines/experiment.py',
                 'scripts/run_external_experiment.py','scripts/score_external_experiment.py',
                 'scripts/analyze_external_experiment.py','docs/EXTERNAL_BASELINES_FORMAL_SCORING_V2_ZH.md'}
        for relative,sha in prior['code_sha256'].items():
            if relative.replace('\\','/') not in allowed and file_digest(confined(ROOT,relative))!=sha:
                raise ValueError('Cannot reuse preflight after changing method/tool source: '+relative)
        for method in METHODS:
            if method_identity(method)[0]!=prior['methods'][method]:raise ValueError('Method changed')
        for relative,sha in prior['preparation_files_sha256'].items():
            if file_digest(confined(ROOT,relative))!=sha:raise ValueError('Prior preflight evidence changed')
    else:
        async with asyncio.TaskGroup() as tasks:
            for method in METHODS:
                tasks.create_task(execute(SimpleNamespace(condition=method,models=list(MODELS),
                    run_id=args.run_id+'-'+method.lower(),tool_base=args.tool_base,limit=None,
                    preflight_only=True,resume_unstarted_from=[])))
    frozen=prepare_freeze()
    from scripts.external_baselines.scitoolagent import embedding_identity
    frozen['embedding_identity']=embedding_identity()
    frozen['packages']=packages(); resources={};evidence={}
    for method in METHODS:
        folder=out.parent/(evidence_prefix+'-'+method.lower())
        manifest=read(folder/'manifest.final.json')
        if manifest['status']!='preflight_only':raise ValueError('Incomplete local preflight')
        for path in folder.glob('*preflight.json'):
            report=read(path)
            if report.get('failed_tools'):raise ValueError('Failed local preflight')
            if 'external_model_calls' in report and report['external_model_calls']!=0:
                raise ValueError('Preparation made unexpected online calls')
        for path in folder.glob('*.json'):
            evidence[str(path.relative_to(ROOT))]=file_digest(path)
        if method=='F1':
            resources['f1_resources_file']=str((folder/'f1_resources.json').relative_to(ROOT))
            f1_resource=read(folder/'f1_resources.json')
            for path in Path(f1_resource['result']['persist_dir']).iterdir():
                if path.is_file():evidence[str(path.relative_to(ROOT))]=file_digest(path)
    frozen['resources']=resources
    frozen['preparation_files_sha256']=evidence
    if getattr(args,'reuse_preflight',None):
        frozen['supersedes']='evaluation/biotooleval/external_baselines_freeze_v1.json'
        frozen['pre_execution_amendment']='Scorer only: modified protein residues, normalized chain IDs, largest available-CA ANM chain. No Test content inspected; methods/tools/prompts/budgets unchanged.'
    freeze_path=confined(ROOT,args.freeze)
    freeze_path.parent.mkdir(parents=True,exist_ok=True)
    if freeze_path.exists():raise ValueError('Freeze already exists; do not overwrite')
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for relative in frozen['code_sha256']:
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/relative,target)
    save_new(freeze_path,frozen)
    save_new(out/'acceptance.json',{'status':'PASS','created_at':now(),
        'freeze_path':str(freeze_path),'freeze_sha256':file_digest(freeze_path),
        'planned_units':4500,'metadata_only_test_access':True,'external_model_calls':0,
        'methods_preflight_passed':list(METHODS)})
    print('FREEZE',str(freeze_path),'PLANNED',4500,'ONLINE_CALLS',0,flush=True)


def assert_safe_resume(batch):
    lock=batch/'active.lock'
    if not lock.exists():return
    pid=read(lock)['pid']
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes=(ctypes.c_void_p,ctypes.c_uint32)
        kernel.CloseHandle.argtypes=(ctypes.c_void_p,)
        handle=kernel.OpenProcess(0x100000,False,pid)
        if handle:
            alive=kernel.WaitForSingleObject(handle,0)!=0;kernel.CloseHandle(handle)
        else:
            # Invalid PID permits recovery; access denied does not.
            alive=ctypes.get_last_error()!=87
    else:
        try:os.kill(pid,0);alive=True
        except ProcessLookupError:alive=False
    if alive:raise ValueError('A live or inaccessible owner holds the batch lock')
    recovery=batch/('stale-lock-'+str(time.time_ns())+'.json')
    shutil.move(str(lock),str(recovery))


async def execute_unit(unit,item,batch,frozen,tool_base):
    from scripts.run_external_baselines import make_request
    folder=unit_path(batch,unit);folder.mkdir(parents=True,exist_ok=False)
    save_new(folder/'claim.json',{'unit':unit,'claimed_at':now()})
    if unit['condition']=='F1':
        shutil.copyfile(confined(ROOT,frozen['resources']['f1_resources_file']),folder/'f1_resources.json')
    request,files=make_request(item)
    run_key=f"{unit['model']}|{item['id']}|{unit['repeat']}"
    recorder=Recorder(folder/'events.jsonl',run_key);budget=Budget()
    facade=ToolFacade(tool_base,files,recorder,budget)
    revision = frozen.get('resource_amendment', {}).get('version')
    revision_fields = {'resource_runtime_version': revision} if revision else {}
    recorder.record('run_start',unit=unit,question_id=item['id'],model=unit['model'],repeat=unit['repeat'],
                    **revision_fields,
                    request=request,input_files=[{'path':p,'sha256':file_digest(Path(p)),
                    'raw_text':Path(p).read_text(encoding='utf-8-sig')} for p in files])
    started=time.perf_counter();_,agent=method_identity(unit['condition'])
    try:
        result=await asyncio.wait_for(agent(request,unit['model'],ModelClient(),facade,recorder,budget),
                                      timeout=budget.task_timeout_seconds)
    except Exception as exc:
        result={'final_status':'task_timeout' if isinstance(exc,TimeoutError) else 'adapter_error',
                'final_response':None,'attempted_steps':facade.tool_calls,
                'latency_ms':(time.perf_counter()-started)*1000,
                'llm_calls':sum(e['kind']=='llm_request' for e in recorder.events)}
        recorder.record('run_error',exception_type=type(exc).__name__,message=str(exc))
    usage={}
    responses=[e for e in recorder.events if e['kind']=='llm_response']
    requests=[e for e in recorder.events if e['kind']=='llm_request']
    for field in ('input_tokens','output_tokens','total_tokens'):
        values=[e.get('usage',{}).get(field) for e in responses]
        usage[field]=sum(values) if values and all(type(v) is int for v in values) else 0 if not requests else None
    row={'unit':unit,'run_key':run_key,'question_id':item['id'],'model':unit['model'],
         'repeat':unit['repeat'],'condition':unit['condition'],'scope':'frozen_benchmark_comparison',
         **revision_fields,
         'result':result,'usage':usage,'llm_response_count':len(responses)}
    recorder.record('run_complete',summary=row)
    # fsynced immutable result is the commit record. A directory lacking this
    # record remains interrupted-started and is never automatically repeated.
    save_new(folder/'result.json',row)
    print('UNIT',unit['unit_id'],result['final_status'],flush=True)


async def launch(args):
    if not args.execute_frozen_test:raise ValueError('Explicit Test execution flag is required')
    freeze_path=confined(ROOT,args.freeze);frozen=read(freeze_path);verify_preparation(frozen)
    batch=confined(ROOT/'evaluation/results',args.run_id)
    if args.resume:
        saved=read(batch/'manifest.start.json')
        verify_batch_freeze(frozen, freeze_path, saved, ROOT)
        assert_safe_resume(batch)
    else:
        batch.mkdir(parents=True,exist_ok=False)
        save_new(batch/'manifest.start.json',{'schema_version':frozen['schema_version'],
            'started_at':now(),'freeze_path':str(freeze_path.relative_to(ROOT)),
            'freeze_sha256':file_digest(freeze_path),'planned_units':4500,
            'automatic_run_retry':False})
    with RunLock(batch):
        # Only this explicit online path parses the benchmark content.
        from scripts.run_biotooleval import load_biotooleval_dataset
        loaded=load_biotooleval_dataset(confined(ROOT,frozen['dataset']['path']))
        if loaded.get('status')!='success':raise ValueError('Frozen dataset validation failed')
        items=loaded['items']
        if len(items)!=150:raise ValueError('Frozen question count changed')
        if len({i['id'] for i in items})!=150:raise ValueError('Duplicate frozen question ID')
        from scripts.run_external_baselines import preflight
        preflight_folder=batch/('launch-preflight-'+str(time.time_ns()));preflight_folder.mkdir()
        await preflight(args.tool_base,preflight_folder,Budget())
        states=ledger_state(batch,frozen['matrix'])
        started=now()
        stopped_for_memory = None
        memory_log = None
        amendment = frozen.get('resource_amendment')
        if amendment:
            memory_log = (batch/('resource-samples-'+str(time.time_ns())+'.jsonl')).open('x',encoding='utf-8')
        def record_memory(sample):
            memory_log.write(json.dumps({'recorded_at':now(),**sample},allow_nan=False)+'\n')
            memory_log.flush()
        async def work():
            await dispatch(frozen['matrix'],states,
                lambda unit:execute_unit(unit,items[unit['question_index']],batch,frozen,args.tool_base),
                frozen['runtime']['agent_concurrency'])
        try:
            if amendment:
                await run_guarded(work, amendment['memory_policy'], record_memory)
            else:
                await work()
        except ResourcePressure as exc:
            stopped_for_memory = {'version':amendment['version'],'paused_at':now(),
                                  'reasons':exc.reasons,'sample':exc.sample,
                                  'automatic_retry':False}
            save_new(batch/('resource-stop-'+str(time.time_ns())+'.json'),stopped_for_memory)
            print('RESOURCE_PRESSURE_PAUSED',json.dumps(stopped_for_memory),flush=True)
        finally:
            if memory_log:
                memory_log.close()
            current=ledger_state(batch,frozen['matrix'])
            counts=dict(collections.Counter(current.values()))
            report={'session_started_at':started,'session_finished_at':now(),
                    'counts':counts,'planned_units':4500,
                    'status':'resource_pressure_paused' if stopped_for_memory else
                             'completed' if counts.get('completed')==4500 else 'incomplete',
                    'frozen_source_integrity':True,
                    'execution_freeze_sha256':file_digest(freeze_path),
                    'resource_runtime_version':amendment['version'] if amendment else None}
            verify_preparation(frozen)
            save_new(batch/('session-'+str(time.time_ns())+'.json'),report)
            print('BATCH_COUNTS',json.dumps(counts),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','check','run','status'))
    parser.add_argument('--freeze',default='evaluation/biotooleval/external_baselines_freeze_v3_1_resource.json')
    parser.add_argument('--run-id',default='external-formal-test150-20261002-v1')
    parser.add_argument('--tool-base',default='http://127.0.0.1:18080')
    parser.add_argument('--execute-frozen-test',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reuse-preflight',help='Reuse unchanged-method local preparation evidence after scorer-only amendment')
    args=parser.parse_args()
    if args.action=='prepare':asyncio.run(prepare(args))
    elif args.action=='check':
        verify_preparation(read(confined(ROOT,args.freeze)));print('FROZEN_MATRIX_PASS',4500)
    elif args.action=='status':
        frozen=read(confined(ROOT,args.freeze))
        print(json.dumps(dict(collections.Counter(ledger_state(
            confined(ROOT/'evaluation/results',args.run_id),frozen['matrix']).values()))))
    else:asyncio.run(launch(args))


if __name__=='__main__':main()
