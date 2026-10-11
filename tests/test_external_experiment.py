"""Offline acceptance fixtures. No Test questions and no model API calls."""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.external_baselines.common import file_digest, now
from scripts.external_baselines.experiment import (METHODS, MODELS, matrix, ledger_state,
    unit_path, save_new, RunLock, dispatch, prepare_freeze, confined)
from scripts.external_baselines.scoring import evidence_digest
from scripts.external_baselines.scoring_v2 import score_run
from scripts.external_baselines.statistics import rates, paired_effect


def fixture():
    return {'id':'unit-fixture','question_zh':'unit request requires multiple analyses',
            'input':{'input_type':'amino_acid_sequence','sequence':'ACDE'},
            'gold':{'expected_behavior':'execute','expected_output':{'gravy':-1.0}}}


def row(answer='unit answer',steps=1):
    return {'result':{'final_response':answer,'attempted_steps':steps,'final_status':'answered'}}


def event(sequence='ACDE',value=-1.0,status='success'):
    return {'kind':'tool_call','event_index':0,'call_id':'unit-call',
            'tool_name':'compute_gravy','resolved_arguments':{'sequence':sequence},
            'status':status,'output':{'status':'success','gravy':value}}


def review(item,r,events,labels):
    return {'evidence_sha256':evidence_digest(item,r,events),'reviewer_id':'unit-fixture',
            'reviewer_kind':'assistant','reviewed_at':now(),'reason':'offline test fixture',
            'evidence_locations':['events[0]'],'labels':labels}


class MatrixLedgerTests(unittest.TestCase):
    def test_full_design_unique_balanced_and_each_question_repeat_has_all_methods_models(self):
        design=matrix();self.assertEqual(len(design),4500)
        self.assertEqual(len({u['unit_id'] for u in design}),4500)
        for model in MODELS:
            for method in METHODS:
                self.assertEqual(sum(u['model']==model and u['condition']==method for u in design),450)
        for q in (0,1,149):
            for r in (0,1,2):
                group=[u for u in design if u['question_index']==q and u['repeat']==r]
                self.assertEqual(len(group),10)
                self.assertEqual(len({(u['condition'],u['model']) for u in group}),10)

    def test_crash_resume_preserves_started_failed_and_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=Path(tmp);design=matrix(1,1)
            interrupted=unit_path(batch,design[0]);interrupted.mkdir(parents=True)
            finished=unit_path(batch,design[1]);finished.mkdir()
            save_new(finished/'result.json',{'unit':design[1],'result':{'final_status':'llm_budget_exhausted'}})
            (finished/'events.jsonl').write_text('{}\n')
            states=ledger_state(batch,design)
            self.assertEqual(states[design[0]['unit_id']],'interrupted_started')
            self.assertEqual(states[design[1]['unit_id']],'completed')
            self.assertEqual(sum(v=='never_started' for v in states.values()),8)

    def test_exclusive_lock_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with RunLock(tmp):
                with self.assertRaises(FileExistsError):RunLock(tmp).__enter__()
            self.assertFalse((Path(tmp)/'active.lock').exists())
            save_new(Path(tmp)/'result.json',{'original':True})
            with self.assertRaises(FileExistsError):save_new(Path(tmp)/'result.json',{'original':False})

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):confined(tmp,'../outside')

    def test_prepare_hashes_but_does_not_parse_test_content(self):
        from scripts import run_external_baselines as legacy
        from scripts.external_baselines import experiment as exp
        from scripts.external_baselines import f1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'evaluation/biotooleval';d.mkdir(parents=True)
            dataset=d/'test.json';dataset.write_text('This is deliberately not valid JSON; never parse it.\n')
            metadata={'status':'frozen_benchmark_ready','item_count':150,
                'dataset_file':str(dataset.relative_to(root)),
                'dataset_sha256':'sha256:'+file_digest(dataset,canonical=True),'fixtures':[]}
            save_new(d/'test_150_freeze_manifest_v5_3.json',metadata)
            for name in ['scripts/run_external_experiment.py','scripts/score_external_experiment.py',
                         'scripts/analyze_external_experiment.py','docs/EXTERNAL_BASELINES_FORMAL_SCORING_V2_ZH.md',
                         'evaluation/biotooleval/question_schema_v1.json']:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('unit fixture')
            with patch.object(legacy,'code_snapshot',return_value={}), \
                 patch.object(exp,'method_identity',side_effect=lambda m:({'adapter':'unit','upstream':None},None)), \
                 patch.object(exp,'public_runtime_settings',return_value={'unit':True}), \
                 patch.object(f1,'native_controls',return_value={'unit':True}):
                frozen=prepare_freeze(root)
            self.assertEqual(frozen['planned_units'],4500)
            self.assertTrue(frozen['full_dev70_waived_by_user'])


class DispatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_four_slots_method_isolation_and_preserve_incomplete(self):
        design=matrix(2,1);states={u['unit_id']:'never_started' for u in design}
        states[design[0]['unit_id']]='interrupted_started'
        seen=[];active=set();maximum=0
        async def unit(u):
            nonlocal maximum
            self.assertNotIn(u['condition'],active);active.add(u['condition'])
            maximum=max(maximum,len(active));seen.append(u['unit_id'])
            await asyncio.sleep(0.001);active.remove(u['condition'])
        await dispatch(design,states,unit,4)
        self.assertEqual(len(seen),19);self.assertEqual(len(set(seen)),19)
        self.assertNotIn(design[0]['unit_id'],seen);self.assertEqual(maximum,4)

    async def test_dispatch_failure_cancels_other_units_without_resampling(self):
        design=matrix(1,1);states={u['unit_id']:'never_started' for u in design};seen=[]
        async def unit(u):
            seen.append(u['unit_id'])
            if u==design[0]:raise RuntimeError('offline injected crash')
            await asyncio.sleep(0.01)
        with self.assertRaises(RuntimeError):await dispatch(design,states,unit,4)
        self.assertEqual(len(seen),4);self.assertEqual(len(set(seen)),4)


class WholePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_4500_unit_mock_execution_resume_scoring_and_analysis(self):
        """All data/calls here are labelled synthetic, inside an isolated temp root."""
        import contextlib
        import io
        from types import SimpleNamespace
        from scripts import run_external_experiment as runner
        from scripts import run_external_baselines as legacy
        from scripts import score_external_experiment as scorer
        from scripts import analyze_external_experiment as analyzer
        from scripts import run_biotooleval as native
        items=[]
        for q in range(150):
            i=fixture();i['id']=f'offline_unit_{q:03d}'
            i['gold']['expected_behavior']='reject' if q>=125 else 'execute'
            items.append(i)
        called=[]
        async def fake_agent(request,model,client,facade,recorder,budget):
            self.assertEqual(set(request),{'question','input','supplied_files'})
            self.assertNotIn('gold',request['input']);called.append(recorder.run_key)
            q=int(recorder.run_key.split('|')[1].rsplit('_',1)[1])
            if q<125:
                recorder.record('tool_request',call_id='unit-call',tool_name='compute_gravy',
                                resolved_arguments={'sequence':'ACDE'})
                e=event();recorder.record(e.pop('kind'),**{k:v for k,v in e.items() if k!='event_index'})
            answer='unit synthetic result' if q<125 else 'unit synthetic rejection'
            return {'final_status':'answered','final_response':answer,'attempted_steps':int(q<125),
                    'latency_ms':1,'llm_calls':0,'usage_complete':True}
        async def fake_preflight(*args):return {'successful_tools':list(range(20))}
        with tempfile.TemporaryDirectory(prefix='formal-offline-unit-') as tmp:
            root=Path(tmp);data=root/'unit-data.json';save_new(data,{'items':items})
            scorer_source=root/'scripts/external_baselines/scoring_v2.py'
            scorer_source.parent.mkdir(parents=True)
            scorer_source.write_bytes((Path(scorer.__file__).parent/'external_baselines/scoring_v2.py').read_bytes())
            frozen={'schema_version':'external-baselines-formal/1','matrix':matrix(),
                    'planned_units':4500,'dataset':{'path':'unit-data.json',
                    'raw_sha256':file_digest(data),'fixtures':[]},'resources':{'f1_resources_file':'unit-resources.json'},
                    'runtime':{'agent_concurrency':4}}
            freeze=root/'unit-freeze.json';save_new(freeze,frozen)
            args=SimpleNamespace(execute_frozen_test=True,freeze='unit-freeze.json',
                                 run_id='unit-batch',resume=False,tool_base='http://unused')
            # No official method is invoked; no external models or HTTP transports.
            with patch.object(runner,'ROOT',root),patch.object(scorer,'ROOT',root), \
                 patch.object(runner,'verify_preparation'),patch.object(legacy,'preflight',fake_preflight), \
                 patch.object(native,'load_biotooleval_dataset',return_value={'status':'success','items':items}), \
                 patch.object(runner,'method_identity',return_value=({},fake_agent)), \
                 patch.object(runner.shutil,'copyfile'),contextlib.redirect_stdout(io.StringIO()):
                await runner.launch(args)
                self.assertEqual(len(called),4500)
                args.resume=True;await runner.launch(args)
                self.assertEqual(len(called),4500)
                batch=root/'evaluation/results/unit-batch';automatic=root/'unit-auto'
                scorer.score_archive(freeze,batch,automatic)
                packets=[json.loads(l) for l in (automatic/'review_queue.blind.jsonl').read_text().splitlines()]
                self.assertEqual(len(packets),4500)
                labels=[]
                for p in packets:
                    reject=p['rubric']['expected_behavior']=='reject'
                    labels.append({'review_id':p['review_id'],'evidence_sha256':p['evidence_sha256'],
                        'reviewer_id':'offline-synthetic-unit-test','reviewer_kind':'assistant',
                        'reviewed_at':now(),'reason':'synthetic unit assertion, not scientific evidence',
                        'evidence_locations':['unit fixture'],
                        'labels':{'execution_contract_correct':True,
                                  'rejection_response_correct' if reject else 'final_answer_correct':True}})
                label_path=root/'unit-labels.jsonl'
                label_path.write_text('\n'.join(json.dumps(l) for l in labels)+'\n',encoding='utf-8')
                reviewed=root/'unit-reviewed';scorer.score_archive(freeze,batch,reviewed,label_path)
                summary=json.loads((reviewed/'summary.json').read_text())
                self.assertEqual(summary['primary_resolved_units'],4500)
                analyzer.analyze(reviewed,root/'unit-analysis')
                results=json.loads((root/'unit-analysis/analysis.json').read_text())
                self.assertEqual(len(results['groups']),10);self.assertEqual(len(results['effects']),8)
                self.assertTrue(all(g['scientific_task_success']['lower']==1 for g in results['groups']))
                self.assertTrue(all(e['complete_pair_ci95_pp']==[0,0] for e in results['effects']))


class FormalScoringTests(unittest.TestCase):
    def pdb_fixture(self,path):
        import numpy as np
        from Bio.PDB.StructureBuilder import StructureBuilder
        from Bio.PDB import PDBIO
        builder=StructureBuilder();builder.init_structure('offline-unit');builder.init_model(0)
        builder.init_chain('A')
        for number,name in enumerate(('MSE','ALA','ALA'),1):
            builder.init_residue(name,'H' if name=='MSE' else ' ',number,' ')
            atom='CA' if number==1 else 'N'
            builder.init_atom(atom,np.array([number,0.,0.]),1.,1.,' ',atom.rjust(4),number,element='C' if atom=='CA' else 'N')
        builder.init_chain('B')
        for number in (1,2):
            builder.init_residue('ALA',' ',number,' ')
            builder.init_atom('CA',np.array([number,1.,0.]),1.,1.,' ',' CA ',number+3,element='C')
        io=PDBIO();io.set_structure(builder.get_structure());io.save(str(path))

    def test_modified_hetero_residue_in_ent_matches_registered_parsing(self):
        from scripts.external_baselines.scoring_v2 import reference_objects
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'offline-unit.ent';self.pdb_fixture(p)
            objects=reference_objects({'input':{}},[str(p)])
            a=next(o for o in objects if o['chain_id']=='A')
            self.assertEqual(a['sequence'],'MAA');self.assertEqual(a['ca_residue_count'],1)

    def test_anm_default_identity_uses_ca_count_not_sequence_length(self):
        from scripts.external_baselines.scoring_v2 import reference_objects,call_identity
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'offline-unit.pdb';self.pdb_fixture(p)
            i={'input':{}};objects=reference_objects(i,[str(p)])
            e={'kind':'tool_call','event_index':0,'call_id':'unit','tool_name':'calculate_protein_anm',
               'resolved_arguments':{'file_path':str(p)},'output':{'chain_id':'B'},'status':'success'}
            self.assertTrue(call_identity(i,[e],objects)[0]['identity_correct'])
            e['output']['chain_id']='A'
            self.assertFalse(call_identity(i,[e],objects)[0]['identity_correct'])

    def test_same_gold_match_requires_full_execution_contract(self):
        i=fixture();r=row();ev=[event()]
        labels=review(i,r,ev,{'final_answer_correct':True})
        s=score_run(i,r,ev,[],labels)
        self.assertTrue(s['scientific_output_correct']);self.assertIsNone(s['scientific_task_success'])
        labels['labels']['execution_contract_correct']=False
        self.assertFalse(score_run(i,r,ev,[],labels)['scientific_task_success'])

    def test_complete_contract_and_answer_can_pass_equivalent_tool(self):
        i=fixture();r=row();ev=[event()];ev[0]['tool_name']='unit-compound-equivalent'
        labels=review(i,r,ev,{'final_answer_correct':True,'execution_contract_correct':True})
        self.assertTrue(score_run(i,r,ev,[],labels)['scientific_task_success'])

    def test_no_task_id_specific_augmentation(self):
        for id in ('bte_108','bte_120','bte_116','arbitrary-unseen-task'):
            i=fixture();i['id']=id;s=score_run(i,row(),[event()],[])
            self.assertEqual([f['field'] for f in s['field_checks']],['gravy'])

    def test_wrong_identity_and_failed_result_still_fail(self):
        self.assertFalse(score_run(fixture(),row(),[event('AAAA')],[])['scientific_task_success'])
        self.assertFalse(score_run(fixture(),row(),[event(status='tool_error')],[])['scientific_task_success'])

    def test_rejection_requires_correct_response_and_no_scientific_execution(self):
        i=fixture();i['gold']['expected_behavior']='reject';r=row('provide record_id',0);ev=[]
        labels=review(i,r,ev,{'rejection_response_correct':True,'execution_contract_correct':True})
        self.assertTrue(score_run(i,r,ev,[],labels)['scientific_task_success'])
        labels['labels']['execution_contract_correct']=False
        self.assertFalse(score_run(i,r,ev,[],labels)['scientific_task_success'])

    def test_zero_call_crash_not_rejection_and_invoked_error_not_prereject(self):
        i=fixture();i['gold']['expected_behavior']='reject'
        self.assertFalse(score_run(i,row(None,0),[],[])['scientific_task_success'])
        self.assertFalse(score_run(i,row('provide record',1),[event(status='tool_error')],[])['scientific_task_success'])

    def test_real_python_equivalent_can_be_reviewed_without_http(self):
        i=fixture();r=row();ev=[{'kind':'kernel_execution_start','event_index':0,'code':'# offline unit fixture'},
                               {'kind':'kernel_execution_end','event_index':1,'observation':'unit value'}]
        labels=review(i,r,ev,{k:True for k in ('identity_correct','scientific_output_correct',
                                             'final_answer_correct','execution_contract_correct')})
        self.assertTrue(score_run(i,r,ev,[],labels)['scientific_task_success'])

    def test_stale_label_and_automatic_conflict_rejected(self):
        i=fixture();r=row(None);ev=[event()]
        labels=review(i,r,ev,{'final_answer_correct':True})
        with self.assertRaises(ValueError):score_run(i,r,ev,[],labels)
        labels['evidence_sha256']='stale';labels['labels']={}
        with self.assertRaises(ValueError):score_run(i,r,ev,[],labels)


class StatisticalTests(unittest.TestCase):
    def test_unresolved_all_planned_bounds(self):
        self.assertEqual(rates([True,False,None])['lower'],1/3)
        self.assertEqual(rates([True,False,None])['upper'],2/3)

    def test_bootstrap_uses_questions_not_repeat_runs_and_reproducible(self):
        rows=[]
        for q in range(4):
            for method in ('F1','ReAct'):
                for r in range(3):rows.append({'question_index':q,'model':MODELS[0],
                    'condition':method,'repeat':r,'scientific_task_success':method=='F1'})
        result=paired_effect(rows,MODELS[0],'ReAct',500)
        self.assertEqual(result['shared_complete_questions'],4)
        self.assertEqual(result['complete_pair_effect_pp'],100)
        self.assertEqual(result['complete_pair_ci95_pp'],[100,100])
        self.assertEqual(result,paired_effect(rows,MODELS[0],'ReAct',500))
        rows[0]['scientific_task_success']=None
        result=paired_effect(rows,MODELS[0],'ReAct',500)
        self.assertEqual(result['shared_complete_questions'],3)
        self.assertFalse(result['complete_pair_is_primary_full_matrix'])
        self.assertLess(result['all_question_effect_bounds_pp'][0],100)

    def test_no_complete_pairs_does_not_invent_ci(self):
        rows=[{'question_index':0,'model':MODELS[0],'condition':m,'repeat':r,
               'scientific_task_success':None} for m in ('F1','ReAct') for r in range(3)]
        result=paired_effect(rows,MODELS[0],'ReAct',500)
        self.assertIsNone(result['complete_pair_ci95_pp'])
        self.assertEqual(result['all_question_effect_bounds_pp'],[-100,100])


if __name__=='__main__':unittest.main()
