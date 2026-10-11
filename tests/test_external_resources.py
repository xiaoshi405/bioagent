"""Resource repair tests; no Test content, scientific tools or model API."""
import asyncio
import copy
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from scripts.external_baselines.scitoolagent import infer_embedding
from scripts.external_baselines.resource_control import (POLICY, RESOURCE_VERSION, ResourcePressure,
    check_pressure, run_guarded, validate_resource_amendment, verify_batch_freeze)
from scripts.external_baselines.common import file_digest
from scripts.external_baselines.experiment import save_new


def sample(**updates):
    values = {'root_private_bytes': 100, 'process_tree_private_bytes': 200,
              'system_commit_bytes': 1000, 'system_commit_limit_bytes': 20*1024**3,
              'system_commit_headroom_bytes': 10*1024**3}
    values.update(updates)
    return values


class EmbeddingThreadTests(unittest.TestCase):
    def test_fresh_callers_use_one_persistent_inference_thread_and_keep_values(self):
        seen = []
        class Embedding:
            def get_text_embedding(self, text):
                seen.append(threading.get_ident())
                return [len(text), 1.0]
            def get_query_embedding(self, text):
                seen.append(threading.get_ident())
                return [len(text), 2.0]
        embed = Embedding()
        for _ in range(12):
            with ThreadPoolExecutor(max_workers=5) as callers:
                out = list(callers.map(lambda t: infer_embedding(embed, t), ['a','bb','ccc']))
            self.assertEqual(out, [[1,1.0],[2,1.0],[3,1.0]])
        self.assertEqual(infer_embedding(embed,'query',True),[5,2.0])
        self.assertEqual(len(set(seen)),1)
        self.assertNotEqual(seen[0],threading.get_ident())

    def test_inference_exception_is_propagated_without_replacement_value(self):
        class Embedding:
            def get_text_embedding(self,text):raise ValueError('offline fixture failure')
        with self.assertRaisesRegex(ValueError,'offline fixture failure'):
            infer_embedding(Embedding(),'text')


class MemoryPolicyTests(unittest.TestCase):
    @unittest.skipUnless(__import__('os').name=='nt','Windows commit sampler')
    def test_repeated_sampling_reuses_the_same_ctypes_pointer_type(self):
        import ctypes
        from scripts.external_baselines.resource_control import memory_sample
        memory_sample()
        count=len(ctypes._pointer_type_cache)
        for _ in range(100):memory_sample()
        self.assertEqual(len(ctypes._pointer_type_cache),count)

    def test_each_resource_threshold_stops_batch(self):
        check_pressure(sample())
        cases = [sample(root_private_bytes=POLICY['max_root_private_bytes']),
                 sample(process_tree_private_bytes=POLICY['max_process_tree_private_bytes']),
                 sample(system_commit_headroom_bytes=POLICY['minimum_system_commit_headroom_bytes']),
                 sample(system_commit_bytes=19*1024**3)]
        for values in cases:
            with self.subTest(values=values),self.assertRaises(ResourcePressure):
                check_pressure(values)


class GuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_pressure_prevents_any_new_unit(self):
        seen=[]
        async def work():seen.append('claimed')
        with self.assertRaises(ResourcePressure):
            await run_guarded(work,POLICY,lambda s:None,
                              sampler=lambda:sample(root_private_bytes=POLICY['max_root_private_bytes']))
        self.assertEqual(seen,[])

    async def test_runtime_pressure_cancels_and_cleans_up_without_retry(self):
        samples=iter([sample(),sample(root_private_bytes=POLICY['max_root_private_bytes'])])
        seen=[]
        async def work():
            seen.append('started')
            try:await asyncio.sleep(10)
            finally:seen.append('cancelled-cleanup')
        policy={**POLICY,'sample_interval_seconds':0.001}
        with self.assertRaises(ResourcePressure):
            await run_guarded(work,policy,lambda s:None,sampler=lambda:next(samples))
        self.assertEqual(seen,['started','cancelled-cleanup'])

    async def test_monitor_failure_pauses_instead_of_disabling_protection(self):
        def fail():raise OSError('offline sample failure')
        async def work():self.fail('must not claim')
        with self.assertRaises(ResourcePressure) as failure:
            await run_guarded(work,POLICY,lambda s:None,sampler=fail)
        self.assertIn('monitor unavailable',str(failure.exception))

    async def test_completed_work_stops_monitor_and_returns_normally(self):
        calls=[]
        async def work():return 'finished'
        out=await run_guarded(work,POLICY,lambda s:calls.append(s),sampler=sample)
        self.assertEqual(out,'finished');self.assertEqual(len(calls),1)


def freezes():
    old={'created_at':'old','dataset':{'hash':'unchanged'},'runtime':{'agent_concurrency':4},
         'methods':{'SciToolAgent':{'adapter':'scitoolagent-official-protein-v1','upstream':{'sha':'official'}},
                    'F1':{'adapter':'native','upstream':{'sha':'original'}}},
         'packages':{'torch':'fixed'},'code_sha256':{'scripts/external_baselines/scitoolagent.py':'old',
         'scientific_algorithm.py':'same'}}
    new=copy.deepcopy(old);new['created_at']='new'
    new['status']='FROZEN_RESOURCE_AMENDMENT_AFTER_INFRASTRUCTURE_INCIDENT'
    new['methods']['SciToolAgent']['adapter']='scitoolagent-official-protein-v1.1-resource'
    new['packages']['psutil']='fixed'
    new['code_sha256'].update({'scripts/external_baselines/scitoolagent.py':'new',
        'scripts/external_baselines/resource_control.py':'new','scripts/prepare_external_resource_repair.py':'new'})
    new['resource_amendment']={'version':RESOURCE_VERSION,'memory_policy':POLICY,'automatic_retry':False}
    return old,new


class LineageTests(unittest.TestCase):
    def test_resource_repair_cannot_change_data_algorithm_budget_or_official_source(self):
        old,new=freezes();validate_resource_amendment(old,new)
        for kind in ('data','algorithm','upstream','concurrency','dependency','retry'):
            amended=copy.deepcopy(new)
            if kind=='data':amended['dataset']['hash']='changed'
            elif kind=='algorithm':amended['code_sha256']['scientific_algorithm.py']='changed'
            elif kind=='upstream':amended['methods']['SciToolAgent']['upstream']['sha']='changed'
            elif kind=='concurrency':amended['runtime']['agent_concurrency']=1
            elif kind=='dependency':amended['packages']['torch']='changed'
            else:amended['resource_amendment']['automatic_retry']=True
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                validate_resource_amendment(old,amended)

    def test_resume_only_accepts_recorded_original_freeze_and_keeps_old_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old_path=root/'old.json';new_path=root/'new.json'
            save_new(old_path,{'version':'old'});save_new(new_path,{'version':'new'})
            amended={'resource_amendment':{'base_freeze_sha256':file_digest(old_path),'base_freeze_path':'old.json'}}
            start={'freeze_sha256':file_digest(old_path),'freeze_path':'old.json'}
            original=copy.deepcopy(start)
            with patch('scripts.external_baselines.resource_control.verify_resource_lineage'):
                verify_batch_freeze(amended,new_path,start,root)
                with self.assertRaises(ValueError):
                    verify_batch_freeze(amended,new_path,{**start,'freeze_sha256':'other'},root)
            self.assertEqual(start,original)


if __name__=='__main__':unittest.main()
