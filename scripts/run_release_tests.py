"""Run explicitly selected offline regression groups; never launch an experiment."""
from pathlib import Path
import argparse,json,sys,unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
GROUPS={
 'core':'artifact_pool type_checker structured_plan initial_artifacts typed_executor fasta_analysis tool_parameters topology_v2_tools tool_request_guard release_maintenance'.split(),
 'runtime':'typed_planning_runtime hierarchical_recovery controlled_repair controlled_repair_v2 plan_sampling evaluation_metrics evaluation_scoring evaluation_trace llm_client embedding_runtime release_entrypoints'.split(),
 'baselines':'external_baselines external_experiment external_scoring external_resources external_operational_analysis codeact_adapter biomni_adapter scitoolagent_adapter f1_audit'.split(),
}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group',choices=[*GROUPS,'all'],default='core');parser.add_argument('--report')
    args=parser.parse_args()
    names=sum(GROUPS.values(),[]) if args.group=='all' else GROUPS[args.group]
    suite=unittest.TestSuite()
    for name in names:suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_'+name+'.py',top_level_dir=str(ROOT/'tests')))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    summary={'group':args.group,'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'success':result.wasSuccessful()}
    print(json.dumps(summary))
    if args.report:
        path=(ROOT/args.report).resolve();path.relative_to(ROOT.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
