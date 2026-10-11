"""Reconstruct archived execution metrics and final plots without model calls."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,importlib.util,json,math,sys

ROOT=Path(__file__).resolve().parents[1]
METRICS=('expected_behavior_outcome','execute_task_completion','safe_rejection')

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def boolean(s):
    if s in ('True','true','1'):return True
    if s in ('False','false','0'):return False
    if s in ('','None','null'):return None
    raise ValueError('Unknown archived boolean: '+str(s))
def write_csv(path,records):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
def close(actual,expected):assert math.isclose(float(actual),float(expected),rel_tol=1e-9,abs_tol=1e-9),(actual,expected)

def aggregate_runs(runs):
    groups=defaultdict(list)
    for r in runs:
        for metric in METRICS:
            if metric=='execute_task_completion' and r['expected_behavior']=='reject':continue
            if metric=='safe_rejection' and r['expected_behavior']!='reject':continue
            groups[r['model'],r['condition'],metric].append(boolean(r[metric]))
    result=[]
    for (model,method,metric),values in sorted(groups.items()):
        yes=sum(v is True for v in values);no=sum(v is False for v in values);unknown=sum(v is None for v in values);n=len(values)
        result.append(dict(model=model,method=method,metric=metric,planned=n,confirmed_success=yes,confirmed_failure=no,unresolved=unknown,coverage=(yes+no)/n,lower=yes/n,upper=(yes+unknown)/n))
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/reproduced');parser.add_argument('--no-plots',action='store_true');args=parser.parse_args()
    out=(ROOT/args.output).resolve();out.relative_to(ROOT.resolve());out.mkdir(parents=True,exist_ok=True)
    base=ROOT/'results/framework';runs=rows(base/'run_analysis.csv');resources=rows(base/'resource_runs.csv')
    assert len(runs)==4500 and len({r['unit_id'] for r in runs})==4500
    recorded=sum(r['scoring_status']=='recorded' for r in runs);assert recorded==4492
    actual=aggregate_runs(runs);expected={(r['model'],r['method'],r['metric']):r for r in rows(base/'main_metrics.csv')}
    assert len(actual)==len(expected)==30
    for row in actual:
        reference=expected[row['model'],row['method'],row['metric']]
        for key in ['planned','confirmed_success','confirmed_failure','unresolved']:assert int(reference[key])==row[key],(key,row)
        for key in ['coverage','lower','upper']:close(reference[key],row[key])
    write_csv(out/'main_metrics.csv',actual)
    by_unit={r['unit_id']:r for r in resources}
    assert len(by_unit)==recorded and set(by_unit)<={r['unit_id'] for r in runs}
    plotted=[];checks=0
    for name in ['fig4_source_data.csv','fig5_source_data.csv']:
        values=rows(ROOT/'results/figures'/name);plotted.append(values)
        for row in values:
            if row['figure']!='framework':continue
            if row['metric'] in METRICS:
                chosen=[r for r in runs if r['model']==row['model'] and r['condition']==row['method'] and int(r['repeat'])==int(row['repeat']) and (row['metric']!='execute_task_completion' or r['expected_behavior']!='reject') and (row['metric']!='safe_rejection' or r['expected_behavior']=='reject')]
                n=len(chosen);yes=sum(boolean(r[row['metric']]) is True for r in chosen)
                assert n==int(row['denominator']) and yes==int(row['numerator']);close(row['value'],100*yes/n)
                close(row['aggregate_value'],100*float(expected[row['model'],row['method'],row['metric']]['lower']))
            elif row['metric']=='observed_mean_total':
                chosen=[r for r in resources if r['model']==row['model'] and r['condition']==row['method'] and int(r['repeat'])==int(row['repeat'])]
                for r in chosen:
                    if not r['observed_input_tokens'] or not r['observed_output_tokens']:
                        assert r['observed_input_tokens']==r['observed_output_tokens']=='' and int(r['llm_responses'])==0
                tokens=[int(r['observed_input_tokens'] or 0)+int(r['observed_output_tokens'] or 0) for r in chosen]
                assert len(chosen)==int(row['denominator']) and sum(tokens)==int(row['numerator'])
                close(row['value'],sum(tokens)/len(tokens)/1000)
                pooled=[int(r['observed_input_tokens'] or 0)+int(r['observed_output_tokens'] or 0) for r in resources if r['model']==row['model'] and r['condition']==row['method']]
                close(row['aggregate_value'],sum(pooled)/len(pooled)/1000)
            else:raise AssertionError('Unknown framework plot metric: '+row['metric'])
            checks+=1
    report={'status':'PASS','planned_units':4500,'recorded_units':recorded,'historical_missing':4500-recorded,'summary_cells_verified':len(actual),'framework_repeat_plot_cells_verified':checks,'online_model_calls':0,'new_experimental_outcomes':0}
    if not args.no_plots:
        folder=ROOT/'reproduction/plotting';sys.path.insert(0,str(folder))
        spec=importlib.util.spec_from_file_location('release_plot',folder/'fig_results_selected_style.py');plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
        plot.OUT=out;plot.BLUE,plot.ORANGE='#7C9BB6','#D98C85';plot.COLORS=('#7C9BB6','#81B5A4','#A99BC1','#C8B47A','#D98C85')
        def export(fig,axes,titles,stem):
            for suffix in ['pdf','png','svg']:fig.savefig(out/(stem+'.'+suffix),dpi=600)
            plot.plt.close(fig)
        plot.export=export
        for values in plotted:
            groups=defaultdict(list)
            for row in values:groups[row['model'],row['method'],row['metric']].append(row)
            for group in groups.values():group.sort(key=lambda row:int(row['repeat']));assert [int(r['repeat']) for r in group]==[0,1,2]
            if values[0]['figure']=='framework':plot.framework(groups)
            else:plot.ablation(groups)
        report['plots_generated']=2
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
