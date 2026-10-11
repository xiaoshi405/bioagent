"""Question-level paired analysis, conservative unresolved bounds, no p-values."""
from __future__ import annotations

import numpy as np


def rates(values):
    values=list(values);n=len(values)
    yes=sum(v is True for v in values);no=sum(v is False for v in values)
    return {'planned':n,'confirmed_success':yes,'confirmed_failure':no,'unresolved':n-yes-no,
            'coverage':(yes+no)/n if n else None,'lower':yes/n if n else None,
            'upper':(n-no)/n if n else None}


def question_pairs(scores,model,baseline,repeats=3):
    """Require all repeats of both methods. Missing is neither zero nor dropped silently."""
    groups={}
    for score in scores:
        if score['model']==model and score['condition'] in ('F1',baseline):
            key=(score['question_index'],score['condition'])
            group=groups.setdefault(key,{})
            if score['repeat'] in group:raise ValueError('Duplicate statistical unit')
            group[score['repeat']]=score['scientific_task_success']
    questions=sorted({q for q,c in groups});paired=[];bounds=[]
    for q in questions:
        values=[groups.get((q,method),{}) for method in ('F1',baseline)]
        raw=[[v.get(r) for r in range(repeats)] for v in values]
        endpoints=[rates(v) for v in raw]
        bounds.append((endpoints[0]['lower']-endpoints[1]['upper'],
                       endpoints[0]['upper']-endpoints[1]['lower']))
        if all(all(v is not None for v in x) for x in raw):
            paired.append((q,float(np.mean(raw[0])-np.mean(raw[1]))))
    return questions,paired,bounds


def paired_effect(scores,model,baseline,replicates=10000,seed=20261001):
    questions,paired,bounds=question_pairs(scores,model,baseline)
    effect_bounds=[100*float(np.mean([b[k] for b in bounds])) for k in (0,1)] if bounds else None
    ci=None;effect=None
    if paired:
        delta=np.asarray([d for q,d in paired],dtype=float)
        rng=np.random.default_rng(seed)
        samples=np.mean(delta[rng.integers(0,len(delta),size=(replicates,len(delta)))],axis=1)
        ci=(100*np.quantile(samples,[0.025,0.975])).tolist()
        effect=100*float(delta.mean())
    return {'model':model,'baseline':baseline,'contrast':'F1 minus baseline',
            'all_question_effect_bounds_pp':effect_bounds,'shared_complete_questions':len(paired),
            'planned_questions':len(questions),'shared_question_coverage':len(paired)/len(questions) if questions else None,
            'complete_pair_effect_pp':effect,'complete_pair_ci95_pp':ci,
            'complete_pair_is_primary_full_matrix':len(paired)==len(questions) and bool(questions),
            'bootstrap_replicates':replicates,'seed':seed,'p_value_reported':False}
