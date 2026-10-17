#!/usr/bin/env python3
"""Plot the archived ablation and framework results."""
from pathlib import Path
import csv
import argparse
from collections import defaultdict
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/figures'
MODELS = ('qwen3.7-plus-2026-05-26', 'deepseek-v4-flash-0731')
NAMES = ('Qwen 3.7 Plus', 'DeepSeek V4 Flash')
METHODS = ('ReAct', 'CodeAct', 'SciToolAgent', 'Biomni-A1', 'F1')
METHOD_NAMES = ('ReAct', 'CodeAct', 'SciToolAgent', 'Biomni-A1', 'BioAgent (F1)')
CONDITIONS = ('E0', 'E1', 'E2', 'E3', 'S1', 'L1', 'F1')
METRICS = ('execute_task_completion', 'safe_rejection', 'expected_behavior_outcome')
METRIC_NAMES = ('Execute-task\ncompletion', 'Safe\nrejection', 'Expected-behavior\noutcome')
BLUE, ORANGE = '#7C9BB6', '#D98C85'
COLORS = ('#7C9BB6', '#81B5A4', '#A99BC1', '#C8B47A', '#D98C85')
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['DejaVu Sans'],
    'font.size':7, 'axes.linewidth':.65, 'axes.edgecolor':'0.20',
    'axes.spines.top':False, 'axes.spines.right':False, 'figure.facecolor':'white',
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none'})


def bar(ax, x, rows, color, width, size, decimals, gap=3.0, label_anchor=None):
    mean = float(rows[0]['aggregate_value'])
    points = np.array([float(r['value']) for r in rows])
    ax.bar(x, mean, width=width, color=color, edgecolor='0.20', linewidth=.3, zorder=2)
    ax.scatter(x + np.array([-.22,0,.22])*width, points, s=3.4,
               color='0.12', edgecolors='none', zorder=4)
    anchor = max(mean,float(max(points))) if label_anchor is None else label_anchor
    return ax.annotate(f'{mean:.{decimals}f}', (x,anchor),
        xytext=(0,gap), textcoords='offset points', ha='center', va='bottom',
        fontsize=size, fontfamily='Arial', color='0.10', clip_on=False, zorder=5)


def clear_labels(fig, texts, has_grid=False):
    # Preserve each label's own bar-center anchor. Stagger only when measured
    # glyphs would overlap, then clear the original horizontal grid lines.
    for iteration in range(15):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        moved = False
        for index, text in enumerate(texts):
            bbox = text.get_window_extent(renderer)
            for previous in texts[:index]:
                if previous.axes is not text.axes:
                    continue
                other = previous.get_window_extent(renderer)
                if bbox.x0 < other.x1+.7 and bbox.x1 > other.x0-.7 and bbox.y0 < other.y1+1.0 and bbox.y1 > other.y0-1.0:
                    dx,dy = text.get_position()
                    text.set_position((dx,dy+(other.y1+1.5-bbox.y0)*72/fig.dpi))
                    moved = True
                    bbox = text.get_window_extent(renderer)
            if has_grid:
                grids = [text.axes.transData.transform((0,y))[1] for y in text.axes.get_yticks()]
                crossed = [y for y in grids if bbox.y0-1 <= y <= bbox.y1+1]
                if crossed:
                    dx,dy = text.get_position()
                    text.set_position((dx,dy+(max(crossed)+1.5*fig.dpi/72-bbox.y0)*72/fig.dpi))
                    moved = True
        if not moved:
            break
    fig.canvas.draw()


def export(fig, axes, titles, name):
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(OUT / f'{name}.{suffix}', dpi=600)
    plt.close(fig)


def framework(groups):
    fig,axes = plt.subplots(2,2,figsize=(180/25.4,130/25.4))
    fig.subplots_adjust(left=.09,right=.985,top=.865,bottom=.15,wspace=.24,hspace=.55)
    titles,texts = [],[]
    for mi,(model,name) in enumerate(zip(MODELS,NAMES)):
        ax=axes[0,mi]
        ax.set_ylim(0,114); ax.set_yticks(range(0,101,20)); ax.set_xlim(-.5,2.5)
        ax.set_ylabel('Rate (%)',fontsize=7,labelpad=4)
        ax.tick_params(length=2.5,width=.65,labelsize=6.5)
        for center,metric in enumerate(METRICS):
            pitch=.91/5
            for j,(method,color) in enumerate(zip(METHODS,COLORS)):
                texts.append(bar(ax,center+(j-2)*pitch,groups[model,method,metric],color,pitch*.89,5.4,1,gap=6.2))
        ax.set_xticks(range(3),METRIC_NAMES,fontsize=6.5)
        titles.append(ax.set_title(f'{chr(65+mi)}. {name}',loc='left',fontweight='bold',fontsize=8,pad=7))
    for mi,(model,name) in enumerate(zip(MODELS,NAMES)):
        ax=axes[1,mi]
        ax.set_ylim(0,40); ax.set_yticks(range(0,41,10)); ax.set_xlim(-.6,4.6)
        ax.set_ylabel('Mean tokens per run (thousands)',fontsize=7,labelpad=4)
        ax.tick_params(length=2.5,width=.65,labelsize=6.5)
        for j,(method,color) in enumerate(zip(METHODS,COLORS)):
            texts.append(bar(ax,j,groups[model,method,'observed_mean_total'],color,.72*.89,6.1,2,gap=6.2))
        ax.set_xticks(range(5),['ReAct','CodeAct','SciTool\nAgent','Biomni-A1','BioAgent\n(F1)'],fontsize=6.2)
        titles.append(ax.set_title(f'{chr(67+mi)}. {name}: model interaction',loc='left',fontweight='bold',fontsize=8,pad=7))
    fig.legend([Patch(facecolor=c,edgecolor='0.20',linewidth=.3) for c in COLORS],METHOD_NAMES,
        loc='upper center',bbox_to_anchor=(.52,.986),ncol=5,fontsize=7,frameon=False,
        columnspacing=1.25,handlelength=1.15,handletextpad=.45)
    fig.text(.09,.065,'Rates: planned-denominator lower bounds; dots: three repeats.',fontsize=6,color='0.3')
    fig.text(.09,.030,'Tokens: pooled recorded means; dots: repeat means.',fontsize=6,color='0.3')
    clear_labels(fig,texts)
    export(fig,list(axes.flat),titles,'framework_comparison_selected')


def ablation(groups):
    fig,axes = plt.subplots(1,3,figsize=(180/25.4,67/25.4))
    fig.subplots_adjust(left=.055,right=.995,top=.75,bottom=.17,wspace=.30)
    titles,texts = [],[]
    labels=('Completion rate (%)','Safe rejection rate (%)','Expected-behavior outcome (%)')
    for mi,(ax,metric,label) in enumerate(zip(axes,METRICS,labels)):
        ax.axvspan(-.55,3.55,color='0.93',linewidth=0,zorder=.2)
        ax.set_axisbelow(True); ax.grid(axis='y',color='0.82',linewidth=.32)
        ax.set_xlim(-.62,6.62); ax.set_ylim(0,100); ax.set_yticks(range(0,101,20))
        ax.tick_params(length=1.6,width=.45,labelsize=5.8)
        ax.set_ylabel(label,fontsize=6,labelpad=3)
        ax.set_xticks(range(7),CONDITIONS)
        # Qwen is on the left, DeepSeek on the right.
        for center,condition in enumerate(CONDITIONS):
            pair_top=max(float(r[field]) for model in MODELS
                         for r in groups[model,condition,metric] for field in ('value','aggregate_value'))
            for offset,model,color in zip((-.18,.18),MODELS,(BLUE,ORANGE)):
                texts.append(bar(ax,center+offset,groups[model,condition,metric],color,.34,5.0,1,
                                 gap=2.5,label_anchor=pair_top))
        title=METRIC_NAMES[mi].replace('\n',' ')
        titles.append(ax.set_title(f'{chr(65+mi)}. {title}',loc='left',fontsize=6.5,fontweight='bold',pad=20))
    fig.legend([Patch(facecolor=c,edgecolor='0.20',linewidth=.3) for c in (BLUE,ORANGE)],NAMES,
        loc='upper center',bbox_to_anchor=(.5,.995),ncol=2,frameon=False,fontsize=6,
        columnspacing=2.2,handlelength=1.4,handletextpad=.6)
    fig.text(.055,.055,'Shaded region: staged BioAgent conditions E0–E3. Panel C: all 450 model-condition units. Dots: three repeats.',
        fontsize=5,color='0.35',style='italic')
    clear_labels(fig,texts,has_grid=True)
    export(fig,list(axes),titles,'component_ablation_selected')


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/figures')
    args = parser.parse_args()
    OUT = (ROOT / args.output).resolve()
    OUT.relative_to(ROOT)
    if not OUT.relative_to(ROOT).parts or OUT.relative_to(ROOT).parts[0] != 'outputs':
        parser.error('Choose an output directory under outputs/.')
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ('fig4_source_data.csv', 'fig5_source_data.csv'):
        groups = defaultdict(list)
        with (ROOT / 'results/figures' / filename).open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
            for row in records:
                groups[row['model'], row['method'], row['metric']].append(row)
        kinds = {row['figure'] for row in records}
        assert len(kinds) == 1
        draw = {'framework': framework, 'ablation': ablation}[kinds.pop()]
        for rows in groups.values():
            rows.sort(key=lambda row: int(row['repeat']))
            assert [int(row['repeat']) for row in rows] == [0, 1, 2]
        draw(groups)
    print(OUT)


if __name__=='__main__':
    main()
