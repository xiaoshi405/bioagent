#!/usr/bin/env python3
"""Restyle two archived result plots; no scoring, SEM, CI, or paper changes.

Shared typography and physical geometry apply to both figures. Framework
comparison retains four panels. Ablation restores metric panels, ordered
conditions, paired model bars, and the E0--E3 background band.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "paper/figure/drafts/scitoolagent_style_20261003"
OUT = ROOT / "paper/figure/drafts/unified_style_20261003"
MODELS = ("qwen3.7-plus-2026-05-26", "deepseek-v4-flash-0731")
MODEL_NAMES = ("Qwen 3.7 Plus", "DeepSeek V4 Flash")
METHODS = ("ReAct", "CodeAct", "SciToolAgent", "Biomni-A1", "F1")
METHOD_NAMES = ("ReAct", "CodeAct", "SciToolAgent", "Biomni-A1", "BioAgent (F1)")
CONDITIONS = ("E0", "E1", "E2", "E3", "S1", "L1", "F1")
FRAMEWORK_METRICS = ("expected_behavior_outcome", "execute_task_completion", "safe_rejection")
FRAMEWORK_LABELS = ("Expected\nbehavior", "Completion", "Safe\nrejection")
ABLATION_METRICS = ("execute_task_completion", "safe_rejection", "expected_behavior_outcome")
ABLATION_LABELS = ("Execute-task completion", "Safe rejection", "Expected-behavior outcome")

# All physical lengths and typography are shared, independent of grid shape.
STYLE = dict(
    primary="#3E6FB0", accent="#E58B47", text="#333333", grid="#E6E6E6",
    baselines=["#6E8FAF", "#88A2BA", "#A1B3C3", "#BAC5CF"],
    neutral="#B8B8B8", shade="#F3F6FA", font="Arial",
    title_pt=8.0, axis_pt=7.0, tick_pt=6.5, value_pt=6.0, legend_pt=7.0,
    note_pt=6.0, title_pad_pt=12.0,
    bar_width_mm=3.8, paired_gap_mm=.8, bar_edge_pt=.35, axis_pt_width=.65,
    grid_pt=.45, dot_area_pt2=4.0, dot_alpha=.7,
    width_mm=180.0, panel_height_mm=34.0,
    left_mm=13.0, right_mm=3.0, top_mm=18.0, bottom_mm=18.0,
    horizontal_gap_mm=13.0, vertical_gap_mm=18.0, dpi=600,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_source(name):
    # Read the already-verified plotted numbers; never recompute a metric.
    rows = read_csv(INPUT / f"{name}_source_data.csv")
    assert len(rows) == (120 if name == "framework" else 126)
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"], row["method"], row["metric"]].append(row)
    for group in groups.values():
        group.sort(key=lambda r: int(r["repeat"]))
        assert [int(r["repeat"]) for r in group] == [0, 1, 2]
        assert len({r["aggregate_value"] for r in group}) == 1
    return rows, groups


def make_grid(nrows, ncols):
    width = STYLE["width_mm"]
    height = STYLE["top_mm"] + STYLE["bottom_mm"] + nrows * STYLE["panel_height_mm"] + (nrows-1)*STYLE["vertical_gap_mm"]
    plot_width = (width-STYLE["left_mm"]-STYLE["right_mm"]-(ncols-1)*STYLE["horizontal_gap_mm"])/ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(width/25.4,height/25.4), squeeze=False)
    fig.subplots_adjust(left=STYLE["left_mm"]/width, right=1-STYLE["right_mm"]/width,
        bottom=STYLE["bottom_mm"]/height, top=1-STYLE["top_mm"]/height,
        wspace=STYLE["horizontal_gap_mm"]/plot_width,
        hspace=STYLE["vertical_gap_mm"]/STYLE["panel_height_mm"])
    return fig, axes, plot_width, height


def axis_style(ax, width, tokens=False):
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=STYLE["grid"], linewidth=STYLE["grid_pt"])
    ax.tick_params(length=2.5, width=STYLE["axis_pt_width"], labelsize=STYLE["tick_pt"], colors=STYLE["text"])
    ax.set_xlim(0,width)
    ax.set_ylim(0,35 if tokens else 100)
    ax.set_yticks(np.arange(0,36,10) if tokens else np.arange(0,101,20))
    ylabel = "Mean model-interaction tokens\nper run (thousands)" if tokens else "Rate (%)"
    ax.set_ylabel(ylabel,fontsize=STYLE["axis_pt"],labelpad=4,color=STYLE["text"])


def draw_bar(ax, center, rows, color, decimals):
    value = float(rows[0]["aggregate_value"])
    points = np.array([float(r["value"]) for r in rows])
    assert np.all(np.isfinite(points)) and np.isfinite(value)
    ax.bar(center,value,width=STYLE["bar_width_mm"],color=color,
        edgecolor=STYLE["text"],linewidth=STYLE["bar_edge_pt"],zorder=2)
    ax.scatter(center+np.array([-.22,0,.22])*STYLE["bar_width_mm"],points,
        s=STYLE["dot_area_pt2"],color=STYLE["text"],alpha=STYLE["dot_alpha"],
        edgecolors="none",zorder=3)
    # The label retains the frozen aggregate even if a repeat mean is higher.
    return ax.annotate(f"{value:.{decimals}f}",(center,float(np.max(points))),
        xytext=(0,6.0),textcoords="offset points",ha="center",va="bottom",
        fontsize=STYLE["value_pt"],color=STYLE["text"],clip_on=False,zorder=4)


def clear_numeric_labels_from_grids(fig, annotations):
    # Keep the full requested grid. Move only text, using measured glyph boxes,
    # just above a crossed gridline. No white masks, altered axes, or data edits.
    for _ in range(3):
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        moved=False
        for text in annotations:
            ax=text.axes
            bbox=text.get_window_extent(renderer)
            grid_y=[ax.transData.transform((0,y))[1] for y in ax.get_yticks()]
            crossed=[y for y in grid_y if bbox.y0-1 <= y <= bbox.y1+1]
            if crossed:
                offset_px=max(crossed)+1.5*fig.dpi/72-bbox.y0
                x,y=text.get_position()
                text.set_position((x,y+offset_px*72/fig.dpi))
                moved=True
        if not moved:
            break
    fig.canvas.draw()


def shared_legend(fig, labels, colors, height):
    handles=[Patch(facecolor=c,edgecolor=STYLE["text"],linewidth=STYLE["bar_edge_pt"]) for c in colors]
    fig.legend(handles,labels,ncol=len(labels),loc="upper center",
        bbox_to_anchor=(.5,1-3/height),frameon=False,fontsize=STYLE["legend_pt"],
        handlelength=1.3,handletextpad=.5,columnspacing=1.3)


def export(fig, axes, titles, annotations, name, qa_tools):
    clear_numeric_labels_from_grids(fig,annotations)
    sys.path.insert(0,str(qa_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment, audit_layout_manifest, write_json_report, exit_code
    letters=[chr(65+i) for i in range(len(axes))]
    report=require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=letters,
        tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    layout=report["layout"]
    for panel,title,letter in zip(layout["panels"],titles,letters):
        xy=title.get_transform().transform(title.get_position())
        point=fig.dpi_scale_trans.inverted().transform(xy)*72
        panel["panel_label"]=letter
        panel["panel_label_anchor_pt"]=list(map(float,point))
    report=audit_layout_manifest(layout,tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True)
    prefix=OUT/name
    write_json_report(report,prefix.with_suffix(".alignment.json"))
    assert exit_code(report,strict=True)==0
    fig.savefig(prefix.with_suffix(".pdf"))
    fig.savefig(prefix.with_suffix(".svg"))
    fig.savefig(prefix.with_suffix(".png"),dpi=STYLE["dpi"])
    plt.close(fig)


def framework(groups,qa_tools):
    fig,axes,width,height=make_grid(2,2)
    titles,annotations=[],[]
    colors=STYLE["baselines"]+[STYLE["accent"]]
    for i,model in enumerate(MODELS):
        ax=axes[0,i]
        axis_style(ax,width)
        centers=(np.arange(3)+.5)*width/3
        for center,metric in zip(centers,FRAMEWORK_METRICS):
            for j,(method,color) in enumerate(zip(METHODS,colors)):
                x=center+(j-2)*(STYLE["bar_width_mm"]+STYLE["paired_gap_mm"])
                annotations.append(draw_bar(ax,x,groups[model,method,metric],color,1))
        ax.set_xticks(centers,FRAMEWORK_LABELS,fontsize=STYLE["tick_pt"])
        titles.append(ax.set_title(f"{chr(65+i)}. {'Qwen' if i==0 else 'DeepSeek'} – execution behavior",
            loc="left",fontsize=STYLE["title_pt"],fontweight="bold",pad=STYLE["title_pad_pt"]))
    for i,model in enumerate(MODELS):
        ax=axes[1,i]
        axis_style(ax,width,tokens=True)
        centers=(np.arange(5)+.5)*width/5
        for x,method,color in zip(centers,METHODS,colors):
            annotations.append(draw_bar(ax,x,groups[model,method,"observed_mean_total"],color,2))
        ax.set_xticks(centers,["ReAct","CodeAct","SciToolAgent","Biomni-A1","BioAgent\n(F1)"],fontsize=STYLE["tick_pt"])
        titles.append(ax.set_title(f"{chr(67+i)}. {'Qwen' if i==0 else 'DeepSeek'} – model-interaction tokens",
            loc="left",fontsize=STYLE["title_pt"],fontweight="bold",pad=STYLE["title_pad_pt"]))
    shared_legend(fig,METHOD_NAMES,colors,height)
    fig.text(STYLE["left_mm"]/180,5/height,
        "Rates: planned-denominator lower bounds. Tokens: recorded pooled means. Dots: three repeats.",
        fontsize=STYLE["note_pt"],color=STYLE["text"])
    export(fig,list(axes.flat),titles,annotations,"framework_comparison_final",qa_tools)
    return height


def ablation(groups,qa_tools):
    fig,axes,width,height=make_grid(3,1)
    titles,annotations=[],[]
    for i,(metric,label) in enumerate(zip(ABLATION_METRICS,ABLATION_LABELS)):
        ax=axes[i,0]
        axis_style(ax,width)
        ax.axvspan(0,4*width/7,color=STYLE["shade"],linewidth=0,zorder=.1)
        centers=(np.arange(7)+.5)*width/7
        for center,condition in zip(centers,CONDITIONS):
            for j,(model,color) in enumerate(zip(MODELS,[STYLE["primary"],STYLE["accent"]])):
                x=center+(j-.5)*(STYLE["bar_width_mm"]+STYLE["paired_gap_mm"])
                annotations.append(draw_bar(ax,x,groups[model,condition,metric],color,1))
        ax.set_xticks(centers,CONDITIONS,fontsize=STYLE["tick_pt"])
        titles.append(ax.set_title(f"{chr(65+i)}. {label}",loc="left",fontsize=STYLE["title_pt"],
            fontweight="bold",pad=STYLE["title_pad_pt"]))
    shared_legend(fig,MODEL_NAMES,[STYLE["primary"],STYLE["accent"]],height)
    fig.text(STYLE["left_mm"]/180,5/height,
        "Shaded region: staged BioAgent conditions E0–E3. Dots: three repeats.",
        fontsize=STYLE["note_pt"],color=STYLE["text"])
    export(fig,list(axes.flat),titles,annotations,"component_ablation_final",qa_tools)
    return height


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-tools-dir",type=Path,default=Path.home()/".codex/skills/nature-figure/scripts")
    args=parser.parse_args()
    font_manager.findfont(STYLE["font"],fallback_to_default=False)
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":[STYLE["font"]],
        "font.size":STYLE["axis_pt"],"text.color":STYLE["text"],"axes.labelcolor":STYLE["text"],
        "axes.linewidth":STYLE["axis_pt_width"],"axes.edgecolor":STYLE["text"],
        "axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white",
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[ROOT/"paper/bioagent_main.tex",ROOT/"paper/bioagent_main.pdf",ROOT/"paper/bioagent_supplement.tex",ROOT/"paper/bioagent_supplement.pdf",ROOT/"paper/references.bib"]
    protected+=list((ROOT/"paper/figure").glob("*.*"))
    protected+=list(INPUT.glob("*"))
    before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    framework_rows,framework_groups=load_source("framework")
    ablation_rows,ablation_groups=load_source("ablation")
    for name,rows in [("framework",framework_rows),("ablation",ablation_rows)]:
        keys=[k for k in rows[0] if not k.startswith("error_")]
        with (OUT/f"{name}_source_data.csv").open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    heights={"framework":framework(framework_groups,args.qa_tools_dir),"ablation":ablation(ablation_groups,args.qa_tools_dir)}
    assert all(sha(ROOT/p)==h for p,h in before.items())
    manifest=dict(style=STYLE,height_mm=heights,error_bars="none",
        data_changes=False,scope="visual redesign only; no paper or original figure replacement",
        plotted_rates="unchanged planned-denominator lower bounds; one decimal label",
        plotted_tokens="unchanged archived pooled observed means; two decimal labels, thousands",
        dots="three archived repeat values; no statistical uncertainty implied",
        protected_before_after_sha256=before,
        sources_sha256={name:sha(INPUT/f"{name}_source_data.csv") for name in ("framework","ablation")})
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved two unified-style previews to {OUT}")


if __name__=="__main__":
    main()
