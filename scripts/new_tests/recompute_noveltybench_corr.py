"""Recompute NoveltyBench validity/specificity (Overall block) from CACHED test
scores + current benchmarks.json. Faithful to make_figures.load_composite_scores
+ score_evals.partial_pearson_multi (controls = arena_overall, mmlu_pro).
No re-embedding, no main env.
"""
import json, numpy as np
from pathlib import Path
from scipy import stats

RESULTS = Path("data/dat_eval/run_v1/downstream/scores_v1/results")
BENCH = json.load(open("configs/comb_eval/benchmarks.json"))
BM = "noveltybench_utility"

def load_composite():
    me = json.load(open(RESULTS/"multi_embed_scores.json"))
    embs = sorted(me); tasks=["dat","cdat","cdat_novelty","cdat_appropriateness","pace"]
    models = sorted({m for e in embs for m in me[e]})
    comp={}
    for t in tasks:
        stt={}
        for e in embs:
            vals=[me[e].get(m,{}).get(t) for m in models]
            vals=[v for v in vals if v is not None and not (isinstance(v,float) and (np.isnan(v) or v==0))]
            if vals: stt[e]=(float(np.mean(vals)), float(np.std(vals)) or 1.0)
        for m in models:
            zs=[]
            for e in embs:
                if e not in stt: continue
                v=me[e].get(m,{}).get(t)
                if v is None or (isinstance(v,float) and (np.isnan(v) or v==0)): continue
                mu,sd=stt[e]; zs.append((v-mu)/sd)
            if zs: comp.setdefault(m,{})[t]=float(np.mean(zs))
    return comp

def _pp(r,n,k):
    df=n-2-k
    if df<=0 or not -1<r<1: return float("nan")
    t=r*np.sqrt(df/(1-r*r)); return float(2*stats.t.sf(abs(t),df))

def specificity(target,predictor,controls):
    X=np.column_stack([np.ones(len(target))]+[np.asarray(c) for c in controls])
    beta,*_=np.linalg.lstsq(X,target,rcond=None); resid=target-X@beta
    r=float(stats.pearsonr(predictor,resid).statistic); return r,_pp(r,len(target),len(controls))

comp=load_composite()
LAB={"dat":"DAT","cdat":"CDAT","cdat_novelty":"CDAT-N","cdat_appropriateness":"CDAT-A","pace":"PACE"}
print(f"{'test':8s} {'n':>3s} {'validity':>18s} {'specificity':>18s}")
for t,lab in LAB.items():
    xs,ys,ao,mp=[],[],[],[]
    for m,md in comp.items():
        b=BENCH.get(m,{})
        if t not in md: continue
        if any(b.get(f) in (None,"","---") for f in (BM,"arena_overall","mmlu_pro")): continue
        xs.append(md[t]); ys.append(b[BM]); ao.append(b["arena_overall"]); mp.append(b["mmlu_pro"])
    n=len(xs)
    if n<4: print(f"{lab:8s} {n:>3d}  (too few)"); continue
    xs,ys,ao,mp=map(np.asarray,(xs,ys,ao,mp))
    v=float(stats.pearsonr(xs,ys).statistic); vp=float(stats.pearsonr(xs,ys).pvalue)
    s,sp=specificity(ys,xs,[ao,mp])
    def star(p): return "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else ""
    print(f"{lab:8s} {n:>3d}   {v:+.2f}{star(vp):<3s} (p={vp:.3f})   {s:+.2f}{star(sp):<3s} (p={sp:.3f})")
