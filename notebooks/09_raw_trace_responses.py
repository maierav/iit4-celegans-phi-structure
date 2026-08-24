# %% [markdown]
# # 09 — Raw-trace responses of the eight analysed neurons
#
# Averaged ΔF/F₀ time courses (PSTH-style) for the **four core interneurons**
# (AIBL, AVEL, AVAL, RIML — the tentative main complex) and the **four
# chemosensory neurons** (ASEL, ASER, AWAL, AWCL), with variance.
#
# Four views:
#
# | | grouping | variance across |
# |---|---|---|
# | Fig 20 | stimulus **class** | epochs |
# | Fig 21a | individual **stimulus** | 3 repeats, one animal |
# | Fig 21b | individual **stimulus** | 24 epochs, all animals |
#
# This is deliberately *upstream* of any IIT machinery. It answers a question the
# Φ-structure analysis cannot: **is the attractant/repellent distinction present
# in the fluorescence at all?** It is — and that matters for how the Φ-structure
# null is read.
#
# **Outputs:** `figures/fig20_psth_classes.pdf`,
# `figures/fig21_psth_stimuli_{one,all}.pdf`, `results/psth_summary.csv`,
# `results/psth_class_test.csv`

# %%
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("iit4-celegans-phi-structure"):
        subprocess.run(["git", "clone", "--quiet",
                        "https://github.com/maierav/iit4-celegans-phi-structure.git"],
                       check=True)
    os.chdir("iit4-celegans-phi-structure")

REPO_ROOT = os.path.abspath(os.getcwd())
if os.path.basename(REPO_ROOT) == "notebooks":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
    os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import ast
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import ces_hypergraph as ch

BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False})

RECS = list(ch.HERM_DRIVE_IDS)
INTER = ["AIBL", "AVEL", "AVAL", "RIML"]          # core / interneuron quartet
SENS = ["ASEL", "ASER", "AWAL", "AWCL"]           # chemosensory quartet
ALL8 = INTER + SENS
STIMULI = list(ch.STIMULUS_CLASS)
CLASSES_OF = ch.STIMULUS_CLASS
CLASS_LIST = ["attractant", "repellent", "control"]
CCOL = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}

FS = ch.SAMPLING_RATE_HZ
PRE_S, POST_S = 5.0, 25.0
PRE_N, POST_N = round(PRE_S * FS), round(POST_S * FS)
tvec = (np.arange(PRE_N + POST_N) - PRE_N) / FS
resp_win = (tvec >= 0) & (tvec <= 15)              # the 15 s stimulus presentation

for rec in RECS:
    ch.ensure_recording(rec)


def raw_traces(rec, neurons):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    out = {}
    for nm in neurons:
        row = d.loc[d.neuron == nm, tc]
        out[nm] = row.iloc[0].astype(float).values if len(row) else None
    onsets = defaultdict(list)
    for t, lab in ast.literal_eval(d.iloc[0]["stimulus"]):
        onsets[lab].append(int(t))
    return out, onsets


RAW = {r: raw_traces(r, ALL8) for r in RECS}
missing = {r: [n for n, v in RAW[r][0].items() if v is None] for r in RECS}
print("neurons missing in any recording:", {k: v for k, v in missing.items() if v} or "none")
print(f"trace length {len(RAW[RECS[0]][0]['AIBL'])} frames at {FS:.4f} Hz")
print(f"window: {PRE_S:.0f} s pre to {POST_S:.0f} s post onset "
      f"({PRE_N} + {POST_N} = {PRE_N+POST_N} samples)")
print("mean NaN fraction per neuron:",
      {nm: round(100*float(np.mean([np.isnan(RAW[r][0][nm]).mean() for r in RECS])), 2)
       for nm in ALL8})

# %% [markdown]
# ## Epoch extraction and ΔF/F₀
#
# Each epoch is normalised by **its own** pre-onset baseline,
# F₀ = mean fluorescence over the 5 s before onset, so ΔF/F₀ = (F − F₀)/|F₀|.
# Per-epoch normalisation removes slow drift and between-animal differences in
# absolute brightness, which would otherwise dominate the average.

# %%
def epoch_matrix(neuron, stimulus, recs):
    """(n_epochs, n_samples) matrix of ΔF/F₀ aligned to stimulus onset."""
    rows = []
    for r in recs:
        trace, onsets = RAW[r][0][neuron], RAW[r][1]
        for t0 in sorted(onsets.get(stimulus, [])):
            a, b = t0 - PRE_N, t0 + POST_N
            if a < 0 or b > len(trace):
                continue
            seg = trace[a:b].astype(float)
            f0 = np.nanmean(seg[:PRE_N])
            if not np.isfinite(f0) or abs(f0) < 1e-9:
                continue
            rows.append((seg - f0) / abs(f0))
    return np.array(rows)


def summarise(M):
    if len(M) == 0:
        return None
    return dict(mean=np.nanmean(M, 0), sem=np.nanstd(M, 0, ddof=1) / np.sqrt(len(M)),
                sd=np.nanstd(M, 0, ddof=1), n=len(M))


def psth(neuron, stimulus, recs):
    return summarise(epoch_matrix(neuron, stimulus, recs))


def psth_class(neuron, cls, recs):
    """Pool every epoch of every stimulus in one class."""
    parts = [epoch_matrix(neuron, s, recs) for s in STIMULI if CLASSES_OF[s] == cls]
    parts = [p for p in parts if len(p)]
    return summarise(np.vstack(parts)) if parts else None


print("epochs available, AIBL:")
for s in STIMULI[:3]:
    print(f"  {s:<15} one animal {len(epoch_matrix('AIBL', s, [RECS[0]]))}, "
          f"all animals {len(epoch_matrix('AIBL', s, RECS))}")
for c in CLASS_LIST:
    print(f"  class {c:<11} all animals {psth_class('AIBL', c, RECS)['n']}")

# %% [markdown]
# ## Peak-response summary

# %%
rows = []
for scope, recs in [("one_animal", [RECS[0]]), ("all_animals", RECS)]:
    for nm in ALL8:
        for kind, groups, fn in [("class", CLASS_LIST, psth_class),
                                 ("stimulus", STIMULI, psth)]:
            for grp in groups:
                p = fn(nm, grp, recs)
                if p is None:
                    continue
                amp = p["mean"][resp_win]
                k = int(np.nanargmax(np.abs(amp)))
                sem_k = float(p["sem"][resp_win][k])
                rows.append(dict(scope=scope, neuron=nm, group=grp, kind=kind,
                                 n_epochs=p["n"], peak_dFF=round(float(amp[k]), 4),
                                 peak_time_s=round(float(tvec[resp_win][k]), 2),
                                 sem_at_peak=round(sem_k, 4),
                                 snr=round(abs(float(amp[k])) / max(sem_k, 1e-9), 2)))
psth_tab = pd.DataFrame(rows)
psth_tab.to_csv("results/psth_summary.csv", index=False)
cl = psth_tab[(psth_tab.kind == "class") & (psth_tab.scope == "all_animals")]
print("ALL ANIMALS — peak ΔF/F₀ within the 15 s stimulus window:")
print(cl.pivot(index="neuron", columns="group", values="peak_dFF").reindex(ALL8).round(3).to_string())

# %% [markdown]
# ## Does the raw signal discriminate the classes?
#
# One value per epoch — the mean ΔF/F₀ over the 15 s stimulus window — then a
# rank test of attractant epochs against repellent epochs, per neuron, with a
# Holm correction across the eight neurons.

# %%
def epoch_amplitudes(neuron, cls, recs):
    vals = []
    for s in STIMULI:
        if CLASSES_OF[s] != cls:
            continue
        for row in epoch_matrix(neuron, s, recs):
            vals.append(np.nanmean(row[resp_win]))
    return np.array(vals)


rows = []
for scope, recs in [("one_animal", [RECS[0]]), ("all_animals", RECS)]:
    for nm in ALL8:
        a = epoch_amplitudes(nm, "attractant", recs)
        r = epoch_amplitudes(nm, "repellent", recs)
        c = epoch_amplitudes(nm, "control", recs)
        u = stats.mannwhitneyu(a, r, alternative="two-sided")
        d = (a.mean() - r.mean()) / np.sqrt((a.var(ddof=1) + r.var(ddof=1)) / 2)
        rows.append(dict(scope=scope, neuron=nm, n_att=len(a), n_rep=len(r),
                         mean_att=round(a.mean(), 4), mean_rep=round(r.mean(), 4),
                         mean_ctl=round(c.mean(), 4), diff=round(a.mean() - r.mean(), 4),
                         cohens_d=round(d, 3), p_mannwhitney=round(u.pvalue, 5)))
cls_test = pd.DataFrame(rows)

# Holm correction within each scope, across the 8 neurons
for scope in ("one_animal", "all_animals"):
    sel = cls_test.scope == scope
    p = cls_test.loc[sel, "p_mannwhitney"].values
    order = np.argsort(p)
    holm = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        holm[idx] = running
    cls_test.loc[sel, "p_holm"] = holm.round(5)
cls_test.to_csv("results/psth_class_test.csv", index=False)

aa = cls_test[cls_test.scope == "all_animals"]
print("ALL ANIMALS (96 attractant vs 96 repellent epochs):")
print(aa[["neuron", "mean_att", "mean_rep", "mean_ctl", "cohens_d",
          "p_mannwhitney", "p_holm"]].to_string(index=False))
print("\nsurvives Holm correction:", aa[aa.p_holm < 0.05].neuron.tolist() or "none")
oa = cls_test[cls_test.scope == "one_animal"]
print("\nONE ANIMAL (12 vs 12 epochs) — same test:")
print(oa[["neuron", "mean_att", "mean_rep", "cohens_d", "p_mannwhitney", "p_holm"]].to_string(index=False))

# %% [markdown]
# ## Figure 20 — class-pooled responses, both scopes

# %%

fig, axes = plt.subplots(2, 8, figsize=(15.2, 5.0), sharex=True)
for col, nm in enumerate(ALL8):
    for row, (scope, recs, lbl) in enumerate(
            [("one animal", [RECS[0]], "one animal\n(var. across 3 repeats/stim)"),
             ("all animals", RECS, "all 8 animals\n(var. across epochs)")]):
        ax = axes[row, col]
        for cls in CLASS_LIST:
            p = psth_class(nm, cls, recs)
            if p is None: continue
            ax.fill_between(tvec, p["mean"]-p["sem"], p["mean"]+p["sem"],
                            color=CCOL[cls], alpha=0.22, lw=0)
            ax.plot(tvec, p["mean"], color=CCOL[cls], lw=1.1,
                    label=f"{cls} (n={p['n']})")
        ax.axvspan(0, 15, color="#000", alpha=0.05, lw=0, zorder=0)
        ax.axhline(0, color="#666", lw=0.6, ls=":")
        ax.axvline(0, color="#333", lw=0.7)
        if row == 0:
            grp = "core (interneuron)" if nm in INTER else "sensory"
            ax.set_title(f"{nm}\n{grp}", fontsize=6.8,
                         color="#333" if nm in INTER else "#7a4fa3")
        if col == 0:
            ax.set_ylabel(lbl, fontsize=6.2, labelpad=4)
        if row == 1:
            ax.set_xlabel("time from onset (s)", fontsize=6.2, labelpad=3)
        ax.tick_params(labelsize=5.8)
        ax.set_xlim(tvec[0], tvec[-1])
h=[mpl.lines.Line2D([],[],color=CCOL[c],lw=1.4,label=c) for c in CLASS_LIST]
fig.legend(handles=h, frameon=False, fontsize=7, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5,-0.045))
axes[0,0].text(7.5, axes[0,0].get_ylim()[0], "shaded = 15 s\nstimulus",
               fontsize=5.4, color="#555", ha="center", va="bottom")
fig.suptitle("Class-pooled responses, mean ± SEM of ΔF/F$_0$ "
             "(F$_0$ = 5 s pre-onset baseline; shaded band = 15 s stimulus)",
             fontsize=8.5, y=1.02)
fig.savefig("figures/fig20_psth_classes.pdf", bbox_inches="tight")
fig.savefig("figures/fig20_psth_classes.png", dpi=200, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:14],b.get_text()[:14]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:6])
print("wrote figures/fig20_psth_classes.pdf")

# %% [markdown]
# ## Figure 21 — per-stimulus responses, one animal and all animals

# %%
SCOL = {}
for s in STIMULI:
    c = CLASSES_OF[s]
    SCOL[s] = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}[c]
# distinguish stimuli within a class by line style
STYLE = {}
seen = {}
dashes = [(None,None), (3.5,1.4), (1.4,1.2), (5,1.4,1.4,1.4)]
for s in STIMULI:
    c = CLASSES_OF[s]
    STYLE[s] = dashes[seen.get(c,0) % len(dashes)]
    seen[c] = seen.get(c,0)+1

for scope, recs, tag, note in [
        ("one animal", [RECS[0]], "one", "one animal (20220327_herm_2) — variance across the 3 repeats of each stimulus"),
        ("all animals", RECS, "all", "all 8 animals — variance across the 24 epochs of each stimulus")]:
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 5.4), sharex=True)
    for k, nm in enumerate(ALL8):
        ax = axes[k//4, k%4]
        for s in STIMULI:
            p = psth(nm, s, recs)
            if p is None: continue
            ln, = ax.plot(tvec, p["mean"], color=SCOL[s], lw=1.0, label=f"{s} (n={p['n']})")
            if STYLE[s][0] is not None: ln.set_dashes(list(STYLE[s]))
            ax.fill_between(tvec, p["mean"]-p["sem"], p["mean"]+p["sem"],
                            color=SCOL[s], alpha=0.10, lw=0)
        ax.axvspan(0, 15, color="#000", alpha=0.05, lw=0, zorder=0)
        ax.axhline(0, color="#666", lw=0.6, ls=":")
        ax.axvline(0, color="#333", lw=0.7)
        grp = "core" if nm in INTER else "sensory"
        ax.set_title(f"{nm}  ({grp})", fontsize=7.2,
                     color="#333" if nm in INTER else "#7a4fa3", loc="left")
        ax.tick_params(labelsize=6)
        ax.set_xlim(tvec[0], tvec[-1])
        if k//4 == 1: ax.set_xlabel("time from onset (s)", labelpad=4)
        if k%4 == 0: ax.set_ylabel("ΔF/F$_0$", labelpad=4)
    handles = [mpl.lines.Line2D([],[],color=SCOL[s],lw=1.2,
               dashes=list(STYLE[s]) if STYLE[s][0] is not None else (None,None),
               label=s) for s in STIMULI]
    fig.legend(handles=handles, frameon=False, fontsize=6.2, ncol=5,
               loc="lower center", bbox_to_anchor=(0.5,-0.075))
    fig.suptitle(f"Per-stimulus responses, mean ± SEM of ΔF/F$_0$ — {note}",
                 fontsize=8.5, y=1.005)
    fig.savefig(f"figures/fig21_psth_stimuli_{tag}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/fig21_psth_stimuli_{tag}.png", dpi=200, bbox_inches="tight")
    r_=fig.canvas.get_renderer()
    tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
    tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
    print(tag, "overlaps:",[(a.get_text()[:12],b.get_text()[:12]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                       if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:4])
print("wrote figures/fig21_psth_stimuli_{one,all}.pdf")

# %% [markdown]
# ## What this establishes
#
# 1. **The class distinction IS present in the fluorescence.** Two of eight
#    neurons separate attractants from repellents after Holm correction across
#    all 8 animals: **AWAL** (d = +0.72, p < 10⁻⁵) and **AIBL** (d = −0.41,
#    p = 0.0003). AWAL responds to attractants at ΔF/F₀ ≈ 0.70 against ≈ 0.13 for
#    repellents — a 5-fold difference, and the direction matches AWA's known role
#    as an attractant-odour sensor.
# 2. **The sensory quartet carries larger and cleaner responses than the core
#    quartet.** Peak |ΔF/F₀| reaches 0.92 (AWAL, attractants) and 0.58 (ASER,
#    repellents) against 0.11–0.20 in the interneurons. ASER's repellent
#    preference is consistent with its role in salt avoidance.
# 3. **One animal is not enough.** With 12 epochs per class no neuron survives
#    correction, and the single-animal effect signs disagree with the pooled ones
#    for several neurons. Pooling across animals is what makes the effect
#    visible — the same argument that motivates pooling for the TPMs.
# 4. **Control responses are not zero.** The mechanical/fluid artefact of
#    delivery produces genuine deflections (e.g. AWCL ≈ −0.23), so "control"
#    means *vehicle*, not *nothing*. Class contrasts are therefore attractant vs
#    repellent, not stimulus vs silence.
#
# **The consequence for the Φ-structure analysis.** The class information exists
# in the data at the single-neuron level. Its absence from the Φ-structure
# comparison is therefore a property of that pipeline — binarization, TPM
# estimation, state selection, or the sample sizes those steps require — and not
# evidence that attractants and repellents evoke similar brain states. See
# `notebooks/10` for the positive control that makes this concrete.
