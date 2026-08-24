# %% [markdown]
# # 11 — Response time courses, flattening, and binarization
#
# This notebook produces the figures that now open the README, in the order the
# analysis actually depends on them:
#
# | figure | question |
# |---|---|
# | 26 | What does each flattening method do to a single animal's traces? |
# | 27 | ...and to the cycle-triggered average over all 232 epochs? |
# | 28 | Where are the two response windows? |
# | 29 | Does a response visible in the continuous trace survive binarization? |
#
# Three findings drive the rest of the repository:
#
# 1. The stimulus design is periodic at **exactly 60.0 s**, so the full cycle can
#    be averaged — the 15 s presentation *and* the 45 s that follows.
# 2. Sensory neurons respond in an **early** window (1–15 s); the core
#    interneurons carry their class difference in a **late** one (16–31 s).
# 3. **Binarization is not the lossy step.** In the late window it *increases* the
#    mean class contrast. What matters is the *detrending* that precedes it.

# %%
import os, sys, ast, subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.ndimage import percentile_filter, median_filter
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath("__file__"))) \
    if "__file__" not in dir() else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(REPO_ROOT, "src")):
    REPO_ROOT = "."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.makedirs(os.path.join(REPO_ROOT, "figures"), exist_ok=True)
os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)

import ces_hypergraph as ch
for r in ch.HERM_DRIVE_IDS:
    ch.ensure_recording(r)

# %%
RECS  = list(ch.HERM_DRIVE_IDS)
INTER = ["AIBL", "AVEL", "AVAL", "RIML"]
SENS  = ["ASEL", "ASER", "AWAL", "AWCL"]
ALL8  = INTER + SENS
STIMULI = list(ch.STIMULUS_CLASS)
CLS   = ch.STIMULUS_CLASS
FS    = ch.SAMPLING_RATE_HZ
EPOCH_N = round(15 * FS)
CYC_N   = round(60 * FS)
EARLY, LATE = (1, 15), (16, 31)
e0, e1 = round(EARLY[0] * FS), round(EARLY[1] * FS)
l0, l1 = round(LATE[0] * FS), round(LATE[1] * FS)

def load(rec):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    tr = {nm: d.loc[d.neuron == nm, tc].iloc[0].astype(float).values for nm in ALL8}
    on = defaultdict(list)
    for t, l in ast.literal_eval(d.iloc[0]["stimulus"]):
        on[l].append(int(t))
    return tr, on

DAT = {r: load(r) for r in RECS}
REC0 = RECS[0]

# the design is periodic -- verify it before relying on it
iv = np.array([v for r in RECS
               for v in np.diff(sorted(t for s in STIMULI for t in DAT[r][1].get(s, []))) / FS])
print(f"inter-onset interval: median {np.median(iv):.2f} s | sd {iv.std():.2f} | n = {len(iv)}")
print(f"epoch = {EPOCH_N} samples ({EPOCH_N/FS:.1f} s) | cycle = {CYC_N} samples ({CYC_N/FS:.1f} s)")

# %% [markdown]
# ## The three flattening methods
#
# `flat_midrange` is the original rule expressed as a continuous residual.
# `flat_pct_offset` is the field convention (rolling low percentile) but
# **subtracted** rather than divided: AWAL's fluorescence reaches exactly 0 in
# three recordings, so a ratio ΔF/F₀ is undefined there, and division by a
# low-percentile baseline also cannot represent decreases.

# %%
def flat_midrange(x, win_s=300):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    w = max(3, round(win_s * FS))
    hi = percentile_filter(xf, 100, size=w, mode="nearest")
    lo = percentile_filter(xf, 0, size=w, mode="nearest")
    return xf - (hi + lo) / 2

def flat_pct_offset(x, win_s=30, pct=8):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - percentile_filter(xf, pct, size=max(3, round(win_s * FS)), mode="nearest")

def flat_hp(x, win_s=60):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - median_filter(xf, size=max(3, round(win_s * FS)), mode="nearest")

METHODS = [("raw fluorescence", lambda x: np.where(np.isfinite(x), x, np.nanmedian(x)), "raw"),
           ("mid-range 300 s residual\n(ORIGINAL rule)", lambda x: flat_midrange(x, 300), "resid"),
           ("rolling 8th pct 30 s,\nSUBTRACTED", lambda x: flat_pct_offset(x, 30, 8), "resid"),
           ("high-pass, 60 s median", lambda x: flat_hp(x, 60), "resid")]
FLAT = {lbl: {r: {nm: fn(DAT[r][0][nm]) for nm in ALL8} for r in RECS}
        for lbl, fn, _ in METHODS}
LBL_HP = "high-pass, 60 s median"
LBL_MID = "mid-range 300 s residual\n(ORIGINAL rule)"

for nm in ALL8:
    z = sum(int(np.nansum(DAT[r][0][nm] <= 0)) for r in RECS)
    if z:
        print(f"{nm}: {z} samples at or below zero -> ratio dF/F0 undefined, subtraction used")
print("NaNs after flattening:",
      {lbl: int(sum(np.isnan(FLAT[lbl][r][nm]).sum() for r in RECS for nm in ALL8))
       for lbl, _, _ in METHODS})

# %% [markdown]
# ## Response latency: the two neuron groups are on different timescales

# %%
LAGS = np.arange(0, round(55 * FS))
rows = []
for nm in ALL8:
    xc = np.zeros(len(LAGS))
    for r in RECS:
        y = FLAT[LBL_HP][r][nm]
        y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-12)
        ind = np.zeros(len(y))
        for s in STIMULI:
            for t0 in DAT[r][1].get(s, []):
                ind[t0:t0 + EPOCH_N] = 1
        ind = ind - ind.mean()
        for k, L in enumerate(LAGS):
            yy, ii = (y[L:], ind[:len(y) - L]) if L > 0 else (y, ind)
            n_ = min(len(yy), len(ii))
            xc[k] += float(np.nansum(yy[:n_] * ii[:n_]) / n_)
    xc /= len(RECS)
    k = int(np.argmax(xc))
    rows.append(dict(neuron=nm, group="core" if nm in INTER else "sensory",
                     peak_lag_s=round(float(LAGS[k] / FS), 1),
                     xcorr_at_peak=round(float(xc[k]), 4),
                     xcorr_at_0s=round(float(xc[0]), 4)))
lag = pd.DataFrame(rows)
lag.to_csv(os.path.join(REPO_ROOT, "results/response_latency.csv"), index=False)
print(lag.to_string(index=False))
print(f"\nsensory peak lags: {sorted(lag[lag.group=='sensory'].peak_lag_s)}")
print(f"core    peak lags: {sorted(lag[lag.group=='core'].peak_lag_s)}")
_pk = float(lag[lag.group == "core"].peak_lag_s.max())
print(f"-> the design is periodic at {np.median(iv):.0f} s, so a lag of +{_pk:.1f} s "
      f"is also {_pk - np.median(iv):+.1f} s;")
print("   either reading places the core response outside the 15 s presentation.")

# %% [markdown]
# ## Class contrast by window
#
# One value per epoch — the mean flattened signal over the window — attractant
# against repellent, for each method and each window.

# %%
rows = []
for lbl, _, kind in METHODS:
    if kind == "raw":
        continue
    for nm in ALL8:
        for wname, (a_, b_) in [("early", (e0, e1)), ("late", (l0, l1))]:
            va, vr = [], []
            for r in RECS:
                y = FLAT[lbl][r][nm]
                for s in STIMULI:
                    if CLS[s] == "control":
                        continue
                    for t0 in sorted(DAT[r][1].get(s, [])):
                        if t0 + b_ > len(y):
                            continue
                        (va if CLS[s] == "attractant" else vr).append(np.nanmean(y[t0 + a_:t0 + b_]))
            va, vr = np.array(va), np.array(vr)
            d = (va.mean() - vr.mean()) / np.sqrt((va.var(ddof=1) + vr.var(ddof=1)) / 2)
            rows.append(dict(method=lbl.splitlines()[0], neuron=nm,
                             group="core" if nm in INTER else "sensory", window=wname,
                             d=round(d, 3), p=round(stats.mannwhitneyu(va, vr).pvalue, 5)))
wtab = pd.DataFrame(rows)
wtab.to_csv(os.path.join(REPO_ROOT, "results/two_window_contrast.csv"), index=False)
for meth in wtab.method.unique():
    sub = wtab[wtab.method == meth]
    p_ = sub.pivot(index="neuron", columns="window", values="d").reindex(ALL8)
    print(f"\n{meth}:")
    print(p_.round(3).to_string())
    print(f"  core  early {p_.loc[INTER,'early'].abs().mean():.3f}  late {p_.loc[INTER,'late'].abs().mean():.3f}")
    print(f"  sens  early {p_.loc[SENS,'early'].abs().mean():.3f}  late {p_.loc[SENS,'late'].abs().mean():.3f}")

# %% [markdown]
# ## Does binarization preserve the contrast?
#
# Same epochs, same windows, but the value per epoch is the **fraction of the
# window with the bit ON** instead of the mean continuous signal.

# %%
rows = []
for nm in ALL8:
    for wname, (a_, b_) in [("early", (e0, e1)), ("late", (l0, l1))]:
        cv, bv = {}, {}
        for cls in ["attractant", "repellent"]:
            c_, b_l = [], []
            for r in RECS:
                y = FLAT[LBL_HP][r][nm]
                bt = (y > 0).astype(int)
                for s in STIMULI:
                    if CLS[s] != cls:
                        continue
                    for t0 in sorted(DAT[r][1].get(s, [])):
                        if t0 + b_ > len(y):
                            continue
                        c_.append(np.nanmean(y[t0 + a_:t0 + b_]))
                        b_l.append(np.mean(bt[t0 + a_:t0 + b_]))
            cv[cls], bv[cls] = np.array(c_), np.array(b_l)
        dc = ((cv["attractant"].mean() - cv["repellent"].mean())
              / np.sqrt((cv["attractant"].var(ddof=1) + cv["repellent"].var(ddof=1)) / 2))
        db = ((bv["attractant"].mean() - bv["repellent"].mean())
              / np.sqrt((bv["attractant"].var(ddof=1) + bv["repellent"].var(ddof=1)) / 2))
        rows.append(dict(neuron=nm, group="core" if nm in INTER else "sensory", window=wname,
                         d_continuous=round(dc, 3), d_binary=round(db, 3),
                         retained=round(abs(db) / max(abs(dc), 1e-9), 3),
                         p_binary=round(stats.mannwhitneyu(bv["attractant"], bv["repellent"]).pvalue, 5)))
bw = pd.DataFrame(rows)
bw.to_csv(os.path.join(REPO_ROOT, "results/binarized_window_contrast.csv"), index=False)
print(bw.to_string(index=False))
for w in ("early", "late"):
    s_ = bw[bw.window == w]
    print(f"\n{w}: mean |d| continuous {s_.d_continuous.abs().mean():.3f} "
          f"-> binary {s_.d_binary.abs().mean():.3f} "
          f"({100*s_.d_binary.abs().mean()/s_.d_continuous.abs().mean():.0f}% retained)")
print(f"\nsignificant after binarization, late window: "
      f"{int((bw[bw.window=='late'].p_binary<0.05).sum())} of 8")

# %% [markdown]
# ## Figure 26 — each method on one animal's traces

# %%
plt.close("all")
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
CCOL={"attractant":BLUE,"repellent":ORANGE,"control":GREY}

SHOW=["AIBL","RIML","ASER","AWAL"]
T0,T1 = 50, 520                      # covers 8 stimulus presentations
t_s = np.arange(len(DAT[REC0][0]["AIBL"]))/FS
m = (t_s>=T0)&(t_s<=T1)
onsets0=sorted((t,l) for l,ts in DAT[REC0][1].items() for t in ts)

fig, axes = plt.subplots(len(METHODS), len(SHOW), figsize=(13.8, 9.0), sharex=True)
for i,(lbl,fn,kind) in enumerate(METHODS):
    for j,nm in enumerate(SHOW):
        ax = axes[i,j]
        y = FLAT[lbl][REC0][nm][m]
        for t0,lab in onsets0:
            if T0 <= t0/FS <= T1:
                ax.axvspan(t0/FS, t0/FS+15, color=CCOL[CLS[lab]], alpha=0.16, lw=0, zorder=0)
        if kind != "raw":
            ax.axhline(0, color="#c00", lw=0.8, ls="--", zorder=1)
        ax.plot(t_s[m], y, color="#111", lw=0.6, zorder=2)
        ax.tick_params(labelsize=5.6)
        ax.set_xlim(T0, T1)
        if i == 0:
            grp = "core" if nm in INTER else "sensory"
            ax.set_title(f"{nm}  ({grp})", fontsize=7.4,
                         color="#333" if nm in INTER else "#7a4fa3")
        if j == 0:
            ax.set_ylabel(lbl, fontsize=6.2, labelpad=5)
        if i == len(METHODS)-1:
            ax.set_xlabel("time (s)", labelpad=3)

h=[mpl.patches.Patch(color=CCOL[c], alpha=0.45, label=c) for c in ["attractant","repellent","control"]]
h.append(mpl.lines.Line2D([],[],color="#c00",lw=0.9,ls="--",label="binarization boundary (0)"))
fig.legend(handles=h, frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5,-0.030))
fig.suptitle("How each flattening method renders the same data — one animal (20220327_herm_2), "
             "8 consecutive stimulus presentations\n"
             "Shaded = 15 s stimulus. For the residual/ΔF/F₀ rows the dashed line is the threshold "
             "the binarizer applies, so a response is 'visible' to IIT only if the trace crosses it "
             "inside a shaded band.",
             fontsize=8.2, y=0.985)
fig.savefig(os.path.join(REPO_ROOT,"figures/fig26_flattening_timecourses.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT,"figures/fig26_flattening_timecourses.png"), dpi=190, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:18],b.get_text()[:18]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:6])

print("wrote figures/fig26_flattening_timecourses.pdf")

# %% [markdown]
# ## Figure 27 — each method on the 60 s cycle-triggered average

# %%
def cycle_avg(nm, lbl, recs, cls=None):
    rows = []
    for r in recs:
        y = FLAT[lbl][r][nm]
        for s in STIMULI:
            if cls is not None and CLS[s] != cls:
                continue
            for t0 in sorted(DAT[r][1].get(s, [])):
                if t0 + CYC_N > len(y):
                    continue
                rows.append(y[t0:t0 + CYC_N].astype(float))
    M = np.array(rows)
    return np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(len(M)), len(M)

plt.close("all")
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
CCOL={"attractant":BLUE,"repellent":ORANGE,"control":GREY}
SHOW=["AIBL","RIML","ASER","AWAL"]
tcyc=np.arange(CYC_N)/FS

fig, axes = plt.subplots(len(METHODS), len(SHOW), figsize=(12.6, 9.4), sharex=True)
for i,(lbl,fn,kind) in enumerate(METHODS):
    for j,nm in enumerate(SHOW):
        ax=axes[i,j]
        ax.axvspan(0,15,color="#000",alpha=0.06,lw=0,zorder=0)
        for cls in ["attractant","repellent","control"]:
            mu,se,N = cycle_avg(nm, lbl, RECS, cls=cls)
            ax.fill_between(tcyc, mu-se, mu+se, color=CCOL[cls], alpha=0.20, lw=0)
            ax.plot(tcyc, mu, color=CCOL[cls], lw=1.0)
        if kind!="raw":
            ax.axhline(0, color="#c00", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(0,60); ax.tick_params(labelsize=5.8)
        if i==0:
            ax.set_title(f"{nm}  ({'core' if nm in INTER else 'sensory'})", fontsize=7.4,
                         color="#333" if nm in INTER else "#7a4fa3")
        if j==0: ax.set_ylabel(lbl, fontsize=6.2, labelpad=5)
        if i==len(METHODS)-1: ax.set_xlabel("time from onset (s)", labelpad=3)
h=[mpl.lines.Line2D([],[],color=CCOL[c],lw=1.5,label=c) for c in ["attractant","repellent","control"]]
h.append(mpl.patches.Patch(color="#000",alpha=0.10,label="15 s epoch used for the TPM"))
h.append(mpl.lines.Line2D([],[],color="#c00",lw=0.9,ls="--",label="binarization boundary (0)"))
fig.legend(handles=h, frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5,-0.028))
fig.suptitle("Cycle-triggered average over the full 60 s stimulus period, mean ± SEM of 232 epochs per neuron\n"
             "The design is periodic at exactly 60.0 s, so this shows the WHOLE cycle: "
             "the 15 s epoch we feed to IIT (shaded) and the 45 s we treat as baseline.",
             fontsize=8.2, y=0.982)
fig.savefig(os.path.join(REPO_ROOT,"figures/fig27_cycle_by_method.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT,"figures/fig27_cycle_by_method.png"), dpi=190, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:5])

print("wrote figures/fig27_cycle_by_method.pdf")

# %% [markdown]
# ## Figure 28 — the two response windows

# %%
plt.close("all")
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
EWIN, LWIN = "#4a9b6e", "#8a5fa8"
CCOL={"attractant":BLUE,"repellent":ORANGE,"control":GREY}
SHOW=["AIBL","RIML","ASER","AWAL"]
tcyc=np.arange(CYC_N)/FS

def cyc2(nm, lbl, cls):
    rows=[]
    for r in RECS:
        y=FLAT[lbl][r][nm]
        for s in STIMULI:
            if CLS[s]!=cls: continue
            for t0 in sorted(DAT[r][1].get(s,[])):
                if t0+CYC_N>len(y): continue
                rows.append(y[t0:t0+CYC_N].astype(float))
    M=np.array(rows)
    return np.nanmean(M,0), np.nanstd(M,0,ddof=1)/np.sqrt(len(M)), len(M)

fig, axes = plt.subplots(len(METHODS), len(SHOW), figsize=(12.6, 8.0), sharex=True)
for i,(lbl,fn,kind) in enumerate(METHODS):
    for j,nm in enumerate(SHOW):
        ax=axes[i,j]
        ax.axvspan(EARLY[0], EARLY[1], color=EWIN, alpha=0.18, lw=0, zorder=0)
        ax.axvspan(LATE[0],  LATE[1],  color=LWIN, alpha=0.18, lw=0, zorder=0)
        for cls in ["attractant","repellent","control"]:
            mu,se,N = cyc2(nm, lbl, cls)
            ax.fill_between(tcyc, mu-se, mu+se, color=CCOL[cls], alpha=0.18, lw=0)
            ax.plot(tcyc, mu, color=CCOL[cls], lw=1.0)
        if kind!="raw": ax.axhline(0, color="#c00", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(0,60); ax.tick_params(labelsize=5.8)
        if i==0:
            ax.set_title(f"{nm}  ({'core' if nm in INTER else 'sensory'})", fontsize=7.4,
                         color="#333" if nm in INTER else "#7a4fa3")
        if j==0: ax.set_ylabel(lbl, fontsize=6.2, labelpad=5)
        if i==len(METHODS)-1: ax.set_xlabel("time from onset (s)", labelpad=3)
axes[0,0].text(8, axes[0,0].get_ylim()[1], "EARLY\n1–15 s", fontsize=5.6, color=EWIN,
               ha="center", va="top", fontweight="bold")
axes[0,0].text(23.5, axes[0,0].get_ylim()[1], "LATE\n16–31 s", fontsize=5.6, color=LWIN,
               ha="center", va="top", fontweight="bold")
h=[mpl.lines.Line2D([],[],color=CCOL[c],lw=1.5,label=c) for c in ["attractant","repellent","control"]]
h += [mpl.patches.Patch(color=EWIN,alpha=0.45,label="early window 1–15 s"),
      mpl.patches.Patch(color=LWIN,alpha=0.45,label="late window 16–31 s"),
      mpl.lines.Line2D([],[],color="#c00",lw=0.9,ls="--",label="binarization boundary")]
fig.legend(handles=h, frameon=False, fontsize=7, ncol=6, loc="lower center", bbox_to_anchor=(0.5,-0.035))
fig.suptitle("Two response windows on the 60 s stimulus cycle, mean ± SEM (232 epochs/neuron)\n"
             "Sensory neurons respond in the EARLY window; core interneurons carry their class "
             "difference in the LATE window.", fontsize=8.2, y=0.980)
fig.savefig(os.path.join(REPO_ROOT,"figures/fig28_two_windows.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT,"figures/fig28_two_windows.png"), dpi=190, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:5])

print("wrote figures/fig28_two_windows.pdf")

# %% [markdown]
# ## Figure 29 — the binarized PSTH
#
# Middle row binarizes **first** and then averages, so it shows P(bit = 1) rather
# than the bit of an average. Bottom row is one presentation in one animal — the
# actual input to a TPM — with the continuous signal rescaled so its zero
# crossing coincides with the bit flip.

# %%
plt.close("all")
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
EWIN, LWIN = "#4a9b6e", "#8a5fa8"
CCOL={"attractant":BLUE,"repellent":ORANGE,"control":GREY}
LBL_HP = "high-pass, 60 s median"

def bin_of(y): return (y > 0).astype(int)

def cyc_bin(nm, cls):
    """Binarize FIRST, then average across epochs -> fraction of epochs ON at each lag."""
    rows=[]
    for r in RECS:
        b = bin_of(FLAT[LBL_HP][r][nm])
        for s in STIMULI:
            if CLS[s]!=cls: continue
            for t0 in sorted(DAT[r][1].get(s,[])):
                if t0+CYC_N>len(b): continue
                rows.append(b[t0:t0+CYC_N])
    M=np.array(rows, float)
    return M.mean(0), M.std(0,ddof=1)/np.sqrt(len(M)), len(M)

SHOW=["AIBL","RIML","ASER","AWAL"]
tcyc=np.arange(CYC_N)/FS
fig, axes = plt.subplots(3, len(SHOW), figsize=(12.6, 6.6), sharex=True)

# row 0: continuous cross-animal average (for reference)
for j,nm in enumerate(SHOW):
    ax=axes[0,j]
    ax.axvspan(*EARLY, color=EWIN, alpha=0.16, lw=0, zorder=0)
    ax.axvspan(*LATE,  color=LWIN, alpha=0.16, lw=0, zorder=0)
    for cls in ["attractant","repellent","control"]:
        rows=[]
        for r in RECS:
            y=FLAT[LBL_HP][r][nm]
            for s in STIMULI:
                if CLS[s]!=cls: continue
                for t0 in sorted(DAT[r][1].get(s,[])):
                    if t0+CYC_N>len(y): continue
                    rows.append(y[t0:t0+CYC_N])
        M=np.array(rows); mu=M.mean(0); se=M.std(0,ddof=1)/np.sqrt(len(M))
        ax.fill_between(tcyc, mu-se, mu+se, color=CCOL[cls], alpha=0.18, lw=0)
        ax.plot(tcyc, mu, color=CCOL[cls], lw=1.0)
    ax.axhline(0, color="#c00", lw=0.8, ls="--")
    ax.set_title(f"{nm}  ({'core' if nm in INTER else 'sensory'})", fontsize=7.4,
                 color="#333" if nm in INTER else "#7a4fa3")
    if j==0: ax.set_ylabel("continuous\n(high-pass)", fontsize=6.2, labelpad=5)
    ax.tick_params(labelsize=5.8); ax.set_xlim(0,60)

# row 1: BINARIZED cross-animal average -> P(state = 1)
for j,nm in enumerate(SHOW):
    ax=axes[1,j]
    ax.axvspan(*EARLY, color=EWIN, alpha=0.16, lw=0, zorder=0)
    ax.axvspan(*LATE,  color=LWIN, alpha=0.16, lw=0, zorder=0)
    for cls in ["attractant","repellent","control"]:
        mu,se,N = cyc_bin(nm, cls)
        ax.fill_between(tcyc, mu-se, mu+se, color=CCOL[cls], alpha=0.18, lw=0)
        ax.plot(tcyc, mu, color=CCOL[cls], lw=1.0)
    ax.axhline(0.5, color="#666", lw=0.7, ls=":")
    ax.set_ylim(0,1)
    if j==0: ax.set_ylabel("BINARIZED,\naveraged\nP(bit = 1)", fontsize=6.2, labelpad=5)
    ax.tick_params(labelsize=5.8)

# row 2: ONE presentation in ONE animal -- the actual binary input to the TPM
EX_REC=RECS[0]; EX_STIM="100mM NaCl"
t_ex=sorted(DAT[EX_REC][1][EX_STIM])[0]
for j,nm in enumerate(SHOW):
    ax=axes[2,j]
    ax.axvspan(*EARLY, color=EWIN, alpha=0.16, lw=0, zorder=0)
    ax.axvspan(*LATE,  color=LWIN, alpha=0.16, lw=0, zorder=0)
    y=FLAT[LBL_HP][EX_REC][nm][t_ex:t_ex+CYC_N]
    b=bin_of(y)
    # scale the continuous signal so its ZERO maps to 0.5 -> the bit flips exactly
    # where the grey trace crosses the red line. No twin axis, no misalignment.
    yscale = 0.5 + 0.44*y/max(np.max(np.abs(y)), 1e-9)
    ax.plot(tcyc, yscale, color="#999", lw=0.6, zorder=1)
    ax.axhline(0.5, color="#c00", lw=0.7, ls="--", zorder=1)
    ax.step(tcyc, b, where="mid", color="#111", lw=1.1, zorder=3)
    ax.fill_between(tcyc, 0, b, step="mid", color="#111", alpha=0.22, lw=0, zorder=2)
    ax.set_ylim(-0.08,1.35); ax.set_yticks([0,1])
    ax.set_xlabel("time from onset (s)", labelpad=3)
    if j==0: ax.set_ylabel("ONE epoch,\none animal\nbit", fontsize=6.2, labelpad=5)
    ax.tick_params(labelsize=5.8)
    e_on=b[e0:e1].mean(); l_on=b[l0:l1].mean()
    ax.text(8, 1.20, f"{e_on:.2f}", fontsize=6, color=EWIN, ha="center", fontweight="bold")
    ax.text(23.5, 1.20, f"{l_on:.2f}", fontsize=6, color=LWIN, ha="center", fontweight="bold")

h=[mpl.lines.Line2D([],[],color=CCOL[c],lw=1.5,label=c) for c in ["attractant","repellent","control"]]
h += [mpl.patches.Patch(color=EWIN,alpha=0.4,label="early 1–15 s"),
      mpl.patches.Patch(color=LWIN,alpha=0.4,label="late 16–31 s"),
      mpl.lines.Line2D([],[],color="#111",lw=1.2,label="binary bit"),
      mpl.lines.Line2D([],[],color="#999",lw=0.8,label="continuous signal, rescaled so 0 → the red line")]
fig.legend(handles=h, frameon=False, fontsize=6.6, ncol=7, loc="lower center", bbox_to_anchor=(0.5,-0.045))
fig.suptitle("Binarization of the PSTH: cross-animal average (rows 1–2) and a single presentation "
             "in one animal (row 3)\nHigh-pass flattening, 60 s rolling median. Row 3 is "
             f"{EX_STIM}, first presentation, {EX_REC} — the actual binary input to the TPM. "
             "Numbers = fraction of that window ON.", fontsize=8.0, y=0.995)
fig.savefig(os.path.join(REPO_ROOT,"figures/fig29_binarized_psth.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT,"figures/fig29_binarized_psth.png"), dpi=190, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:5])

print("wrote figures/fig29_binarized_psth.pdf")

# %% [markdown]
# ## What this establishes for the rest of the repository
#
# * The class distinction is present in the fluorescence, and it is present on
#   **two timescales** — an early sensory response and a late interneuron one. A
#   single 15 s window is right for one group and wrong for the other.
# * **Binarization is not the lossy step.** In the late window it *increases* the
#   mean class contrast (|d| 0.35 -> 0.42, six of eight neurons significant).
# * What matters is the **detrending** that precedes it: the original mid-range
#   rule lets a tonic neuron's threshold ride up with its own response, producing
#   binary runs longer than the interval between stimuli.

# %% [markdown]
# ## The high-pass window is the parameter that matters
#
# A slow window makes the bit encode **level** — an elevated response is uniformly
# "up" — rather than fluctuation about a local reference. A transition matrix needs
# the latter. This sweeps the window from 3 s to 300 s and asks, at each setting,
# whether the stimulus is detectable in the transition structure.

# %%
from scipy.spatial.distance import jensenshannon

WINS = [3, 5, 8, 12, 20, 30, 45, 60, 120, 300]

def hp_bits(x, win_s):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return (xf - median_filter(xf, size=max(3, round(win_s * FS)), mode="nearest") > 0).astype(int)

BW = {w: {r: {nm: hp_bits(DAT[r][0][nm], w) for nm in ALL8} for r in RECS} for w in WINS}

rows = []
for w in WINS:
    for neurons, tag in [(INTER, "core"), (SENS, "sens")]:
        selft, const, nst, runs = [], [], [], []
        for r in RECS:
            st = ch.combine_states([BW[w][r][nm] for nm in neurons])
            for s in STIMULI:
                for t0 in sorted(DAT[r][1].get(s, [])):
                    seg = st[t0:t0 + EPOCH_N]
                    if len(seg) < 2:
                        continue
                    selft.append(np.mean(seg[:-1] == seg[1:]))
                    const.append(len(set(seg)) == 1)
                    nst.append(len(set(seg)))
            for nm in neurons:
                b = BW[w][r][nm]
                chg = np.flatnonzero(np.diff(b)) + 1
                runs += list(np.diff(np.concatenate([[0], chg, [len(b)]])) / FS)
        rows.append(dict(window_s=w, substrate=tag,
                         mean_run_s=round(float(np.mean(runs)), 2),
                         frac_const_epochs=round(float(np.mean(const)), 3),
                         self_trans=round(float(np.mean(selft)), 3),
                         states_per_epoch=round(float(np.mean(nst)), 2)))
dw = pd.DataFrame(rows)
print(dw.pivot(index="window_s", columns="substrate",
               values=["states_per_epoch", "self_trans", "frac_const_epochs"]).round(3).to_string())

# %% [markdown]
# ### Is a fast window tracking signal or noise?
#
# Two checks that bound the window from below. A bit whose lag-1 autocorrelation
# is negative is alternating sample to sample — noise, not dynamics. And the bit
# should still track the underlying response.

# %%
rows = []
for w in WINS:
    ac1, cc = [], []
    for r in RECS:
        for nm in ALL8:
            b = BW[w][r][nm].astype(float)
            ac1.append(np.corrcoef(b[:-1], b[1:])[0, 1])
            xf = np.where(np.isfinite(DAT[r][0][nm]), DAT[r][0][nm], np.nanmedian(DAT[r][0][nm]))
            sig = xf - median_filter(xf, size=round(20 * FS), mode="nearest")
            cc.append(np.corrcoef(b, sig)[0, 1])
    rows.append(dict(window_s=w, lag1_autocorr=round(float(np.mean(ac1)), 3),
                     corr_with_20s_signal=round(float(np.mean(cc)), 3)))
nz = pd.DataFrame(rows)
nz.to_csv(os.path.join(REPO_ROOT, "results/window_sweep_noise_check.csv"), index=False)
print(nz.to_string(index=False))
print(f"\nlag-1 autocorr turns negative below "
      f"{nz[nz.lag1_autocorr>0].window_s.min():.0f} s -> alternating noise")
print(f"bit-response correlation peaks at "
      f"{nz.loc[nz.corr_with_20s_signal.idxmax(),'window_s']:.0f} s")

# %% [markdown]
# ### The permutation test at the TPM level, per window
#
# Epoch stim/base labels are shuffled **within** each animal, so data volume and
# animal identity are held fixed and only the condition assignment varies.

# %%
def wins_for(w, neurons):
    out = {}
    for r in RECS:
        st = ch.combine_states([BW[w][r][nm] for nm in neurons])
        marks = sorted((t, l) for l, ts in DAT[r][1].items() for t in ts)
        segs = []
        for i, (t0, lab) in enumerate(marks):
            nxt = marks[i + 1][0] if i + 1 < len(marks) else len(st)
            s_, b_ = st[t0:t0 + EPOCH_N], st[t0 + EPOCH_N:nxt]
            if len(s_) > 1:
                segs.append(("stim", s_))
            if len(b_) > 1:
                segs.append(("base", b_))
        out[r] = segs
    return out

def cnt(segs, K=16):
    C = np.zeros((K, K))
    for s in segs:
        for a, b in zip(s[:-1], s[1:]):
            C[a, b] += 1
    return C

def tpmd(C1, C2):
    P1 = (C1 + 0.5) / (C1 + 0.5).sum(1, keepdims=True)
    P2 = (C2 + 0.5) / (C2 + 0.5).sum(1, keepdims=True)
    rw = [k for k in range(len(C1)) if C1[k].sum() > 0 and C2[k].sum() > 0]
    return float(np.mean([jensenshannon(P1[k], P2[k], base=2) for k in rw])), len(rw)

NPERM = 1000
rows = []
for w in WINS:
    for neurons, tag in [(INTER, "core 4n"), (SENS, "sensory 4n")]:
        W = wins_for(w, neurons)
        rs = [s for r in RECS for l, s in W[r] if l == "stim"]
        rb = [s for r in RECS for l, s in W[r] if l == "base"]
        obs, nr = tpmd(cnt(rs), cnt(rb))
        rg = np.random.default_rng(0)
        null = []
        for _ in range(NPERM):
            A, B = [], []
            for r in RECS:
                labs = np.array([l for l, _ in W[r]])
                sg = [s for _, s in W[r]]
                for l, s in zip(rg.permutation(labs), sg):
                    (A if l == "stim" else B).append(s)
            null.append(tpmd(cnt(A), cnt(B))[0])
        null = np.array(null)
        rows.append(dict(window_s=w, substrate=tag, rows_compared=nr, observed=round(obs, 4),
                         null_mean=round(float(null.mean()), 4), null_sd=round(float(null.std()), 4),
                         z=round(float((obs - null.mean()) / null.std()), 2),
                         p=round(float((np.sum(null >= obs) + 1) / (NPERM + 1)), 4)))
        print(f"  w={w:>3}s {tag:<11} z={rows[-1]['z']:+6.2f}  p={rows[-1]['p']:.4f}", flush=True)
tp = pd.DataFrame(rows)
tp.to_csv(os.path.join(REPO_ROOT, "results/window_sweep_tpm_permutation.csv"), index=False)
core = tp[tp.substrate == "core 4n"]
print(f"\ncore quartet: z = {core[core.window_s==300].z.iloc[0]:+.2f} at 300 s, "
      f"{core.z.max():+.2f} at {int(core.loc[core.z.idxmax(),'window_s'])} s")

# %% [markdown]
# ### The tension: class coding prefers a slower window
#
# Under a **level** code (fraction of the window with the bit ON) the class
# contrast peaks at a slower window than the transition dynamics do, because class
# information in these neurons lives in slow amplitude differences. Under a
# **rate** code (flips per sample) it is weaker everywhere.

# %%
def epoch_vals(w, nm, kind, a_, b_):
    out = {"attractant": [], "repellent": [], "control": []}
    for r in RECS:
        b = BW[w][r][nm]
        for s in STIMULI:
            for t0 in sorted(DAT[r][1].get(s, [])):
                seg = b[t0 + a_:t0 + b_]
                if len(seg) < 2:
                    continue
                out[CLS[s]].append(seg.mean() if kind == "level" else np.mean(np.diff(seg) != 0))
    return {k: np.array(v) for k, v in out.items()}

rows = []
for w in WINS:
    for kind in ("level", "rate"):
        for wname, (a_, b_) in [("early", (e0, e1)), ("late", (l0, l1))]:
            ds, ps = [], []
            for nm in ALL8:
                v = epoch_vals(w, nm, kind, a_, b_)
                a, r_ = v["attractant"], v["repellent"]
                ds.append((a.mean() - r_.mean()) / np.sqrt((a.var(ddof=1) + r_.var(ddof=1)) / 2))
                ps.append(stats.mannwhitneyu(a, r_).pvalue)
            ds, ps = np.array(ds), np.array(ps)
            rows.append(dict(window_s=w, code=kind, window=wname,
                             mean_abs_d=round(float(np.abs(ds).mean()), 3),
                             core_abs_d=round(float(np.abs(ds[:4]).mean()), 3),
                             sens_abs_d=round(float(np.abs(ds[4:]).mean()), 3),
                             n_sig=int((ps < 0.05).sum())))
cd = pd.DataFrame(rows)
cd.to_csv(os.path.join(REPO_ROOT, "results/window_sweep_coding.csv"), index=False)
for kind in ("level", "rate"):
    print(f"\n{kind.upper()} coding, class contrast (mean |d|):")
    print(cd[cd.code == kind].pivot(index="window_s", columns="window",
                                    values=["core_abs_d", "sens_abs_d"]).round(3).to_string())

# %% [markdown]
# ## Figure 30 — the window sweep

# %%
plt.close("all")
BLUE, ORANGE, GREY, LIGHT = "#1f6fb4", "#c2571a", "#8a8a8a", "#9bb8d4"
GREEN, PURPLE = "#4a9b6e", "#8a5fa8"

fig = plt.figure(figsize=(11.6, 6.8))
g = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42)

dc = dw[dw.substrate=="core"].set_index("window_s")
ds = dw[dw.substrate=="sens"].set_index("window_s")
W = np.array(WINS)

# (a) the problem your objection names: a slow window makes the bit a level, not a dynamic
ax = fig.add_subplot(g[0,0])
ax.semilogx(W, [dc.loc[w,"self_trans"] for w in W], "o-", color=BLUE, lw=1.3, ms=4, label="core 4n")
ax.semilogx(W, [ds.loc[w,"self_trans"] for w in W], "s-", color=ORANGE, lw=1.3, ms=4, label="sensory 4n")
ax.axvline(300, ls=":", lw=1, color="#c00")
ax.text(300, 0.05, " original\n 300 s", fontsize=5.6, color="#c00", ha="right", va="bottom")
ax.set_xlabel("high-pass window (s)", labelpad=5)
ax.set_ylabel("self-transition fraction", labelpad=5)
ax.legend(frameon=False, fontsize=6, loc="upper left")
ax.set_title("a  A slow window freezes\n   the joint state", loc="left")

# (b) states per epoch
ax = fig.add_subplot(g[0,1])
ax.semilogx(W, [dc.loc[w,"states_per_epoch"] for w in W], "o-", color=BLUE, lw=1.3, ms=4)
ax.semilogx(W, [ds.loc[w,"states_per_epoch"] for w in W], "s-", color=ORANGE, lw=1.3, ms=4)
ax.axhline(16, ls="--", lw=0.8, color="#888")
ax.text(4, 15.4, "all 16 states", fontsize=5.6, color="#888", va="top")
ax.axvline(300, ls=":", lw=1, color="#c00")
ax.set_xlabel("high-pass window (s)", labelpad=5)
ax.set_ylabel("distinct joint states\nper 15 s epoch", labelpad=5)
ax.set_title("b  ...and starves the TPM\n   of state coverage", loc="left")

# (c) THE RESULT: TPM permutation z vs window
ax = fig.add_subplot(g[0,2])
for tag, col, mk in [("core 4n", BLUE, "o"), ("sensory 4n", ORANGE, "s")]:
    sub = tp[tp.substrate==tag].sort_values("window_s")
    ax.semilogx(sub.window_s, sub.z, mk+"-", color=col, lw=1.3, ms=4, label=tag)
ax.axhline(1.96, ls="--", lw=0.9, color="#333")
ax.text(220, 2.5, "p = 0.05", fontsize=5.8, color="#333", ha="right")
ax.axvline(300, ls=":", lw=1, color="#c00")
ax.axvspan(14, 30, color=GREEN, alpha=0.13, lw=0)
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(-1, 90)
ax.set_xlabel("high-pass window (s)", labelpad=5)
ax.set_ylabel("permutation z\n(stimulus vs baseline)", labelpad=5)
ax.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0.02,0.86))
ax.set_title("c  Stimulus detectability peaks\n   near 20 s", loc="left")

# (d) core detail, linear -- the finding is invisible on a symlog axis
ax = fig.add_subplot(g[1,0])
sub = tp[tp.substrate=="core 4n"].sort_values("window_s")
cols = [GREEN if z_ > 1.96 else LIGHT for z_ in sub.z]
ax.bar(range(len(sub)), sub.z, 0.68, color=cols)
ax.axhline(1.96, ls="--", lw=0.9, color="#333")
ax.text(len(sub)-0.4, 2.25, "p = 0.05", fontsize=5.8, color="#333", ha="right")
ax.set_xticks(range(len(sub)))
ax.set_xticklabels([f"{int(w)}" for w in sub.window_s], fontsize=5.8)
ax.tick_params(axis="x", pad=1.5)
for i, w in enumerate(sub.window_s):
    if w == 300:
        ax.text(i, sub.z.iloc[i]+0.35, "original", fontsize=5.4, color="#c00", ha="center")
ax.set_xlabel("high-pass window (s)", labelpad=7)
ax.set_ylabel("permutation z", labelpad=5)
ax.set_title("d  Core quartet: z = +0.47 at 300 s,\n   +6.92 at 20 s", loc="left")

# (e) the noise check -- a fast window is not free
ax = fig.add_subplot(g[1,1])
ax.semilogx(W, nz.lag1_autocorr, "o-", color=PURPLE, lw=1.3, ms=4, label="lag-1 autocorr of the bit")
ax.semilogx(W, nz.corr_with_20s_signal, "s-", color=GREEN, lw=1.3, ms=4, label="corr(bit, response)")
ax.axhline(0, color="#333", lw=0.8, ls="--")
ax.axvspan(14, 30, color=GREEN, alpha=0.13, lw=0)
ax.set_xlabel("high-pass window (s)", labelpad=5)
ax.set_ylabel("correlation", labelpad=5)
ax.legend(frameon=False, fontsize=5.2, loc="upper left", handlelength=1.6, labelspacing=0.22, borderaxespad=0.2)
ax.set_ylim(-0.30, 1.24)
ax.set_title("e  Below ~8 s the bit\n   anti-correlates: noise", loc="left")

# (f) class contrast: level coding wants a SLOW window -- the tension
ax = fig.add_subplot(g[1,2])
lv = cd[(cd.code=="level")&(cd.window=="late")].sort_values("window_s")
rt = cd[(cd.code=="rate")&(cd.window=="late")].sort_values("window_s")
ax.semilogx(lv.window_s, lv.core_abs_d, "o-", color=BLUE, lw=1.3, ms=4, label="level code, core")
ax.semilogx(lv.window_s, lv.sens_abs_d, "o--", color=BLUE, lw=1.1, ms=3.4, alpha=0.65, label="level code, sensory")
ax.semilogx(rt.window_s, rt.core_abs_d, "s-", color=ORANGE, lw=1.3, ms=4, label="rate code, core")
ax.axvspan(14, 30, color=GREEN, alpha=0.13, lw=0)
ax.set_xlabel("high-pass window (s)", labelpad=5)
ax.set_ylabel("class contrast, mean |d|\n(late window)", labelpad=5)
ax.legend(frameon=False, fontsize=5.2, loc="upper left", handlelength=1.6, labelspacing=0.22, borderaxespad=0.2)
ax.set_ylim(0, 0.70)
ax.set_title("f  But class coding wants\n   a slower window", loc="left")

fig.savefig(os.path.join(REPO_ROOT,"figures/fig30_window_sweep.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT,"figures/fig30_window_sweep.png"), dpi=200, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:6])

print("wrote figures/fig30_window_sweep.pdf")

# %% [markdown]
# ## What this leaves open
#
# The window is now optimised for the transition dynamics, and at that setting the
# TPM separates stimulus from baseline at z = +6.9 (core) and z = +35 (sensory).
# The Phi-structure computed from the same matrices does not (z = +0.5 and -0.4;
# see `results/tpm_vs_phi_by_window.csv`). So the loss has moved from the
# binarization to the **unfolding** -- the first time in this project it has landed
# on a theory-facing step rather than a preprocessing one.
