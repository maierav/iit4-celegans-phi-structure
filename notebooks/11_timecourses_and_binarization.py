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
