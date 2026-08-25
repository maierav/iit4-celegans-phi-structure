# %% [markdown]
# # 13 — Robustness of the Φ(t) picture: median, all stimuli, raw traces, TPM stability
#
# Four checks requested after the offset-dip finding:
#
# 1. **Median instead of mean** for the Φ(t) grand averages.
# 2. **All stimuli**, not just 100 mM NaCl, with the explicit STIMULUS vs
#    NO-STIMULUS contrast (each epoch against its own pre-stimulus baseline).
# 3. **Raw fluorescence** (no flattening) as mean and median with variance, for
#    the same eight neurons on the same −15 to 60 s timescale.
# 4. **TPM stability**: drop k random transitions, recompute, compare — from
#    k = 1 to k ≈ N; plus the same question asked of the φ-per-state map,
#    which is what Φ(t) actually consumes.
#
# Setup is identical to notebook 12 (20 s high-pass bits, one giant TPM per
# substrate, Φ per state).

# %%
import os, sys, ast, subprocess, time
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.ndimage import median_filter
from scipy.spatial.distance import jensenshannon
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = "."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import ces_hypergraph as ch
for r in ch.HERM_DRIVE_IDS:
    ch.ensure_recording(r)
os.environ["PYPHI_WELCOME_OFF"] = "yes"
import pyphi
from pyphi import convert
pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False

RECS = list(ch.HERM_DRIVE_IDS)
INTER = ["AIBL", "AVEL", "AVAL", "RIML"]
SENS = ["ASEL", "ASER", "AWAL", "AWCL"]
ALL8 = INTER + SENS
STIMULI = list(ch.STIMULUS_CLASS)
CLS = ch.STIMULUS_CLASS
FS = ch.SAMPLING_RATE_HZ
EPOCH_N = round(15 * FS)
CYC_N = round(60 * FS)
PRE_S = 15
PRE_N = round(PRE_S * FS)
text = np.arange(-PRE_N, CYC_N) / FS
W_HP = 20
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
CCOL = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}

def load(rec):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    tr = {nm: d.loc[d.neuron == nm, tc].iloc[0].astype(float).values for nm in ALL8}
    on = defaultdict(list)
    for t, l in ast.literal_eval(d.iloc[0]["stimulus"]):
        on[l].append(int(t))
    return tr, on

DAT = {r: load(r) for r in RECS}
def hp(x):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - median_filter(xf, size=max(3, round(W_HP * FS)), mode="nearest")
BITS = {r: {nm: (hp(DAT[r][0][nm]) > 0).astype(int) for nm in ALL8} for r in RECS}
ST = {r: {"core": ch.combine_states([BITS[r][nm] for nm in INTER]),
          "sens": ch.combine_states([BITS[r][nm] for nm in SENS])} for r in RECS}

def giant_tpm(key, K=16):
    C = np.zeros((K, K))
    for r in RECS:
        st = ST[r][key]
        for a, b in zip(st[:-1], st[1:]):
            C[a, b] += 1
    return C

PHI_OF = {}
for key, neurons in [("core", INTER), ("sens", SENS)]:
    C = giant_tpm(key)
    P = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    net = pyphi.Network(convert.state_by_state2state_by_node(P), node_labels=neurons)
    PHI_OF[key] = {si: float(pyphi.new_big_phi.phi_structure(
        pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4)))).big_phi)
        for si in range(16)}
print("Phi per state recomputed")

def phi_epochs(key):
    out, meta = [], []
    for r in RECS:
        for s in STIMULI:
            for t0 in sorted(DAT[r][1].get(s, [])):
                if t0 - PRE_N >= 0 and t0 + CYC_N <= len(ST[r][key]):
                    seg = ST[r][key][t0 - PRE_N:t0 + CYC_N]
                    out.append([PHI_OF[key][x] for x in seg])
                    meta.append((r, s, CLS[s]))
    return np.array(out), meta
PE = {key: phi_epochs(key) for key in ("core", "sens")}

def raw_epochs(nm):
    out = []
    for r in RECS:
        tr = DAT[r][0][nm]
        for s in STIMULI:
            for t0 in sorted(DAT[r][1].get(s, [])):
                if t0 - PRE_N >= 0 and t0 + CYC_N <= len(tr):
                    seg = tr[t0 - PRE_N:t0 + CYC_N].astype(float)
                    out.append(seg - np.nanmean(seg[:PRE_N]))
    return np.array(out)
RE = {nm: raw_epochs(nm) for nm in ALL8}
print("epoch tensors:", PE["sens"][0].shape, RE["AIBL"].shape)

# %% [markdown]
# ## Figure 35 — mean vs median Φ(t); the ON-vs-OFF contrast; every stimulus

# %%
segs = {"stim": slice(round((0 + PRE_S) * FS), round((15 + PRE_S) * FS)),
        "post": slice(round((16 + PRE_S) * FS), round((31 + PRE_S) * FS))}
fig = plt.figure(figsize=(12.8, 8.2))
g = fig.add_gridspec(3, 2, hspace=0.52, wspace=0.26)
for col, key in enumerate(("core", "sens")):
    P, meta = PE[key]
    ax = fig.add_subplot(g[0, col])
    mu, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    md = np.median(P, 0)
    q1, q3 = np.percentile(P, 25, 0), np.percentile(P, 75, 0)
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    ax.fill_between(text, mu - se, mu + se, color=BLUE, alpha=0.20, lw=0)
    ax.plot(text, mu, color=BLUE, lw=1.1, label="mean ± SEM")
    ax.fill_between(text, q1, q3, color=ORANGE, alpha=0.15, lw=0)
    ax.plot(text, md, color=ORANGE, lw=1.1, label="median [IQR]")
    ax.axvline(0, color="#c00", lw=0.7); ax.set_yscale("log")
    ax.set_title(f"{'ab'[col]}  {key.upper()} — mean vs median Φ(t)", loc="left")
    if col == 0:
        ax.legend(frameon=False, fontsize=6, loc="center right")
    ax.set_ylabel("Φ(t)", labelpad=5)

    ax = fig.add_subplot(g[1, col])
    Pb = P - P[:, :PRE_N].mean(1, keepdims=True)
    mu, se = Pb.mean(0), Pb.std(0, ddof=1) / np.sqrt(len(Pb))
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    ax.axvspan(16, 31, color="#8a5fa8", alpha=0.08, lw=0)
    ax.fill_between(text, mu - se, mu + se, color="#555", alpha=0.25, lw=0)
    ax.plot(text, mu, color="#111", lw=1.0)
    ax.axhline(0, color="#333", lw=0.7, ls=":"); ax.axvline(0, color="#c00", lw=0.7)
    ax.set_title(f"{'cd'[col]}  ΔΦ(t) vs own pre-stimulus baseline", loc="left")
    ax.set_ylabel("ΔΦ(t)", labelpad=5)

    ax = fig.add_subplot(g[2, col])
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    for s in STIMULI:
        idx = [i for i, (r_, s_, c_) in enumerate(meta) if s_ == s]
        v = (P[idx] - P[idx, :PRE_N].mean(1, keepdims=True)).mean(0)
        ax.plot(text, np.convolve(v, np.ones(3) / 3, mode="same"),
                color=CCOL[CLS[s]], lw=0.8, alpha=0.75)
    ax.axhline(0, color="#333", lw=0.7, ls=":"); ax.axvline(0, color="#c00", lw=0.7)
    ax.set_title(f"{'ef'[col]}  ΔΦ(t) per stimulus (colour = class)", loc="left")
    ax.set_ylabel("ΔΦ(t)", labelpad=5); ax.set_xlabel("time from onset (s)", labelpad=4)
    if col == 1:
        h = [mpl.lines.Line2D([], [], color=CCOL[c], lw=1.3, label=c)
             for c in ["attractant", "repellent", "control"]]
        ax.legend(handles=h, frameon=False, fontsize=5.8, loc="upper right")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig35_phi_mean_median_contrast.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig35_phi_mean_median_contrast.png"), dpi=185, bbox_inches="tight")
print("wrote figures/fig35")

# the contrast, quantified
outc = []
for key in ("core", "sens"):
    P, meta = PE[key]
    Pb = P - P[:, :PRE_N].mean(1, keepdims=True)
    for lbl, v in [("stim window", Pb[:, segs["stim"]].mean(1)),
                   ("post window", Pb[:, segs["post"]].mean(1))]:
        outc.append(dict(substrate=key, window=lbl, mean_dPhi=round(float(v.mean()), 4),
                         p_wilcoxon=round(float(stats.wilcoxon(v).pvalue), 5)))
oc = pd.DataFrame(outc)
oc.to_csv(os.path.join(REPO_ROOT, "results/phi_on_off_contrast.csv"), index=False)
print(oc.to_string(index=False))

# %% [markdown]
# ## Figure 36 — raw fluorescence, mean and median, same neurons and timescale

# %%
fig, axes = plt.subplots(2, 8, figsize=(15.6, 4.6), sharex=True)
for j, nm in enumerate(ALL8):
    M = RE[nm]
    for row, stat in enumerate(["mean", "median"]):
        ax = axes[row, j]
        ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
        if stat == "mean":
            mu = np.nanmean(M, 0)
            se = np.nanstd(M, 0, ddof=1) / np.sqrt(len(M))
            lo, hi, col = mu - se, mu + se, BLUE
        else:
            mu = np.nanmedian(M, 0)
            lo, hi, col = np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), ORANGE
        ax.fill_between(text, lo, hi, color=col, alpha=0.20, lw=0)
        ax.plot(text, mu, color=col, lw=0.95)
        ax.axhline(0, color="#333", lw=0.6, ls=":"); ax.axvline(0, color="#c00", lw=0.6)
        ax.tick_params(labelsize=5.4)
        if row == 0:
            ax.set_title(f"{nm}\n({'core' if nm in INTER else 'sensory'})", fontsize=6.6,
                         color="#333" if nm in INTER else "#7a4fa3")
        if j == 0:
            ax.set_ylabel(f"raw ΔF\n{stat}", fontsize=6.2, labelpad=5)
        if row == 1:
            ax.set_xlabel("time from onset (s)", labelpad=3, fontsize=6.2)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig36_raw_mean_median.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig36_raw_mean_median.png"), dpi=185, bbox_inches="tight")
print("wrote figures/fig36")

# %% [markdown]
# ## TPM stability: drop-k dropout, and the φ-per-state map under subsampling

# %%
def all_transitions(key):
    T = []
    for r in RECS:
        st = ST[r][key]
        T += list(zip(st[:-1], st[1:]))
    return np.array(T)
TR = {key: all_transitions(key) for key in ("core", "sens")}

def tpm_from(T, K=16):
    C = np.zeros((K, K)); np.add.at(C, (T[:, 0], T[:, 1]), 1); return C
def row_jsd(C1, C2):
    P1 = (C1 + 0.5) / (C1 + 0.5).sum(1, keepdims=True)
    P2 = (C2 + 0.5) / (C2 + 0.5).sum(1, keepdims=True)
    return float(np.mean([jensenshannon(P1[k], P2[k], base=2) for k in range(16)]))

DROPS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 20000, 30000, 35000, 38000, 39000, 39500, 39700]
rng = np.random.default_rng(0)
rows = []
for key in ("core", "sens"):
    T = TR[key]; N = len(T); Cfull = tpm_from(T)
    for k in DROPS:
        ds = [row_jsd(Cfull, tpm_from(T[rng.choice(N, N - k, replace=False)])) for _ in range(20)]
        rows.append(dict(substrate=key, dropped=k, remaining=N - k,
                         jsd_mean=round(float(np.mean(ds)), 6), jsd_sd=round(float(np.std(ds)), 6)))
stab = pd.DataFrame(rows)
stab.to_csv(os.path.join(REPO_ROOT, "results/tpm_stability_dropout.csv"), index=False)
print(stab.pivot(index="dropped", columns="substrate", values="jsd_mean").to_string())

# %%
def phis_scalar(C, neurons):
    P = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    net = pyphi.Network(convert.state_by_state2state_by_node(P), node_labels=neurons)
    return np.array([float(pyphi.new_big_phi.sia(
        pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4)))).phi) for si in range(16)])

SIZES = [500, 1000, 3000, 10000, 30000]
rows = []
for key, neurons in [("core", INTER), ("sens", SENS)]:
    T = TR[key]; full = phis_scalar(tpm_from(T), neurons)
    top2 = set(np.argsort(full)[-2:])
    for n in SIZES:
        for rep in range(5):
            p = phis_scalar(tpm_from(T[rng.choice(len(T), n, replace=False)]), neurons)
            rows.append(dict(substrate=key, n_transitions=n, rep=rep,
                             spearman=round(float(stats.spearmanr(full, p).statistic), 4),
                             top2_recovered=int(len(set(np.argsort(p)[-2:]) & top2))))
    print(key, "done", flush=True)
ps = pd.DataFrame(rows)
ps.to_csv(os.path.join(REPO_ROOT, "results/tpm_stability_phi.csv"), index=False)
print(ps.groupby(["substrate", "n_transitions"]).spearman.mean().round(3).to_string())

# %% [markdown]
# ## Reading
#
# * **Median vs mean:** the median Φ(t) is flat at ~1 on both substrates while
#   the mean shows all the structure. This is exactly what a top-2-state
#   indicator predicts: the high-Φ states occupy 16–21% of samples, so the
#   median sits in the low-Φ bulk at every time point. The mean IS the correct
#   statistic for a quantity carried by a minority of samples; the flat median
#   confirms the mechanism rather than undermining the effect.
# * **All stimuli, ON vs OFF:** baseline-correcting each epoch against its own
#   pre-stimulus window and pooling all 10 stimuli reproduces the two findings
#   in cleaner form — no stimulus-window effect (core +0.67 p = 0.22, sensory
#   −0.04 p = 0.66) and the significant sensory post-offset dip (−0.32,
#   p = 1e-5). Per-stimulus traces show the dip in every class.
# * **Raw traces:** mean and median agree on the response shapes (sensory ON
#   transients, AWCL OFF rebound, slow core drifts), so the Φ-level story is
#   not an artifact of a few outlier epochs at the fluorescence level either.
# * **TPM stability:** dropping k of 39,824 transitions moves the TPM by a JSD
#   that grows as √k and stays below 0.01 until ~3,000 transitions are removed.
#   The matrix itself is the most robust object in the pipeline. **But the
#   φ-per-state map is far more fragile:** subsampled to 30k transitions
#   (75% of the data) the per-state φ ranking correlates only ρ = 0.82 (core) /
#   0.69 (sensory) with the full-data map, and at one-stimulus scale (~1,000
#   transitions) ρ = 0.30 / 0.13.
#   Φ inherits none of the TPM's √k robustness — it is a highly nonlinear
#   readout that amplifies small row perturbations. Any per-stimulus or
#   per-condition Φ comparison must budget for THIS instability, not the TPM's.

# %% [markdown]
# ## Figure 37 — TPM stability vs φ-map stability

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), constrained_layout=True)
ax = axes[0]
for key, col, mk in [("core", BLUE, "o"), ("sens", ORANGE, "s")]:
    sub = stab[stab.substrate == key]
    ax.errorbar(sub.dropped, sub.jsd_mean, yerr=sub.jsd_sd, fmt=mk + "-", color=col,
                lw=1.2, ms=3.5, capsize=1.5, label=f"{key} substrate")
ax.set_xscale("log"); ax.set_yscale("log")
kk = np.array([10, 30000])
ax.plot(kk, 3.5e-4 * np.sqrt(kk / 10), ls=":", lw=0.9, color="#555")
ax.text(1800, 0.004, "∝ √k", fontsize=6.2, color="#555", rotation=18)
ax.set_xlabel("transitions dropped (of 39,824)", labelpad=5)
ax.set_ylabel("mean row-wise JSD", labelpad=5)
ax.legend(frameon=False, fontsize=6, loc="upper left")
ax.set_title("a  The TPM is very stable:\n   drop-k error grows as √k", loc="left")

ax = axes[1]
for key, col, mk in [("core", BLUE, "o"), ("sens", ORANGE, "s")]:
    sub = stab[stab.substrate == key]
    ax.errorbar(sub.remaining, sub.jsd_mean, yerr=sub.jsd_sd, fmt=mk + "-", color=col,
                lw=1.2, ms=3.5, capsize=1.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
ax.axvline(936, ls="--", lw=0.9, color="#c00")
ax.text(936, 2.2e-4, " one stimulus", fontsize=5.6, color="#c00", va="bottom")
ax.axvline(4979, ls="--", lw=0.9, color="#888")
ax.text(4979, 2.2e-4, " one animal", fontsize=5.6, color="#888", va="bottom", ha="right")
ax.set_xlabel("transitions remaining", labelpad=5)
ax.set_ylabel("mean row-wise JSD", labelpad=5)
ax.set_title("b  ...read as sample size", loc="left")

ax = axes[2]
for key, col, mk in [("core", BLUE, "o"), ("sens", ORANGE, "s")]:
    sub = ps[ps.substrate == key].groupby("n_transitions").agg(
        rho=("spearman", "mean"), rho_sd=("spearman", "std")).reset_index()
    ax.errorbar(sub.n_transitions, sub.rho, yerr=sub.rho_sd, fmt=mk + "-", color=col,
                lw=1.2, ms=3.5, capsize=1.5, label=f"{key}")
ax.set_xscale("log"); ax.axhline(1.0, color="#333", lw=0.7, ls=":")
ax.set_ylim(0, 1.05)
ax.set_xlabel("transitions in subsampled TPM", labelpad=5)
ax.set_ylabel("Spearman ρ vs full-data\nφ_s-per-state map", labelpad=5)
ax.legend(frameon=False, fontsize=5.8, loc="lower right")
ax.set_title("c  ...but the φ-per-state map is NOT", loc="left")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig37_tpm_stability.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig37_tpm_stability.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig37")

# %% [markdown]
# ## Figure 38 — median Φ(t) under the same state rasters

# %%
def epochs_with_pre(nm):
    out = []
    for r in RECS:
        src = BITS[r][nm]
        for s in STIMULI:
            for t0 in sorted(DAT[r][1].get(s, [])):
                if t0 - PRE_N >= 0 and t0 + CYC_N <= len(src):
                    out.append(src[t0 - PRE_N:t0 + CYC_N])
    return np.array(out)
AVG_BIT = {key: np.array([epochs_with_pre(nm).mean(0) for nm in neurons])
           for key, neurons in [("core", INTER), ("sens", SENS)]}

fig = plt.figure(figsize=(12.8, 4.6))
g = fig.add_gridspec(3, 2, hspace=0.16, wspace=0.24, height_ratios=[0.55, 1.0, 1.0])
for col, key in enumerate(("core", "sens")):
    neurons = INTER if key == "core" else SENS
    ax = fig.add_subplot(g[0, col])
    ax.imshow(1 - AVG_BIT[key][::-1], aspect="auto", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest", extent=[text[0], text[-1], -0.5, 3.5])
    ax.set_yticks(range(4)); ax.set_yticklabels(neurons[::-1], fontsize=5.0)
    ax.set_xticks([]); ax.axvline(0, color="#c00", lw=0.8)
    ax.axvline(15, color="#c00", lw=0.8, ls=":")
    ax.set_title(f"{key.upper()} — mean and median Φ(t), same raster", loc="left",
                 fontsize=7.2, pad=3)
    P, meta = PE[key]
    ax = fig.add_subplot(g[1, col])
    mu, se = P.mean(0), P.std(0, ddof=1) / np.sqrt(len(P))
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    ax.fill_between(text, mu - se, mu + se, color="#555", alpha=0.25, lw=0)
    ax.plot(text, mu, color="#111", lw=1.0)
    ax.set_xlim(text[0], text[-1]); ax.axvline(0, color="#c00", lw=0.7)
    ax.set_ylabel("mean Φ(t)", labelpad=4); ax.set_xticklabels([])
    ax = fig.add_subplot(g[2, col])
    md = np.median(P, 0)
    q1, q3 = np.percentile(P, 25, 0), np.percentile(P, 75, 0)
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    ax.fill_between(text, q1, q3, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(text, md, color=ORANGE, lw=1.0)
    ax.set_xlim(text[0], text[-1]); ax.axvline(0, color="#c00", lw=0.7)
    ax.set_ylabel("median Φ(t)\n[IQR]", labelpad=4)
    ax.set_xlabel("time from onset (s)", labelpad=4)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig38_phi_median_rasters.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig38_phi_median_rasters.png"), dpi=185, bbox_inches="tight")
print("wrote figures/fig38")
