# %% [markdown]
# # 21 — The Markov-order test
#
# Every TPM in this repository is a **first-order** object: P(state at t+τ |
# state at t). This notebook tests that assumption directly, at two levels,
# and checks whether the scale-up's 57 interactions survive a
# history-augmented model.
#
# * **Per-target** (all 107 neurons): logistic P(bit_i at t+1) from bits at t
#   vs + bits at t−1. Sensitive but confounded: the first-order model is a
#   main-effects approximation, so t−1 can look informative by proxying
#   within-t interactions.
# * **State-level** (both quartets): second- vs first-order Markov on the
#   saturated 16-state kernel — the clean memory test. Calibrated by
#   parametric bootstrap (first-order chains simulated from the estimated
#   TPM, same lengths and initial states).
# * **τ-sweep**: where in time does the memory live?

# %%
import os, sys, time
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.special import expit
from scipy import stats as sstats

REPO_ROOT = "."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import ces_hypergraph as ch
for r in ch.HERM_DRIVE_IDS:
    ch.ensure_recording(r)
RECS = list(ch.HERM_DRIVE_IDS)
FS = ch.SAMPLING_RATE_HZ; W_HP = 20

def hp(x):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - median_filter(xf, size=max(3, round(W_HP * FS)), mode="nearest")

# the 107-neuron set (as notebook 20)
DATA = {}; sets = []
for r in RECS:
    d = pd.read_csv(ch.recording_path(r)); tc = d.columns[9:-1]
    d = d[d.conf == 1.0]
    ok = set()
    for _, row in d.iterrows():
        tr = row[tc].astype(float).values
        if np.mean(~np.isfinite(tr)) < 0.05:
            ok.add(row.neuron); DATA.setdefault(r, {})[row.neuron] = tr
    sets.append(ok)
COMMON = sorted(set.intersection(*sets)); N = len(COMMON)
A_list, Y_list = [], []
for r in RECS:
    B = np.stack([(hp(DATA[r][nm]) > 0).astype(np.float32) for nm in COMMON], axis=1)
    A_list.append(B[:-1]); Y_list.append(B[1:])
Xp = np.concatenate([a[:-1] for a in A_list])
Xc = np.concatenate([a[1:] for a in A_list])
Yn = np.concatenate([y[1:] for y in Y_list])
n = len(Xc)
X_1st = np.c_[np.ones(n, dtype=np.float32), Xc]
X_2nd = np.c_[X_1st, Xp]
print(f"{N} neurons, {n} aligned (t-1, t, t+1) triples")

# %% [markdown]
# ## Per-target test (confounded — reported with its caveat)

# %%
def fit_logistic(Xd, y, iters=25, ridge=1e-4):
    w = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = expit(Xd @ w)
        g = Xd.T @ (y - p) - ridge * w
        H = (Xd * (p * (1 - p))[:, None]).T @ Xd + ridge * np.eye(Xd.shape[1])
        step = np.linalg.solve(H, g); w += step
        if np.max(np.abs(step)) < 1e-8: break
    return w
def ll_of(Xd, y):
    w = fit_logistic(Xd, y)
    p = np.clip(expit(Xd @ w), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

rows = []
for i in range(N):
    y = Yn[:, i].astype(float)
    p0 = y.mean(); ll_m = float(n * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0)))
    ll1 = ll_of(X_1st, y); ll2 = ll_of(X_2nd, y)
    rows.append(dict(target=COMMON[i],
                     info_1st_bits=(ll1 - ll_m) / n / np.log(2),
                     gain_2nd_bits=(ll2 - ll1) / n / np.log(2),
                     G2_2nd=2 * (ll2 - ll1)))
mo = pd.DataFrame(rows)
mo.to_csv(os.path.join(REPO_ROOT, "results/markov_per_target.csv"), index=False)
h0_bits = N / (2 * n * np.log(2))
print(f"first-order info median {mo.info_1st_bits.median():.4f} bits | 2nd-order gain median "
      f"{mo.gain_2nd_bits.median():.5f} (H0 overfit floor {h0_bits:.5f})")
print(f"targets beating chi2 Bonferroni (df {N}): "
      f"{int((mo.G2_2nd > sstats.chi2.isf(0.05 / N, N)).sum())}/{N}")

# %% [markdown]
# ## State-level test — the clean instrument

# %%
QS = {"sens": ["ASEL", "ASER", "AWAL", "AWCL"], "core": ["AIBL", "AVEL", "AVAL", "RIML"]}
STQ = {}
for key, qq in QS.items():
    STQ[key] = {}
    for r in RECS:
        d = pd.read_csv(ch.recording_path(r)); tc = d.columns[9:-1]
        tr = {nm: d.loc[d.neuron == nm, tc].iloc[0].astype(float).values for nm in qq}
        STQ[key][r] = ch.combine_states([(hp(tr[nm]) > 0).astype(int) for nm in qq])

def g2_secondorder(seqs, K=16):
    C1 = np.zeros((K, K)); C2 = np.zeros((K, K, K))
    for s in seqs:
        np.add.at(C1, (s[:-1], s[1:]), 1)
        np.add.at(C2, (s[:-2], s[1:-1], s[2:]), 1)
    P1 = np.where(C1.sum(1, keepdims=True) > 0, C1 / np.maximum(C1.sum(1, keepdims=True), 1), 0)
    G = 0.0
    for sp in range(K):
        for scur in range(K):
            row = C2[sp, scur]; tot = row.sum()
            if tot == 0: continue
            e = tot * P1[scur]; m_ = row > 0
            G += 2 * np.sum(row[m_] * np.log(row[m_] / np.maximum(e[m_], 1e-12)))
    return float(G)
def g2_first(seqs, K=16):
    C1 = np.zeros((K, K))
    for s in seqs: np.add.at(C1, (s[:-1], s[1:]), 1)
    pm = C1.sum(0) / C1.sum()
    G = 0.0
    for scur in range(K):
        row = C1[scur]; tot = row.sum()
        if tot == 0: continue
        e = tot * pm; m_ = row > 0
        G += 2 * np.sum(row[m_] * np.log(row[m_] / np.maximum(e[m_], 1e-12)))
    return float(G)
def sim_firstorder(seqs, rng, K=16):
    C1 = np.zeros((K, K))
    for s in seqs: np.add.at(C1, (s[:-1], s[1:]), 1)
    cum = np.cumsum((C1 + 0.5) / (C1 + 0.5).sum(1, keepdims=True), axis=1)
    out = []
    for s in seqs:
        sim = np.empty(len(s), dtype=int); sim[0] = s[0]
        u = rng.random(len(s))
        for t in range(1, len(s)):
            sim[t] = np.searchsorted(cum[sim[t - 1]], u[t])
        out.append(sim)
    return out

rng = np.random.default_rng(1)
mo_state = []
for key in QS:
    seqs = [STQ[key][r] for r in RECS]
    G_real = g2_secondorder(seqs)
    null = np.array([g2_secondorder(sim_firstorder(seqs, rng)) for _ in range(20)])
    ntr = sum(len(s) - 2 for s in seqs)
    bits_raw = G_real / (2 * ntr * np.log(2)); bits_null = null.mean() / (2 * ntr * np.log(2))
    bits_first = g2_first(seqs) / (2 * sum(len(s) - 1 for s in seqs) * np.log(2))
    mo_state.append(dict(quartet=key, G2_real=round(G_real), null_mean=round(float(null.mean())),
                         null_sd=round(float(null.std())),
                         z=round(float((G_real - null.mean()) / null.std()), 1),
                         bits_first_order=round(bits_first, 4),
                         bits_memory_debiased=round(bits_raw - bits_null, 4),
                         memory_pct_of_first=round(100 * (bits_raw - bits_null) / bits_first, 1)))
    print(mo_state[-1])
mos = pd.DataFrame(mo_state)
mos.to_csv(os.path.join(REPO_ROOT, "results/markov_state_level.csv"), index=False)

# %% [markdown]
# ## Where the memory lives: the τ-sweep

# %%
rng = np.random.default_rng(2)
rows = []
for key in QS:
    for tau_s in [1, 2, 4, 8, 16]:
        seqs = [STQ[key][r][::tau_s] for r in RECS]
        G_real = g2_secondorder(seqs)
        null = np.array([g2_secondorder(sim_firstorder(seqs, rng)) for _ in range(12)])
        ntr = sum(len(s) - 2 for s in seqs)
        bits = (G_real - null.mean()) / (2 * ntr * np.log(2))
        bf = g2_first(seqs) / (2 * sum(len(s) - 1 for s in seqs) * np.log(2))
        rows.append(dict(quartet=key, tau_samples=tau_s, tau_s=round(tau_s * 0.375, 2),
                         n_trans=ntr, bits_first=round(bf, 4), bits_memory=round(float(bits), 4),
                         memory_pct=round(100 * bits / bf, 1),
                         z=round(float((G_real - null.mean()) / null.std()), 1)))
tsw = pd.DataFrame(rows)
tsw.to_csv(os.path.join(REPO_ROOT, "results/markov_tau_sweep.csv"), index=False)
print(tsw.to_string(index=False))

# %% [markdown]
# ## Do the 57 interactions survive a history-augmented model?

# %%
fin = pd.read_csv(os.path.join(REPO_ROOT, "results/ho_scale_interactions_final.csv"))
lam = 1.526   # calibration from notebook 20
G_hist = []; p_hist = []
for _, r_ in fin.iterrows():
    j = COMMON.index(r_.src1); k = COMMON.index(r_.src2); i = COMMON.index(r_.target)
    y = Yn[:, i].astype(float)
    G = 2 * (ll_of(np.c_[X_2nd, (Xc[:, j] * Xc[:, k]).astype(np.float32)], y) - ll_of(X_2nd, y))
    G_hist.append(round(G, 1)); p_hist.append(float(sstats.chi2.sf(G / lam, 1)))
fin["G2_hist"] = G_hist; fin["p_hist"] = p_hist
fin.to_csv(os.path.join(REPO_ROOT, "results/ho_scale_interactions_final.csv"), index=False)
print(f"history-robust: {int((fin.p_hist < 0.05).sum())}/{len(fin)} interactions survive "
      f"with t-1 main effects in the model")

# %% [markdown]
# ## Reading
#
# * **First-order is violated, substantially.** The clean state-level test:
#   memory beyond the current state carries 69% (sensory) and 77% (core) as
#   much predictive information as the entire first-order structure
#   (z = +32 / +46 vs first-order chains). The per-target version agrees
#   directionally (104/107) but conflates memory with main-effects
#   misspecification, so the state-level numbers are the citable ones.
# * **The memory is fast.** It collapses with lag — 69→34→19% (sensory) and
#   77→49→28% (core) at τ = 0.38/0.75/1.5 s, with z falling from ~40–55 to
#   1.7–3.7 by 1.5 s (τ = 6 s estimates are n = 2.5k and unreliable). That
#   timescale is the GCaMP-decay/moving-median window, pointing at indicator
#   and binarization smearing rather than deep neural history.
# * **The 57 interactions are not memory artifacts**: all 57 survive the
#   conditional LRT with t−1 main effects added (57/57 at p < 0.05,
#   λ-calibrated).
# * **What this means for the TPMs.** The τ = 375 ms TPM is the one-lag
#   conditional of a process that is not first-order Markov: a well-defined
#   *descriptive* object (everything built on it — sensitivity, interactions
#   — is a one-lag quantity), but not the generative kernel. A genuine
#   tension to carry: the lag where first-order roughly holds (≥1.5 s) is
#   the lag where the atlas agreement fails (notebook 16's sweep), so one
#   cannot have the externally-validated TPM and the Markov-clean TPM at the
#   same τ with this indicator.

# %% [markdown]
# ## Figure 51

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter
BLUE, ORANGE, GREY, PURPLE = "#1f6fb4", "#c2571a", "#8a8a8a", "#7a4a8a"
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), constrained_layout=True)
ax = axes[0]
ax.scatter(mo.info_1st_bits, mo.gain_2nd_bits, s=10, color=BLUE, alpha=0.6, lw=0)
ax.axhline(h0_bits, ls=":", lw=1, color="#333")
ax.text(mo.info_1st_bits.max() * 0.97, h0_bits * 1.15, "overfitting floor (H₀)", fontsize=5.8, color="#333", ha="right")
ax.set_xlabel("first-order information (bits/transition)", labelpad=5)
ax.set_ylabel("gain from adding t−1 (bits)", labelpad=5)
ax.set_title("a  104/107 targets beat the first-order\n   model — but this test conflates memory\n   with model misspecification", loc="left", fontsize=8)
ax = axes[1]
x = np.arange(2)
ax.bar(x - 0.17, mos.G2_real, 0.32, color=ORANGE, label="real G²")
ax.bar(x + 0.17, mos.null_mean, 0.32, color=GREY, yerr=mos.null_sd,
       label="first-order chains (20 sims)", error_kw=dict(lw=0.8))
for xi, (g, z) in zip(x - 0.17, zip(mos.G2_real, mos.z)):
    ax.text(xi, g + 120, f"z = {z:+.0f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(["sensory quartet", "core quartet"], fontsize=7)
ax.set_ylabel("second-order G² (saturated state kernel)", labelpad=5)
ax.legend(frameon=False, fontsize=6)
ax.set_ylim(0, float(mos.G2_real.max()) * 1.13)
ax.set_title("b  The clean test: memory is real —\n   69–77% of the first-order information", loc="left")
ax = axes[2]
for key, col, lab in [("sens", BLUE, "sensory"), ("core", PURPLE, "core")]:
    sub = tsw[tsw.quartet == key]
    solid = sub[sub.tau_samples < 16]; weak = sub[sub.tau_samples == 16]
    ax.plot(solid.tau_s, solid.memory_pct, "o-", color=col, lw=1.4, ms=4.5, label=lab)
    ax.plot(weak.tau_s, weak.memory_pct, "o", mfc="none", mec=col, ms=4.5)
ax.axvspan(0.3, 0.8, color="#000", alpha=0.05, lw=0)
ax.text(0.52, 118, "TPM regime\n(atlas-validated)", fontsize=5.6, ha="center", color="#333")
ax.set_xscale("log")
ax.xaxis.set_minor_formatter(NullFormatter()); ax.xaxis.set_minor_locator(FixedLocator([]))
ax.xaxis.set_major_locator(FixedLocator([0.375, 0.75, 1.5, 3, 6]))
ax.xaxis.set_major_formatter(FixedFormatter(["0.38", "0.75", "1.5", "3", "6"]))
ax.set_xlabel("lag τ (s)", labelpad=5)
ax.set_ylabel("memory, % of first-order information", labelpad=5)
ax.legend(frameon=False, fontsize=6.5, loc="upper center")
ax.set_ylim(0, 132)
ax.text(5.9, 18, "open markers:\nn = 2.5k, unreliable", fontsize=5.4, ha="right", color=GREY)
ax.set_title("c  The memory lives at sub-1.5 s lags —\n   the indicator/binarization timescale", loc="left")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig51_markov_order.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig51_markov_order.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig51")
