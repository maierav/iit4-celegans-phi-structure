# %% [markdown]
# # 19 — Higher-order effective connectivity: the interim-paper seed analysis
#
# The interim paper's two-part structure: (1) the TPM-derived pairwise
# sensitivity matches the interventional signal-propagation atlas (notebook
# 16); (2) the same TPM natively defines **beyond-pairwise** terms that no
# pairwise effective-connectivity measure can express. This notebook
# establishes part 2 honestly: the decomposition, the estimation-bias null
# it must clear, and the proper likelihood-ratio test.
#
# Definitions (sensory quartet, giant TPM, standing pipeline):
# * **1st order** S1[j,i]: mean |ΔP(i ON at t+1)| over the 8 state pairs
#   differing only in bit j (notebook 15's sensitivity matrix).
# * **2nd order** D2[j,k→i]: the mixed discrete partial
#   P(i|j,k) − P(i|j) − P(i|k) + P(i|base), |·| averaged over the 4 base
#   contexts — what every pairwise measure assumes to be zero.
# * **LRT**: per-target logistic models of P(i ON at t+1) from the four
#   current bits — additive vs +2-way vs +3-way vs saturated — tested on the
#   39,824 real transitions. This is the claim-bearing test; |D2| maps are
#   descriptive.

# %%
import os, sys, ast
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from scipy.ndimage import median_filter
from scipy.special import expit
from scipy import stats as sstats

REPO_ROOT = "."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import ces_hypergraph as ch
for r in ch.HERM_DRIVE_IDS:
    ch.ensure_recording(r)

RECS = list(ch.HERM_DRIVE_IDS)
Q = ["ASEL", "ASER", "AWAL", "AWCL"]
FS = ch.SAMPLING_RATE_HZ; W_HP = 20; TAU = 1

def load(rec):
    d = pd.read_csv(ch.recording_path(rec)); tc = d.columns[9:-1]
    return {nm: d.loc[d.neuron == nm, tc].iloc[0].astype(float).values for nm in Q}
def hp(x):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - median_filter(xf, size=max(3, round(W_HP * FS)), mode="nearest")
ST = {r: ch.combine_states([(hp(load(r)[nm]) > 0).astype(int) for nm in Q]) for r in RECS}
def C_of(recs):
    C = np.zeros((16, 16))
    for r in recs:
        np.add.at(C, (ST[r][:-TAU], ST[r][TAU:]), 1)
    return C
Cfull = C_of(RECS)
TRANS = np.concatenate([np.stack([ST[r][:-TAU], ST[r][TAU:]], axis=1) for r in RECS])
X = np.stack([(TRANS[:, 0] >> j) & 1 for j in range(4)], axis=1).astype(float)
Ynext = np.stack([(TRANS[:, 1] >> i) & 1 for i in range(4)], axis=1).astype(float)
print("transitions:", len(TRANS))

# %% [markdown]
# ## The decomposition and its split-half stability

# %%
def sbn_of(C):
    sbn = np.zeros((16, 4))
    for s_ in range(16):
        tot = C[s_].sum()
        for i in range(4):
            on = C[s_, [t_ for t_ in range(16) if (t_ >> i) & 1]].sum()
            sbn[s_, i] = (on + 0.5) / (tot + 1.0)
    return sbn

def orders(C):
    sbn = sbn_of(C)
    S1 = np.zeros((4, 4)); S2 = np.zeros((4, 4, 4))
    for j in range(4):
        pairs = [(s_, s_ | (1 << j)) for s_ in range(16) if not (s_ >> j) & 1]
        for i in range(4):
            S1[j, i] = np.mean([abs(sbn[b, i] - sbn[a, i]) for a, b in pairs])
    for j, k in combinations(range(4), 2):
        base = [s_ for s_ in range(16) if not (s_ >> j) & 1 and not (s_ >> k) & 1]
        for i in range(4):
            S2[j, k, i] = np.mean(np.abs([sbn[s_ | (1 << j) | (1 << k), i] - sbn[s_ | (1 << j), i]
                                          - sbn[s_ | (1 << k), i] + sbn[s_, i] for s_ in base]))
    return S1, S2

S1f, S2f = orders(Cfull)
halves = list(combinations(range(8), 4))[:35]
S1h, S2h = [], []
for h in halves:
    for recs in ([RECS[i] for i in h], [RECS[i] for i in range(8) if i not in h]):
        a, b = orders(C_of(recs)); S1h.append(a); S2h.append(b)
S1h = np.array(S1h); S2h = np.array(S2h)
SNR1 = S1f / np.maximum(S1h.std(0), 1e-9)
SNR2 = S2f / np.maximum(S2h.std(0), 1e-9)
triples = [(j, k, i) for j, k in combinations(range(4), 2) for i in range(4)]
cross = ~np.eye(4, dtype=bool)
print(f"1st order cross terms: SNR>2 in {(SNR1[cross] > 2).sum()}/12")
print(f"2nd order terms:       SNR>2 in {sum(SNR2[t] > 2 for t in triples)}/24; "
      f"median |D2| {np.median([S2f[t] for t in triples]):.4f} "
      f"({np.median([S2f[t] for t in triples]) / np.median(S1f[cross]):.2f}x the 1st-order cross median)")
pd.DataFrame(S1f, index=Q, columns=Q).to_csv(os.path.join(REPO_ROOT, "results/ho_order1.csv"))
pd.DataFrame([dict(pair=f"{Q[j]}×{Q[k]}", target=Q[i], D2=round(float(S2f[j, k, i]), 5),
                   snr_splithalf=round(float(SNR2[j, k, i]), 2)) for j, k, i in triples]
             ).to_csv(os.path.join(REPO_ROOT, "results/ho_order2.csv"), index=False)

# %% [markdown]
# ## The additive null: |D2| of a mechanism with NO interactions
#
# |mixed partial| of a smoothed, finite-sample estimate is positively biased,
# so raw |D2| magnitude is NOT evidence of higher-order structure. The null:
# fit a main-effects-only logistic mechanism per target, regenerate next-bits
# AT THE REAL CURRENT STATES (occupancy preserved exactly, interactions zero
# by construction), recompute |D2|. 40 replicates.

# %%
def fit_logistic(Xd, y, iters=80):
    w = np.zeros(Xd.shape[1])
    for _ in range(iters):
        p = expit(Xd @ w)
        g = Xd.T @ (y - p)
        H = (Xd * (p * (1 - p))[:, None]).T @ Xd + 1e-8 * np.eye(Xd.shape[1])
        step = np.linalg.solve(H, g); w += step
        if np.max(np.abs(step)) < 1e-10: break
    return w

X1 = np.c_[np.ones(len(X)), X]
W_ADD = np.stack([fit_logistic(X1, Ynext[:, i]) for i in range(4)])
rng = np.random.default_rng(0)
null_S2 = []
for _ in range(40):
    P_on = np.stack([expit(X1 @ W_ADD[i]) for i in range(4)], axis=1)
    bits = (rng.random(P_on.shape) < P_on).astype(int)
    nxt = sum(bits[:, i] * (2 ** i) for i in range(4))
    Cn = np.zeros((16, 16)); np.add.at(Cn, (TRANS[:, 0], nxt), 1)
    null_S2.append(orders(Cn)[1])
null_S2 = np.array(null_S2)
rows = []
for j, k, i in triples:
    nv = null_S2[:, j, k, i]
    rows.append(dict(pair=f"{Q[j]}×{Q[k]}", target=Q[i], D2=round(float(S2f[j, k, i]), 4),
                     null_mean=round(float(nv.mean()), 4), null_sd=round(float(nv.std()), 5),
                     z=round(float((S2f[j, k, i] - nv.mean()) / max(nv.std(), 1e-12)), 2),
                     p_emp=round(float((np.sum(nv >= S2f[j, k, i]) + 1) / 41), 4)))
nulltab = pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)
nulltab.to_csv(os.path.join(REPO_ROOT, "results/ho_additive_null.csv"), index=False)
print(f"null |D2| mean ~{null_S2[null_S2 > 0].mean():.4f} — comparable to the measured values")
print(f"terms with z>2 vs additive null: {(nulltab.z > 2).sum()}/24; z>3: {(nulltab.z > 3).sum()}/24")
print("-> per-term, nothing survives multiple comparisons. The LRT below is the proper test.")

# %% [markdown]
# ## The claim-bearing test: likelihood ratios on the real transitions

# %%
def design(X, order):
    cols = [np.ones(len(X))] + [X[:, j] for j in range(4)]
    if order >= 2:
        for j, k in combinations(range(4), 2): cols.append(X[:, j] * X[:, k])
    if order >= 3:
        for c in combinations(range(4), 3): cols.append(np.prod(X[:, c], axis=1))
    if order >= 4:
        cols.append(np.prod(X, axis=1))
    return np.stack(cols, axis=1)

def ll_of(Xd, y):
    w = fit_logistic(Xd, y)
    p = np.clip(expit(Xd @ w), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

lrt_rows = []
for i in range(4):
    lls = {o: ll_of(design(X, o), Ynext[:, i]) for o in (1, 2, 3, 4)}
    for lo, hi, df in [(1, 2, 6), (2, 3, 4), (3, 4, 1)]:
        G = 2 * (lls[hi] - lls[lo])
        lrt_rows.append(dict(target=Q[i], order=hi, G2=round(G, 1), df=df,
                             p=float(sstats.chi2.sf(G, df))))
lrt = pd.DataFrame(lrt_rows)
lrt.to_csv(os.path.join(REPO_ROOT, "results/ho_lrt.csv"), index=False)
print(lrt.to_string(index=False))
print("\nBonferroni over 4 targets (alpha = 0.0125):")
for order in (2, 3, 4):
    sig = lrt[(lrt.order == order) & (lrt.p < 0.0125)].target.tolist()
    print(f"  order {order}: {sig if sig else 'none'}")

# %% [markdown]
# ## The one surviving interaction, shown directly

# %%
emp = np.zeros((2, 2))
a_bit = (TRANS[:, 0] >> 0) & 1; e_bit = (TRANS[:, 0] >> 1) & 1; w_next = (TRANS[:, 1] >> 2) & 1
for a in (0, 1):
    for e in (0, 1):
        m = (a_bit == a) & (e_bit == e)
        emp[a, e] = w_next[m].mean()
print("P(AWAL ON at t+1):")
print(pd.DataFrame(emp, index=["ASEL=0", "ASEL=1"], columns=["ASER=0", "ASER=1"]).round(4))
print(f"ASEL effect on AWAL: +{emp[1,0]-emp[0,0]:.4f} when ASER OFF, +{emp[1,1]-emp[0,1]:.4f} when ASER ON")

# %% [markdown]
# ## Reading — what the interim paper can and cannot claim
#
# * **Raw higher-order magnitude is mostly estimation bias.** Measured |D2|
#   (median 0.018) sits barely above an additive mechanism's |D2| at matched
#   volume and smoothing (~0.017): the absolute value of a noisy estimate is
#   positively biased. Per-term, 0/24 survive multiple comparisons against
#   the additive null. **This is itself a publishable warning** — the same
#   nonlinearity-manufactures-structure lesson as the Φ amplification result
#   (notebook 15), now at the connectivity level, and directly relevant to
#   the higher-order-interactions literature, which rarely runs matched
#   additive nulls.
# * **One genuine interaction survives the proper test**: AWAL's mechanism is
#   non-additive (LRT G² = 18.8, df 6, p = 0.0045, Bonferroni-safe), and the
#   effect is interpretable — ASEL's influence on AWAL halves when ASER is ON
#   (+0.039 → +0.019), the two salt sensors jointly gating the odor sensor.
#   No 3-way or 4-way term is detectable at this volume.
# * **For the paper**: part 2 is therefore "a principled, bias-corrected
#   test for mechanism-level higher-order connectivity" — demonstrated by one
#   validated interaction and an honest negative for the rest — rather than
#   "rich higher-order structure". The scale-up (all identified triples
#   across the recordings) is where the distribution of genuine interactions
#   gets measured.

# %% [markdown]
# ## Figure 49

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), constrained_layout=True)
ax = axes[0]
nt = nulltab.sort_values("z", ascending=True).reset_index(drop=True)
y = np.arange(len(nt))
ax.errorbar(nt.null_mean, y, xerr=1.96 * nt.null_sd, fmt="|", color=GREY, elinewidth=3.5,
            alpha=0.55, label="additive null (95%)")
ax.scatter(nt.D2, y, s=14, color=ORANGE, zorder=3, label="measured |D2|")
for yi, r_ in nt.iterrows():
    if r_.z > 2: ax.text(r_.D2 + 0.0012, yi, f"{r_['pair']}→{r_.target}", fontsize=5.2, va="center")
ax.set_yticks([]); ax.set_xlabel("second-order term |D2| (Δ probability)", labelpad=5)
ax.set_xlim(0, 0.047)
ax.legend(frameon=False, fontsize=6, loc="lower right")
ax.set_title("a  Most raw 3-way magnitude is estimation\n   bias — the additive null is not ~0", loc="left")
ax = axes[1]
x = np.arange(4); w = 0.26
for oi, (order, col, lab) in enumerate([(2, ORANGE, "2-way (df 6)"), (3, "#7a4a8a", "3-way (df 4)"),
                                        (4, GREY, "4-way (df 1)")]):
    v = [lrt[(lrt.target == q) & (lrt.order == order)].G2.iloc[0] for q in Q]
    ax.bar(x + (oi - 1) * w, v, w, color=col, label=lab)
ax.axhline(sstats.chi2.isf(0.0125, 6), ls=":", lw=0.9, color=ORANGE)
ax.text(3.45, sstats.chi2.isf(0.0125, 6) + 0.4, "p = 0.0125 (df 6)", fontsize=5.5, color=ORANGE, ha="right")
ax.set_xticks(x); ax.set_xticklabels(Q, fontsize=7)
ax.set_ylabel("likelihood-ratio G² (vs one order lower)", labelpad=5)
ax.legend(frameon=False, fontsize=6)
ax.set_title("b  One mechanism is genuinely non-additive:\n   AWAL (G² = 18.8, p = 0.0045)", loc="left")
ax = axes[2]
for e, (col, lab) in enumerate([(BLUE, "ASER OFF"), (ORANGE, "ASER ON")]):
    ax.plot([0, 1], [emp[0, e], emp[1, e]], "o-", color=col, lw=1.5, ms=5, label=lab)
ax.set_xticks([0, 1]); ax.set_xticklabels(["ASEL OFF", "ASEL ON"], fontsize=7.5)
ax.set_ylabel("P(AWAL ON at t+1)", labelpad=5)
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
ax.set_ylim(0.398, 0.462)
ax.text(0.5, 0.403, "non-parallel = interaction: ASEL's effect on AWAL\nhalves when ASER is ON (+0.039 → +0.019)\nn = 8.0k–12.6k transitions per cell",
        ha="center", fontsize=6, color="#333")
ax.set_title("c  The salt sensors jointly gate the\n   odor sensor's next state", loc="left")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig49_higher_order.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig49_higher_order.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig49")
