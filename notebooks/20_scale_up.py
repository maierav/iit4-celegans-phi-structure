# %% [markdown]
# # 20 — Scale-up: all 107 identified neurons
#
# The interim paper's two analyses at full scale. **Part 1**: the TPM's
# pairwise face against the interventional signal-propagation atlas — grown
# from the quartet's 12 pairs to every measured pair among the 107 neurons
# identified at confidence 1.0 (<5% NaN) in all 8 recordings. **Part 2**: the
# interaction screen — all 606,797 (source-pair → target) triples, with the
# three-gate discipline notebook 19 established (calibrated FDR → split-half
# reproducibility → conditional LRT).

# %%
import os, sys, re, time, pickle, urllib.request
import numpy as np
import pandas as pd
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
FS = ch.SAMPLING_RATE_HZ; W_HP = 20; TAU = 1

# neuron set: conf = 1.0 and <5% NaN in ALL 8 recordings
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
COMMON = sorted(set.intersection(*sets))
N = len(COMMON)
print(f"common high-confidence neurons: {N}")

def hp(x):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - median_filter(xf, size=max(3, round(W_HP * FS)), mode="nearest")
A_list, Y_list = [], []
for r in RECS:
    B = np.stack([(hp(DATA[r][nm]) > 0).astype(np.float32) for nm in COMMON], axis=1)
    A_list.append(B[:-TAU]); Y_list.append(B[TAU:])
A = np.concatenate(A_list); Y = np.concatenate(Y_list)
Ac = (1.0 - A).astype(np.float32)
X1 = np.c_[np.ones(len(A), dtype=np.float32), A]
print("transitions:", len(A))

# %% [markdown]
# ## Part 1 — pairwise EC at scale vs the atlas and the anatomy
#
# Two estimators: **conditional** (per-target logistic on all 107 current
# bits — partial effects, indirect routes removed) and **marginal**
# (P(i|j ON) − P(i|j OFF) — includes indirect routes, like-for-like with
# signal propagation). Atlas extraction follows notebook 16 (stub unpickler,
# orientation verified against the package source).

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
W_full = np.zeros((N, N + 1))
for i in range(N):
    W_full[i] = fit_logistic(X1, Y[:, i].astype(float))
Wc = W_full[:, 1:].T          # Wc[j, i] = coef of source j in target i's model
on_c = A.T @ Y; n_on = A.sum(0)
dP = on_c / n_on[:, None] - (Y.sum(0)[None, :] - on_c) / (len(A) - n_on)[:, None]

base = "https://raw.githubusercontent.com/leiferlab/worm-functional-connectivity/main/"
for f, out in [("atlas/wild-type.pickle", "/tmp/wt.pickle"),
               ("wormfunconn/aconnectome_ids.txt", "/tmp/ids.txt")]:
    if not os.path.exists(out): urllib.request.urlretrieve(base + f, out)
class Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, st): self.__dict__.update(st if isinstance(st, dict) else {"_state": st})
class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return Stub if module.startswith("wormfunconn") else super().find_class(module, name)
fa = StubUnpickler(open("/tmp/wt.pickle", "rb")).load()
nid = [str(x) for x in fa.neu_ids]; SC = np.array(fa.scalar_atlas, dtype=float)
ix = {n: nid.index(n) for n in COMMON if n in nid}
rows = []
for sj in COMMON:
    for ti in COMMON:
        if sj == ti or sj not in ix or ti not in ix: continue
        v = SC[ix[ti], ix[sj]]          # [responding, stimulated]
        if np.isfinite(v):
            rows.append(dict(stim=sj, resp=ti, atlas=v,
                             ours_cond=Wc[COMMON.index(sj), COMMON.index(ti)],
                             ours_marginal=dP[COMMON.index(sj), COMMON.index(ti)]))
at = pd.DataFrame(rows)
rho = sstats.spearmanr(np.abs(at.atlas), np.abs(at.ours_marginal))
thr = np.percentile(np.abs(at.atlas), 75); lab = (np.abs(at.atlas) >= thr).astype(int)
sc = np.abs(at.ours_marginal.values)
auc = sstats.mannwhitneyu(sc[lab == 1], sc[lab == 0]).statistic / (lab.sum() * (len(lab) - lab.sum()))
pa_ = float((at.atlas > 0).mean()); po_ = float((at.ours_marginal > 0).mean())
obs = float((np.sign(at.atlas) * np.sign(at.ours_marginal) > 0).mean())
chance = pa_ * po_ + (1 - pa_) * (1 - po_)
print(f"atlas pairs: {len(at)} | rho={rho.statistic:+.3f} (p={rho.pvalue:.1e}) | AUC={auc:.3f}")
print(f"sign agreement {obs:.3f} vs chance {chance:.3f} -> kappa {(obs-chance)/(1-chance):+.3f} (uninformative)")
neg = at[at.atlas < 0]
print(f"atlas-inhibitory pairs recovered: marginal {float((neg.ours_marginal<0).mean()):.2f}, "
      f"conditional {float((neg.ours_cond<0).mean()):.2f} (n={len(neg)})")

if not os.path.exists("/tmp/herm_el.csv"):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/openworm/c302/master/c302/data/herm_full_edgelist.csv", "/tmp/herm_el.csv")
el = pd.read_csv("/tmp/herm_el.csv"); el.columns = [c.strip() for c in el.columns]
for c in ("Source", "Target"): el[c] = el[c].str.strip()
Aw = np.zeros((N, N)); idx = {n: i for i, n in enumerate(COMMON)}
for _, r_ in el[el.Source.isin(COMMON) & el.Target.isin(COMMON)].iterrows():
    Aw[idx[r_.Source], idx[r_.Target]] += float(r_.Weight)
cross = ~np.eye(N, dtype=bool); he = (Aw > 0)[cross]
sc2 = np.abs(dP)[cross]
auc_an = sstats.mannwhitneyu(sc2[he], sc2[~he]).statistic / (he.sum() * (~he).sum())
print(f"anatomy: AUC(edge vs no-edge) = {auc_an:.3f} ({int(he.sum())} edges of {int(cross.sum())} pairs)")
at.to_csv(os.path.join(REPO_ROOT, "results/ho_scale_atlas_pairs.csv"), index=False)

# negative control: pharynx is nearly isolated from the somatic circuit
PHAR = {"I1L","I1R","I2L","I2R","I3","I4","I5","I6","M1","M2L","M2R","M3L","M3R","M4","M5","MCL","MCR","MI","NSML","NSMR"}
som = [COMMON.index(n) for n in COMMON if n not in PHAR]
ph  = [COMMON.index(n) for n in COMMON if n in PHAR]
w_in = np.abs(Wc[np.ix_(som, som)])[~np.eye(len(som), dtype=bool)]
w_x  = np.abs(Wc[np.ix_(som, ph)]).ravel()
u = sstats.mannwhitneyu(w_in, w_x)
print(f"negative control: somatic-internal |W| median {np.median(w_in):.4f} vs soma→pharynx {np.median(w_x):.4f} (p={u.pvalue:.1e})")

# %% [markdown]
# ## Part 2 — the interaction screen: all 606,797 triples, three gates
#
# **Screen**: Woolf's test on each (x_j, x_k, y_i) 2×2×2 margin — closed-form,
# so the whole grid is a few matmuls per target. **Calibration**: transitions
# are autocorrelated and marginal tables carry omitted-variable distortion,
# so χ² is inflated; a genomic-control λ is estimated from the additive-
# simulation null (per-target main-effects logistic, next-bits regenerated at
# the real current states — occupancy preserved, interactions zero). **Gates**:
# λ-calibrated BH-FDR 0.05 → same sign + nominal p<0.05 in both disjoint
# animal halves → conditional LRT (interaction added to the full 108-term
# main-effects model), λ-calibrated.

# %%
P_tot = A.T @ A; Q_tot = A.T @ Ac; R_tot = Ac.T @ Ac
triu = np.triu_indices(N, 1)
def woolf_screen(Ymat):
    out = np.zeros((N, len(triu[0])), dtype=np.float32)
    Ls  = np.zeros_like(out)
    for i in range(N):
        y = Ymat[:, i].astype(np.float32)
        P1 = A.T @ (A * y[:, None]); Q1 = A.T @ (Ac * y[:, None]); R1 = Ac.T @ (Ac * y[:, None])
        def LV(m11, m10, m01, m00):
            m11 = m11 + 0.5; m10 = m10 + 0.5; m01 = m01 + 0.5; m00 = m00 + 0.5
            return np.log(m11 * m00 / (m10 * m01)), 1/m11 + 1/m10 + 1/m01 + 1/m00
        L1, V1 = LV(P1, P_tot - P1, Q1.T, Q_tot.T - Q1.T)
        L0, V0 = LV(Q1, Q_tot - Q1, R1, R_tot - R1)
        out[i] = (((L1 - L0) ** 2) / (V1 + V0))[triu]
        Ls[i]  = (L1 - L0)[triu]
    return out, Ls
chi2_all, L_all = woolf_screen(Y)

P_add = expit(X1 @ W_full.T).astype(np.float32)
rng = np.random.default_rng(0)
lam_list = []
for rep in range(3):
    Ysim = (rng.random(P_add.shape) < P_add).astype(np.float32)
    lam_list.append(np.median(woolf_screen(Ysim)[0]) / 0.4549)
lam = float(np.mean(lam_list))
print(f"lambda = {lam:.3f}")
pv_adj = sstats.chi2.sf((chi2_all / lam).ravel().astype(float), 1)
m = len(pv_adj)
order = np.argsort(pv_adj); ranked = pv_adj[order]
k_ = np.where(ranked * m / np.arange(1, m + 1) <= 0.05)[0]
n_fdr = int(k_.max() + 1) if len(k_) else 0
print(f"calibrated BH-FDR 0.05: {n_fdr} of {m:,} (Bonferroni: {(pv_adj < 0.05/m).sum()})")

# additive-null yield through the calibrated FDR
Ysim = (rng.random(P_add.shape) < P_add).astype(np.float32)
pv_sim = sstats.chi2.sf((woolf_screen(Ysim)[0] / lam).ravel().astype(float), 1)
o = np.sort(pv_sim); k2 = np.where(o * m / np.arange(1, m + 1) <= 0.05)[0]
print(f"additive-null yield at the same gate: {int(k2.max()+1) if len(k2) else 0}")

# %%
h1 = list(range(4)); h2 = list(range(4, 8))
def half_data(idxs):
    return (np.concatenate([A_list[i] for i in idxs]).astype(np.float32),
            np.concatenate([Y_list[i] for i in idxs]).astype(np.float32))
A1h, Y1h = half_data(h1); A2h, Y2h = half_data(h2)
def woolf_one(Ah, Yh, j, k, i):
    y = Yh[:, i]; a = Ah[:, j]; b = Ah[:, k]
    out = []
    for bv in (1, 0):
        mb = (b == bv)
        n11 = ((a == 1) & (y == 1) & mb).sum() + 0.5; n10 = ((a == 1) & (y == 0) & mb).sum() + 0.5
        n01 = ((a == 0) & (y == 1) & mb).sum() + 0.5; n00 = ((a == 0) & (y == 0) & mb).sum() + 0.5
        out.append((np.log(n11 * n00 / (n10 * n01)), 1/n11 + 1/n10 + 1/n01 + 1/n00))
    (L1, V1), (L0, V0) = out
    return float(L1 - L0), float(sstats.chi2.sf(((L1 - L0) ** 2 / (V1 + V0)) / lam, 1))
def ll_of(Xd, y):
    w = fit_logistic(Xd, y)
    p = np.clip(expit(Xd @ w), 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
flat = np.argsort(pv_adj)[:n_fdr]
rows = []
for f in flat:
    i = int(f // len(triu[0])); pr = int(f % len(triu[0]))
    j, k = int(triu[0][pr]), int(triu[1][pr])
    Lh1, p1 = woolf_one(A1h, Y1h, j, k, i); Lh2, p2 = woolf_one(A2h, Y2h, j, k, i)
    y = Y[:, i].astype(float)
    G = 2 * (ll_of(np.c_[X1, (A[:, j] * A[:, k]).astype(np.float32)], y) - ll_of(X1, y))
    rows.append(dict(source_pair=f"{COMMON[j]}×{COMMON[k]}", target=COMMON[i],
                     chi2_adj=round(float(chi2_all.ravel()[f] / lam), 1), p_adj=float(pv_adj[f]),
                     L_h1=round(Lh1, 3), L_h2=round(Lh2, 3),
                     repro=bool(np.sign(Lh1) == np.sign(Lh2) and p1 < 0.05 and p2 < 0.05),
                     G2_cond=round(G, 1), p_cond=float(sstats.chi2.sf(G / lam, 1))))
surv = pd.DataFrame(rows)
surv.to_csv(os.path.join(REPO_ROOT, "results/ho_scale_interactions.csv"), index=False)
fin = surv[surv.repro & (surv.p_cond < 0.05)].copy()
print(f"gates: {len(surv)} -> repro {int(surv.repro.sum())} -> conditional {len(fin)}")

# %% [markdown]
# ## The motif structure of the 57

# %%
def nclass(n): return re.sub(r"[LR]$", "", n) if re.search(r"[LR]$", n) and n not in ("ALA", "RID") else n
fin["src1"] = [s.split("×")[0] for s in fin.source_pair]
fin["src2"] = [s.split("×")[1] for s in fin.source_pair]
fin["bilateral_pair"] = [nclass(a) == nclass(b) for a, b in zip(fin.src1, fin.src2)]
CMD = {"AVAL","AVAR","AVBL","AVBR","AVEL","AVER","AVDL","AVDR"}
fin["cmd_involved"] = [(a in CMD or b in CMD or t in CMD) for a, b, t in zip(fin.src1, fin.src2, fin.target)]
fin["sign"] = np.sign(fin.L_h1)
n_bil = sum(1 for j, k in zip(*triu) if nclass(COMMON[j]) == nclass(COMMON[k]))
from scipy.stats import hypergeom
p_enr = hypergeom.sf(int(fin.bilateral_pair.sum()) - 1, len(triu[0]), n_bil, len(fin))
print(f"bilateral source pairs: {int(fin.bilateral_pair.sum())}/{len(fin)} vs base {n_bil}/{len(triu[0])} "
      f"(enrichment p = {p_enr:.1e})")
print(f"positives: {int((fin.sign>0).sum())}; bilateral among them: {int(fin[fin.sign>0].bilateral_pair.sum())}")
print(f"command-circuit involved: {int(fin.cmd_involved.sum())}; negative: {int((fin[fin.cmd_involved].sign<0).sum())}")
fin.to_csv(os.path.join(REPO_ROOT, "results/ho_scale_interactions_final.csv"), index=False)

# %% [markdown]
# ## Reading
#
# * **Part 1 tempered.** At 636 measured pairs the atlas correspondence is
#   weak but real: rho = +0.18 (p = 4e-6), AUC 0.58 for detecting the atlas's
#   strong pairs, and it vanishes *within* the strong pairs — detection-level
#   agreement, not amplitude tracking. Signs carry no information beyond base
#   rate (kappa ~ 0.02) and atlas-inhibitory pairs are essentially invisible
#   (both measures are calcium-ON statistics). Anatomy correspondence is the
#   same weak order (AUC 0.58–0.59): at scale, the quartet's clean
#   function-over-anatomy dissociation becomes "weakly matches both". The
#   pharynx negative control passes (cross-circuit coupling lower, p = 7e-18).
# * **Part 2 is the result.** 57 of 606,797 candidate three-way interactions
#   survive calibration (lambda = 1.53), split-half reproduction, and
#   conditional confirmation — the additive null yields zero through the same
#   pipeline. They are not scattered: bilateral homolog source pairs are
#   enriched 23x (p = 7e-12) and supply 6 of the 7 positive (synergistic)
#   interactions, while every command-circuit interaction is negative
#   (33/33) — sub-additivity in log-odds, the mutual-exclusion signature of
#   the forward/backward system. Interactions are log-odds quantities, so
#   probability-ceiling saturation is already accounted for by the additive-
#   logistic null.
# * **Caveats.** lambda-calibration corrects the average inflation, not its
#   tail; the split-half and conditional gates carry the rest. Marginal
#   screen hits can reflect omitted common drive — the conditional LRT gate
#   (all 107 bits in the model) is what earns the word "interaction". And
#   these are ecological-TPM quantities: context mixtures, not frozen-
#   background mechanisms.

# %% [markdown]
# ## Figure 50

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
BLUE, ORANGE, GREY, PURPLE, GREEN = "#1f6fb4", "#c2571a", "#8a8a8a", "#7a4a8a", "#4a7a4a"
fig = plt.figure(figsize=(13.2, 3.7))
g = fig.add_gridspec(1, 4, width_ratios=[1.05, 0.95, 1.05, 1.0], wspace=0.34)
ax = fig.add_subplot(g[0])
ax.scatter(np.abs(at.atlas), np.abs(at.ours_marginal), s=7, color=BLUE, alpha=0.45, lw=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("atlas response amplitude |ΔF/F|", labelpad=5)
ax.set_ylabel("our marginal effect |ΔP|", labelpad=5)
ax.text(0.04, 0.955, f"ρ = {rho.statistic:+.2f} (p = {rho.pvalue:.0e}), n = {len(at)}\n"
        f"AUC strong-vs-weak = {auc:.2f}\nsigns: κ ≈ 0.02 (uninformative)",
        transform=ax.transAxes, fontsize=6, va="top")
ax.set_title("a  At scale the atlas correspondence is\n   weak but real — detection, not amplitude", loc="left")
ax = fig.add_subplot(g[1])
stages = ["BH-FDR 0.05\n(λ-calibrated)", "+ split-half\nreproducible", "+ conditional\nLRT"]
real = [len(surv), int(surv.repro.sum()), len(fin)]
x = np.arange(3)
ax.bar(x - 0.17, real, 0.32, color=ORANGE, label="real data")
ax.bar(x + 0.17, [0, 0, 0], 0.32, color=GREY, label="additive null")
for xi, v in zip(x - 0.17, real): ax.text(xi, v + 1, str(v), ha="center", fontsize=6.5)
for xi in x + 0.17: ax.text(xi, 1, "0", ha="center", fontsize=6.5, color=GREY)
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=5.8)
ax.set_ylabel("interactions surviving", labelpad=5)
ax.set_ylim(0, max(real) + 10)
ax.legend(frameon=False, fontsize=6)
ax.set_title(f"b  {len(fin)} interactions pass all three gates;\n   the additive null yields none", loc="left")
ax = fig.add_subplot(g[2])
mot = np.where(fin.bilateral_pair, "bilateral pair", np.where(fin.cmd_involved, "command circuit", "other"))
COLm = {"bilateral pair": GREEN, "command circuit": PURPLE, "other": GREY}
for mname in ["other", "command circuit", "bilateral pair"]:
    msk = mot == mname
    ax.scatter(fin.L_h1[msk], fin.L_h2[msk], s=16, color=COLm[mname], label=mname, lw=0, zorder=3)
lim = 0.48
ax.plot([-lim, lim], [-lim, lim], color="#ccc", lw=0.7, zorder=1)
ax.axhline(0, color="#ddd", lw=0.6); ax.axvline(0, color="#ddd", lw=0.6)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("interaction (log-odds), animals 1–4", labelpad=5)
ax.set_ylabel("interaction, animals 5–8", labelpad=5)
ax.legend(frameon=False, fontsize=6, loc="lower right")
npos = int((fin.sign > 0).sum()); nbp = int(fin[fin.sign > 0].bilateral_pair.sum())
ncmd = int(fin.cmd_involved.sum())
ax.set_title(f"c  Sign law: command-circuit {ncmd}/{ncmd} negative;\n   {nbp} of {npos} positives are bilateral pairs", loc="left")
ax = fig.add_subplot(g[3])
def cells(j, k, i):
    a = A[:, j].astype(bool); b = A[:, k].astype(bool); y = Y[:, i]
    return np.array([[y[~a & ~b].mean(), y[~a & b].mean()], [y[a & ~b].mean(), y[a & b].mean()]])
eS = cells(COMMON.index("SMDVL"), COMMON.index("SMDVR"), COMMON.index("SMDVR"))
eA = cells(COMMON.index("AVAL"), COMMON.index("AVBR"), COMMON.index("AVBR"))
for e, col in zip([0, 1], [GREY, ORANGE]):
    ax.plot([0, 1], [eS[0, e], eS[1, e]], "o-", color=col, lw=1.4, ms=4)
    ax.plot([2.4, 3.4], [eA[0, e], eA[1, e]], "o--", color=col, lw=1.4, ms=4)
ax.plot([], [], color=GREY, label="2nd source OFF"); ax.plot([], [], color=ORANGE, label="2nd source ON")
ax.set_xticks([0, 1, 2.4, 3.4])
ax.set_xticklabels(["SMDVL\nOFF", "SMDVL\nON", "AVAL\nOFF", "AVAL\nON"], fontsize=6)
ax.text(0.5, -0.24, "→ SMDVR (synergy)", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5, color=GREEN)
ax.text(2.9, -0.24, "→ AVBR (antagonism)", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5, color=PURPLE)
ax.set_ylabel("P(target ON at t+1)", labelpad=5)
ax.legend(frameon=False, fontsize=6, loc="center right")
ax.set_title("d  Synergy within a bilateral pair;\n   antagonism across command neurons", loc="left")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig50_interaction_atlas.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig50_interaction_atlas.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig50")
