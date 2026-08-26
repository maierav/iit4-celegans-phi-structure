# %% [markdown]
# # 16 — Our effective connectivity vs the literature
#
# The TPM-derived effective sensitivity matrix (notebook 15) compared against
# two published connectivities of the same four neurons:
#
# * **Anatomy** — synapse counts from the hermaphrodite connectome
#   (Cook et al. 2019, *Nature* 571:63), via OpenWorm's c302 edge list.
# * **Function** — the signal-propagation atlas measured by single-neuron
#   optogenetic activation with whole-brain imaging
#   (Randi, Sharma, Dvali & Leifer 2023, *Nature* 623:406; data from
#   leiferlab/worm-functional-connectivity, wild-type pickle).
#
# The functional atlas paper's own headline is that signal propagation differs
# from anatomical predictions, partly through extrasynaptic (peptidergic)
# signalling — so disagreement among the three is the expected outcome, not a
# failure of any one of them.

# %%
import os, sys
import numpy as np
import pandas as pd
import urllib.request, pickle
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = "."
Q = ["ASEL", "ASER", "AWAL", "AWCL"]

# anatomy (Cook 2019 via OpenWorm c302)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/openworm/c302/master/c302/data/herm_full_edgelist.csv",
    "/tmp/herm_full_edgelist.csv")
el = pd.read_csv("/tmp/herm_full_edgelist.csv")
el.columns = [c.strip() for c in el.columns]
for c in ("Source", "Target"):
    el[c] = el[c].str.strip()
# Electrical edges are ALREADY listed reciprocally in the source (verified:
# 99.8% of electrical rows have their mirror row present), so rows are added
# exactly as listed — re-symmetrizing would double-count each gap junction.
ee = el[el.Type.str.strip() == "electrical"]
pairs = set(zip(ee.Source, ee.Target))
recip = sum((b, a) in pairs for a, b in pairs)
print(f"electrical reciprocal-listing check: {recip}/{len(pairs)} rows mirrored")
sub = el[el.Source.isin(Q) & el.Target.isin(Q)]
A = np.zeros((4, 4)); idx = {n: i for i, n in enumerate(Q)}
for _, r_ in sub.iterrows():
    A[idx[r_.Source], idx[r_.Target]] += float(r_.Weight)
pd.DataFrame(A, index=Q, columns=Q).to_csv(
    os.path.join(REPO_ROOT, "results/anatomical_weights_quartet.csv"))
print("anatomy (rows source, cols target):")
print(pd.DataFrame(A, index=Q, columns=Q).astype(int).to_string())

# %%
# functional atlas (Randi 2023). The pickle references the wormfunconn package,
# whose compiled extension is NumPy-1-only; a stub unpickler extracts the
# arrays without importing it.
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/leiferlab/worm-functional-connectivity/main/atlas/wild-type.pickle",
    "/tmp/wild-type.pickle")

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {"_state": st})

class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("wormfunconn"):
            return type(name, (_Stub,), {})
        return super().find_class(module, name)

fa = _StubUnpickler(open("/tmp/wild-type.pickle", "rb")).load()
nid = [str(x) for x in fa.neu_ids]
ix = {n: nid.index(n) for n in Q}
# scalar_atlas is indexed [responding, stimulated] (see FunctionalAtlas.get_responses:
# self.ec[rn_i, sn_i]); transpose to [stimulated, responding] = [source, target]
F = np.array([[fa.scalar_atlas[ix[a], ix[b]] for b in Q] for a in Q], dtype=float).T
Fd = pd.DataFrame(F, index=Q, columns=Q)
Fd.to_csv(os.path.join(REPO_ROOT, "results/functional_atlas_quartet.csv"))
print("functional atlas (rows stimulated, cols responding; NaN = not measured):")
print(Fd.round(4).to_string())

# %%
S = pd.read_csv(os.path.join(REPO_ROOT, "results/effective_sensitivity_sens.csv"),
                index_col=0).values
rows = []
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        rows.append(dict(pair=f"{Q[i]}->{Q[j]}", anatomy=A[i, j],
                         functional=F[i, j] if np.isfinite(F[i, j]) else np.nan,
                         ours=S[i, j]))
three = pd.DataFrame(rows)
three.to_csv(os.path.join(REPO_ROOT, "results/connectivity_three_way.csv"), index=False)
print(three.round(4).to_string(index=False))
mask = ~np.eye(4, dtype=bool)
print("\nSpearman rho over the 12 cross pairs (ours vs anatomy):",
      round(float(stats.spearmanr(A[mask], S[mask]).statistic), 3))
m = three.dropna()
print(f"pairs with all three measured: {len(m)} (functional atlas covers 4 of 12)")


# %%
# Diagonal-blocked comparison (the published data cannot contain self-activation
# by design: you cannot stimulate a neuron and read its own evoked response as
# a propagation edge, and anatomy has no self-synapses). Our diagonal is the
# largest entry in the matrix, so it dominates any shared colour scale and
# hides the cross-term pattern.
meas = [(i, j) for i in range(4) for j in range(4)
        if i != j and np.isfinite(F[i, j])]
ourv = {(i, j): S[i, j] for i in range(4) for j in range(4) if i != j}
top4 = set(sorted(ourv, key=ourv.get, reverse=True)[:4])
det = set(meas)
ov = top4 & det
from math import comb
p_hyp = sum(comb(4, k) * comb(8, 4 - k) for k in range(len(ov), 5)) / comb(12, 4)
print(f"our top-4 cross pairs vs atlas's detected pairs: overlap {len(ov)}/4, "
      f"hypergeometric P(>= {len(ov)}) = {p_hyp:.4f}")
print("shared:", sorted(f"{Q[i]}->{Q[j]}" for i, j in ov))
fin = lambda x: x if np.isfinite(x) else 0.0
recip_ours = max(((i, j) for i in range(4) for j in range(i + 1, 4)),
                 key=lambda p_: S[p_[0], p_[1]] + S[p_[1], p_[0]])
recip_atlas = max(((i, j) for i in range(4) for j in range(i + 1, 4)),
                  key=lambda p_: fin(F[p_[0], p_[1]]) + fin(F[p_[1], p_[0]]))
print(f"strongest reciprocal pair: ours {Q[recip_ours[0]]}<->{Q[recip_ours[1]]}, "
      f"atlas {Q[recip_atlas[0]]}<->{Q[recip_atlas[1]]}")
fv = [F[i, j] for i, j in meas]; sv = [S[i, j] for i, j in meas]
print("amplitude ranks over the 4 measured pairs: rho =",
      round(float(stats.spearmanr(fv, sv).statistic), 2),
      "(driven by AWCL->ASER; n = 4)")
pd.DataFrame([dict(test="set overlap top4 vs detected", value=len(ov), p=round(p_hyp, 4)),
              dict(test="strongest reciprocal pair agrees", value=int(recip_ours == recip_atlas), p=np.nan),
              dict(test="amplitude rank rho (n=4)",
                   value=round(float(stats.spearmanr(fv, sv).statistic), 2), p=0.60)]).to_csv(
    os.path.join(REPO_ROOT, "results/functional_agreement.csv"), index=False)

# %% [markdown]
# ## Reading
#
# * **Terminology first: the Randi atlas IS effective connectivity.** The
#   Leifer group's own follow-up (Dvali, Seguin, Betzel, Leifer,
#   arXiv:2412.14498) describes the atlas's connections as the "effective"
#   connection between pairs of neurons — causal, perturbation-derived,
#   including poly-synaptic and extrasynaptic routes — in contrast to
#   correlation-based functional connectivity. No separate Granger/DCM-style
#   whole-brain effective-connectivity dataset with neuron identity exists for
#   C. elegans; the atlas is the field's reference for exactly this quantity.
# * **With the diagonal blocked, ours and the atlas agree on the pattern.**
#   The published matrices have structural zeros on the diagonal (anatomy has
#   no self-synapses; the atlas cannot report self-activation as propagation),
#   while our largest entries are the self-terms — so a shared colour scale
#   hides our cross structure. Masked to cross terms only: our three strongest
#   cross pairs are three of the atlas's four detected pairs
#   (ASEL->ASER, ASEL->AWAL, AWAL->ASEL; hypergeometric P(>=3 of 4) = 0.067),
#   and both matrices name ASEL<->AWAL the strongest reciprocal pair. The one
#   disagreement is AWCL->ASER: the atlas's strongest quartet edge, our
#   near-floor entry — plausibly because optogenetic forcing of AWCL recruits
#   routes that natural AWCL fluctuations at one 375 ms lag do not.
# * **Amplitudes do not track (rho = -0.40 over n = 4)** — expected across a
#   forced-peak-response regime vs a passive-association regime; the
#   agreement is in WHICH pairs communicate, not in how much.
# * **Anatomy remains the outlier**: its strongest edge (ASEL->AWCL, 17
#   contacts) is undetected by the atlas and near-floor for us — consistent
#   with the atlas paper's finding that propagation deviates from anatomy.
# * **Caveats.** One lateral quartet (L cells only); the atlas measures 4 of
#   12 directed pairs; our matrix is regime- and preprocessing-specific.


# %% [markdown]
# ## Did we get lucky? The agreement under parameter sweeps
#
# If the pattern agreement with the effective atlas were an accident of our
# preprocessing choices, it should be fragile: change the high-pass window or
# the lag and it should collapse to chance (expected overlap 4·4/12 = 1.33 of
# 4). Sweep both axes and recompute the agreement statistics at each setting.
#
# Crucially, the 20 s window was selected MONTHS before this comparison, by an
# independent criterion — stimulus-present-vs-absent detection at the TPM level
# (results/window_sweep_tpm_permutation.csv: sensory z = +35 at 20 s) — never
# by atlas agreement. This sweep is a check, not the selection.

# %%
from scipy.ndimage import median_filter
from math import comb
import subprocess
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import ces_hypergraph as ch
FS = ch.SAMPLING_RATE_HZ
RECS = list(ch.HERM_DRIVE_IDS)
os.makedirs(os.path.join(REPO_ROOT, "data"), exist_ok=True)
for k_, v_ in ch.HERM_DRIVE_IDS.items():
    p_ = os.path.join(REPO_ROOT, f"data/{k_}.csv")
    if os.path.exists(p_) and os.path.getsize(p_) > 1_000_000:
        continue
    subprocess.run(["curl", "-sSL", "-o", p_,
                    f"https://drive.google.com/uc?export=download&id={v_}"], check=True)
    if os.path.getsize(p_) < 1_000_000:
        subprocess.run(["curl", "-sSL", "-o", p_,
                        f"https://drive.usercontent.google.com/download?id={v_}&export=download&confirm=t"],
                       check=True)
TR = {}
for r in RECS:
    d_ = pd.read_csv(os.path.join(REPO_ROOT, f"data/{r}.csv"))
    names = d_["neuron"].tolist()
    TR[r] = {nm: d_.iloc[names.index(nm)][9:-1].astype(float).values for nm in Q}

def sens_matrix(win_s, tau):
    Cg = np.zeros((16, 16))
    for r in RECS:
        bits = [(TR[r][nm] - median_filter(
                    np.where(np.isfinite(TR[r][nm]), TR[r][nm], np.nanmedian(TR[r][nm])),
                    size=max(3, round(win_s * FS)), mode="nearest") > 0).astype(int)
                for nm in Q]
        st = sum(b * (2 ** i) for i, b in enumerate(bits))
        for a_, b_ in zip(st[:-tau], st[tau:]):
            Cg[a_, b_] += 1
    sbn = np.zeros((16, 4))
    for s_ in range(16):
        for i in range(4):
            on = Cg[s_, [t_ for t_ in range(16) if (t_ >> i) & 1]].sum()
            sbn[s_, i] = (on + 0.5) / (Cg[s_].sum() + 1.0)
    S_ = np.zeros((4, 4))
    for j in range(4):
        pairs = [(s_, s_ | (1 << j)) for s_ in range(16) if not (s_ >> j) & 1]
        for i in range(4):
            S_[j, i] = np.mean([abs(sbn[b_, i] - sbn[a_, i]) for a_, b_ in pairs])
    return S_

DET = {(0, 1), (0, 2), (2, 0), (3, 1)}
def agree(S_):
    ourv = {(i, j): S_[i, j] for i in range(4) for j in range(4) if i != j}
    top4 = set(sorted(ourv, key=ourv.get, reverse=True)[:4])
    ov = len(top4 & DET)
    recip = max(((i, j) for i in range(4) for j in range(i + 1, 4)),
                key=lambda p_: S_[p_[0], p_[1]] + S_[p_[1], p_[0]])
    return ov, recip == (0, 2), sum(comb(4, k) * comb(8, 4 - k) for k in range(ov, 5)) / comb(12, 4)

rows = []
for win in [5, 10, 20, 30, 60, 120, 300]:
    ov, rec, p_ = agree(sens_matrix(win, 1))
    rows.append(dict(axis="window_s", value=win, overlap_top4=ov,
                     recip_is_ASEL_AWAL=rec, p_hyp=round(p_, 4)))
for tau in [1, 2, 4, 8, 16]:
    ov, rec, p_ = agree(sens_matrix(20, tau))
    rows.append(dict(axis="tau_s", value=round(tau / FS, 2), overlap_top4=ov,
                     recip_is_ASEL_AWAL=rec, p_hyp=round(p_, 4)))
sw = pd.DataFrame(rows)
sw.to_csv(os.path.join(REPO_ROOT, "results/agreement_robustness.csv"), index=False)
print(sw.to_string(index=False))

# %% [markdown]
# ### Reading
#
# * **Not luck, but not parameter-free either — the answer is "tuned by an
#   independent criterion".** The full 3/4 + reciprocal-pair agreement holds on
#   a plateau: windows of 20–30 s at τ = 1–2 samples (0.37–0.75 s). Away from
#   it the agreement degrades toward chance (1.33/4) — very fast windows (5–10
#   s) turn the bits into noise, very slow ones (60–300 s) re-freeze the tonic
#   cells; lags ≥ 1.5 s lose the interaction.
# * **The window was not chosen to match the atlas.** It was fixed by stimulus
#   detection (sensory z = +35 at 20 s, the sweep's peak region) long before
#   this comparison existed. That the same setting maximises agreement with an
#   independent optogenetic measurement is two selection criteria — one
#   internal, one external — pointing at the same operating point.
# * **The honest statement of the luck involved:** the ASEL<->AWAL reciprocal
#   identification is robust across every slow-enough window (20–300 s) and
#   fails only at τ ≥ 1.5 s or fast windows. The 3/4 top-pair overlap is the
#   more parameter-sensitive statistic. Both should be quoted with the plateau,
#   not as free-standing results.

# %% [markdown]
# ## Anatomy split by synapse type
#
# The Cook et al. edge list distinguishes chemical synapses (directed) from
# gap junctions (electrical, listed reciprocally). For the quartet:

# %%
Cq = np.zeros((4, 4)); Eq = np.zeros((4, 4))
for _, r_ in sub.iterrows():
    M = Cq if r_.Type.strip() == "chemical" else Eq
    M[idx[r_.Source], idx[r_.Target]] += float(r_.Weight)
Tq = Cq + Eq
pd.DataFrame(Cq, index=Q, columns=Q).to_csv(
    os.path.join(REPO_ROOT, "results/anatomy_chemical_quartet.csv"))
pd.DataFrame(Eq, index=Q, columns=Q).to_csv(
    os.path.join(REPO_ROOT, "results/anatomy_electrical_quartet.csv"))
print("chemical:\n", pd.DataFrame(Cq, index=Q, columns=Q).astype(int).to_string())
print("electrical:\n", pd.DataFrame(Eq, index=Q, columns=Q).astype(int).to_string())
assert np.array_equal(Tq, A), "combined must equal the matrix used in fig43"

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), constrained_layout=True)
for k, (ax, (M, title, cmap)) in enumerate(zip(axes, [
        (Cq, "a  Chemical synapses (directed)", "Blues"),
        (Eq, "b  Gap junctions (electrical, symmetric)", "Greens"),
        (Tq, "c  Combined", "Purples")])):
    V = M.astype(float)
    ax.imshow(V, cmap=cmap, aspect="equal", vmin=0)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"→{n_}" for n_ in Q], fontsize=6.4)
    ax.set_yticks(range(4)); ax.set_yticklabels(Q, fontsize=6.4)
    vmax = V.max() if V.max() > 0 else 1
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{V[i, j]:.0f}", ha="center", va="center", fontsize=6.2,
                    color="#fff" if V[i, j] > 0.55 * vmax else "#222")
    ax.set_title(title, loc="left")
    ax.set_xlabel("target", labelpad=4, fontsize=6.6)
    if k == 0:
        ax.set_ylabel("source", labelpad=4, fontsize=6.6)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig44_anatomy_by_type.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig44_anatomy_by_type.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig44")

# %% [markdown]
# ### Reading
#
# * Within the quartet the wiring is almost entirely **chemical** (30 of 32
#   contacts). Exactly one gap junction exists: **ASEL–AWCL** (1 contact each
#   way) — so the quartet's only electrical coupling sits inside its heaviest
#   chemical pathway (ASEL→AWCL, 16).
# * AWAL is purely presynaptic within the quartet (7 chemical contacts onto
#   ASEL, none incoming), and no anatomical edge touches AWAL as a target —
#   yet ASEL↔AWAL is the strongest reciprocal pair in both the effective atlas
#   and our matrix. Whatever carries it (contralateral partners via the AWCR/
#   ASER route, extrasynaptic peptidergic signalling, or common drive), it is
#   invisible to the quartet's own anatomy in BOTH synapse classes.

# %% [markdown]
# ## Figure 43 — the three connectivities, with the diagonal-matched panel

# %%
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6), constrained_layout=True)
def panel(ax, M, title, cmap, fmt, k, mask_diag=False):
    V = np.asarray(M, dtype=float).copy()
    if mask_diag:
        np.fill_diagonal(V, np.nan)
    ax.imshow(np.ma.masked_invalid(V), cmap=cmap, aspect="equal")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"→{n_}" for n_ in Q], fontsize=6)
    ax.set_yticks(range(4)); ax.set_yticklabels(Q, fontsize=6)
    vmax = np.nanmax(V) if np.nanmax(V) > 0 else 1
    for i in range(4):
        for j in range(4):
            v = V[i, j]
            if i == j and mask_diag: txt = "—"
            elif not np.isfinite(v): txt = "n.m."
            else: txt = fmt(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color="#fff" if np.isfinite(v) and not (i == j and mask_diag)
                          and v > 0.55 * vmax else "#222")
    ax.set_title(title, loc="left")
    ax.set_xlabel("target / responding", labelpad=4, fontsize=6.4)
    if k == 0:
        ax.set_ylabel("source / stimulated", labelpad=4, fontsize=6.4)

panel(axes[0], A, "a  Anatomy: synaptic contacts\n   (Cook 2019 via OpenWorm)", "Blues", lambda v: f"{v:.0f}", 0)
panel(axes[1], F, "b  Effective (signal propagation)\n   (Randi 2023; n.m. = not measured)", "Oranges", lambda v: f"{v:.3f}", 1)
panel(axes[2], S, "c  Ours, full: diagonal dominates\n   the colour scale", "Purples", lambda v: f"{v:.3f}", 2)
panel(axes[3], S, "d  Ours, diagonal blocked to match b:\n   the cross-pattern emerges", "Purples", lambda v: f"{v:.3f}", 3, mask_diag=True)
for ax in (axes[1], axes[3]):
    for (i, j) in sorted(ov):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1a9641", lw=1.6))
fig.suptitle("Blocking the self-terms (absent from published data by design) exposes the agreement: "
             "our three strongest cross pairs are three of the atlas's four detected pairs (green)",
             fontsize=8.0)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig43_connectivity_comparison.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig43_connectivity_comparison.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig43")
