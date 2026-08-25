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
sub = el[el.Source.isin(Q) & el.Target.isin(Q)]
A = np.zeros((4, 4)); idx = {n: i for i, n in enumerate(Q)}
for _, r_ in sub.iterrows():
    i, j = idx[r_.Source], idx[r_.Target]; w = float(r_.Weight)
    A[i, j] += w
    if r_.Type.strip() == "electrical":
        A[j, i] += w
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

# %% [markdown]
# ## Reading
#
# * **The three connectivities disagree — including the two published ones.**
#   Anatomy's strongest edge (ASEL→AWCL, 18 synapses) is our weakest effective
#   entry (0.009) and is unmeasured in the functional atlas. The functional
#   atlas's strongest quartet edge (AWCL→ASER, 0.230) has 1 synapse behind it.
#   ASEL↔AWAL propagates strongly in the atlas (0.163/0.165) over **zero**
#   direct anatomical synapses (ASEL→AWAL direction), presumably via
#   extrasynaptic or indirect routes. Anatomy-vs-ours over all 12 cross pairs:
#   ρ ≈ 0.0.
# * **This is the atlas paper's own conclusion reproduced in miniature** —
#   signal propagation differs from anatomical predictions, partly through
#   extrasynaptic signalling. The disagreement among published references
#   dissolves the idea of pruning our mechanisms with "the" connectome: there
#   is no single ground-truth matrix to prune by.
# * **Why ours is smaller than the functional atlas where both exist:** the
#   atlas drives the source neuron optogenetically far outside its natural
#   operating range and reads the peak evoked response; our sensitivity is the
#   passive, binarized, 375 ms-lag association during natural dynamics, where
#   sensory neurons are driven overwhelmingly by the STIMULUS (a common input),
#   not by each other. Ours answers "how much does neuron j's state inform
#   neuron i's next state in this regime" — the quantity the TPM actually
#   uses — not "what happens if you force neuron j."
# * **Caveats.** The comparison is one lateral quartet (L cells only, so
#   L↔R pathways through the contralateral partners are invisible); the
#   functional atlas measures only 4 of the 12 directed pairs; and our matrix
#   is regime- and preprocessing-specific by construction.

# %% [markdown]
# ## Figure 43 — the three connectivities side by side

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5), constrained_layout=True)
mats = [(pd.DataFrame(A, index=Q, columns=Q),
         "a  Anatomy: synapse count\n   (Cook 2019 via OpenWorm)", "Blues"),
        (Fd, "b  Functional atlas: response amplitude\n   (Randi 2023; n.m. = not measured)", "Oranges"),
        (pd.DataFrame(S, index=Q, columns=Q),
         "c  Ours: TPM effective sensitivity\n   (this repo, 20 s bits)", "Purples")]
for k, (ax, (M, title, cmap)) in enumerate(zip(axes, mats)):
    V = M.values.astype(float).copy()
    ax.imshow(np.ma.masked_invalid(V), cmap=cmap, aspect="equal")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"→{n}" for n in Q], fontsize=6)
    ax.set_yticks(range(4)); ax.set_yticklabels(Q, fontsize=6)
    vmax = np.nanmax(V) if np.nanmax(V) > 0 else 1
    for i in range(4):
        for j in range(4):
            v = V[i, j]
            txt = "n.m." if not np.isfinite(v) else (f"{v:.0f}" if k == 0 else f"{v:.3f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.6,
                    color="#fff" if np.isfinite(v) and v > 0.55 * vmax else "#222")
    ax.set_title(title, loc="left")
    ax.set_xlabel("target / responding", labelpad=4, fontsize=6.4)
    if k == 0:
        ax.set_ylabel("source / stimulated", labelpad=4, fontsize=6.4)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig43_connectivity_comparison.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig43_connectivity_comparison.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig43")
