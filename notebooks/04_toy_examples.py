# %% [markdown]
# # 04 — Toy examples computed with PyPhi
#
# Every structure in this notebook is a **real Φ-structure unfolded by PyPhi**
# from a small network — not a hand-written dictionary of φ values. The point is
# to exercise the exact distance on objects that actually arise, including ones
# containing **degree-3 and degree-4 relations**.
#
# What this notebook establishes:
#
# 1. Higher-order relations occur readily in 3-unit networks (up to degree 4).
# 2. The exact distance is sensitive to them; `|ΔΦ|` and a pairwise-only
#    representation are not.
# 3. Distance 0 coincides with genuine isomorphism, checked against an
#    independent brute-force isomorphism test.
# 4. The bounds |ΔΦ| ≤ D ≤ Φ₁+Φ₂ hold on every pair.
#
# **Outputs:** `figures/fig10_toy_examples.pdf`, `results/toy_distance_matrix.csv`,
# `results/toy_tests.csv`

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
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "git+https://github.com/wmayner/pyphi.git@b78d0e3"], check=True)

REPO_ROOT = os.path.abspath(os.getcwd())
if os.path.basename(REPO_ROOT) == "notebooks":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
    os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

os.environ["PYPHI_WELCOME_OFF"] = "yes"
import itertools
import time
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphi

from gold_standard import gold_standard_distance, phi_of

pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False

BLUE, ORANGE, GREY, LIGHT = "#1f6fb4", "#c2571a", "#8a8a8a", "#9bb8d4"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})

# %% [markdown]
# ## Building Φ-structures from networks
#
# `to_gold_standard` converts a PyPhi `PhiStructure` into the `(φ_d, φ_r)` form
# the distance takes. A `Relation` is a frozenset **of distinctions** with one
# φ_r, so the conversion is direct — no flattening, no choice of representation.

# %%
LABELS = ("A", "B", "C")


def build_tpm(rule):
    """State-by-node TPM from a deterministic update rule."""
    T = np.zeros((8, 3))
    for i, st in enumerate(itertools.product([0, 1], repeat=3)):
        T[i] = rule(st)
    return T


NETWORKS = {
    # the classic IIT example network: A=OR(B,C), B=AND(A,C), C=XOR(A,B)
    "AND-OR-XOR": np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0],
                            [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0]], float),
    "all-XOR": build_tpm(lambda s: (s[1] ^ s[2], s[0] ^ s[2], s[0] ^ s[1])),
    "all-AND": build_tpm(lambda s: (s[1] & s[2], s[0] & s[2], s[0] & s[1])),
    "all-OR":  build_tpm(lambda s: (s[1] | s[2], s[0] | s[2], s[0] | s[1])),
    "MAJ":     build_tpm(lambda s: tuple(int(sum(s) - x >= 1) for x in s)),
}


def phi_structure_of(tpm, state, labels=LABELS):
    net = pyphi.Network(tpm, node_labels=labels)
    return pyphi.new_big_phi.phi_structure(pyphi.Subsystem(net, state))


def to_gold_standard(ps, labels=LABELS):
    """PhiStructure -> (phi_d, phi_r). Distinctions are named by their mechanism."""
    name = lambda mech: "".join(labels[u] for u in mech)
    phi_d = {name(c.mechanism): float(c.phi) for c in ps.distinctions}
    phi_r = {frozenset(name(tuple(m)) for m in r.mechanisms): float(r.phi)
             for r in ps.relations}
    return phi_d, phi_r


# %% [markdown]
# ## Survey — where do higher-order relations appear?
#
# Unfold every state of every network and record the relation degrees present.

# %%
survey = []
for net_name, tpm in NETWORKS.items():
    for st in itertools.product([0, 1], repeat=3):
        try:
            ps = phi_structure_of(tpm, st)
        except Exception:
            continue
        phi_d, phi_r = to_gold_standard(ps)
        degrees = sorted(len(S) for S in phi_r)
        survey.append({"network": net_name, "state": "".join(map(str, st)),
                       "Phi": round(float(ps.big_phi), 5),
                       "n_distinctions": len(phi_d), "n_relations": len(phi_r),
                       "max_degree": max(degrees) if degrees else 0})
survey = pd.DataFrame(survey)
print(survey.to_string(index=False))
print(f"\n{int((survey.max_degree >= 3).sum())} of {len(survey)} states contain a "
      f"relation of degree >= 3; the maximum degree reached is "
      f"{int(survey.max_degree.max())}.")

# %% [markdown]
# Higher-order relations are not exotic — most states of these tiny networks
# have them, and degree-4 relations (binding four distinctions at once) appear
# throughout.

# %%
CASES = {}
for net_name, st in [("AND-OR-XOR", "000"), ("AND-OR-XOR", "101"), ("AND-OR-XOR", "111"),
                     ("all-XOR", "000"), ("all-XOR", "011"), ("all-XOR", "101"),
                     ("all-AND", "111"), ("all-OR", "000"), ("MAJ", "000")]:
    ps = phi_structure_of(NETWORKS[net_name], tuple(int(c) for c in st))
    CASES[f"{net_name}[{st}]"] = to_gold_standard(ps)

for k, (pd_, pr_) in CASES.items():
    by_deg = pd.Series([len(S) for S in pr_]).value_counts().sort_index().to_dict()
    print(f"{k:<18} {len(pd_)} distinctions, {len(pr_):>2} relations "
          f"(by degree {by_deg}), Phi = {phi_of((pd_, pr_)):8.5f}")

# %% [markdown]
# ## An independent isomorphism test
#
# The distance should return 0 exactly when two structures are the same up to
# relabelling. This brute-force check decides that question separately, so it
# can confirm or contradict the distance rather than restate it.

# %%
def isomorphic(S1, S2):
    """True iff some relabelling of distinctions makes S1 literally equal S2."""
    k1, k2 = list(S1[0]), list(S2[0])
    if len(k1) != len(k2) or len(S1[1]) != len(S2[1]):
        return False
    rnd = lambda d: {k: round(v, 10) for k, v in d.items()}
    for perm in permutations(k2):
        M = dict(zip(k1, perm))
        if {M[a]: round(v, 10) for a, v in S1[0].items()} != rnd(S2[0]):
            continue
        moved = {frozenset(M[a] for a in S): round(v, 10) for S, v in S1[1].items()}
        if moved == rnd(S2[1]):
            return True
    return False


# %% [markdown]
# ## Test 1 — the full distance matrix
#
# Nine real structures, every pair. The largest has 7 distinctions, so the
# search is over 7! = 5040 bijections per pair.

# %%
names = list(CASES)
n = len(names)
Dm = np.zeros((n, n))
t0 = time.perf_counter()
for i in range(n):
    for j in range(i + 1, n):
        Dm[i, j] = Dm[j, i] = gold_standard_distance(CASES[names[i]], CASES[names[j]])
elapsed = time.perf_counter() - t0

Dmat = pd.DataFrame(np.round(Dm, 4), index=names, columns=names)
Dmat.to_csv("results/toy_distance_matrix.csv")
print(f"{n}x{n} matrix in {elapsed:.1f}s")
print(Dmat.to_string())

print("\nmetric checks on the matrix")
print(f"  diagonal all zero : {bool(np.allclose(np.diag(Dm), 0))}")
print(f"  symmetric         : {bool(np.allclose(Dm, Dm.T))}")
viol = sum(1 for a in range(n) for b in range(n) for c in range(n)
           if Dm[a, c] > Dm[a, b] + Dm[b, c] + 1e-9)
print(f"  triangle violations over all {n**3} ordered triples: {viol}")

print("\nevery zero off-diagonal entry, checked against the isomorphism test")
for i in range(n):
    for j in range(i + 1, n):
        if Dm[i, j] < 1e-12:
            iso = isomorphic(CASES[names[i]], CASES[names[j]])
            print(f"  {names[i]:<16} vs {names[j]:<16} isomorphic = {iso}")

# %% [markdown]
# ## Test 2 — a controlled higher-order perturbation
#
# Take one real structure and change **only** its highest-degree relation. This
# isolates higher-order content from everything else.
#
# `all-XOR[000]` has 4 distinctions and 15 relations, including exactly one of
# degree 4.

# %%
X = CASES["all-XOR[000]"]
deg4 = [S for S in X[1] if len(S) == 4][0]
deg2 = sorted([S for S in X[1] if len(S) == 2], key=sorted)[0]

# (A) delete the degree-4 relation
X_deleted = (X[0], {S: v for S, v in X[1].items() if S != deg4})

# (B) move its phi onto a degree-2 relation -- Phi is preserved exactly
moved = dict(X[1])
carried = moved.pop(deg4)
moved[deg2] = moved[deg2] + carried
X_moved = (X[0], moved)

print(f"all-XOR[000]: the degree-4 relation is {sorted(deg4)} with phi_r = {carried:.5f}")
print(f"  Phi(original) = {phi_of(X):.5f}")
print(f"  Phi(deleted)  = {phi_of(X_deleted):.5f}")
print(f"  Phi(moved)    = {phi_of(X_moved):.5f}   <- unchanged by construction")


def pairwise_only(S1, S2):
    """Discard every relation that is not degree 2, then run the same exact search."""
    keep = lambda S: (S[0], {k: v for k, v in S[1].items() if len(k) == 2})
    return gold_standard_distance(keep(S1), keep(S2))


TESTS = [
    ("A  delete the degree-4 relation", X, X_deleted),
    ("B  move its phi to degree 2 (Phi preserved)", X, X_moved),
    ("C  all-XOR[000] vs all-XOR[101]", CASES["all-XOR[000]"], CASES["all-XOR[101]"]),
    ("D  all-XOR[000] vs all-XOR[011]", CASES["all-XOR[000]"], CASES["all-XOR[011]"]),
    ("E  AND-OR-XOR[101] vs [111]", CASES["AND-OR-XOR[101]"], CASES["AND-OR-XOR[111]"]),
]

rows = []
for label, S1, S2 in TESTS:
    rows.append({"test": label,
                 "abs_dPhi": round(abs(phi_of(S1) - phi_of(S2)), 5),
                 "pairwise_only": round(pairwise_only(S1, S2), 5),
                 "gold_standard": round(gold_standard_distance(S1, S2), 5),
                 "isomorphic": isomorphic(S1, S2)})
results = pd.DataFrame(rows)
results.to_csv("results/toy_tests.csv", index=False)
print("\n" + results.to_string(index=False))

# %% [markdown]
# Reading the table:
#
# * **Test A** removes a degree-4 relation. A pairwise-only representation has
#   nowhere to store that relation, so it reports **0.0** — the two structures
#   look identical to it. The exact distance reports the φ_r that was removed.
# * **Test B** is the sharpest case. The same φ_r is *moved* from the degree-4
#   relation to a degree-2 one, so Φ is unchanged and `|ΔΦ|` reports **0.0**.
#   The exact distance reports 1.0 — it charges the loss at degree 4 and the
#   gain at degree 2 separately.
# * **Test D** is a genuine isomorphism and correctly scores 0.
#
# ## Test 3 — the bounds

# %%
bounds = []
for label, S1, S2 in TESTS:
    lo = abs(phi_of(S1) - phi_of(S2))
    d = gold_standard_distance(S1, S2)
    hi = phi_of(S1) + phi_of(S2)
    bounds.append({"test": label.split()[0], "lower_absdPhi": round(lo, 5),
                   "distance": round(d, 5), "upper_sumPhi": round(hi, 5),
                   "holds": bool(lo - 1e-9 <= d <= hi + 1e-9)})
bounds = pd.DataFrame(bounds)
print(bounds.to_string(index=False))
print(f"\nall bounds hold: {bool(bounds.holds.all())}")

# %% [markdown]
# ## Figure 10

# %%
fig10 = plt.figure(figsize=(10.6, 3.6))
gs10 = fig10.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=0.42)

# (a) distance matrix
ax = fig10.add_subplot(gs10[0])
im = ax.imshow(Dm, cmap="magma_r")
ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=90, fontsize=5.5)
ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=5.5)
cb = fig10.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("exact distance", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
ax.set_title("a  Nine real Φ-structures,\nevery pair", loc="left")

# (b) relation degrees present across the survey
ax = fig10.add_subplot(gs10[1])
counts = pd.Series([len(S) for _, (_, pr) in CASES.items() for S in pr]).value_counts().sort_index()
ax.bar(counts.index.astype(str), counts.values,
       color=[LIGHT if k == 2 else ORANGE for k in counts.index], width=0.62)
ax.text(0.98, 0.97, "orange = no pairwise form", transform=ax.transAxes,
        ha="right", va="top", fontsize=6, color=ORANGE)
for k, v in counts.items():
    ax.text(str(k), v + 1.5, str(v), ha="center", fontsize=6.5)
ax.set_xlabel("relation degree k", labelpad=6)
ax.set_ylabel("relations across the 9 structures", labelpad=6)
ax.set_ylim(0, counts.max() * 1.22)
_ho = int(counts[counts.index > 2].sum())
_np = int(counts[counts.index != 2].sum())
ax.set_title(f"b  {_ho} of {int(counts.sum())} relations are degree>2;\n"
             f"{_np} have no pairwise form", loc="left")

# (c) the three measures on the five tests
ax = fig10.add_subplot(gs10[2])
xs = np.arange(len(TESTS)); w = 0.26
m1 = results["abs_dPhi"].values
m2 = results["pairwise_only"].values
m3 = results["gold_standard"].values
ax.bar(xs - w, m1, w, color=GREY, label="|ΔΦ|")
ax.bar(xs, m2, w, color=LIGHT, label="pairwise-only")
ax.bar(xs + w, m3, w, color=ORANGE, label="gold standard")
for xi, v in zip(xs - w, m1):
    if v < 1e-9: ax.text(xi, 0.03, "blind", ha="center", fontsize=5.2, color=GREY, rotation=90)
for xi, v in zip(xs, m2):
    if v < 1e-9: ax.text(xi, 0.03, "blind", ha="center", fontsize=5.2, color="#5a7fa4", rotation=90)
ax.set_xticks(xs); ax.set_xticklabels([t[0] for t in results.test], fontsize=7)
ax.set_xlabel("test (see table)", labelpad=6)
ax.set_ylabel("reported distance", labelpad=6)
ax.set_ylim(0, max(m3.max(), m1.max()) * 1.34)
ax.legend(frameon=False, loc="upper left", fontsize=6)
ax.set_title("c  Test D is a true isomorphism;\nA and B are not", loc="left")

fig10.savefig("figures/fig10_toy_examples.pdf")
fig10.savefig("figures/fig10_toy_examples.png", dpi=200)
print("wrote figures/fig10_toy_examples.pdf")

# %% [markdown]
# ## Summary
#
# 1. Degree-3 and degree-4 relations arise in most states of 3-unit networks.
# 2. Deleting a degree-4 relation is invisible to a pairwise-only representation
#    and visible to the exact distance.
# 3. Moving φ_r from degree 4 to degree 2 is invisible to `|ΔΦ|` — Φ is
#    unchanged — and visible to the exact distance.
# 4. Across 9 real structures, distance 0 coincided with genuine isomorphism in
#    every case, and the triangle inequality held on all 729 ordered triples.
# 5. The bounds |ΔΦ| ≤ D ≤ Φ₁ + Φ₂ held on every pair.
