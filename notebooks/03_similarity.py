# %% [markdown]
# # 03 — Comparing Φ-structures: why the current measures fail
#
# **What this notebook does:** tests the three similarity measures used so far
# against cases where the right answer is known, and shows exactly where each
# one breaks. It ends with a prototype distance that is sensitive to
# higher-order structure — offered as a starting point for discussion, not a
# finished metric.
#
# **The three measures under test**
#
# | # | Measure | Where | Representation |
# |---|---|---|---|
# | 1 | \|ΔΦ\| | notebook 6 | a single scalar |
# | 2 | brute-force bijection | notebooks 1, 7 | distinctions + **pairwise** relations |
# | 3 | **gold standard** | this repo | distinctions + relations at **every degree** |
#
# **Outputs:** `figures/fig05_measure_failures.pdf`,
# `figures/fig06_scaling.pdf`, `results/measure_comparison.csv`

# %%
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("iit4-celegans-phi-structure"):
        subprocess.run(
            ["git", "clone", "--quiet",
             "https://github.com/maierav/iit4-celegans-phi-structure.git"],
            check=True,
        )
    os.chdir("iit4-celegans-phi-structure")

REPO_ROOT = os.path.abspath(os.getcwd())
if os.path.basename(REPO_ROOT) == "notebooks":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
    os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

import json
import math
import time

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import ces_hypergraph as ch

BLUE, ORANGE, GREY, GREEN = "#1f6fb4", "#c2571a", "#8a8a8a", "#2a7a2a"
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "pdf.fonttype": 42, "ps.fonttype": 42,
})

# %% [markdown]
# ## Test 1 — Φ alone cannot distinguish structures
#
# Two Φ-structures are built with **identical Φ** but genuinely different
# content: A spreads its φ over two first-order distinctions, B over three
# distinctions including a second-order mechanism. Any measure defined as |ΔΦ|
# calls them identical.
#
# This is not a contrived worry: the original notebook 7 produced two
# 5-node structures with *exactly* equal Φ (2.1779535637765157) and 10
# distinctions each, differing only in which units carried them.
#
# Note that the pairwise-only measure *can* see this particular difference — the distinction
# count differs, which its zero-padding registers. Test 2 is the case it cannot
# see.

# %%
def make_structure(node_spec, face_spec):
    """Build a (nodes, faces) hypergraph by hand for testing."""
    nodes = {(tuple(m), tuple(c), tuple(e)): phi for m, c, e, phi in node_spec}
    faces = [
        {"degree": len(relata), "purview": tuple(pv), "phi": phi,
         "relata": tuple(sorted((tuple(m), d) for m, d in relata))}
        for relata, pv, phi in face_spec
    ]
    return nodes, faces


# Structure A: two first-order distinctions.
# Structure B: same total Phi, but spread over THREE distinctions, one of which
# is second-order (mechanism {0,1}). This is a genuine structural difference,
# not a relabeling of A.
A = make_structure(
    [((0,), (0,), (0,), 0.30), ((1,), (1,), (1,), 0.20)],
    [([((0,), "CAUSE"), ((1,), "CAUSE")], (0,), 0.10)],
)
B = make_structure(
    [((0,), (0,), (0,), 0.15), ((1,), (1,), (1,), 0.15),
     ((0, 1), (0, 1), (0, 1), 0.20)],
    [([((0,), "CAUSE"), ((1,), "CAUSE")], (0,), 0.10)],
)

phi_A = sum(A[0].values()) + sum(f["phi"] for f in A[1])
phi_B = sum(B[0].values()) + sum(f["phi"] for f in B[1])

print(f"Phi(A) = {phi_A:.4f}   Phi(B) = {phi_B:.4f}")
print(f"  |dPhi|                            = {abs(phi_A - phi_B):.4f}   <- calls them IDENTICAL")

pdA, prA, _ = ch.pairwise_projection(*A)
pdB, prB, _ = ch.pairwise_projection(*B)
d_pair_ab, _ = ch.ces_distance_pairwise(pdA, prA, pdB, prB)
print(f"  pairwise-only                     = {d_pair_ab:.4f}")

d_hyp, d_term, r_term = ch.ces_distance_hypergraph(A, B)
print(f"  earlier prototype (kept for ref)  = {d_hyp:.4f}   "
      f"(distinctions {d_term:.4f} + relations {r_term:.4f})")
print("  the exact gold standard is scored for all three tests in Figure 5 below")

# %% [markdown]
# ## Test 2 — the pairwise representation cannot see higher-order faces
#
# Now two structures that are **identical except for one degree-3 face**. A
# degree-3 face joins three distinctions at once; it has no pairwise equivalent.
# A representation keyed by ordered pairs has nowhere to put it.

# %%
base_nodes = [((0,), (0,), (0,), 0.30), ((1,), (1,), (1,), 0.30),
              ((0, 1), (0,), (1,), 0.10)]
shared_face = ([((0,), "CAUSE"), ((1,), "CAUSE")], (0,), 0.10)
extra_face = ([((0,), "CAUSE"), ((1,), "CAUSE"), ((0, 1), "EFFECT")], (0,), 0.10)

X = make_structure(base_nodes, [shared_face])
Y = make_structure(base_nodes, [shared_face, extra_face])

print("X and Y differ by exactly ONE degree-3 face.\n")
print("degree distribution  X:", ch.face_degree_distribution(X[1]),
      " Y:", ch.face_degree_distribution(Y[1]))

# |dPhi|
phi_X = sum(X[0].values()) + sum(f["phi"] for f in X[1])
phi_Y = sum(Y[0].values()) + sum(f["phi"] for f in Y[1])
print(f"\n  |dPhi|                            = {abs(phi_X - phi_Y):.4f}")

# pairwise-only: discard non-degree-2 relations, then brute-force bijection
pdX, prX, dropX = ch.pairwise_projection(*X)
pdY, prY, dropY = ch.pairwise_projection(*Y)
print(f"  pairwise projection: X keeps {len(prX)} edges (drops {dropX}), "
      f"Y keeps {len(prY)} edges (drops {dropY})")
d_pair, _ = ch.ces_distance_pairwise(pdX, prX, pdY, prY)
print(f"  pairwise-only                     = {d_pair:.4f}   <- BLIND to the difference")

d_hyp2, d_t2, r_t2 = ch.ces_distance_hypergraph(X, Y)
print(f"  earlier prototype (kept for ref)  = {d_hyp2:.4f}   <- detects it")

# %% [markdown]
# Note that |dPhi| does register a difference here only because the extra
# face adds φ_r mass. Make the comparison φ-preserving — move the same φ from a
# pairwise relation into a higher-order one — and |dPhi| goes blind too.

# %%
Z = make_structure(
    base_nodes,
    [([((0,), "CAUSE"), ((1,), "CAUSE"), ((0, 1), "EFFECT")], (0,), 0.10)],
)
phi_Z = sum(Z[0].values()) + sum(f["phi"] for f in Z[1])
pdZ, prZ, dropZ = ch.pairwise_projection(*Z)

print("X: one degree-2 face   Z: one degree-3 face   (same total phi)")
print(f"  Phi(X) = {phi_X:.4f}   Phi(Z) = {phi_Z:.4f}")
print(f"  |dPhi|                            = {abs(phi_X - phi_Z):.4f}   <- blind")
print(f"  pairwise: X {len(prX)} edges, Z {len(prZ)} edges (Z dropped {dropZ})")
d_pair_xz, _ = ch.ces_distance_pairwise(pdX, prX, pdZ, prZ)
print(f"  pairwise-only                     = {d_pair_xz:.4f}   <- sees only the missing edge")
d_hyp_xz, _, _ = ch.ces_distance_hypergraph(X, Z)
print(f"  earlier prototype (kept for ref)  = {d_hyp_xz:.4f}   <- sees the degree change")

# %% [markdown]
# ## Test 3 — real data
#
# The same comparison on the actual *C. elegans* Φ-structure from notebook 02.

# %%
PRIMARY = "20220327_herm_2"
hg_path = f"results/ces_hypergraph_{PRIMARY}.json"

if os.path.exists(hg_path):
    with open(hg_path) as fh:
        payload = json.load(fh)
    real_nodes = {
        (tuple(n["mechanism"]), tuple(n["cause_purview"]), tuple(n["effect_purview"])): n["phi"]
        for n in payload["nodes"]
    }
    real_faces = [
        {"degree": f["degree"], "purview": tuple(f["purview"]), "phi": f["phi"],
         "relata": tuple(sorted((tuple(m), d) for m, d in f["relata"]))}
        for f in payload["faces"]
    ]
    real = (real_nodes, real_faces)
    # Report at the RELATION level: faces are an internal enumeration over
    # cause/effect sides and all inherit their parent relation's phi.
    _rels = {}
    for f in real_faces:
        _rels.setdefault(frozenset(m for m, _d in f["relata"]), set()).add(round(f["phi"], 12))
    for _d, _p in _rels.items():
        assert len(_p) == 1, f"faces disagree on phi within one relation: {_d}"
    _rels = {d: max(p) for d, p in _rels.items()}
    _n_pair = sum(1 for d in _rels if len(d) == 2)
    _phi_r = sum(_rels.values())
    _phi_lost = sum(v for d, v in _rels.items() if len(d) != 2)
    print(f"real Phi-structure ({PRIMARY}, state {payload['state']}): "
          f"Phi = {payload['big_phi']:.5f}")
    print(f"  {len(real_nodes)} distinctions, {len(_rels)} relations "
          f"(degrees {sorted(len(d) for d in _rels)})")
    print(f"  no pairwise form: {len(_rels) - _n_pair} of {len(_rels)} relations, "
          f"carrying {_phi_lost:.5f} of {_phi_r:.5f} phi_r "
          f"({100 * _phi_lost / _phi_r:.0f}%)")
else:
    real = None
    print("run notebook 02 first to generate", hg_path)

# %%
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(REPO_ROOT, "src"))
from gold_standard import gold_standard_distance
from matplotlib.ticker import MaxNLocator
from collections import Counter
from scipy.optimize import linear_sum_assignment
from itertools import combinations as _combos


def _rand_struct(nd, seed, maxdeg=3, density=0.5):
    """Random Phi-structure for timing and validation."""
    r = np.random.default_rng(seed)
    labs = [f"d{i}" for i in range(nd)]
    phi_d = {l: round(float(r.uniform(0.05, 0.5)), 3) for l in labs}
    phi_r = {}
    for k in range(1, min(nd, maxdeg) + 1):
        for S in _combos(labs, k):
            if r.random() < density:
                phi_r[frozenset(S)] = round(float(r.uniform(0.02, 0.3)), 3)
    return (phi_d, phi_r)

# %% [markdown]
# ## Test 4 — how the exact distance scales
#
# The search is over bijections between **distinctions**: n! where
# n = max(N_dist1, N_dist2). It is not over 2^n mechanisms and not over
# relations — relations follow once the distinction mapping is fixed.
#
# A natural question is whether the factorial search can be replaced by optimal
# assignment (Hungarian, O(n^3)). It cannot: assignment is exact only for a cost
# that is **linear** in the pairing, and the relation term is not. Relation
# {a, b} is scored against {M(a), M(b)}, so its contribution depends on *two*
# assignment decisions at once. The cell below measures how often that matters.

# %%
rows = []
for n in [3, 4, 6, 8, 9, 10, 12]:
    rows.append({"n_distinctions": n, "permutations": math.factorial(n)})
scaling = pd.DataFrame(rows)

# measured runtime of the exact distance
timings = []
for n in [3, 4, 6, 8, 9]:
    A_n, B_n = _rand_struct(n, 11), _rand_struct(n, 22)
    t0 = time.perf_counter()
    gold_standard_distance(A_n, B_n)
    timings.append({"n_distinctions": n, "seconds": round(time.perf_counter() - t0, 4)})
timing_df = pd.DataFrame(timings)
print("measured runtime of the exact distance:")
print(timing_df.to_string(index=False))

# does Hungarian on the distinction term alone reproduce the exact distance?
def _assignment_only(S1, S2):
    """Hungarian on the DISTINCTION term, then score relations under that mapping."""
    d1, r1 = S1
    d2, r2 = S2
    k1, k2 = list(d1), list(d2)
    n = max(len(k1), len(k2))
    p1 = k1 + [f"\0n{i}" for i in range(n - len(k1))]
    p2 = k2 + [f"\0m{i}" for i in range(n - len(k2))]
    C = np.array([[abs(d1.get(a, 0.0) - d2.get(b, 0.0)) for b in p2] for a in p1])
    ri, ci = linear_sum_assignment(C)
    M = {p1[i]: p2[ci[list(ri).index(i)]] for i in range(n)}
    cost = sum(abs(d1.get(a, 0.0) - d2.get(b, 0.0)) for a, b in M.items())
    for S, v in r1.items():
        cost += abs(v - r2.get(frozenset(M[a] for a in S), 0.0))
    inv = {v: k for k, v in M.items()}
    for T, v in r2.items():
        if frozenset(inv[b] for b in T) not in r1:
            cost += v
    return cost


_rng = np.random.default_rng(3)
n_worse, excess = 0, []
for _s in range(400):
    A_, B_ = _rand_struct(int(_rng.integers(2, 5)), _s), _rand_struct(int(_rng.integers(2, 5)), _s + 3000)
    ex, ap = gold_standard_distance(A_, B_), _assignment_only(A_, B_)
    if ap > ex + 1e-12:
        n_worse += 1
        excess.append(ap - ex)
print(f"\nHungarian on the distinction term alone, 400 random pairs:")
print(f"  larger than the exact distance in {n_worse} cases ({100*n_worse/400:.0f}%)")
print(f"  mean excess when wrong: {np.mean(excess):.4f}, max {np.max(excess):.4f}")
print("  -> assignment is NOT a valid substitute; the relation term couples the pairs")

# %% [markdown]
# ## Figure 5 — where each measure fails

# %%
# Rebuild the three tests at the RELATION level and score them with the exact
# gold standard alongside the two superseded measures.
def _phi(S):
    return sum(S[0].values()) + sum(S[1].values())


def _pairwise_only(S1, S2):
    """Measure 2: discard everything that is not a degree-2 relation, then run
    the same exact bijection search on what is left."""
    keep = lambda S: (S[0], {k: v for k, v in S[1].items() if len(k) == 2})
    return gold_standard_distance(keep(S1), keep(S2))


# T1 same Phi, different content | T2 one extra degree-3 relation
# T3 a degree-2 relation replaced by a degree-3 one, Phi preserved
_A = ({"a": 0.30, "b": 0.20}, {frozenset({"a", "b"}): 0.10})
_B = ({"a": 0.15, "b": 0.15, "c": 0.20}, {frozenset({"a", "b"}): 0.10})
_base = {"a": 0.30, "b": 0.30, "c": 0.10}
_X = (_base, {frozenset({"a", "b"}): 0.10})
_Y = (_base, {frozenset({"a", "b"}): 0.10, frozenset({"a", "b", "c"}): 0.10})
_Z = (_base, {frozenset({"a", "b", "c"}): 0.10})

TESTS = [("same Φ,\ndifferent content", _A, _B),
         ("one extra\ndegree-3 relation", _X, _Y),
         ("degree 2 → 3,\nΦ preserved", _X, _Z)]

m1, m2, m3, table = [], [], [], []
for _name, _S1, _S2 in TESTS:
    a_, b_, c_ = (abs(_phi(_S1) - _phi(_S2)),
                  _pairwise_only(_S1, _S2),
                  gold_standard_distance(_S1, _S2))
    m1.append(a_); m2.append(b_); m3.append(c_)
    table.append({"test": _name.replace("\n", " "), "measure_1_absdPhi": round(a_, 5),
                  "measure_2_pairwise_only": round(b_, 5),
                  "measure_3_gold_standard": round(c_, 5)})
pd.DataFrame(table).to_csv("results/measure_comparison.csv", index=False)
print(pd.DataFrame(table).to_string(index=False))

# Relation-level facts about the real structure (recomputed, not hardcoded).
REL_REAL, PHI_R_BY_K = None, None
if real is not None:
    _rels = {}
    for f in real_faces:
        dset = frozenset(mech for mech, _dir in f["relata"])
        _rels.setdefault(dset, set()).add(round(float(f["phi"]), 12))
    for _d, _p in _rels.items():
        assert len(_p) == 1, f"faces disagree on phi within a relation: {_d}"
    REL_REAL = {d: max(p) for d, p in _rels.items()}
    PHI_R_BY_K = Counter()
    for d, v in REL_REAL.items():
        PHI_R_BY_K[len(d)] += v

fig5, axes5 = plt.subplots(1, 3, figsize=(10.6, 3.5))
LIGHT = "#9bb8d4"

# (a) three tests x three measures
ax = axes5[0]
x = np.arange(3)
w = 0.26
ax.bar(x - w, m1, w, color=GREY, label="|ΔΦ| (scalar)")
ax.bar(x, m2, w, color=LIGHT, label="pairwise-only")
ax.bar(x + w, m3, w, color=ORANGE, label="gold standard")
for xi, v in zip(x - w, m1):
    if v < 1e-9:
        ax.text(xi, 0.006, "blind", ha="center", fontsize=5.5, color=GREY, rotation=90)
for xi, v in zip(x, m2):
    if v < 1e-9:
        ax.text(xi, 0.006, "blind", ha="center", fontsize=5.5, color="#5a7fa4", rotation=90)
ax.set_xticks(x)
ax.set_xticklabels([t[0] for t in TESTS], fontsize=6)
ax.set_ylabel("reported distance", labelpad=7)
ax.set_ylim(0, max(m3) * 1.42)
ax.legend(frameon=False, loc="upper right", fontsize=6)
ax.set_title("a  Only the gold standard is\nnon-zero on all three tests")

# (b) real structure, counted in RELATIONS
ax = axes5[1]
if REL_REAL is not None:
    n_rel = len(REL_REAL)
    n_pair = sum(1 for d in REL_REAL if len(d) == 2)
    ax.bar([0, 1], [n_pair, n_rel - n_pair], color=[LIGHT, ORANGE], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["representable\n(degree 2)", "unrepresentable\n(k=1 or k>2)"])
    for i, v in enumerate([n_pair, n_rel - n_pair]):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=7)
    ax.set_ylabel("relations in the real Φ-structure", labelpad=7)
    ax.set_ylim(0, max(n_pair, n_rel - n_pair) * 1.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"b  Real worm structure:\n{n_rel - n_pair} of {n_rel} relations "
                 "have no pairwise form")
else:
    ax.set_axis_off()

# (c) phi_r mass by relation degree
ax = axes5[2]
if PHI_R_BY_K is not None:
    ks = sorted(PHI_R_BY_K)
    ax.bar(range(len(ks)), [PHI_R_BY_K[k] for k in ks],
           color=[LIGHT if k == 2 else ORANGE for k in ks], width=0.6)
    for i, k in enumerate(ks):
        ax.text(i, PHI_R_BY_K[k] + 0.006, f"{PHI_R_BY_K[k]:.3f}",
                ha="center", fontsize=6.5)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"{k}\n{'self' if k == 1 else ('pairwise' if k == 2 else 'higher-order')}"
                        for k in ks])
    ax.set_xlabel("relation degree k", labelpad=7)
    ax.set_ylabel("summed φ$_r$", labelpad=7)
    ax.set_ylim(0, max(PHI_R_BY_K.values()) * 1.22)
    _share = 100 * PHI_R_BY_K[1] / sum(PHI_R_BY_K.values())
    ax.set_title(f"c  The self-relation alone carries\n{_share:.0f}% of the φ$_r$ mass")
else:
    ax.set_axis_off()

fig5.tight_layout()
fig5.savefig("figures/fig05_measure_failures.pdf", bbox_inches="tight")
fig5.savefig("figures/fig05_measure_failures.png", dpi=200, bbox_inches="tight")
print("wrote figures/fig05_measure_failures.pdf")

# %% [markdown]
# ## Figure 6 — how the exact distance scales

# %%
fig6, axes6 = plt.subplots(1, 2, figsize=(8.8, 3.3))

ax = axes6[0]
ns = scaling["n_distinctions"].values
ax.semilogy(ns, scaling["permutations"], "o-", color=ORANGE, lw=1.4,
            markersize=4.5, label="bijections searched: n!")
ax.set_xlabel("number of distinctions n", labelpad=7)
ax.set_ylabel("bijections", labelpad=7)
ax.legend(frameon=False, loc="upper left")
ax.set_title("a  The search is factorial in n")
ax.annotate("real worm\nstructures (n=3)", xy=(3.05, math.factorial(3)),
            xytext=(3.55, 3.0e2), fontsize=6, color="#444", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#888", shrinkA=1, shrinkB=2))
ax.set_ylim(3, 2e9)

ax = axes6[1]
ax.plot(timing_df["n_distinctions"], timing_df["seconds"], "o-",
        color=ORANGE, lw=1.4, markersize=4.5)
for _, r_ in timing_df.iterrows():
    _lbl = "<0.001s" if r_["seconds"] < 1e-3 else f"{r_['seconds']:.3g}s"
    ax.annotate(_lbl, xy=(r_["n_distinctions"], r_["seconds"]),
                xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6)
ax.set_xlabel("number of distinctions n", labelpad=7)
ax.set_ylabel("seconds (one distance)", labelpad=7)
ax.set_ylim(bottom=0)
ax.set_title("b  Measured runtime of the exact distance")
ax.margins(0.10, 0.18)

fig6.tight_layout()
fig6.savefig("figures/fig06_scaling.pdf", bbox_inches="tight")
fig6.savefig("figures/fig06_scaling.png", dpi=200, bbox_inches="tight")
print("wrote figures/fig06_scaling.pdf")

# %% [markdown]
# ## What the prototype does, and what it does not
#
# `ces_distance_hypergraph` works in two terms:
#
# 1. **Distinctions** are matched by optimal assignment (Hungarian, O(n³)),
#    with a penalty for pairing distinctions whose mechanism order or purview
#    size differ. This replaces the factorial search and is exact for a linear
#    cost.
# 2. **Relation faces** are grouped by `(degree, purview size)` and compared
#    within group as sorted φ spectra, weighted by degree. Higher-order faces
#    are therefore compared as higher-order objects.
#
# **Known limitations — these are the open questions for discussion:**
#
# * The bucketed spectrum comparison is permutation-invariant *within* a bucket,
#   so it ignores *which* distinctions a face joins. Two structures whose
#   degree-3 faces connect entirely different triples score as identical.
# * The degree weighting (`w(k) = k`) and the structural penalty (0.5 per unit
#   of mismatch) are free parameters with no principled justification yet.
# * It is not proven to be a metric — the triangle inequality has not been
#   checked.
# * It is not invariant to relabeling units, which matters when comparing
#   structures across animals.
#
# ### Candidate directions
#
# * **Simplicial / topological:** treat the CES as a filtered simplicial complex
#   (faces are already simplices) and compare persistence diagrams. Naturally
#   handles all degrees and is relabeling-invariant.
# * **Optimal transport on the face poset:** Gromov–Wasserstein between
#   hypergraphs, using φ as mass. Respects which distinctions a face joins.
# * **Hypergraph kernels:** e.g. a Weisfeiler–Leman-style refinement on the
#   incidence structure, giving a positive-definite similarity.
#
# These are the options to weigh next.
