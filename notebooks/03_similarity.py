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
# | 3 | degree-graded assignment | this repo | distinctions + **faces of any degree** |
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
# Note that measure 2 *can* see this particular difference — the distinction
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
print(f"  measure 1, |dPhi|                 = {abs(phi_A - phi_B):.4f}   <- calls them IDENTICAL")

pdA, prA, _ = ch.pairwise_projection(*A)
pdB, prB, _ = ch.pairwise_projection(*B)
d_pair_ab, _ = ch.ces_distance_pairwise(pdA, prA, pdB, prB)
print(f"  measure 2, bijection distance     = {d_pair_ab:.4f}")

d_hyp, d_term, r_term = ch.ces_distance_hypergraph(A, B)
print(f"  measure 3, degree-graded distance = {d_hyp:.4f}   "
      f"(distinctions {d_term:.4f} + relations {r_term:.4f})")

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

# measure 1
phi_X = sum(X[0].values()) + sum(f["phi"] for f in X[1])
phi_Y = sum(Y[0].values()) + sum(f["phi"] for f in Y[1])
print(f"\n  measure 1, |dPhi|                 = {abs(phi_X - phi_Y):.4f}")

# measure 2: pairwise projection, then brute-force bijection
pdX, prX, dropX = ch.pairwise_projection(*X)
pdY, prY, dropY = ch.pairwise_projection(*Y)
print(f"  pairwise projection: X keeps {len(prX)} edges (drops {dropX}), "
      f"Y keeps {len(prY)} edges (drops {dropY})")
d_pair, _ = ch.ces_distance_pairwise(pdX, prX, pdY, prY)
print(f"  measure 2, bijection distance     = {d_pair:.4f}   <- BLIND to the difference")

# measure 3
d_hyp2, d_t2, r_t2 = ch.ces_distance_hypergraph(X, Y)
print(f"  measure 3, degree-graded distance = {d_hyp2:.4f}   <- detects it")

# %% [markdown]
# Note that measure 1 does register a difference here only because the extra
# face adds φ_r mass. Make the comparison φ-preserving — move the same φ from a
# pairwise face into a higher-order one — and measure 1 goes blind too.

# %%
Z = make_structure(
    base_nodes,
    [([((0,), "CAUSE"), ((1,), "CAUSE"), ((0, 1), "EFFECT")], (0,), 0.10)],
)
phi_Z = sum(Z[0].values()) + sum(f["phi"] for f in Z[1])
pdZ, prZ, dropZ = ch.pairwise_projection(*Z)

print("X: one degree-2 face   Z: one degree-3 face   (same total phi)")
print(f"  Phi(X) = {phi_X:.4f}   Phi(Z) = {phi_Z:.4f}")
print(f"  measure 1, |dPhi|                 = {abs(phi_X - phi_Z):.4f}   <- blind")
print(f"  pairwise: X {len(prX)} edges, Z {len(prZ)} edges (Z dropped {dropZ})")
d_pair_xz, _ = ch.ces_distance_pairwise(pdX, prX, pdZ, prZ)
print(f"  measure 2, bijection distance     = {d_pair_xz:.4f}   <- sees only the missing edge")
d_hyp_xz, _, _ = ch.ces_distance_hypergraph(X, Z)
print(f"  measure 3, degree-graded distance = {d_hyp_xz:.4f}   <- sees the degree change")

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
    deg = ch.face_degree_distribution(real_faces)
    higher = sum(v for k, v in deg.items() if k > 2)
    _, _, dropped_real = ch.pairwise_projection(*real)
    print(f"real Phi-structure ({PRIMARY}, state {payload['state']}): "
          f"Phi = {payload['big_phi']:.5f}")
    print(f"  {len(real_nodes)} distinctions, {len(real_faces)} faces, degrees {deg}")
    print(f"  higher-order faces: {higher} ({100 * higher / len(real_faces):.0f}%)")
    print(f"  a pairwise representation discards {dropped_real} of them")
else:
    real = None
    print("run notebook 02 first to generate", hg_path)

# %% [markdown]
# ## Test 4 — scaling
#
# Measure 2 searches all bijections: O(n! · n²), with a default guard at 8! =
# 40,320. Notebook 7's own toy models produced 10 and 12 distinctions, so the
# guard already blocks them. Optimal assignment (Hungarian) is O(n³) and solves
# the same matching problem exactly for a linear cost.

# %%
rows = []
for n in [4, 6, 8, 10, 12, 16, 22]:
    perms = math.factorial(n)
    rows.append({
        "n_distinctions": n,
        "permutations": perms,
        "brute_force_feasible": perms <= 40320,
        "hungarian_ops_approx": n ** 3,
    })
scaling = pd.DataFrame(rows)
print(scaling.to_string(index=False))

# empirical timing of measure 3 on growing structures
timings = []
for n in [4, 8, 16, 32, 64]:
    nodes_n = {((i,), (i,), (i,)): 0.1 + 0.01 * i for i in range(n)}
    faces_n = [
        {"degree": 2 + (i % 3), "purview": (i % 4,), "phi": 0.05,
         "relata": tuple(sorted(((j,), "CAUSE") for j in range(i % 3 + 2)))}
        for i in range(3 * n)
    ]
    t0 = time.perf_counter()
    ch.ces_distance_hypergraph((nodes_n, faces_n), (nodes_n, faces_n))
    timings.append({"n_distinctions": n,
                    "seconds": round(time.perf_counter() - t0, 4)})
timing_df = pd.DataFrame(timings)
print("\nmeasure 3 runtime:")
print(timing_df.to_string(index=False))

# %% [markdown]
# ## Summary table

# %%
summary = pd.DataFrame([
    {"test": "same Phi, different content",
     "measure_1_absdPhi": abs(phi_A - phi_B),
     "measure_2_pairwise": d_pair_ab,
     "measure_3_hypergraph": d_hyp},
    {"test": "differ by one degree-3 face",
     "measure_1_absdPhi": abs(phi_X - phi_Y),
     "measure_2_pairwise": d_pair,
     "measure_3_hypergraph": d_hyp2},
    {"test": "degree-2 face vs degree-3 face (phi preserved)",
     "measure_1_absdPhi": abs(phi_X - phi_Z),
     "measure_2_pairwise": d_pair_xz,
     "measure_3_hypergraph": d_hyp_xz},
]).round(5)
summary.to_csv("results/measure_comparison.csv", index=False)
print(summary.to_string(index=False))
print("\n0.0 means the measure cannot distinguish the two structures.")

# %% [markdown]
# ## Figure 5 — where each measure fails

# %%
fig5, axes5 = plt.subplots(1, 3, figsize=(11, 3.5))

# (a) the three test cases
ax = axes5[0]
tests = ["same Φ,\ndifferent content", "one extra\ndegree-3 face",
         "degree 2 → 3,\nΦ preserved"]
x = np.arange(3)
w = 0.27
m1 = [abs(phi_A - phi_B), abs(phi_X - phi_Y), abs(phi_X - phi_Z)]
m2 = [d_pair_ab, d_pair, d_pair_xz]
m3 = [d_hyp, d_hyp2, d_hyp_xz]
ax.bar(x - w, m1, w, color=GREY, label="1  |ΔΦ|")
ax.bar(x, m2, w, color="#9bb8d4", label="2  pairwise bijection")
ax.bar(x + w, m3, w, color=ORANGE, label="3  degree-graded")
for xi, v in zip(x - w, m1):
    if v < 1e-9:
        ax.text(xi, 0.008, "blind", ha="center", fontsize=5.5,
                color=GREY, rotation=90)
for xi, v in zip(x, m2):
    if v < 1e-9:
        ax.text(xi, 0.008, "blind", ha="center", fontsize=5.5,
                color="#5a7fa4", rotation=90)
ax.set_xticks(x)
ax.set_xticklabels(tests, fontsize=6)
ax.set_ylabel("reported distance", labelpad=7)
ax.legend(frameon=False, loc="upper left")
ax.set_title("a  Only measure 3 is non-zero\non every test  (higher = more sensitive)")

# (b) real structure: what a pairwise view keeps
ax = axes5[1]
if real is not None:
    kept = len(real_faces) - dropped_real
    bars = ax.bar(["kept\n(degree 2)", "discarded\n(degree > 2)"],
                  [kept, dropped_real], color=["#9bb8d4", ORANGE], width=0.55)
    for b, v in zip(bars, [kept, dropped_real]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, str(v),
                ha="center", fontsize=7)
    ax.set_ylabel("relation faces", labelpad=7)
    ax.set_ylim(0, max(kept, dropped_real) * 1.3)
    ax.set_title(f"b  Real Φ-structure: {100 * dropped_real / len(real_faces):.0f}%"
                 "\nof faces lost to a pairwise view")
else:
    ax.set_axis_off()

# (c) phi_r mass by degree in the real structure
ax = axes5[2]
if real is not None:
    by_deg = {}
    for f in real_faces:
        by_deg[f["degree"]] = by_deg.get(f["degree"], 0.0) + f["phi"]
    ks = sorted(by_deg)
    ax.bar([str(k) for k in ks], [by_deg[k] for k in ks],
           color=["#9bb8d4" if k == 2 else ORANGE for k in ks], width=0.6)
    ax.set_xlabel("face degree", labelpad=7)
    ax.set_ylabel("summed φ_r", labelpad=7)
    ax.set_title("c  Higher-order faces carry\nreal integrated information")
else:
    ax.set_axis_off()

fig5.tight_layout()
fig5.savefig("figures/fig05_measure_failures.pdf")
fig5.savefig("figures/fig05_measure_failures.png", dpi=200)
print("wrote figures/fig05_measure_failures.pdf")

# %% [markdown]
# ## Figure 6 — scaling

# %%
fig6, axes6 = plt.subplots(1, 2, figsize=(8.4, 3.2))

ax = axes6[0]
ns = scaling["n_distinctions"].values
ax.semilogy(ns, scaling["permutations"], "o-", color=GREY, lw=1.3,
            markersize=4, label="bijections: n!")
ax.semilogy(ns, scaling["hungarian_ops_approx"], "s-", color=ORANGE, lw=1.3,
            markersize=4, label="assignment: n³")
ax.axhline(40320, ls=":", lw=1, color="#444")
ax.text(ns[-1], 40320 * 2.2, "default guard (8!)", ha="right", fontsize=6, color="#444")
ax.set_xlabel("number of distinctions", labelpad=7)
ax.set_ylabel("operations", labelpad=7)
ax.legend(frameon=False, loc="upper left")
ax.set_title("a  Brute force is factorial")

ax = axes6[1]
ax.plot(timing_df["n_distinctions"], timing_df["seconds"], "o-",
        color=ORANGE, lw=1.3, markersize=4)
ax.set_xlabel("number of distinctions", labelpad=7)
ax.set_ylabel("seconds", labelpad=7)
ax.set_title("b  Measured runtime, degree-graded measure")
ax.margins(0.08)

fig6.tight_layout()
fig6.savefig("figures/fig06_scaling.pdf")
fig6.savefig("figures/fig06_scaling.png", dpi=200)
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
