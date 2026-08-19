# %% [markdown]
# # 05 — A complete example with PyPhi 2.0, from scratch
#
# This notebook builds two Φ-structures from nothing but a pair of update rules,
# unfolds them with **PyPhi 2.0**, and compares them with the exact distance —
# showing every step visually.
#
# Nothing is hand-written: both structures come out of PyPhi, and every number
# below is computed in the cells that print it.
#
# **The two systems**
#
# | | rule | state |
# |---|---|---|
# | Structure 1 | A = OR(B,C), B = AND(A,C), C = XOR(A,B) | 101 |
# | Structure 2 | every unit = XOR of the other two | 011 |
#
# Both yield **4 distinctions** and contain relations up to **degree 4**, so the
# comparison exercises higher-order structure directly.
#
# **Outputs:** `figures/fig11_two_structures.pdf`,
# `figures/fig12_mapping_search.pdf`, `figures/fig13_cost_breakdown.pdf`,
# `results/example_mappings.csv`, `results/example_terms.csv`

# %% [markdown]
# ## Setup
#
# PyPhi 2.0 is not on PyPI (the latest release there is 1.2.0), so it is
# installed from the `2.0` branch. It requires Python ≥ 3.13 and, unlike the
# `feature/iit-4.0` branch used elsewhere in this repo, it does **not** depend
# on `graphillion`.

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
                    "git+https://github.com/wmayner/pyphi.git@2.0"], check=True)

REPO_ROOT = os.path.abspath(os.getcwd())
if os.path.basename(REPO_ROOT) == "notebooks":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
    os.chdir(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

os.environ["PYPHI_WELCOME_OFF"] = "yes"
import itertools
from collections import Counter
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphi

from gold_standard import gold_standard_distance, phi_of

pyphi.config.PROGRESS_BARS = False

BLUE, ORANGE, GREY, LIGHT, GREEN = "#1f6fb4", "#c2571a", "#8a8a8a", "#9bb8d4", "#2a7a2a"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})

print("formalism in use:", pyphi.config.formalism.iit.version)

# %% [markdown]
# **A note on the formalism.** PyPhi 2.0 ships two IIT 4.0 variants. The
# installed default here is `IIT_4_0_2023`, which is what the rest of this repo
# uses, so the numbers are directly comparable. The 2026 refinement adds an
# intrinsic-information requirement under which deterministic systems return
# φ_s = 0 — worth knowing before comparing against published values.

# %% [markdown]
# ## Step 1 — build the two systems
#
# The PyPhi 2.0 API is `Substrate` (the network) → `System` (network + state) →
# `.ces()` (the Φ-structure). This replaces the `Network`/`Subsystem`/
# `phi_structure()` sequence of the 4.0 branch.

# %%
LABELS = ("A", "B", "C")


def tpm_from_rule(rule, n=3):
    """State-by-node TPM from a deterministic update rule."""
    T = np.zeros((2 ** n, n))
    for i, st in enumerate(itertools.product([0, 1], repeat=n)):
        T[i] = rule(st)
    return T


# Structure 1: the classic IIT example network
TPM_1 = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0],
                  [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0]], float)
STATE_1 = (1, 0, 1)

# Structure 2: every unit is the XOR of the other two
TPM_2 = tpm_from_rule(lambda s: (s[1] ^ s[2], s[0] ^ s[2], s[0] ^ s[1]))
STATE_2 = (0, 1, 1)


def compute_ces(tpm, state, labels=LABELS):
    substrate = pyphi.Substrate(tpm=tpm, node_labels=labels)
    return pyphi.System(substrate, state=state).ces()


ces1 = compute_ces(TPM_1, STATE_1)
ces2 = compute_ces(TPM_2, STATE_2)

for tag, ces in [("Structure 1  AND-OR-XOR[101]", ces1),
                 ("Structure 2  all-XOR[011]", ces2)]:
    print(f"{tag}")
    print(f"   Phi = {float(ces.big_phi):.5f}"
          f"   (distinctions {float(ces.sum_phi_distinctions):.5f}"
          f" + relations {float(ces.sum_phi_relations):.5f})")
    print(f"   {len(ces.distinctions)} distinctions, "
          f"{ces.relations.num_relations()} relations")

# %% [markdown]
# ## Step 2 — convert to the distance's input form
#
# A PyPhi `Relation` is a frozenset **of distinctions** carrying one φ_r, which
# is exactly the form the distance takes: φ_r keyed by a set of distinctions.
# No flattening or choice of representation is involved.

# %%
def to_gold_standard(ces, labels=LABELS):
    name = lambda mech: "".join(labels[u] for u in mech)
    phi_d = {name(d.mechanism): float(d.phi) for d in ces.distinctions}
    phi_r = {frozenset(name(tuple(m)) for m in r.mechanisms): float(r.phi)
             for r in ces.relations}
    return phi_d, phi_r


S1 = to_gold_standard(ces1)
S2 = to_gold_standard(ces2)

for tag, S, ces in [("Structure 1", S1, ces1), ("Structure 2", S2, ces2)]:
    degrees = dict(sorted(Counter(len(k) for k in S[1]).items()))
    print(f"{tag}: phi_d = "
          f"{ {k: round(v, 4) for k, v in sorted(S[0].items())} }")
    print(f"   relations by degree: {degrees}")
    print(f"   sum phi_d + sum phi_r = {phi_of(S):.5f}  "
          f"== PyPhi Phi ({float(ces.big_phi):.5f}): "
          f"{abs(phi_of(S) - float(ces.big_phi)) < 1e-9}")

# %% [markdown]
# ## Step 3 — see the two structures
#
# Each is drawn as the hypergraph it is: distinctions as nodes (area ∝ φ_d),
# pairwise relations as edges, self-relations as loops, and relations of degree
# ≥ 3 as shaded faces spanning the distinctions they bind.

# %%
def ring_positions(names):
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(names), endpoint=False)
    return {nm: (float(np.cos(a)), float(np.sin(a))) for nm, a in zip(sorted(names), ang)}


def draw_structure(ax, S, title, pos=None, phi_scale=1.0):
    phi_d, phi_r = S
    pos = pos or ring_positions(list(phi_d))
    max_r = max(phi_r.values()) if phi_r else 1.0

    # degree >= 3 first, as filled faces underneath
    for T, v in sorted(phi_r.items(), key=lambda kv: -len(kv[0])):
        if len(T) >= 3:
            P = np.array([pos[d] for d in sorted(T)])
            centre = P.mean(axis=0)
            ang = np.arctan2(P[:, 1] - centre[1], P[:, 0] - centre[0])
            P = P[np.argsort(ang)]
            ax.fill(P[:, 0], P[:, 1], color=ORANGE,
                    alpha=0.10 + 0.22 * (v / max_r), lw=0, zorder=1)
    # pairwise edges
    for T, v in phi_r.items():
        if len(T) == 2:
            (x1, y1), (x2, y2) = [pos[d] for d in sorted(T)]
            ax.plot([x1, x2], [y1, y2], color=BLUE,
                    lw=0.6 + 3.2 * (v / max_r), alpha=0.85, zorder=2)
    # self-relations as loops
    for T, v in phi_r.items():
        if len(T) == 1:
            x, y = pos[next(iter(T))]
            th = np.linspace(-0.28 * np.pi, 1.28 * np.pi, 140)
            ax.plot(x + 0.17 * np.cos(th), y + 0.25 + 0.17 * np.sin(th),
                    color=ORANGE, lw=0.8 + 3.0 * (v / max_r),
                    solid_capstyle="round", zorder=3)
    # distinctions
    for d, (x, y) in pos.items():
        ax.scatter([x], [y], s=(120 + 900 * phi_d[d] / max(phi_d.values())) * phi_scale,
                   color="#17527d", edgecolors="white", linewidths=1.2, zorder=4)
        ax.text(x, y, d, ha="center", va="center", fontsize=6,
                color="white", zorder=5)
        # φ_d label: above/below for top/bottom nodes, outside for side nodes
        if abs(y) > 0.5:                       # top or bottom
            lx, ly, ha_, va_ = x, y + (0.62 if y > 0 else -0.42), "center", "center"
        else:                                  # left or right
            lx, ly, ha_, va_ = x * 1.34, y - 0.30, ("right" if x < 0 else "left"), "top"
            lx = x * 1.34
        ax.text(lx, ly, f"φ$_d$={phi_d[d]:.3f}", ha=ha_, va=va_,
                fontsize=5.8, color="#444", zorder=6)
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.15, 2.35)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left")
    return pos


fig11 = plt.figure(figsize=(10.6, 4.2))
gs11 = fig11.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.22)

ax = fig11.add_subplot(gs11[0])
pos1 = draw_structure(ax, S1, f"a  Structure 1 — AND-OR-XOR[101]\n"
                                f"Φ = {phi_of(S1):.4f}")
ax = fig11.add_subplot(gs11[1])
pos2 = draw_structure(ax, S2, f"b  Structure 2 — all-XOR[011]\n"
                                f"Φ = {phi_of(S2):.4f}")

ax = fig11.add_subplot(gs11[2])
deg1 = Counter(len(k) for k in S1[1])
deg2 = Counter(len(k) for k in S2[1])
ks = sorted(set(deg1) | set(deg2))
x = np.arange(len(ks))
ax.bar(x - 0.19, [deg1.get(k, 0) for k in ks], 0.38, color=BLUE, label="Structure 1")
ax.bar(x + 0.19, [deg2.get(k, 0) for k in ks], 0.38, color=ORANGE, label="Structure 2")
for xi, k in zip(x, ks):
    if deg1.get(k, 0): ax.text(xi - 0.19, deg1[k] + 0.12, str(deg1[k]), ha="center", fontsize=6)
    if deg2.get(k, 0): ax.text(xi + 0.19, deg2[k] + 0.12, str(deg2[k]), ha="center", fontsize=6)
ax.set_xticks(x)
ax.set_xticklabels([f"{k}\n{'self' if k == 1 else ('pairwise' if k == 2 else 'higher')}"
                    for k in ks], fontsize=6.5)
ax.set_xlabel("relation degree k", labelpad=6)
ax.set_ylabel("number of relations", labelpad=6)
ax.set_ylim(0, max(max(deg1.values()), max(deg2.values())) * 1.28)
ax.legend(frameon=False, loc="upper right")
ax.set_title("c  Both contain degree-4 relations", loc="left")

fig11.savefig("figures/fig11_two_structures.pdf")
fig11.savefig("figures/fig11_two_structures.png", dpi=200)
print("wrote figures/fig11_two_structures.pdf")

# %% [markdown]
# Node area is φ_d, edge width and loop width are φ_r, and the orange shading is
# a relation of degree ≥ 3 — a face that binds three or four distinctions at
# once and has no pairwise equivalent. Structure 2 is the more regular of the
# two: every distinction is a 2-unit mechanism and all its relations at a given
# degree carry the same φ_r.
#
# ## Step 4 — search over mappings
#
# Both structures have 4 distinctions, so there are 4! = 24 bijections. Each is
# scored, and the distance is the smallest score.

# %%
k1, k2 = list(S1[0]), list(S2[0])


def mapping_terms(M, A, B):
    """Every term contributing to the cost under mapping M, as (label, value)."""
    d1, r1 = A
    d2, r2 = B
    terms = [(f"φ_d  {a} → {b}", abs(d1.get(a, 0.0) - d2.get(b, 0.0)))
             for a, b in M.items()]
    for T, v in sorted(r1.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0]))):
        image = frozenset(M[a] for a in T)
        terms.append((f"φ_r  {{{','.join(sorted(T))}}} → {{{','.join(sorted(image))}}}",
                      abs(v - r2.get(image, 0.0))))
    inverse = {v: k for k, v in M.items()}
    for T, v in sorted(r2.items(), key=lambda kv: (-len(kv[0]), sorted(kv[0]))):
        if frozenset(inverse[b] for b in T) not in r1:
            terms.append((f"φ_r  unmatched {{{','.join(sorted(T))}}}", v))
    return terms


all_mappings = []
for perm in permutations(k2):
    M = dict(zip(k1, perm))
    terms = mapping_terms(M, S1, S2)
    all_mappings.append({"mapping": "  ".join(f"{a}→{b}" for a, b in M.items()),
                         "cost": sum(v for _, v in terms),
                         "_M": M})
mappings = pd.DataFrame(all_mappings).sort_values("cost").reset_index(drop=True)
mappings.drop(columns="_M").to_csv("results/example_mappings.csv", index=False)

D = gold_standard_distance(S1, S2)
best_M = mappings.loc[0, "_M"]
print(f"{len(mappings)} bijections scored")
print(f"  best  : {mappings.loc[0, 'mapping']}   cost = {mappings.loc[0, 'cost']:.5f}")
print(f"  worst : {mappings.iloc[-1]['mapping']}   cost = {mappings.iloc[-1]['cost']:.5f}")
print(f"  spread: {mappings.cost.max() - mappings.cost.min():.5f} "
      f"({mappings.cost.nunique()} distinct costs)")
print(f"\nD(S1, S2) = {D:.5f}   agrees with the best mapping: "
      f"{abs(D - mappings.loc[0, 'cost']) < 1e-12}")

lower, upper = abs(phi_of(S1) - phi_of(S2)), phi_of(S1) + phi_of(S2)
print(f"\nbounds:  |ΔΦ| = {lower:.5f}  ≤  D = {D:.5f}  ≤  Φ₁+Φ₂ = {upper:.5f}   "
      f"{lower - 1e-9 <= D <= upper + 1e-9}")
print(f"|ΔΦ| alone would report {lower:.5f}, understating the difference by "
      f"{100 * (1 - lower / D):.0f}%.")

# %%
fig12 = plt.figure(figsize=(10.6, 3.6))
gs12 = fig12.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.30)

# (a) every mapping's cost
ax = fig12.add_subplot(gs12[0])
order = np.arange(len(mappings))
colors = [ORANGE if i == 0 else LIGHT for i in order]
ax.bar(order, mappings.cost, color=colors, width=0.75)
ax.axhline(lower, ls="--", lw=1.1, color=GREY, zorder=5)
ax.text(-0.35, lower + 0.05, f"|ΔΦ| = {lower:.3f}  (lower bound)", va="bottom", ha="left",
        fontsize=6.5, color="#555", zorder=6)
ax.annotate(f"minimum = D = {D:.4f}", xy=(0, mappings.loc[0, "cost"]),
            xytext=(3.4, mappings.cost.max() * 0.96), fontsize=6.5, color=ORANGE,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=ORANGE))
ax.set_xlabel("bijection, sorted by cost (all 4! = 24)", labelpad=6)
ax.set_ylabel("total cost", labelpad=6)
ax.set_xticks([0, 5, 10, 15, 20, 23])
ax.margins(x=0.01)
ax.set_ylim(0, mappings.cost.max() * 1.14)
ax.set_title("a  The distance is the minimum over mappings", loc="left")

# (b) the winning mapping drawn
ax = fig12.add_subplot(gs12[1])
left = {d: (0.0, y) for d, y in zip(sorted(S1[0]), np.linspace(0.86, 0.14, len(S1[0])))}
right = {d: (1.0, y) for d, y in zip(sorted(S2[0]), np.linspace(0.86, 0.14, len(S2[0])))}
for a, b in best_M.items():
    (x1, y1), (x2, y2) = left[a], right[b]
    ax.annotate("", xy=(x2 - 0.06, y2), xytext=(x1 + 0.06, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=ORANGE,
                                connectionstyle="arc3,rad=0.10"))
for d, (x, y) in left.items():
    ax.scatter([x], [y], s=90 + 700 * S1[0][d] / max(S1[0].values()),
               color=BLUE, edgecolors="white", linewidths=1, zorder=3)
    ax.text(x - 0.11, y, f"{d}\n{S1[0][d]:.3f}", ha="right", va="center", fontsize=6)
for d, (x, y) in right.items():
    ax.scatter([x], [y], s=90 + 700 * S2[0][d] / max(S2[0].values()),
               color=ORANGE, edgecolors="white", linewidths=1, zorder=3)
    ax.text(x + 0.11, y, f"{d}\n{S2[0][d]:.3f}", ha="left", va="center", fontsize=6)
ax.text(0.0, 0.99, "Structure 1", ha="center", fontsize=7, color=BLUE)
ax.text(1.0, 0.99, "Structure 2", ha="center", fontsize=7, color=ORANGE)
ax.set_xlim(-0.42, 1.42)
ax.set_ylim(0.02, 1.10)
ax.set_axis_off()
ax.set_title("b  The winning bijection", loc="left")

fig12.savefig("figures/fig12_mapping_search.pdf")
fig12.savefig("figures/fig12_mapping_search.png", dpi=200)
print("wrote figures/fig12_mapping_search.pdf")

# %% [markdown]
# The 24 bijections do **not** all cost the same — the spread is real, so the
# minimisation is doing work. Note also that the winning mapping is not the one
# that best matches φ_d values pairwise: relations are carried along by the
# distinction mapping, so a locally worse distinction pairing can win if it
# places the relations better.
#
# ## Step 5 — where the distance comes from
#
# The cost under the winning mapping, term by term.

# %%
terms = mapping_terms(best_M, S1, S2)
term_df = pd.DataFrame(terms, columns=["term", "contribution"])
term_df["kind"] = np.where(term_df.term.str.startswith("φ_d"), "distinction", "relation")


def _degree(label):
    if label.startswith("φ_d"):
        return 0
    inside = label.split("{")[1].split("}")[0]
    return len(inside.split(","))


term_df["degree"] = term_df.term.map(_degree)
term_df = term_df.sort_values("contribution", ascending=False).reset_index(drop=True)
term_df.to_csv("results/example_terms.csv", index=False)

print(f"total = {term_df.contribution.sum():.5f}  (= D)")
print(f"  from distinctions : {term_df.loc[term_df.kind == 'distinction', 'contribution'].sum():.5f}")
print(f"  from relations    : {term_df.loc[term_df.kind == 'relation', 'contribution'].sum():.5f}")
print("\nby relation degree:")
for k, sub in term_df[term_df.kind == "relation"].groupby("degree"):
    print(f"   degree {k}: {len(sub)} terms, {sub.contribution.sum():.5f}")
print("\nlargest contributions:")
print(term_df.head(8).to_string(index=False))

# %%
fig13 = plt.figure(figsize=(10.6, 4.4))
gs13 = fig13.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.44)

# (a) every term
ax = fig13.add_subplot(gs13[0])
show = term_df.iloc[::-1]
cols = [BLUE if kd == "distinction" else (LIGHT if dg == 2 else ORANGE)
        for kd, dg in zip(show.kind, show.degree)]
ax.barh(np.arange(len(show)), show.contribution, color=cols, height=0.74)
ax.set_yticks(np.arange(len(show)))
ax.set_yticklabels(show.term, fontsize=5.2, family="monospace")
ax.set_xlabel("contribution to the distance", labelpad=6)
ax.set_xlim(0, term_df.contribution.max() * 1.10)
ax.set_title("a  Every term under the winning mapping", loc="left")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, LIGHT, ORANGE)]
ax.legend(handles, ["distinction", "pairwise relation", "self / higher-order"],
          frameon=False, loc="lower right", fontsize=6)

# (b) split by kind and degree
ax = fig13.add_subplot(gs13[1])
grp = term_df.groupby("degree").contribution.sum()
labels = ["distinctions" if k == 0 else f"degree {k}" for k in grp.index]
cols = [BLUE if k == 0 else (LIGHT if k == 2 else ORANGE) for k in grp.index]
ax.bar(range(len(grp)), grp.values, color=cols, width=0.62)
for i, v in enumerate(grp.values):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5)
ax.set_xticks(range(len(grp)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("summed contribution", labelpad=6)
ax.set_ylim(0, grp.max() * 1.22)
higher = grp[[k for k in grp.index if k > 2]].sum()
ax.set_title(f"b  {100 * higher / grp.sum():.0f}% comes from\nrelations of degree > 2", loc="left")

# (c) the bounds
ax = fig13.add_subplot(gs13[2])
ax.barh([0], [upper], color="#e8e8e8", height=0.42)
ax.barh([0], [D], color=ORANGE, height=0.42)
ax.axvline(lower, color=GREY, ls="--", lw=1.2)
ax.text(lower, 0.32, f"|ΔΦ|\n{lower:.3f}", ha="center", va="bottom", fontsize=6.5, color=GREY)
ax.text(D, -0.34, f"D = {D:.3f}", ha="center", va="top", fontsize=7, color=ORANGE)
ax.text(upper, 0.32, f"Φ₁+Φ₂\n{upper:.3f}", ha="right", va="bottom", fontsize=6.5, color="#666")
ax.set_yticks([])
ax.set_xlabel("distance", labelpad=6)
ax.set_xlim(0, upper * 1.06)
ax.set_ylim(-0.75, 0.75)
ax.spines["left"].set_visible(False)
ax.set_title("c  The distance sits inside\nits theoretical bounds", loc="left")

fig13.savefig("figures/fig13_cost_breakdown.pdf")
fig13.savefig("figures/fig13_cost_breakdown.png", dpi=200)
print("wrote figures/fig13_cost_breakdown.pdf")

# %% [markdown]
# ## Summary
#
# Two Φ-structures were built from update rules, unfolded by PyPhi 2.0, and
# compared — nothing hand-written at any stage.
#
# 1. Both have 4 distinctions and relations up to degree 4.
# 2. All 24 bijections were scored; their costs genuinely differ, so the
#    minimisation matters.
# 3. The distance agrees with the best mapping, and sits inside
#    |ΔΦ| ≤ D ≤ Φ₁ + Φ₂.
# 4. `|ΔΦ|` alone would substantially understate the difference, because it
#    cannot see how the same total Φ is distributed across degrees.
# 5. A large share of the distance comes from relations of degree > 2 — content
#    no pairwise representation could hold.
