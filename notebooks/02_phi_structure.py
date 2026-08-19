# %% [markdown]
# # 02 — Unfolding the Φ-structure, and why the published Φ was 0
#
# **What this notebook does:** installs PyPhi at the IIT 4.0 branch, unfolds the
# Φ-structure from the TPM built in notebook 01, and diagnoses the connectivity
# problem that silently zeroed the original result. It also extracts the
# Φ-structure as a **weighted hypergraph**, which is the object the similarity
# measure has to compare.
#
# **Key concepts, in one line each**
#
# * A **distinction** is a subset of units (a *mechanism*) that specifies a
#   cause and an effect over some *purview*, with integrated information φ_d.
# * A **relation** is an overlap between the purviews of two or more
#   distinctions, with integrated information φ_r. A relation joining *k*
#   distinctions is a **degree-k face**.
# * The **Φ-structure** (or cause–effect structure, CES) is all distinctions
#   plus all relations. Φ = Σφ_d + Σφ_r.
#
# The degree-k faces for k > 2 are the "higher-order" content that makes this
# object a hypergraph rather than a graph — and they are what a standard graph
# similarity cannot see.
#
# **Outputs:** `figures/fig03_connectivity.pdf`,
# `figures/fig04_phi_structure.pdf`, `results/phi_structure_variants.csv`,
# `results/ces_hypergraph_20220327_herm_2.json`

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

# %% [markdown]
# ## Install PyPhi (IIT 4.0 branch)
#
# IIT 4.0 lives on the `feature/iit-4.0` branch, not on PyPhi releases. This
# pins the exact commit the original notebooks used, `b78d0e3`, so results are
# reproducible.
#
# *On macOS/Apple Silicon:* the `graphillion` wheel that PyPhi depends on is
# linked against Homebrew GCC's `libgomp`. If `import pyphi` fails with a
# `libgomp.1.dylib` error, rebuild it from source (`CC=clang CXX=clang++ pip
# install --no-binary :all: graphillion`), which drops OpenMP cleanly. Colab
# (Linux) is unaffected.

# %%
PYPHI_COMMIT = "b78d0e342d37175cbd55cf35a6d52ae035b4c50f"

try:
    import pyphi
except ImportError:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         f"git+https://github.com/wmayner/pyphi.git@{PYPHI_COMMIT}"],
        check=True,
    )
    import pyphi

os.environ["PYPHI_WELCOME_OFF"] = "yes"
pyphi.config.PROGRESS_BARS = False
pyphi.config.WELCOME_OFF = True
pyphi.config.PARALLEL = False  # Ray does not play well with Colab

import json

import numpy as np
from collections import Counter
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx

import ces_hypergraph as ch

BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "pdf.fonttype": 42, "ps.fonttype": 42,
})
print("PyPhi ready")

# %% [markdown]
# ## 1. Load the TPM from notebook 01
#
# If notebook 01 has not been run in this session, rebuild the TPM here.

# %%
PRIMARY = "20220327_herm_2"
tpm_path = f"results/tpm_{PRIMARY}.npy"

if os.path.exists(tpm_path):
    tpm = np.load(tpm_path)
else:
    print("rebuilding TPM (run notebook 01 first to skip this)")
    path = os.path.join("data", f"{PRIMARY}.csv")
    if not os.path.exists(path):
        url = ("https://drive.google.com/uc?export=download&id="
               + ch.HERM_DRIVE_IDS[PRIMARY])
        subprocess.run(["curl", "-sSL", "-o", path, url], check=True)
    df, time_cols = ch.load_recording(path)
    window = round(300 * ch.SAMPLING_RATE_HZ)
    tau = round(3 * ch.SAMPLING_RATE_HZ)
    binary = [ch.moving_window_binarize(ch.neuron_trace(df, time_cols, n), window)
              for n in ch.NOTEBOOK_NEURONS]
    tpm, _ = ch.build_tpm(ch.combine_states(binary), tau, n_units=4)
    np.save(tpm_path, tpm)

STATE = (1, 0, 0, 1)  # the state analyzed in the original notebooks
print("TPM:", tpm.shape, "| state:", dict(zip(ch.NOTEBOOK_NEURONS, STATE)))

# %% [markdown]
# ## 2. The connectivity matrix problem
#
# PyPhi requires the system graph to be **strongly connected** (every unit
# reachable from every other). If it is not, PyPhi short-circuits: it returns a
# `NullSystemIrreducibilityAnalysis` with reason `NO_STRONG_CONNECTIVITY` and
# system-level φ_s = 0.
#
# The original notebooks state this functional connectivity table
# (from [funconn.princeton.edu](https://funconn.princeton.edu/)):
#
# | from \ to | AIBL | AVEL | AVAL | RIML |
# |---|---|---|---|---|
# | **AIBL** | 0 | 0 | 0 | 0 |
# | **AVEL** | 28 | 0 | 22 | 0 |
# | **AVAL** | 0 | 35 | 0 | 0 |
# | **RIML** | 0 | 10 | 0 | 0 |
#
# **AIBL's row is entirely zero** — it has no outgoing edges. So the graph
# cannot be strongly connected. Two separate issues follow.

# %%
# Issue 1: the binary CM contains an edge the stated table does not.
#
# First, a check worth doing explicitly: the DataFrame built from a dict of
# columns, read as rows=source / cols=target, reproduces the markdown table
# exactly. There is NO transposition here.
as_dict = {"AIBL": [0, 28, 0, 0], "AVEL": [0, 0, 35, 10],
           "AVAL": [0, 22, 0, 0], "RIML": [0, 0, 0, 0]}
df_cm = pd.DataFrame(as_dict, index=ch.NOTEBOOK_NEURONS)
print("DataFrame as built in the original notebook (rows=from, cols=to):")
print(df_cm.to_string())

loop_edges = [(i, j, int(df_cm.loc[i, j]))
              for i in df_cm.index for j in df_cm.columns if df_cm.loc[i, j] > 0]
print("\n  read as source->target:", loop_edges)
print("  markdown table         :", [tuple(e) for e in ch.FUNCONN_EDGES])
print("  identical              :",
      sorted(loop_edges) == sorted(tuple(e) for e in ch.FUNCONN_EDGES))

# The discrepancy is in the BINARY matrix that PyPhi actually receives.
cm_nb = ch.CM_NOTEBOOKS
cm_fc = ch.funconn_binary_cm()
disagree = [(ch.NOTEBOOK_NEURONS[i], ch.NOTEBOOK_NEURONS[j],
             int(cm_nb[i, j]), int(cm_fc[i, j]))
            for i, j in np.argwhere(cm_nb != cm_fc)]
print("\nbinary CM used by the notebooks vs. one derived from the table")
print("  cells that disagree (from, to, notebooks, table):", disagree)
print("  -> the binary CM asserts an AVEL -> RIML connection that the table")
print("     does not contain, which is why the two rows of the Phi table below")
print("     differ (0.37367 vs 0.15836).")

# Issue 2: with or without that edge, the graph is not strongly connected.
print("\nstrong connectivity check")
print("  notebooks' binary CM        :", ch.is_strongly_connected(cm_nb))
print("  table-derived CM            :", ch.is_strongly_connected(cm_fc))
print("  AIBL out-degree (both)      :", int(cm_nb[0].sum()), int(cm_fc[0].sum()))
print("  RIML in-degree  (table)     :", int(cm_fc[:, 3].sum()))

g_fc = nx.from_numpy_array(cm_fc, create_using=nx.DiGraph)
sccs = [sorted(ch.NOTEBOOK_NEURONS[i] for i in c)
        for c in nx.strongly_connected_components(g_fc)]
print("  strongly connected components:", sccs)

# %% [markdown]
# ### What each choice yields
#
# We compute Φ four ways to separate "the connectivity is wrong" from "the
# result is small".

# %%
cm_sc = cm_fc.copy()
cm_sc[0, 1] = 1  # AIBL -> AVEL
cm_sc[1, 3] = 1  # AVEL -> RIML  (minimal additions to close the cycles)

variants = {
    "notebooks' CM (as published)": cm_nb,
    "CM derived from the stated table": cm_fc,
    "minimal strongly connected": cm_sc,
    "no CM (all-to-all)": None,
}

rows = []
structures = {}
for name, cm in variants.items():
    sub = ch.make_subsystem(tpm, STATE, ch.NOTEBOOK_NEURONS, cm=cm)
    ps = ch.phi_structure(sub)
    nodes, faces = ch.ces_hypergraph(ps)
    structures[name] = (ps, nodes, faces)
    rows.append({
        "connectivity": name,
        "strongly_connected": ch.is_strongly_connected(cm) if cm is not None else True,
        "Phi": round(float(ps.big_phi), 5),
        "sia_type": type(ps.sia).__name__,
        "phi_s": round(float(ps.sia.phi), 5),
        "n_distinctions": len(nodes),
        "n_faces": len(faces),
        "max_face_degree": max((f["degree"] for f in faces), default=0),
    })

variant_table = pd.DataFrame(rows)
variant_table.to_csv("results/phi_structure_variants.csv", index=False)
print(variant_table.to_string(index=False))

# %% [markdown]
# **Reading the table.** The published configuration reports Φ = 0.3737 while
# simultaneously reporting `NullSystemIrreducibilityAnalysis` and φ_s = 0. The
# Φ value is a sum over distinctions and relations that were computed anyway;
# the *system* was judged reducible, so that number does not mean what it
# appears to mean. This exactly reproduces the original notebook output.

# %% [markdown]
# ## Figure 3 — the connectivity defect

# %%
fig3, axes3 = plt.subplots(1, 3, figsize=(11, 3.4))

def draw_circuit(ax, cm, title, highlight_sinks=True):
    g = nx.DiGraph()
    g.add_nodes_from(ch.NOTEBOOK_NEURONS)
    for i, src in enumerate(ch.NOTEBOOK_NEURONS):
        for j, dst in enumerate(ch.NOTEBOOK_NEURONS):
            if cm[i, j]:
                g.add_edge(src, dst)
    pos = nx.circular_layout(g)
    sinks = [n for k, n in enumerate(ch.NOTEBOOK_NEURONS) if cm[k].sum() == 0]
    colors = [ORANGE if (highlight_sinks and n in sinks) else "#dce6f0"
              for n in g.nodes]
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1150, node_color=colors,
                           edgecolors="#333", linewidths=0.8)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=6.5)
    for u, v in g.edges:
        ax.annotate("", xy=pos[v], xytext=pos[u],
                    arrowprops=dict(arrowstyle="-|>", lw=1.3, color="#555",
                                    connectionstyle="arc3,rad=0.17",
                                    shrinkA=18, shrinkB=18))
    ax.set_axis_off()
    ax.set_title(title)
    return sinks

s1 = draw_circuit(axes3[0], cm_nb, "a  As published (binary CM)")
axes3[0].text(0.5, -0.09, f"no outgoing edges: {', '.join(s1)}\nnot strongly connected",
              transform=axes3[0].transAxes, ha="center", va="top",
              fontsize=6.5, color=ORANGE)

s2 = draw_circuit(axes3[1], cm_fc, "b  CM derived from the stated table")
# Name ALL strongly connected components, not just the singletons.
scc_text = " + ".join("{" + ", ".join(c) + "}" for c in sccs)
axes3[1].text(0.5, -0.09,
              f"AIBL still a sink; RIML has no input\n"
              f"{len(sccs)} components: {scc_text}",
              transform=axes3[1].transAxes, ha="center", va="top",
              fontsize=6.5, color=ORANGE)

draw_circuit(axes3[2], cm_sc, "c  Minimal strongly connected repair", highlight_sinks=False)
axes3[2].text(0.5, -0.09,
              "+AIBL→AVEL, +AVEL→RIML\nPhi is now well defined",
              transform=axes3[2].transAxes, ha="center", va="top",
              fontsize=6.5, color="#2a7a2a")

fig3.tight_layout()
fig3.savefig("figures/fig03_connectivity.pdf")
fig3.savefig("figures/fig03_connectivity.png", dpi=200)
print("wrote figures/fig03_connectivity.pdf")

# %% [markdown]
# ## 3. The Φ-structure as a weighted hypergraph
#
# We use the all-to-all variant, which is the only one here that yields a
# genuinely irreducible system, and extract every distinction and every
# relation **face**.
#
# The correct PyPhi API path (the original notebook 7 crashed guessing at it):
#
# ```python
# for relation in phi_structure.relations:   # pyphi.relations.Relation
#     for face in relation.faces:            # RelationFace
#         face.phi, face.purview, len(face)  # degree = number of relata
#         for mice in face:                  # MaximallyIrreducibleCause/Effect
#             mice.mechanism, mice.direction
# ```

# %%
ps, nodes, faces = structures["no CM (all-to-all)"]

print(f"Phi = {ps.big_phi:.5f}")
print(f"  {len(nodes)} distinctions, sum phi_d = {sum(nodes.values()):.5f}")
print(f"  {len(list(ps.relations))} relations carrying {len(faces)} faces")
print(f"  sum phi_r over relations = {sum(float(r.phi) for r in ps.relations):.5f}")
print("  (Phi = sum phi_d + sum phi_r over RELATIONS, not over faces)")

print("\ndistinctions")
labels = {i: n for i, n in enumerate(ch.NOTEBOOK_NEURONS)}
def units(t):
    return "".join(labels[u][:4] for u in t) if t else "0"
for (mech, cpv, epv), phi in sorted(nodes.items(), key=lambda kv: -kv[1]):
    print(f"  mechanism {units(mech):14s} cause {units(cpv):10s} "
          f"effect {units(epv):10s} phi_d = {phi:.5f}")

deg_dist = ch.face_degree_distribution(faces)
print("\nrelation faces by degree:", deg_dist)
higher = sum(v for k, v in deg_dist.items() if k > 2)
print(f"  higher-order (degree > 2): {higher} of {len(faces)} "
      f"= {100 * higher / len(faces):.0f}% of all faces")

# %% [markdown]
# ### This is the crux of the similarity problem
#
# A representation that keeps only **pairwise** relations discards those
# higher-order faces outright. Notebook 1 and notebook 7 both do exactly that:
# they store relations in a dict keyed by *ordered pairs* of distinctions.

# %%
phi_d, phi_r, dropped = ch.pairwise_projection(nodes, faces)
print(f"pairwise projection keeps {len(phi_r)} edges and DROPS {dropped} faces")
print(f"  discarded phi_r mass: "
      f"{sum(f['phi'] for f in faces if f['degree'] > 2):.5f}")

# save the hypergraph for notebook 03.
# Mechanism entries are plain ints, but relation-face purviews are PyPhi `Unit`
# objects carrying `.index` and `.label`. Normalize both to ints.
def _ints(seq):
    return [int(u.index) if hasattr(u, "index") else int(u) for u in seq]


def _labels(seq):
    return [str(u.label) if hasattr(u, "label") else ch.NOTEBOOK_NEURONS[int(u)]
            for u in seq]


payload = {
    "recording": PRIMARY,
    "state": list(STATE),
    "big_phi": float(ps.big_phi),
    "nodes": [{"mechanism": _ints(m), "cause_purview": _ints(c),
               "effect_purview": _ints(e), "phi": float(v)}
              for (m, c, e), v in nodes.items()],
    "faces": [{"degree": int(f["degree"]), "purview": _ints(f["purview"]),
               "purview_labels": _labels(f["purview"]),
               "phi": float(f["phi"]),
               "relata": [[_ints(mech), direction] for mech, direction in f["relata"]]}
              for f in faces],
}
with open(f"results/ces_hypergraph_{PRIMARY}.json", "w") as fh:
    json.dump(payload, fh, indent=2)
print(f"\nwrote results/ces_hypergraph_{PRIMARY}.json")

# %% [markdown]
# ## Figure 4 — the Φ-structure at the level of RELATIONS
#
# Drawn at the **relation** level, not the face level. A `Relation` in PyPhi is a
# frozenset of distinctions with a single φ_r; its faces are an internal
# enumeration over cause/effect sides and all inherit the parent's φ, so they
# carry no independent information. The structural facts are the relations.

# %%
# Collapse faces to their parent relations: identity = the set of distinctions.
relations = {}
for f in faces:
    dset = frozenset(mech for mech, _dir in f["relata"])
    phis = relations.setdefault(dset, set())
    phis.add(round(float(f["phi"]), 12))
for dset, phis in relations.items():
    assert len(phis) == 1, f"faces of one relation disagree on phi: {dset} {phis}"
relations = {dset: max(phis) for dset, phis in relations.items()}

SHORT = {k[0]: f"D{i+1}" for i, k in enumerate(nodes)}
MECH_LABEL = {SHORT[k[0]]: units(k[0]) for k in nodes}
PHI_D = {SHORT[k[0]]: v for k, v in nodes.items()}
REL = [(frozenset(SHORT[m] for m in dset), v) for dset, v in relations.items()]
REL.sort(key=lambda kv: -kv[1])

n_rel = len(REL)
n_pair = sum(1 for S, _ in REL if len(S) == 2)
phi_total = sum(v for _, v in REL)
phi_nonpair = sum(v for S, v in REL if len(S) != 2)

print(f"{n_rel} relations: "
      f"{sum(1 for S,_ in REL if len(S)==1)} self, {n_pair} pairwise, "
      f"{sum(1 for S,_ in REL if len(S)>2)} higher-order")
print(f"sum phi_r = {phi_total:.5f}; not representable as a pairwise edge: "
      f"{phi_nonpair:.5f} ({100*phi_nonpair/phi_total:.0f}%)")

fig4 = plt.figure(figsize=(10.4, 3.9))
gs4 = fig4.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.05], wspace=0.34)
LIGHT = "#9bb8d4"

# (a) the hypergraph itself
ax = fig4.add_subplot(gs4[0])
ring = {d: (np.cos(a), np.sin(a) * 0.85 + 0.12)
        for d, a in zip(sorted(PHI_D), [np.pi/2, np.pi*7/6, np.pi*11/6])}
for S, v in REL:                                    # filled simplices first
    if len(S) >= 3:
        Q = np.array([ring[d] for d in sorted(S)])
        ax.fill(Q[:, 0], Q[:, 1], color=ORANGE, alpha=0.20, zorder=1, lw=0)
for S, v in REL:
    if len(S) == 2:
        u, w = [ring[d] for d in sorted(S)]
        ax.plot([u[0], w[0]], [u[1], w[1]], color=BLUE, lw=1.4, zorder=2)
    elif len(S) == 1:
        x, y = ring[next(iter(S))]
        th = np.linspace(-0.26 * np.pi, 1.26 * np.pi, 140)
        ax.plot(x + 0.20 * np.cos(th), y + 0.29 + 0.20 * np.sin(th),
                color=ORANGE, lw=3.4, zorder=3, solid_capstyle="round")
for d, (x, y) in ring.items():
    ax.scatter([x], [y], s=330 + 1500 * PHI_D[d], color="#17527d",
               edgecolors="white", linewidths=1.2, zorder=4)
    ax.text(x, y, d, ha="center", va="center", fontsize=6.5, color="white", zorder=5)
    # place φ_d below the node, except for the looped node where the loop sits above
    ax.text(x, y - 0.34, f"{MECH_LABEL[d]}\nφ$_d$={PHI_D[d]:.3f}", ha="center", va="top",
            fontsize=5.6, color="#444", zorder=5)
self_phi = max([v for S, v in REL if len(S) == 1], default=0.0)
ho_phi = max([v for S, v in REL if len(S) > 2], default=0.0)
ax.text(0.62, 1.30, f"self-relation\nφ$_r$ = {self_phi:.3f}", ha="center", va="bottom",
        fontsize=6.5, color=ORANGE)
ax.text(0.0, -1.22, f"degree-3 relation\nφ$_r$ = {ho_phi:.5f}", ha="center", va="top",
        fontsize=6.5, color=ORANGE)
ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.85, 1.98); ax.set_axis_off()
ax.set_title(f"a  {len(PHI_D)} distinctions, {n_rel} relations\n"
             "(node area ∝ φ$_d$; orange = no pairwise equivalent)")

# (b) relation degree
ax = fig4.add_subplot(gs4[1])
by_k = Counter(len(S) for S, _ in REL)
ks = sorted(by_k)
ax.bar(range(len(ks)), [by_k[k] for k in ks],
       color=[LIGHT if k == 2 else ORANGE for k in ks], width=0.6)
for i, k in enumerate(ks):
    ax.text(i, by_k[k] + 0.06, str(by_k[k]), ha="center", fontsize=7)
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([f"{k}\n{'self' if k==1 else ('pairwise' if k==2 else 'higher-order')}"
                    for k in ks])
ax.set_xlabel("relation degree k (distinctions bound)", labelpad=5)
ax.set_ylabel("number of relations", labelpad=7)
ax.set_ylim(0, max(by_k.values()) * 1.34)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f"b  {n_rel - n_pair} of {n_rel} relations have\nno pairwise equivalent")

# (c) what a pairwise keying can hold
ax = fig4.add_subplot(gs4[2])
ax.bar([0, 1], [n_pair, n_rel - n_pair], color=[LIGHT, ORANGE], width=0.55)
ax.set_xticks([0, 1])
ax.set_xticklabels(["kept\n(degree 2)", "unrepresentable\n(k=1 or k>2)"])
for i, v in enumerate([n_pair, n_rel - n_pair]):
    ax.text(i, v + 0.06, str(v), ha="center", fontsize=7)
ax.set_ylabel("relations", labelpad=7)
ax.set_ylim(0, max(n_pair, n_rel - n_pair) * 1.48)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.text(0.5, 0.93,
        f"φ$_r$ lost = {phi_nonpair:.3f} of {phi_total:.3f}"
        f"  ({100*phi_nonpair/phi_total:.0f}%)",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="#7a2f00")
ax.set_title("c  A pairwise representation drops\nmost of the φ$_r$ mass")

fig4.savefig("figures/fig04_phi_structure.pdf", bbox_inches="tight")
fig4.savefig("figures/fig04_phi_structure.png", dpi=200, bbox_inches="tight")
print("wrote figures/fig04_phi_structure.pdf")

# %% [markdown]
# ## Summary
#
# 1. **PyPhi requires strong connectivity.** The published CM has AIBL as a
#    sink, so the system is judged reducible and φ_s = 0 — while a nonzero Φ is
#    still printed. Reproduced exactly (Φ = 0.3737).
# 2. The binary CM additionally asserts an **AVEL -> RIML** edge that the
#    notebooks' own stated connectivity table does not contain (the
#    dict-built DataFrame itself matches the table exactly -- there is no
#    transposition). Removing that edge does not help: AIBL is a sink either
#    way, so neither version is strongly connected.
# 3. The Φ-structure is a **weighted hypergraph**: distinctions as nodes,
#    relations as hyperedges. Counted as RELATIONS (not faces -- faces inherit
#    their parent relation's phi and carry no independent information), this
#    structure has 5 relations, of which 2 have no pairwise form: the degree-3
#    relation and the self-relation on AIBL. Those two carry 98% of the phi_r.
# 4. A pairwise representation discards all of them. That is the gap notebook 03
#    addresses.
