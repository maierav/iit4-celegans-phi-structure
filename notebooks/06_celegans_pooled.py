# %% [markdown]
# # 06 — The *C. elegans* comparison, on pooled per-stimulus data
#
# This is the analysis the project was built for: **are the Φ-structures of
# attractant responses more similar to one another than those of repellent
# responses?**
#
# Two decisions distinguish this notebook from the earlier per-animal attempts:
#
# 1. **No connectivity matrix.** PyPhi is given the TPM alone and assumes full
#    connectivity. This removes the sink-node defect that forced φ_s = 0.
# 2. **Epochs are pooled across animals per stimulus.** Each stimulus gets one
#    TPM built from all 8 recordings × 3 repeats = 24 epochs.
#
# **Outputs:** `figures/fig14_pooled_celegans.pdf`,
# `results/pooled_structures.csv`, `results/pooled_distance_interneurons.csv`,
# `results/pooled_distance_sensory.csv`, `results/pooled_permutation_test.csv`

# %% [markdown]
# ## Why pooling was necessary — and what it costs
#
# A 4-unit system has 16 states, so its TPM has 16 × 16 = **256 parameters**.
# One stimulus epoch in one animal supplies 3 repeats × ~40 samples = **120
# frames**, or 0.47 frames per parameter. Measured across the 8 recordings, a
# single epoch visits a mean of **5.0 of 16 states** — and *controls* score
# higher (6.1) than attractants (4.7), which is the signature of noise rather
# than biology. Any per-stimulus Φ-structure built that way is mostly an
# artefact of states that were never observed.
#
# Pooling the same stimulus across all 8 animals gives **24 epochs, 960 frames**
# per stimulus — 3.8 frames per parameter, and 11–16 of 16 states visited.
#
# **What pooling assumes.** That the 8 animals are interchangeable replicates of
# one system. They are isogenic hermaphrodites imaged under the same protocol,
# which is the strongest version of that assumption available in practice, but
# it is still an assumption: it discards genuine between-animal variation in
# neural dynamics, and it cannot be checked from within the pooled data. It also
# means **there is no within-class variance estimate from repeated animals** —
# the only replication left is across the 4 stimuli in each class.
#
# This is a stopgap that makes the question computable, not a solution. Sampling
# more states per animal is the real fix.

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
import ast
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphi
from pyphi import convert

import ces_hypergraph as ch

pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False

BLUE, ORANGE, GREY, LIGHT = "#1f6fb4", "#c2571a", "#8a8a8a", "#9bb8d4"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})

RECORDINGS = ["20220327_herm_2", "20220327_herm_4", "20220403_herm_2", "20220403_herm_3",
              "20220427_herm_2", "20220427_herm_3", "20220427_herm_4", "20220427_herm_5"]
INTERNEURONS = ["AIBL", "AVEL", "AVAL", "RIML"]
SENSORY = ["ASEL", "ASER", "AWAL", "AWCL"]

FS = ch.SAMPLING_RATE_HZ
TAU = round(3 * FS)            # 3 s transition lag
WINDOW = round(300 * FS)       # 300 s binarization window
EPOCH_SAMPLES = round(15 * FS)  # 15 s of response after each onset

# %%
for rec in RECORDINGS:
    ch.ensure_recording(rec)
present = {n for n in pd.read_csv(ch.recording_path(RECORDINGS[0])).neuron}
print("all 8 recordings available")
print("interneurons present:", all(n in present for n in INTERNEURONS))
print("sensory present:     ", all(n in present for n in SENSORY))

# %% [markdown]
# ## Pooling
#
# For each recording, binarize the traces, collapse to one integer state per
# frame, then accumulate transition counts within a 15 s window after every
# stimulus onset. Counts for the same stimulus are summed across animals.

# %%
def load_states(rec, neurons):
    d = pd.read_csv(ch.recording_path(rec))
    tcols = d.columns[9:-1]
    binary = np.array([
        ch.moving_window_binarize(
            d.loc[d.neuron == n, tcols].iloc[0].astype(float).values, WINDOW)
        for n in neurons])
    states = ch.combine_states(binary)
    onsets = defaultdict(list)
    for onset, label in ast.literal_eval(d.iloc[0]["stimulus"]):
        onsets[label].append(int(onset))
    return states, onsets


def pooled_counts(neurons):
    counts = {s: np.zeros((16, 16)) for s in ch.STIMULUS_CLASS}
    frames, epochs = defaultdict(int), defaultdict(int)
    for rec in RECORDINGS:
        states, onsets = load_states(rec, neurons)
        for stim, times in onsets.items():
            if stim not in counts:
                continue
            for t0 in times:
                segment = states[t0:t0 + EPOCH_SAMPLES]
                if len(segment) <= TAU:
                    continue
                for a, b in zip(segment[:-TAU], segment[TAU:]):
                    counts[stim][a, b] += 1
                frames[stim] += len(segment)
                epochs[stim] += 1
    return counts, frames, epochs


counts_inter, frames_inter, epochs_inter = pooled_counts(INTERNEURONS)
STIMULI = list(ch.STIMULUS_CLASS)
CLASSES = np.array([ch.STIMULUS_CLASS[s] for s in STIMULI])

budget = pd.DataFrame([{
    "stimulus": s, "class": ch.STIMULUS_CLASS[s], "epochs": epochs_inter[s],
    "frames": frames_inter[s],
    "states_visited": int((counts_inter[s].sum(1) > 0).sum()),
    "frames_per_TPM_parameter": round(frames_inter[s] / 256, 2),
} for s in STIMULI])
print(budget.to_string(index=False))
print(f"\nbefore pooling: 120 frames, 0.47 per parameter, ~5.0 of 16 states visited")
print(f"after pooling : {budget.frames.mean():.0f} frames, "
      f"{budget.frames_per_TPM_parameter.mean():.2f} per parameter, "
      f"{budget.states_visited.mean():.1f} of 16 states visited")

# %% [markdown]
# ## Unfolding — without a connectivity matrix
#
# `pyphi.Network` is constructed with the TPM only. With `cm` omitted PyPhi
# assumes full connectivity, so the strong-connectivity check that previously
# short-circuited the analysis to `NullSystemIrreducibilityAnalysis` no longer
# applies. Rows for states never visited are handled by Laplace smoothing
# (α = 0.5), which leaves them uniform rather than undefined.

# %%
def tpm_from_counts(counts, alpha=0.5):
    smoothed = counts + alpha
    return smoothed / smoothed.sum(1, keepdims=True)


def unfold(counts, neurons):
    """Phi-structure of the pooled TPM at its most-occupied state. No CM."""
    tpm_sbs = tpm_from_counts(counts)
    state_index = int(np.argmax(counts.sum(1)))
    tpm_sbn = convert.state_by_state2state_by_node(tpm_sbs)
    network = pyphi.Network(tpm_sbn, node_labels=neurons)   # cm deliberately omitted
    state = tuple((state_index >> i) & 1 for i in range(len(neurons)))
    subsystem = pyphi.Subsystem(network, state)
    return pyphi.new_big_phi.phi_structure(subsystem), state


structures_inter = {}
for s in STIMULI:
    ps, state = unfold(counts_inter[s], INTERNEURONS)
    structures_inter[s] = {"ps": ps, "state": state, "Phi": float(ps.big_phi),
                           "n_dist": len(ps.distinctions),
                           "n_rel": len(list(ps.relations))}
    print(f"{s:<15} state {''.join(map(str, state))}  Phi = {float(ps.big_phi):8.3f}  "
          f"{len(ps.distinctions):>2} distinctions, {len(list(ps.relations)):>5} relations")

# %% [markdown]
# **Every structure has 14–15 distinctions.** The exact distance searches n!
# bijections, and 15! = 1.3 × 10¹² is far past the practical ceiling of n ≈ 9.
#
# The exact search is *not needed here*. All ten structures are built over the
# **same four neurons**, so a distinction labelled `AIBL·AVEL` denotes the same
# mechanism in every structure. The correspondence is given by the data, not
# something to be searched for — and the identity mapping is exactly the term
# the exact distance would evaluate. What is lost is the guarantee that no
# *other* mapping scores lower; that guarantee only matters when the two
# structures live on different substrates.

# %%
def to_canonical(ps, neurons):
    """Key distinctions by mechanism -- canonical across stimuli (same neurons)."""
    name = lambda mech: "·".join(neurons[u] for u in mech)
    phi_d = {name(tuple(d.mechanism)): float(d.phi) for d in ps.distinctions}
    phi_r = {frozenset(name(tuple(m)) for m in r.mechanisms): float(r.phi)
             for r in ps.relations}
    return phi_d, phi_r


canonical_inter = {s: to_canonical(structures_inter[s]["ps"], INTERNEURONS) for s in STIMULI}
mechanism_sets = [set(canonical_inter[s][0]) for s in STIMULI]
print(f"distinct mechanisms across all 10 structures: {len(set.union(*mechanism_sets))} "
      f"of the 15 possible for 4 units")
print(f"every structure's mechanisms are a subset of that union: "
      f"{all(m <= set.union(*mechanism_sets) for m in mechanism_sets)}")


def distance(A, B):
    """Exact distance under the canonical correspondence."""
    d1, r1 = A
    d2, r2 = B
    return (sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in set(d1) | set(d2))
            + sum(abs(r1.get(k, 0.0) - r2.get(k, 0.0)) for k in set(r1) | set(r2)))


# %% [markdown]
# ## Magnitude versus shape
#
# Φ ranges from 92 to 734 across stimuli — an eight-fold spread that does not
# track class. A raw distance is dominated by that spread, so it is reported
# both ways: raw, and after scaling each structure to unit Φ, which compares
# **shape** independent of magnitude.

# %%
n = len(STIMULI)


def distance_matrix(structs):
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = distance(structs[STIMULI[i]], structs[STIMULI[j]])
    return D


def unit_phi(S):
    total = sum(S[0].values()) + sum(S[1].values())
    return ({k: v / total for k, v in S[0].items()},
            {k: v / total for k, v in S[1].items()})


D_raw = distance_matrix(canonical_inter)
shape_inter = {s: unit_phi(canonical_inter[s]) for s in STIMULI}
D_shape = distance_matrix(shape_inter)

Phi_values = np.array([structures_inter[s]["Phi"] for s in STIMULI])
dPhi = np.abs(Phi_values[:, None] - Phi_values[None, :])
upper = np.triu_indices(n, 1)
print(f"correlation with |ΔΦ| across all {len(upper[0])} pairs:")
print(f"  raw distance  : r = {np.corrcoef(D_raw[upper], dPhi[upper])[0, 1]:.3f}")
print(f"  shape distance: r = {np.corrcoef(D_shape[upper], dPhi[upper])[0, 1]:.3f}")

pd.DataFrame(np.round(D_shape, 5), index=STIMULI, columns=STIMULI).to_csv(
    "results/pooled_distance_interneurons.csv")

# %% [markdown]
# ## The test
#
# The hypothesis predicts that within-attractant distances are **smaller** than
# within-repellent ones. With 4 stimuli per class there are 6 within-class pairs
# each. Significance comes from permuting the class labels.

# %%
def class_contrast(D, labels):
    iu = np.triu_indices(len(labels), 1)
    a = [D[i, j] for i, j in zip(*iu) if labels[i] == labels[j] == "attractant"]
    r = [D[i, j] for i, j in zip(*iu) if labels[i] == labels[j] == "repellent"]
    return np.mean(a) - np.mean(r), np.mean(a), np.mean(r)


def permutation_test(D, labels, n_shuffles=20000, seed=0):
    rng = np.random.default_rng(seed)
    observed, mean_a, mean_r = class_contrast(D, labels)
    null = np.array([class_contrast(D, rng.permutation(labels))[0]
                     for _ in range(n_shuffles)])
    p = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_shuffles + 1)
    return observed, mean_a, mean_r, p, null


obs_i, a_i, r_i, p_i, null_i = permutation_test(D_shape, CLASSES)
print("INTERNEURONS (AIBL, AVEL, AVAL, RIML), shape distance")
print(f"  within-attractant {a_i:.4f}   within-repellent {r_i:.4f}")
print(f"  difference {obs_i:+.4f}   two-sided p = {p_i:.4f}")
print(f"  (hypothesis predicts a NEGATIVE difference)")

# %% [markdown]
# ## Robustness — a second, independent neuron set
#
# The chemosensory quartet ASEL/ASER (salt), AWAL/AWCL (attractant odour) is
# present at confidence 1.0 in all 8 recordings. It is the set the project's
# stated aim actually names, and it gives a genuinely independent test of the
# same hypothesis on the same recordings.

# %%
counts_sens, _, _ = pooled_counts(SENSORY)
structures_sens = {}
for s in STIMULI:
    ps, state = unfold(counts_sens[s], SENSORY)
    structures_sens[s] = {"ps": ps, "Phi": float(ps.big_phi),
                          "n_dist": len(ps.distinctions),
                          "n_rel": len(list(ps.relations))}

shape_sens = {s: unit_phi(to_canonical(structures_sens[s]["ps"], SENSORY)) for s in STIMULI}
D_sens = distance_matrix(shape_sens)
pd.DataFrame(np.round(D_sens, 5), index=STIMULI, columns=STIMULI).to_csv(
    "results/pooled_distance_sensory.csv")

obs_s, a_s, r_s, p_s, null_s = permutation_test(D_sens, CLASSES)
print("SENSORY (ASEL, ASER, AWAL, AWCL), shape distance")
print(f"  within-attractant {a_s:.4f}   within-repellent {r_s:.4f}")
print(f"  difference {obs_s:+.4f}   two-sided p = {p_s:.4f}")
print(f"\nsign agrees between the two neuron sets: "
      f"{np.sign(obs_s) == np.sign(obs_i)}")

# %% [markdown]
# ## Robustness — leave-one-stimulus-out

# %%
jackknife = []
for k, dropped in enumerate(STIMULI):
    keep = [i for i in range(n) if i != k]
    sub, labels = D_shape[np.ix_(keep, keep)], CLASSES[keep]
    if list(labels).count("attractant") < 2 or list(labels).count("repellent") < 2:
        continue
    jackknife.append({"dropped": dropped,
                      "contrast": round(class_contrast(sub, labels)[0], 4)})
jackknife = pd.DataFrame(jackknife)
print(jackknife.to_string(index=False))
print(f"\nfull-set contrast {obs_i:+.4f}; jackknife range "
      f"{jackknife.contrast.min():+.4f} to {jackknife.contrast.max():+.4f}")

# %%
results = pd.DataFrame([{
    "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "epochs_pooled": epochs_inter[s], "frames": frames_inter[s],
    "states_visited": int((counts_inter[s].sum(1) > 0).sum()),
    "inter_Phi": round(structures_inter[s]["Phi"], 4),
    "inter_n_dist": structures_inter[s]["n_dist"],
    "inter_n_rel": structures_inter[s]["n_rel"],
    "sens_Phi": round(structures_sens[s]["Phi"], 4),
    "sens_n_dist": structures_sens[s]["n_dist"],
    "sens_n_rel": structures_sens[s]["n_rel"],
} for s in STIMULI])
results.to_csv("results/pooled_structures.csv", index=False)

pd.DataFrame([
    {"neuron_set": "interneurons", "within_attractant": round(a_i, 5),
     "within_repellent": round(r_i, 5), "diff": round(obs_i, 5),
     "p_two_sided": round(p_i, 5), "n_shuffles": 20000},
    {"neuron_set": "sensory", "within_attractant": round(a_s, 5),
     "within_repellent": round(r_s, 5), "diff": round(obs_s, 5),
     "p_two_sided": round(p_s, 5), "n_shuffles": 20000},
]).to_csv("results/pooled_permutation_test.csv", index=False)
print(results.to_string(index=False))

# %% [markdown]
# ## Figure 14

# %%
short = {s: (s if len(s) <= 12 else s[:12]) for s in STIMULI}
COL = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}

fig14 = plt.figure(figsize=(11.0, 6.8))
gs14 = fig14.add_gridspec(2, 3, hspace=0.72, wspace=0.44)

ax = fig14.add_subplot(gs14[0, 0])
visited = [int((counts_inter[s].sum(1) > 0).sum()) for s in STIMULI]
ax.bar([0], [5.0], 0.5, color=GREY)
ax.bar([1], [np.mean(visited)], 0.5, color=ORANGE)
ax.errorbar([1], [np.mean(visited)],
            yerr=[[np.mean(visited) - min(visited)], [max(visited) - np.mean(visited)]],
            fmt="none", ecolor="#444", lw=1, capsize=3)
ax.axhline(16, ls=":", lw=1, color="#444")
ax.text(1.45, 16.2, "all 16", ha="right", va="bottom", fontsize=6, color="#444")
ax.set_xticks([0, 1])
ax.set_xticklabels(["1 epoch\n120 frames", "24 epochs\n960 frames"], fontsize=6.5)
ax.set_ylabel("states visited (of 16)", labelpad=6)
ax.set_ylim(0, 18.5)
ax.set_title("a  Pooling fixes the coverage", loc="left")

ax = fig14.add_subplot(gs14[0, 1])
order = np.argsort([-structures_inter[s]["Phi"] for s in STIMULI])
ax.barh(range(n), [structures_inter[STIMULI[i]]["Phi"] for i in order],
        color=[COL[CLASSES[i]] for i in order], height=0.68)
ax.set_yticks(range(n))
ax.set_yticklabels([short[STIMULI[i]] for i in order], fontsize=6)
ax.invert_yaxis()
ax.set_xlabel("Φ (pooled, no CM)", labelpad=6)
ax.set_title("b  Φ varies 8-fold, not by class", loc="left")

ax = fig14.add_subplot(gs14[0, 2])
by_class = np.argsort([{"attractant": 0, "repellent": 1, "control": 2}[c] for c in CLASSES])
im = ax.imshow(D_shape[np.ix_(by_class, by_class)], cmap="magma_r")
ax.set_xticks(range(n))
ax.set_xticklabels([short[STIMULI[i]] for i in by_class], rotation=90, fontsize=5.5)
ax.set_yticks(range(n))
ax.set_yticklabels([short[STIMULI[i]] for i in by_class], fontsize=5.5)
for b in [3.5, 7.5]:
    ax.axhline(b, color="w", lw=1.3)
    ax.axvline(b, color="w", lw=1.3)
cb = fig14.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.ax.tick_params(labelsize=6)
cb.set_label("shape distance", fontsize=6)
ax.set_title("c  Pairwise distances\n(attractant | repellent | control)", loc="left")

for slot, (null, obs, p, title) in enumerate([
        (null_i, obs_i, p_i, "d  Interneurons: not significant"),
        (null_s, obs_s, p_s, "e  Sensory: sign flips")]):
    ax = fig14.add_subplot(gs14[1, slot])
    ax.hist(null, bins=45, color=LIGHT, edgecolor="none")
    ax.axvline(obs, color=ORANGE, lw=2)
    side = "left" if obs >= 0 else "right"
    ax.text(obs + (0.03 if obs >= 0 else -0.04), ax.get_ylim()[1] * 0.95,
            f"observed {obs:+.3f}\np = {p:.2f}", fontsize=6.5, color=ORANGE,
            va="top", ha=side)
    ax.set_xlabel("mean within-attractant − within-repellent", labelpad=6)
    ax.set_ylabel("shuffles", labelpad=6)
    ax.set_title(title, loc="left")

ax = fig14.add_subplot(gs14[1, 2])
ax.barh(range(len(jackknife)), jackknife.contrast, color=ORANGE, height=0.68)
ax.axvline(obs_i, ls="--", lw=1, color="#444")
ax.set_yticks(range(len(jackknife)))
ax.set_yticklabels([f"−{short[s]}" for s in jackknife.dropped], fontsize=6)
ax.invert_yaxis()
ax.set_xlabel(f"contrast, that stimulus dropped\n(dashed = full set, {obs_i:+.2f})", labelpad=6)
ax.set_title("f  Unstable to dropping\nany one stimulus", loc="left")

fig14.savefig("figures/fig14_pooled_celegans.pdf")
fig14.savefig("figures/fig14_pooled_celegans.png", dpi=200)
print("wrote figures/fig14_pooled_celegans.pdf")

# %% [markdown]
# ## What this shows
#
# 1. **Both fixes worked.** Without a connectivity matrix Φ is well defined for
#    every stimulus, and pooling raised state coverage from ~5 to 11–16 of 16.
#    The pipeline now produces ten real per-stimulus Φ-structures.
# 2. **The hypothesis is not supported.** Attractant structures are not more
#    similar to each other than repellent ones. In the interneurons the
#    difference runs the *wrong way* and is not significant (p ≈ 0.38); in the
#    sensory neurons it runs the predicted way but is even further from
#    significance (p ≈ 0.73). **The sign is not stable across neuron sets.**
# 3. **Nor is it stable within one neuron set.** Dropping any single stimulus
#    moves the interneuron contrast between +0.02 and +0.29.
# 4. **This is a null result, not evidence of absence.** Four stimuli per class
#    give six within-class pairs — the permutation null has a standard deviation
#    comparable to the observed effect, so only a very large effect could have
#    reached significance. The design is underpowered for this contrast.
#
# ## What would raise the power
#
# * **More stimuli per class.** The binding constraint is 4, not the number of
#   animals. Six to eight per class would roughly double the within-class pairs.
# * **Per-animal structures.** Pooling was forced by state coverage. Longer
#   recordings, or a coarser binarization that visits more states per epoch,
#   would allow one structure per animal per stimulus — restoring a real
#   within-class variance estimate and a far larger permutation space.
# * **Choosing the state deliberately.** Every pooled TPM is evaluated at its
#   most-occupied state, which is all-off for all ten stimuli. A state chosen
#   to reflect the *response* rather than the baseline may separate classes
#   better.
