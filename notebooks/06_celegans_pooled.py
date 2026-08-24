# %% [markdown]
# # 06 — The *C. elegans* comparison
#
# The analysis the project was built for: **are the Φ-structures of attractant
# responses more similar to one another than those of repellent responses?**
#
# The short answer is **no**, and the reason is more informative than the
# hypothesis test: **there is no signal above the noise floor.** Comparing the
# same stimulus against itself across different animals gives distances just as
# large as comparing different stimuli. Everything else in this notebook follows
# from that.
#
# **Outputs:** `figures/fig14_pooled_celegans.pdf`, `figures/fig16_epochs.pdf`,
# `figures/fig17_structures_and_labels.pdf`, and five CSVs in `results/`.

# %% [markdown]
# ## What is being compared, concretely
#
# **Four neurons.** Every Φ-structure here is over a **4-unit substrate**, so
# it has at most 2⁴−1 = 15 distinctions. Two substrates are analysed
# *separately*, and they are never compared to each other:
#
# | substrate | neurons | why |
# |---|---|---|
# | interneurons | AIBL, AVEL, AVAL, RIML | the tentative *main complex* identified in awake animals (Kitazono et al. 2023), and the set the project's earlier notebooks used |
# | sensory | ASEL, ASER, AWAL, AWCL | amphid chemosensory neurons: ASEL/ASER for salt, AWAL/AWCL for attractant odour — the set the project's stated aim names |
#
# These are **two independent tests of the same hypothesis on the same
# recordings**, not a decomposition of one analysis. Running both is a
# robustness check: if the effect were real it should appear, with the same
# sign, in a substrate that carries the relevant information. It does not — and
# the disagreement between them is itself a result.
#
# This is *not* an attempt to explain a Φ-structure effect in terms of raw
# neural activity. That inference would be treacherous — the map from traces to
# Φ-structure runs through binarization, a TPM, and the whole IIT unfolding, and
# is nowhere close to linear. No such claim is made here.

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
from itertools import combinations

import matplotlib as mpl
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
                     "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False})

RECS = ["20220327_herm_2", "20220327_herm_4", "20220403_herm_2", "20220403_herm_3",
        "20220427_herm_2", "20220427_herm_3", "20220427_herm_4", "20220427_herm_5"]
INTER = ["AIBL", "AVEL", "AVAL", "RIML"]
SENSORY = ["ASEL", "ASER", "AWAL", "AWCL"]
SUBSTRATE = {"inter": INTER, "sens": SENSORY}
STIMULI = list(ch.STIMULUS_CLASS)
CLASSES = np.array([ch.STIMULUS_CLASS[s] for s in STIMULI])
COL = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}
short = {s: (s if len(s) <= 12 else s[:12]) for s in STIMULI}

FS = ch.SAMPLING_RATE_HZ
WINDOW = round(300 * FS)
EPOCH_N = round(15 * FS)
TAU_N = 1

for rec in RECS:
    ch.ensure_recording(rec)
print(f"sampling rate {FS:.4f} Hz | binarization window {WINDOW} samples (300 s)")
print(f"epoch {EPOCH_N} samples (15 s) | lag τ = {TAU_N} sample ({TAU_N/FS:.3f} s)")

# %% [markdown]
# ## Temporal grain: τ = one sampling interval
#
# Transitions are counted between **consecutive frames** — τ = 1 sample =
# 0.375 s, the native acquisition rate.
#
# This is an **auxiliary assumption**, and it is worth being explicit that it is
# one. Three reasons for this choice:
#
# 1. **Data.** Coarse-graining time discards samples we cannot spare. At τ = 1
#    a 40-frame epoch yields 39 transitions; at τ = 8 (3 s) it yields 32, an 18%
#    loss across the whole dataset.
# 2. **Plausibility.** The only consciousness we can reason about from the
#    inside is our own, and the integration window relevant to an experience —
#    say, the positive valence of an attractant — is very unlikely to be on the
#    order of seconds.
# 3. **Scope.** The published argument for selecting τ by `argmax φ_s` was made
#    for single-animal, single-session TPMs. It does not straightforwardly
#    transfer to a TPM concatenated across animals and sessions, which is what
#    is built here.
#
# `notebooks/07` implements the argmax-φ_s selection and shows that at this data
# volume τ* is not identifiable — it tracks the epoch window rather than the
# dynamics. **Choosing τ properly is left for future runs with more samples per
# epoch.** Like the choice of binarization, the temporal grain here is somewhat
# arbitrary; analysis of experimental data always rests on auxiliary
# assumptions, and sometimes those assumptions are known to be imperfect rather
# than merely unverified.

# %% [markdown]
# ## Epochs
#
# A 15 s window from each stimulus onset — matching the stimulus presentation
# duration used in the reference notebook. Onsets are 60 s apart, so epochs
# never overlap and never span two stimuli. Each animal contributes 3 repeats
# per stimulus; pooling across 8 animals gives 24 epochs = 960 frames.

# %%
def load(rec, neurons):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    binary = [ch.moving_window_binarize(
        d.loc[d.neuron == n, tc].iloc[0].astype(float).values, WINDOW) for n in neurons]
    states = ch.combine_states(binary)
    onsets = defaultdict(list)
    for t, lab in ast.literal_eval(d.iloc[0]["stimulus"]):
        onsets[lab].append(int(t))
    return states, onsets


DATA = {tag: [load(r, nz) for r in RECS] for tag, nz in SUBSTRATE.items()}

d0 = pd.read_csv(ch.recording_path(RECS[0]))
onset_times = sorted(int(t) for t, _ in ast.literal_eval(d0.iloc[0]["stimulus"]))
print(f"{len(onset_times)} presentations per recording, "
      f"{np.diff(onset_times).min()}–{np.diff(onset_times).max()} samples apart "
      f"({np.diff(onset_times).min()/FS:.0f} s)")
print(f"epoch {EPOCH_N} samples fits inside the inter-onset interval: "
      f"{EPOCH_N < np.diff(onset_times).min()}")


def counts(tag, stimulus, recs=None, tau=TAU_N, epoch=EPOCH_N):
    C = np.zeros((16, 16))
    n_epochs = n_frames = 0
    source = DATA[tag] if recs is None else [DATA[tag][i] for i in recs]
    for states, onsets in source:
        for t0 in onsets.get(stimulus, []):
            segment = states[t0:t0 + epoch]
            if len(segment) <= tau:
                continue
            for a, b in zip(segment[:-tau], segment[tau:]):
                C[a, b] += 1
            n_epochs += 1
            n_frames += len(segment)
    return C, n_epochs, n_frames


budget = pd.DataFrame([{
    "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "epochs": counts("inter", s)[1], "frames": counts("inter", s)[2],
    "transitions": int(counts("inter", s)[0].sum()),
    "states_visited": int((counts("inter", s)[0].sum(1) > 0).sum()),
} for s in STIMULI])
print(budget.to_string(index=False))

# %% [markdown]
# ## Figure 16 — how the epochs were chosen

# %%
tc = d0.columns[9:-1]
stim_marks = [(int(t), lab) for t, lab in ast.literal_eval(d0.iloc[0]["stimulus"])]

fig16 = plt.figure(figsize=(11.0, 7.4))
gs16 = fig16.add_gridspec(3, 1, hspace=0.75, height_ratios=[1.25, 1.0, 0.85])

ax = fig16.add_subplot(gs16[0])
for i, nm in enumerate(INTER):
    tr = d0.loc[d0.neuron == nm, tc].iloc[0].astype(float).values
    z = (tr - np.nanmean(tr)) / np.nanstd(tr)
    ax.plot(np.arange(len(z)) / FS, z * 0.42 + (3 - i) * 1.25, lw=0.35, color="#333")
    ax.text(-30, (3 - i) * 1.25, nm, fontsize=6, ha="right", va="center")
for t0, lab in stim_marks:
    ax.axvspan(t0 / FS, (t0 + EPOCH_N) / FS, color=COL[ch.STIMULUS_CLASS[lab]],
               alpha=0.30, lw=0)
ax.set_xlim(-60, len(tc) / FS)
ax.set_ylim(-0.9, 4.5)
ax.set_yticks([])
ax.set_xlabel("time (s)", labelpad=6)
ax.spines["left"].set_visible(False)
ax.legend(handles=[mpl.patches.Patch(color=COL[c], alpha=0.5, label=c)
                   for c in ("attractant", "repellent", "control")],
          frameon=False, loc="upper right", ncol=3, handlelength=1.1, borderaxespad=0.2)
ax.set_title("a  One recording (31 min, 4979 frames). Shading marks the 30 analysed "
             "epochs:\n15 s from each stimulus onset — the stimulus presentation duration",
             loc="left")

ax = fig16.add_subplot(gs16[1])
t_lo, t_hi = stim_marks[0][0] - 40, stim_marks[2][0] + 120
sl = slice(t_lo, t_hi)
tt = np.arange(len(tc))[sl] / FS
for i, nm in enumerate(INTER):
    tr = d0.loc[d0.neuron == nm, tc].iloc[0].astype(float).values
    z = (tr - np.nanmean(tr)) / np.nanstd(tr)
    ax.plot(tt, z[sl] * 0.34 + (3 - i) * 0.95, lw=0.7, color="#333")
    ax.text(t_lo / FS - 4, (3 - i) * 0.95, nm, fontsize=6, ha="right", va="center")
for t0, lab in stim_marks[:3]:
    ax.axvspan(t0 / FS, (t0 + EPOCH_N) / FS, color=COL[ch.STIMULUS_CLASS[lab]],
               alpha=0.30, lw=0)
    ax.annotate("", xy=(t0 / FS, 4.05), xytext=((t0 + EPOCH_N) / FS, 4.05),
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#444"))
    ax.text((t0 + EPOCH_N / 2) / FS, 4.20, f"{lab}\n15 s = {EPOCH_N} samples",
            fontsize=6, ha="center", va="bottom", color="#333")
    ax.axvline(t0 / FS, color="#444", lw=0.8, ls=":")
gap = (stim_marks[1][0] - stim_marks[0][0]) / FS
ax.annotate("", xy=(stim_marks[0][0] / FS, -0.72), xytext=(stim_marks[1][0] / FS, -0.72),
            arrowprops=dict(arrowstyle="<->", lw=0.8, color="#777"))
ax.text((stim_marks[0][0] + stim_marks[1][0]) / 2 / FS, -0.62,
        f"onsets {gap:.0f} s apart — epochs never overlap",
        fontsize=6, ha="center", va="bottom", color="#555")
ax.set_xlim(t_lo / FS - 12, t_hi / FS)
ax.set_ylim(-1.0, 5.3)
ax.set_yticks([])
ax.set_xlabel("time (s)", labelpad=8)
ax.spines["left"].set_visible(False)
ax.set_title("b  Three consecutive epochs. Each is a clean 15 s window after onset;\n"
             "the 45 s between onsets is discarded", loc="left")

ax = fig16.add_subplot(gs16[2])
vals = [EPOCH_N, EPOCH_N * 3, EPOCH_N * 24]
bars = ax.bar(range(3), vals, 0.5, color=[GREY, LIGHT, ORANGE])
for v, b in zip(vals, bars):
    ax.text(b.get_x() + b.get_width() / 2, v + 22,
            f"{v} frames\n{v/256:.2f} per TPM parameter", ha="center", fontsize=6.2)
ax.set_xticks(range(3))
ax.set_xticklabels(["1 animal\n1 presentation", "1 animal\n3 repeats",
                    "8 animals\n24 epochs"], fontsize=6.5)
ax.set_ylabel("frames", labelpad=6)
ax.set_ylim(0, 1250)
ax.set_title("c  Pooling the same stimulus across animals", loc="left")

fig16.savefig("figures/fig16_epochs.pdf")
fig16.savefig("figures/fig16_epochs.png", dpi=200)
print("wrote figures/fig16_epochs.pdf")

# %% [markdown]
# ## Why pooling, and what it assumes
#
# A 4-unit system has a 16 × 16 TPM: **256 parameters**. One stimulus epoch in
# one animal supplies 40 frames — 0.16 per parameter. Pooling the same stimulus
# across all 8 animals gives 960 frames, 3.75 per parameter.
#
# What it assumes is that the 8 animals are interchangeable replicates of one
# system. They are isogenic hermaphrodites imaged under one protocol, which is
# the strongest available version of that assumption — but it is still an
# assumption. It discards between-animal variation and **removes the
# within-class variance estimate that repeated animals would have provided**.
# The split-half analysis below recovers part of what pooling hides.

# %% [markdown]
# ## Unfolding, and what the distinction labels mean
#
# `pyphi.Network` is built from the TPM alone, with `cm` omitted, so PyPhi
# assumes full connectivity. See the README for why, and for what a future
# connectivity constraint would have to settle.

# %%
def unfold(C, neurons, alpha=0.5):
    P = (C + alpha) / (C + alpha).sum(1, keepdims=True)
    network = pyphi.Network(convert.state_by_state2state_by_node(P),
                            node_labels=neurons)
    state_index = int(np.argmax(C.sum(1)))
    state = tuple((state_index >> i) & 1 for i in range(len(neurons)))
    return pyphi.new_big_phi.phi_structure(pyphi.Subsystem(network, state)), state


def canonical(ps, neurons):
    """Key distinctions by the MECHANISM they are over: a subset of the neurons."""
    name = lambda mech: "·".join(neurons[u] for u in mech)
    return ({name(tuple(d.mechanism)): float(d.phi) for d in ps.distinctions},
            {frozenset(name(tuple(m)) for m in r.mechanisms): float(r.phi)
             for r in ps.relations})


def unit_phi(S):
    total = sum(S[0].values()) + sum(S[1].values())
    return ({k: v / total for k, v in S[0].items()},
            {k: v / total for k, v in S[1].items()})


def distance(A, B):
    """Distance under the label-preserving correspondence (see note below)."""
    return (sum(abs(A[0].get(k, 0.0) - B[0].get(k, 0.0)) for k in set(A[0]) | set(B[0]))
            + sum(abs(A[1].get(k, 0.0) - B[1].get(k, 0.0)) for k in set(A[1]) | set(B[1])))


FULL = {}
for tag, neurons in SUBSTRATE.items():
    for s in STIMULI:
        C, _, _ = counts(tag, s)
        ps, state = unfold(C, neurons)
        FULL[(tag, s)] = {"raw": canonical(ps, neurons), "Phi": float(ps.big_phi),
                          "nd": len(ps.distinctions), "nr": len(list(ps.relations)),
                          "state": state}
    print(f"{tag}: {len(STIMULI)} structures unfolded")

summary = pd.DataFrame([{
    "substrate": t, "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "Phi": round(FULL[(t, s)]["Phi"], 2), "n_dist": FULL[(t, s)]["nd"],
    "n_rel": FULL[(t, s)]["nr"],
} for t in SUBSTRATE for s in STIMULI])
print(summary.to_string(index=False))

# %% [markdown]
# ### What the labels do — and what they do NOT do
#
# A fair objection: labels should not matter to a Φ-structure. They do not. The
# labels here are not a reduction, a deduplication, or a relabelling trick.
#
# With 4 neurons there are exactly 2⁴−1 = **15 possible mechanisms** — the 15
# non-empty subsets. A distinction *is* a mechanism, so `AIBL·AVEL` names one
# specific subset and `AIBL·AVAL` names a different one. Every structure here
# has 13–15 distinctions drawn from that same fixed set of 15.
#
# So when two structures are compared, the correspondence
# `AIBL·AVEL ↔ AIBL·AVEL` is **one specific bijection out of the 15! available**
# — the one that pairs each mechanism with itself. It is not a collapse of the
# structure; the structure is untouched.

# %%
mechanisms = {s: set(FULL[("inter", s)]["raw"][0]) for s in STIMULI}
union = set.union(*mechanisms.values())
print(f"distinct mechanisms across all 10 structures: {len(union)} "
      f"(= 2^4 - 1 = {2**4 - 1})")
print(f"distinctions per structure: "
      f"{min(len(m) for m in mechanisms.values())}–{max(len(m) for m in mechanisms.values())}")
print(f"any duplicated labels within a structure: "
      f"{any(len(m) != len(set(m)) for m in mechanisms.values())}")
print("\nthe 15 mechanisms:")
for m in sorted(union, key=lambda x: (x.count('·'), x)):
    print(f"    {m}")

# %% [markdown]
# ### The cost of using the identity bijection
#
# The exact distance minimises over all bijections. With 15 distinctions that is
# 15! = 1.3 × 10¹² — far past the *n* ≈ 9 ceiling measured in `notebooks/03`.
#
# The identity bijection is used instead, on the grounds that the two structures
# share a substrate so the mechanism correspondence is given rather than
# searched for. **This is not free, and the cell below measures what it costs:**
# a random-relabelling comparison shows the identity mapping is far better than
# a typical alternative, but it is **not always the argmin**. So the reported
# value is an **upper bound** on the exact distance, and is described as such.

# %%
import random as _random


def distance_under(A, B, M):
    cost = sum(abs(A[0].get(a, 0.0) - B[0].get(M.get(a, a), 0.0))
               for a in set(A[0]) | set(B[0]))
    B_pulled = {frozenset(M.get(x, x) for x in T): v for T, v in B[1].items()}
    for key in set(A[1]) | set(B_pulled):
        cost += abs(A[1].get(key, 0.0) - B_pulled.get(key, 0.0))
    return cost


A = unit_phi(FULL[("inter", "100mM NaCl")]["raw"])
B = unit_phi(FULL[("inter", "450mM NaCl")]["raw"])
labels = sorted(set(A[0]) | set(B[0]))
d_identity = distance(A, B)
rng_lab = _random.Random(0)
random_costs = []
for _ in range(2000):
    shuffled = labels[:]
    rng_lab.shuffle(shuffled)
    random_costs.append(distance_under(A, B, dict(zip(labels, shuffled))))
random_costs = np.array(random_costs)
print(f"identity bijection            : {d_identity:.4f}")
print(f"2000 random relabellings      : min {random_costs.min():.4f}  "
      f"mean {random_costs.mean():.4f}  max {random_costs.max():.4f}")
print(f"identity beats {100 * (random_costs > d_identity).mean():.1f}% of random mappings")
print(f"identity is the smallest seen : {d_identity <= random_costs.min()}")
print("\n=> the identity bijection is a principled, strongly-performing choice,")
print("   but not provably the argmin. Reported distances are UPPER BOUNDS.")

# %% [markdown]
# ## The TPMs, and how the state was chosen
#
# This section answers two questions directly: **what do the TPMs look like**,
# and **how is one state picked out of 16** to unfold the Φ-structure at.
#
# ### The TPM
#
# For each stimulus, every transition observed inside any of its 24 epochs is
# tallied into a 16 × 16 count matrix `C[a, b]` = times state *a* was followed
# τ frames later by state *b*. Rows are then normalised to probabilities with
# **Laplace smoothing**, α = 0.5:
#
# ```
# P[a, b] = (C[a, b] + α) / Σ_b (C[a, b] + α)
# ```
#
# Smoothing is not cosmetic here. With 960 frames and 256 parameters, several
# rows have **zero** observed transitions — those states were never visited in
# any epoch of that stimulus. Without smoothing those rows are undefined and
# PyPhi cannot proceed; with it, they become the uniform prior. Marked with
# dotted lines in the figure below, they are rows where the TPM carries **no
# data at all**: 0–5 of 16 rows per stimulus.
#
# ### Picking the state
#
# A Φ-structure is defined for a system **in a particular state** — IIT 4.0 is
# state-dependent, and there is no such thing as "the Φ-structure of this TPM".
# So one of the 16 states must be chosen for each stimulus.
#
# The rule used is **argmax occupancy**: the state the system spent the most
# epoch frames in. For all ten stimuli that is **0000**, all four neurons below
# threshold, occupying 42–71% of frames.
#
# **This is the weakest link in the pipeline, and it deserves to be stated
# plainly.** Two things are wrong with it:
#
# 1. **0000 is the baseline, not the response.** The most-occupied state during
#    a stimulus epoch is the quiescent one, so the structure being characterised
#    is closer to "what this circuit is like at rest under this chemical" than
#    "what the response to this chemical is like".
# 2. **The choice matters enormously.** Φ at the 16 states of a single TPM spans
#    more than two orders of magnitude. Different states of the same TPM are
#    genuinely different structures.
#
# The cell below quantifies both, and then re-runs the class contrast under four
# different selection rules to check the conclusion does not hinge on the
# choice.

# %%
def occupancy_and_counts(tag, stimulus, tau=TAU_N, epoch=EPOCH_N):
    C = np.zeros((16, 16))
    occ = np.zeros(16)
    for states, onsets in DATA[tag]:
        for t0 in onsets.get(stimulus, []):
            segment = states[t0:t0 + epoch]
            for s_ in segment:
                occ[s_] += 1
            for a, b in zip(segment[:-tau], segment[tau:]):
                C[a, b] += 1
    return C, occ


CNT = {s: occupancy_and_counts("inter", s) for s in STIMULI}

occupancy = pd.DataFrame([{
    "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "argmax_state": format(int(np.argmax(CNT[s][1])), "04b"),
    "argmax_frac": round(CNT[s][1].max() / CNT[s][1].sum(), 3),
    "second_state": format(int(np.argsort(-CNT[s][1])[1]), "04b"),
    "second_frac": round(np.sort(CNT[s][1])[-2] / CNT[s][1].sum(), 3),
    "states_visited": int((CNT[s][1] > 0).sum()),
    "unvisited_tpm_rows": int((CNT[s][0].sum(1) == 0).sum()),
} for s in STIMULI])
occupancy.to_csv("results/state_occupancy.csv", index=False)
print(occupancy.to_string(index=False))
print(f"\nargmax state is 0000 for "
      f"{(occupancy.argmax_state == '0000').sum()} of {len(STIMULI)} stimuli")
print(f"unvisited (prior-only) TPM rows per stimulus: "
      f"{occupancy.unvisited_tpm_rows.min()}–{occupancy.unvisited_tpm_rows.max()}")

# %% [markdown]
# ### How much does the state choice matter? Φ at all 16 states of one TPM

# %%
def tpm_from(C, alpha=0.5):
    return (C + alpha) / (C + alpha).sum(1, keepdims=True)


def structure_at(C, state_index, neurons=INTER):
    network = pyphi.Network(convert.state_by_state2state_by_node(tpm_from(C)),
                            node_labels=neurons)
    state = tuple((state_index >> i) & 1 for i in range(len(neurons)))
    return pyphi.new_big_phi.phi_structure(pyphi.Subsystem(network, state))


C_ref, occ_ref = CNT["100mM NaCl"]
by_state = []
for si in range(16):
    ps = structure_at(C_ref, si)
    by_state.append(dict(state=format(si, "04b"),
                         occupancy=round(occ_ref[si] / occ_ref.sum(), 4),
                         Phi=round(float(ps.big_phi), 2),
                         n_dist=len(ps.distinctions),
                         n_rel=len(list(ps.relations))))
phi_by_state = pd.DataFrame(by_state)
phi_by_state.to_csv("results/phi_by_state_100mM_NaCl.csv", index=False)
print(phi_by_state.to_string(index=False))
print(f"\nΦ spans {phi_by_state.Phi.min():.1f}–{phi_by_state.Phi.max():.1f} across the "
      f"16 states of ONE TPM — a {phi_by_state.Phi.max()/phi_by_state.Phi.min():.0f}× spread")

# %% [markdown]
# ### Does the conclusion depend on the rule?
#
# Four rules, each applied to all ten stimuli, each giving a full distance
# matrix and permutation test.

# %%
def class_contrast(D, labels_):
    iu = np.triu_indices(len(labels_), 1)
    a = [D[i, j] for i, j in zip(*iu) if labels_[i] == labels_[j] == "attractant"]
    r = [D[i, j] for i, j in zip(*iu) if labels_[i] == labels_[j] == "repellent"]
    return np.mean(a) - np.mean(r), np.mean(a), np.mean(r)


def contrast_under_rule(pick):
    shapes = {}
    used = {}
    for s in STIMULI:
        C, occ = CNT[s]
        si = pick(C, occ)
        used[s] = format(si, "04b")
        shapes[s] = unit_phi(canonical(structure_at(C, si), INTER))
    m = len(STIMULI)
    M = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            M[i, j] = M[j, i] = distance(shapes[STIMULI[i]], shapes[STIMULI[j]])
    obs, mean_a, mean_r = class_contrast(M, CLASSES)
    rng_ = np.random.default_rng(0)
    null = np.array([class_contrast(M, rng_.permutation(CLASSES))[0] for _ in range(5000)])
    return dict(diff=obs, within_attractant=mean_a, within_repellent=mean_r,
                p_two_sided=(np.sum(np.abs(null) >= abs(obs)) + 1) / 5001,
                states_used="|".join(sorted(set(used.values()))))


def max_phi_s_state(C, occ):
    network = pyphi.Network(convert.state_by_state2state_by_node(tpm_from(C)),
                            node_labels=INTER)
    best = (-np.inf, 0)
    for si in range(16):
        if occ[si] == 0:
            continue
        state = tuple((si >> i) & 1 for i in range(4))
        try:
            v = float(pyphi.new_big_phi.sia(pyphi.Subsystem(network, state)).phi)
        except Exception:
            continue
        if v > best[0]:
            best = (v, si)
    return best[1]


RULES = {
    "argmax-occupancy (0000)": lambda C, occ: int(np.argmax(occ)),
    "all-ON (1111)": lambda C, occ: 15,
    "2nd most occupied": lambda C, occ: int(np.argsort(-occ)[1]),
    "max phi_s over states": max_phi_s_state,
}
robustness = pd.DataFrame([dict(rule=name, **contrast_under_rule(fn))
                           for name, fn in RULES.items()]).round(5)
robustness.to_csv("results/state_choice_robustness.csv", index=False)
print(robustness.to_string(index=False))
print("\nAll four rules give a null. The hypothesis predicts a NEGATIVE difference")
print("and all four produce one, but none approaches significance. The conclusion")
print("does not hinge on the state choice -- though the Φ VALUES depend on it hugely.")

# %% [markdown]
# ## Figure 18 — the TPMs, and the state-choice problem

# %%
ROB = robustness.set_index("rule").to_dict("index")

LBL=[format(i,"04b") for i in range(16)]

fig = plt.figure(figsize=(11.4, 8.6))
g = fig.add_gridspec(4, 6, hspace=0.72, wspace=0.55, height_ratios=[1,1,0.16,1.35])

# rows 0-1: the ten TPMs
for k,s in enumerate(STIMULI):
    ax = fig.add_subplot(g[k//5, (k%5)+(1 if k%5>=0 else 0)] if False else g[k//5, k%5])
    C,occ = CNT[s]
    P = tpm_from(C)
    im = ax.imshow(P, cmap="magma_r", vmin=0, vmax=0.5, interpolation="nearest")
    unv = np.where(C.sum(1)==0)[0]
    for u in unv:
        ax.add_patch(mpl.patches.Rectangle((-0.5,u-0.5),16,1,fill=False,
                                            ec="#00a0a0",lw=0.7,ls=":"))
    ax.set_xticks([0,15]); ax.set_xticklabels(["0000","1111"], fontsize=5)
    ax.set_yticks([0,15]); ax.set_yticklabels(["0000","1111"], fontsize=5)
    ax.tick_params(length=1.6, pad=1)
    ax.set_title(f"{s}\n{ch.STIMULUS_CLASS[s]} · {int(C.sum())} transitions",
                 fontsize=6.2, color=COL[ch.STIMULUS_CLASS[s]], loc="center", pad=3)
    if k==0:
        ax.set_ylabel("state at t", fontsize=6, labelpad=2)
        ax.set_xlabel("state at t+τ", fontsize=6, labelpad=2)
cax = fig.add_subplot(g[2, 1:4])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("transition probability (Laplace-smoothed, α = 0.5)", fontsize=6.5, labelpad=3)
cb.ax.tick_params(labelsize=6)
cax.text(1.02, 0.5, "dotted rows = state never visited\n(entirely prior)", transform=cax.transAxes,
         fontsize=5.8, va="center", ha="left", color="#00a0a0")

# row 3: the state-choice problem
ax = fig.add_subplot(g[3,0:3])
x = np.arange(16)
ax.bar(x, phi_by_state.occupancy, 0.72, color=LIGHT, label="occupancy")
ax2 = ax.twinx()
ax2.plot(x, phi_by_state.Phi, "o-", ms=3, lw=1.0, color=ORANGE, label="Φ at that state")
ax2.set_yscale("log")
ax2.set_ylabel("Φ  (log)", color=ORANGE, labelpad=1, fontsize=7)
ax2.tick_params(axis="y", colors=ORANGE, labelsize=6, pad=1)
ax.set_xticks(x); ax.set_xticklabels(LBL, rotation=90, fontsize=5)
ax.set_ylabel("fraction of epoch frames", labelpad=4)
ax.set_xlabel("state", labelpad=4)
ax.axvline(0, color="#333", lw=0.8, ls=":")
ax.text(0.55, 0.73, "chosen:\nargmax occupancy", fontsize=6, color="#333", va="top")
h=[mpl.patches.Patch(color=LIGHT,label="occupancy"),
   mpl.lines.Line2D([],[],marker="o",color=ORANGE,label="Φ at that state")]
ax.legend(handles=h, frameon=False, loc="upper center", fontsize=6, ncol=2,
          borderaxespad=0.1)
ax.set_ylim(0, 0.98)
ax.set_title(f"a  100 mM NaCl: one TPM, 16 possible states.\n"
             f"   Φ spans {phi_by_state.Phi.min():.1f}–{phi_by_state.Phi.max():.1f} — a {phi_by_state.Phi.max()/phi_by_state.Phi.min():.0f}× spread",
             loc="left")

# occupancy of 0000 vs 1111 across stimuli
ax = fig.add_subplot(g[3,3])
w=0.38; xs=np.arange(len(STIMULI))
ax.bar(xs-w/2, occupancy.argmax_frac, w, color=GREY, label="0000 (chosen)")
ax.bar(xs+w/2, occupancy.second_frac, w, color=ORANGE, label="1111")
ax.set_xticks(xs); ax.set_xticklabels([s[:11] for s in STIMULI], rotation=90, fontsize=5.5)
ax.set_ylabel("fraction of frames", labelpad=6)
ax.legend(frameon=False, loc="upper right", fontsize=5.8)
ax.set_ylim(0, 0.92)
ax.set_title("b  0000 dominates\n   every epoch", loc="left")

# the four selection rules
ax = fig.add_subplot(g[3,4:6])
rules = list(ROB)
y = np.arange(len(rules))
diffs = [ROB[r]["diff"] for r in rules]
ax.barh(y, diffs, 0.55, color=[ORANGE if abs(d)>0.1 else GREY for d in diffs])
for yi,r in zip(y, rules):
    ax.text(ROB[r]["diff"] - 0.008, yi, f"p = {ROB[r]['p_two_sided']:.2f}", va="center",
            fontsize=6.2, ha="right", color="#333")
ax.axvline(0, color="#333", lw=0.9)
ax.set_yticks([]); ax.invert_yaxis()
for yi, r in zip(y, rules):
    ax.text(0.012, yi, r, va="center", ha="left", fontsize=6.2, color="#333")
ax.set_xlim(-0.30, 0.22)
ax.set_xlabel("mean within-attractant − within-repellent", labelpad=4)
ax.set_title("c  Four state-selection rules.\n   All null; hypothesis predicts negative", loc="left")

fig.savefig("figures/fig18_tpms_and_state_choice.pdf", bbox_inches="tight")
fig.savefig("figures/fig18_tpms_and_state_choice.png", dpi=200, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:8])

print("wrote figures/fig18_tpms_and_state_choice.pdf")

# %% [markdown]
# ## Magnitude versus shape
#
# Φ varies widely across stimuli without tracking class, so distances are
# reported after scaling each structure to unit Φ — comparing **shape**
# independent of magnitude.

# %%
SHAPE = {k: unit_phi(v["raw"]) for k, v in FULL.items()}
n = len(STIMULI)
DM = {}
for tag in SUBSTRATE:
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = distance(SHAPE[(tag, STIMULI[i])], SHAPE[(tag, STIMULI[j])])
    DM[tag] = M
    pd.DataFrame(np.round(M, 5), index=STIMULI, columns=STIMULI).to_csv(
        f"results/pooled_distance_{'interneurons' if tag == 'inter' else 'sensory'}.csv")

upper = np.triu_indices(n, 1)
for tag in SUBSTRATE:
    Phi = np.array([FULL[(tag, s)]["Phi"] for s in STIMULI])
    dPhi = np.abs(Phi[:, None] - Phi[None, :])
    print(f"{tag}: corr(shape distance, |ΔΦ|) = "
          f"{np.corrcoef(DM[tag][upper], dPhi[upper])[0, 1]:.3f}")

# %% [markdown]
# ## The noise floor — the analysis that matters most
#
# Before asking whether *classes* differ, ask whether **anything** differs. Split
# the 8 animals into two halves, build a structure for the *same* stimulus from
# each half, and measure the distance between them. That is pure sampling noise:
# same stimulus, same substrate, same everything except which animals went in.
#
# If between-stimulus distances are not clearly larger than this, no downstream
# test can mean anything.

# %%
SPLITS = [tuple(sorted(c)) for c in combinations(range(8), 4)][:10]
NOISE = {}
for tag, neurons in SUBSTRATE.items():
    for s in STIMULI:
        ds = []
        for half in SPLITS:
            other = tuple(i for i in range(8) if i not in half)
            Ca, _, _ = counts(tag, s, recs=half)
            Cb, _, _ = counts(tag, s, recs=other)
            pa, _ = unfold(Ca, neurons)
            pb, _ = unfold(Cb, neurons)
            ds.append(distance(unit_phi(canonical(pa, neurons)),
                               unit_phi(canonical(pb, neurons))))
        NOISE[(tag, s)] = ds
    print(f"{tag}: split-half noise floor computed ({len(SPLITS)} splits per stimulus)")

noise_rows = []
for tag in SUBSTRATE:
    between = DM[tag][upper]
    within = np.concatenate([NOISE[(tag, s)] for s in STIMULI])
    noise_rows.append(dict(
        substrate=tag,
        between_mean=round(between.mean(), 4), between_sd=round(between.std(), 4),
        within_mean=round(within.mean(), 4), within_sd=round(within.std(), 4),
        signal_to_noise=round(between.mean() / within.mean(), 3)))
noise_table = pd.DataFrame(noise_rows)
noise_table.to_csv("results/noise_floor.csv", index=False)
print(noise_table.to_string(index=False))
print("\n=> between-stimulus distances are AT the within-stimulus noise floor.")
print("   Ratios near 1 mean the measure cannot currently separate stimuli at all.")

# %% [markdown]
# ## The class test
#
# Reported for completeness. Given the noise floor above, a null here was
# expected; the value of running it is that the *pattern* of failure is
# informative.

# %%
rng = np.random.default_rng(0)
PERM = {}
for tag in SUBSTRATE:
    obs, mean_a, mean_r = class_contrast(DM[tag], CLASSES)
    null = np.array([class_contrast(DM[tag], rng.permutation(CLASSES))[0]
                     for _ in range(20000)])
    PERM[tag] = dict(obs=obs, a=mean_a, r=mean_r, null=null,
                     p=(np.sum(np.abs(null) >= abs(obs)) + 1) / 20001)
    print(f"{tag}: within-attractant {mean_a:.4f}  within-repellent {mean_r:.4f}  "
          f"difference {obs:+.4f}  p = {PERM[tag]['p']:.4f}")
print("\n(the hypothesis predicts a NEGATIVE difference)")

jack = []
for tag in SUBSTRATE:
    for k, dropped in enumerate(STIMULI):
        keep = [i for i in range(n) if i != k]
        labels_ = CLASSES[keep]
        if list(labels_).count("attractant") < 2 or list(labels_).count("repellent") < 2:
            continue
        jack.append(dict(substrate=tag, dropped=dropped,
                         contrast=round(class_contrast(DM[tag][np.ix_(keep, keep)],
                                                       labels_)[0], 4)))
jackknife = pd.DataFrame(jack)
jackknife.to_csv("results/jackknife.csv", index=False)
print(jackknife.groupby("substrate").contrast.agg(["min", "max", "mean"]).round(4).to_string())

pd.DataFrame([{
    "substrate": t,
    "within_attractant": round(PERM[t]["a"], 5),
    "within_repellent": round(PERM[t]["r"], 5),
    "diff": round(PERM[t]["obs"], 5),
    "p_two_sided": round(PERM[t]["p"], 5),
    "between_mean": round(DM[t][upper].mean(), 5),
    "noise_floor": round(np.concatenate([NOISE[(t, s)] for s in STIMULI]).mean(), 5),
    "signal_to_noise": round(DM[t][upper].mean()
                             / np.concatenate([NOISE[(t, s)] for s in STIMULI]).mean(), 4),
    "n_shuffles": 20000,
} for t in SUBSTRATE]).to_csv("results/pooled_permutation_test.csv", index=False)

pd.DataFrame([{
    "substrate": t, "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "Phi": round(FULL[(t, s)]["Phi"], 3), "n_dist": FULL[(t, s)]["nd"],
    "n_rel": FULL[(t, s)]["nr"],
    "noise_floor_mean": round(np.mean(NOISE[(t, s)]), 4),
    "noise_floor_sd": round(np.std(NOISE[(t, s)]), 4),
} for t in SUBSTRATE for s in STIMULI]).to_csv("results/pooled_structures.csv", index=False)
print("\nwrote 5 result files")

# %% [markdown]
# ## Figure 17 — real Φ-structures, and what the labels are

# %%
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


fig = plt.figure(figsize=(11.0, 7.6))
g = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.55, height_ratios=[1.15,1.0])

# --- (a,b) two real Phi-structures in 3D ---
def draw3d(ax, S, title):
    labs = sorted(S[0], key=lambda x:(x.count("·"), x))
    order = {l:i for i,l in enumerate(labs)}
    # position: x = mechanism order (1..4 units), y,z = spread within order
    P={}
    for l in labs:
        k = l.count("·")+1
        same = [m for m in labs if m.count("·")+1==k]
        j = same.index(l); m = len(same)
        ang = 2*np.pi*j/max(1,m)
        P[l] = (k, np.cos(ang)*(0.9 if m>1 else 0), np.sin(ang)*(0.9 if m>1 else 0))
    # relations, thickest first, cap for legibility
    rel = sorted(S[1].items(), key=lambda kv:-kv[1])[:120]
    vmax = max(v for _,v in rel) if rel else 1
    for T,v in rel:
        pts = [P[x] for x in T if x in P]
        if len(pts)==2:
            ax.plot(*zip(*pts), lw=0.3+2.0*v/vmax, color=LIGHT, alpha=0.5)
        elif len(pts)>=3:
            for tri in [pts[:3]]:
                ax.add_collection3d(Poly3DCollection([tri], alpha=0.06,
                                    facecolor=ORANGE, edgecolor="none"))
    dmax = max(S[0].values())
    for l,(x,y,z) in P.items():
        ax.scatter([x],[y],[z], s=8+300*S[0].get(l,0)/dmax, color=BLUE,
                   edgecolor="w", linewidth=0.3, depthshade=False, zorder=5)
    ax.set_xlabel("mechanism order (units in the distinction)", labelpad=2, fontsize=6)
    ax.set_xticks([1,2,3,4]); ax.tick_params(labelsize=5.5, pad=0)
    ax.set_yticks([]); ax.set_zticks([])
    ax.set_title(title, loc="left", fontsize=7.5)
    ax.view_init(elev=18, azim=-62)
    ax.grid(False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.pane.set_visible(False)

axa = fig.add_subplot(g[0,0], projection="3d")
draw3d(axa, unit_phi(FULL[("inter","100mM NaCl")]["raw"]),
       f"a  100 mM NaCl (attractant)\n   Φ={FULL[('inter','100mM NaCl')]['Phi']:.0f}, "
       f"{FULL[('inter','100mM NaCl')]['nd']} distinctions, {FULL[('inter','100mM NaCl')]['nr']} relations")
axb = fig.add_subplot(g[0,1], projection="3d")
draw3d(axb, unit_phi(FULL[("inter","450mM NaCl")]["raw"]),
       f"b  450 mM NaCl (repellent)\n   Φ={FULL[('inter','450mM NaCl')]['Phi']:.0f}, "
       f"{FULL[('inter','450mM NaCl')]['nd']} distinctions, {FULL[('inter','450mM NaCl')]['nr']} relations")

# --- (c) what the labels are ---
axc = fig.add_subplot(g[0,2])
labs = sorted(set(FULL[("inter","100mM NaCl")]["raw"][0]), key=lambda x:(x.count("·"), x))
Mx = np.zeros((len(labs), 4))
for i,l in enumerate(labs):
    for j,nm in enumerate(INTER):
        if nm in l.split("·"): Mx[i,j]=1
axc.imshow(Mx, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
axc.set_xticks(range(4)); axc.set_xticklabels(INTER, rotation=90, fontsize=6)
axc.set_yticks(range(len(labs))); axc.set_yticklabels(labs, fontsize=5)
for i in range(len(labs)+1): axc.axhline(i-0.5, color="w", lw=0.5)
for j in range(5): axc.axvline(j-0.5, color="w", lw=0.5)
axc.set_title("c  The 15 distinction labels are the 15\n"
              "non-empty subsets of the same 4 neurons", loc="left")

# --- (d) between vs within (noise floor) ---
axd = fig.add_subplot(g[1,0])
iu = np.triu_indices(len(STIMULI),1)
for k,tag in enumerate(("inter","sens")):
    b = DM[tag][iu]; w = np.concatenate([NOISE[(tag,s)] for s in STIMULI])
    for off,(vals,c,lab) in enumerate([(w,GREY,"within-stimulus\n(same stimulus,\ndifferent animals)"),
                                        (b,ORANGE,"between-stimulus")]):
        x = k*2.4 + off*0.9
        axd.scatter(np.full(len(vals),x)+np.random.default_rng(1).normal(0,0.07,len(vals)),
                    vals, s=5, color=c, alpha=0.45, lw=0)
        axd.plot([x-0.28,x+0.28],[vals.mean()]*2, color="#222", lw=1.6, zorder=5)
axd.set_xticks([0.45, 2.85]); axd.set_xticklabels(["interneurons","sensory"], fontsize=7)
axd.set_ylabel("shape distance", labelpad=6)
h=[mpl.lines.Line2D([],[],marker="o",ls="",color=GREY,label="within-stimulus (noise floor)"),
   mpl.lines.Line2D([],[],marker="o",ls="",color=ORANGE,label="between-stimulus")]
axd.legend(handles=h, frameon=False, loc="lower center", fontsize=6, handletextpad=0.2)
axd.set_ylim(0.3, 2.3)
axd.set_title("d  Between-stimulus distances sit\nAT the within-stimulus noise floor", loc="left")

# --- (e,f) distance matrices, both substrates ---
oc = np.argsort([{"attractant":0,"repellent":1,"control":2}[c] for c in CLASSES])
for k,(tag,ttl) in enumerate([("inter","e  Interneurons"),("sens","f  Sensory")]):
    ax = fig.add_subplot(g[1,1+k])
    im = ax.imshow(DM[tag][np.ix_(oc,oc)], cmap="magma_r", vmin=0.5, vmax=2.0)
    ax.set_xticks(range(len(STIMULI)))
    ax.set_xticklabels([short[STIMULI[i]] for i in oc], rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(STIMULI)))
    ax.set_yticklabels([short[STIMULI[i]] for i in oc], fontsize=5.5)
    for b_ in [3.5,7.5]: ax.axhline(b_,color="w",lw=1.3); ax.axvline(b_,color="w",lw=1.3)
    cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
    cb.set_label("shape distance", fontsize=6)
    nfm = np.concatenate([NOISE[(tag,s)] for s in STIMULI]).mean()
    ax.set_title(f"{ttl}\n(noise floor = {nfm:.2f})", loc="left")

fig.savefig("figures/fig17_structures_and_labels.pdf", bbox_inches="tight")
fig.savefig("figures/fig17_structures_and_labels.png", dpi=200, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:6])

print("wrote figures/fig17_structures_and_labels.pdf")

# %% [markdown]
# ## Figure 14 — the result

# %%
iu=np.triu_indices(len(STIMULI),1)

fig = plt.figure(figsize=(11.0, 6.6))
g = fig.add_gridspec(2, 3, hspace=0.66, wspace=0.44)

# (a) Phi per stimulus, both substrates
ax = fig.add_subplot(g[0,0])
x=np.arange(len(STIMULI)); w=0.38
ax.bar(x-w/2, [FULL[("inter",s)]["Phi"] for s in STIMULI], w,
       color=[COL[c] for c in CLASSES], edgecolor="none")
ax.bar(x+w/2, [FULL[("sens",s)]["Phi"] for s in STIMULI], w,
       color=[COL[c] for c in CLASSES], alpha=0.45, edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels([short[s] for s in STIMULI], rotation=90, fontsize=5.5)
ax.set_ylabel("Φ", labelpad=6)
h=[mpl.patches.Patch(color="#555",label="interneurons"),
   mpl.patches.Patch(color="#555",alpha=0.45,label="sensory")]
ax.legend(handles=h, frameon=False, loc="upper right", fontsize=6)
ax.set_title("a  Φ varies widely, not by class", loc="left")

# (b,c) permutation nulls with the noise floor marked
for k,(tag,ttl) in enumerate([("inter","b  Interneurons"),("sens","c  Sensory")]):
    ax = fig.add_subplot(g[0,1+k])
    P=PERM[tag]
    ax.hist(P["null"], bins=45, color=LIGHT, edgecolor="none")
    ax.axvline(P["obs"], color=ORANGE, lw=2)
    side = "left" if P["obs"]>=0 else "right"
    ax.text(P["obs"]+(0.02 if P["obs"]>=0 else -0.02), ax.get_ylim()[1]*0.95,
            f"observed {P['obs']:+.3f}\np = {P['p']:.2f}", fontsize=6.5,
            color=ORANGE, va="top", ha=side)
    ax.set_xlabel("mean within-attractant − within-repellent", labelpad=6)
    ax.set_ylabel("shuffles", labelpad=6)
    ax.set_title(f"{ttl}: not significant", loc="left")

# (d) the decisive panel -- signal vs noise
ax = fig.add_subplot(g[1,0])
rngp=np.random.default_rng(1)
for k,tag in enumerate(("inter","sens")):
    b=DM[tag][iu]; wv=np.concatenate([NOISE[(tag,s)] for s in STIMULI])
    for off,(vals,c) in enumerate([(wv,GREY),(b,ORANGE)]):
        xx = k*2.3 + off*0.85
        ax.scatter(np.full(len(vals),xx)+rngp.normal(0,0.07,len(vals)), vals,
                   s=5, color=c, alpha=0.45, lw=0)
        ax.plot([xx-0.26,xx+0.26],[vals.mean()]*2, color="#222", lw=1.8, zorder=5)
    ax.text(k*2.3+0.42, 2.42, f"ratio {b.mean()/wv.mean():.2f}", ha="center", fontsize=6.5)
ax.set_xticks([0.42, 2.72]); ax.set_xticklabels(["interneurons","sensory"], fontsize=7)
ax.set_ylabel("shape distance", labelpad=6); ax.set_ylim(0.25, 2.6)
h=[mpl.lines.Line2D([],[],marker="o",ls="",color=GREY,label="within-stimulus (noise floor)"),
   mpl.lines.Line2D([],[],marker="o",ls="",color=ORANGE,label="between-stimulus (signal)")]
ax.legend(handles=h, frameon=False, loc="lower center", fontsize=6, handletextpad=0.2)
ax.set_title("d  There is no signal above the noise", loc="left")

# (e) jackknife, both substrates
ax = fig.add_subplot(g[1,1])
for k,tag in enumerate(("inter","sens")):
    sub=jackknife[jackknife.substrate==tag]
    ax.scatter(sub.contrast, np.arange(len(sub))+k*0.22-0.11,
               s=14, color=[ORANGE,BLUE][k], alpha=0.8, lw=0,
               label="interneurons" if k==0 else "sensory")
    ax.axvline(PERM[tag]["obs"], ls="--", lw=1, color=[ORANGE,BLUE][k])
ax.axvline(0, color="#333", lw=0.8)
ax.set_yticks(range(len(STIMULI)))
ax.set_yticklabels([f"−{short[s]}" for s in STIMULI], fontsize=6)
ax.invert_yaxis(); ax.set_xlabel("contrast, that stimulus dropped", labelpad=6)
ax.legend(frameon=False, loc="lower right", fontsize=6)
ax.set_title("e  Unstable to dropping one stimulus,\nand the sign disagrees across substrates", loc="left")

# (f) per-stimulus noise floor vs its mean between-stimulus distance
ax = fig.add_subplot(g[1,2])
for tag,mk in [("inter","o"),("sens","^")]:
    for i,s in enumerate(STIMULI):
        nf_ = np.mean(NOISE[(tag,s)])
        bw  = np.mean([DM[tag][i,j] for j in range(len(STIMULI)) if j!=i])
        ax.scatter([nf_],[bw], marker=mk, s=22, color=COL[CLASSES[i]],
                   alpha=0.85, lw=0)
lims=[0.7,1.75]
ax.plot(lims, lims, ls="--", lw=1, color="#333")
ax.text(1.58, 1.52, "y = x", fontsize=6, color="#333", rotation=38)
ax.set_xlim(*lims); ax.set_ylim(*lims)
ax.set_xlabel("within-stimulus noise floor", labelpad=6)
ax.set_ylabel("mean distance to other stimuli", labelpad=6)
h=[mpl.lines.Line2D([],[],marker="o",ls="",color="#555",label="interneurons"),
   mpl.lines.Line2D([],[],marker="^",ls="",color="#555",label="sensory")]
ax.legend(handles=h, frameon=False, loc="upper left", fontsize=6,
          title="colour = stimulus class", title_fontsize=6)
ax.set_title("f  Every stimulus sits on the diagonal", loc="left")

fig.savefig("figures/fig14_pooled_celegans.pdf", bbox_inches="tight")
fig.savefig("figures/fig14_pooled_celegans.png", dpi=200, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:6])

print("wrote figures/fig14_pooled_celegans.pdf")

# %% [markdown]
# ## What this shows
#
# 1. **There is no signal above the noise floor.** Comparing the same stimulus
#    against itself across different animals gives distances as large as
#    comparing different stimuli — signal-to-noise ratios of 1.06 and 0.96. This
#    is the finding that governs everything else. It is not that attractants and
#    repellents fail to separate; it is that **no pair of stimuli separates**
#    beyond what resampling the animals produces.
# 2. **The class test is null, as it must be.** Interneurons give a difference
#    of essentially zero; sensory neurons give a difference in the predicted
#    direction but far from significance. The sign disagrees between substrates
#    and is unstable to dropping any single stimulus.
# 3. **Φ does not track class either.** It varies several-fold across stimuli
#    with controls interleaved among the extremes.
#
# ## Where to go next — build intuition before adding power
#
# The distance is a new measure and its behaviour on real data is not yet
# understood. Adding statistical power to a measure whose noise properties are
# unknown would be premature. In rough order:
#
# 1. **Characterise the noise floor properly.** It is currently one number per
#    stimulus from 10 splits. How does it scale with the number of animals
#    pooled, with epoch length, with τ? A measure whose noise floor is
#    understood can be corrected for; one whose is not, cannot.
# 2. **Within-animal, within-stimulus variance.** Each animal sees each stimulus
#    3 times. Those 3 repeats give a variance estimate that pooling destroys.
#    It requires per-repeat structures, which the current sampling cannot
#    support — but it is the right quantity.
# 3. **A positive control.** Find *any* manipulation that this distance detects
#    reliably in these data — sleep versus wake, early versus late in the
#    session, one animal versus another. Without a positive control, a null on
#    the stimulus contrast is uninformative about the measure.
# 4. **More stimuli per class.** Only after the above. The binding constraint on
#    the class test is 4 stimuli per class = 6 within-class pairs, but more
#    pairs of an uninformative measure buys nothing.
