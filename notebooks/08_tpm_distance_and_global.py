# %% [markdown]
# # 08 — TPM distances, the smoothing problem, and a global-TPM alternative
#
# Three things this notebook settles, all raised as objections to `notebooks/06`:
#
# 1. **Compare the TPMs directly, before any Φ.** If the attractant / repellent
#    / control structure exists at all, it should be visible in the transition
#    matrices themselves. It is not.
# 2. **Quantify the smoothing.** Laplace smoothing on a 4-neuron TPM is not a
#    minor regularisation — it supplies **43% of the probability mass**. That is
#    a genuine violation of IIT's requirements, because the causal structure is
#    partly invented. Dropping to 3 or 2 neurons reduces it to 19% and 2%.
# 3. **The global-TPM alternative.** If anatomical connectivity is static, a
#    single TPM for the whole dataset is defensible, with stimuli distinguished
#    by *which state* they drive the system into. Requires a state criterion
#    other than "most frequent", since 0000 dominates everywhere.
#
# **Outputs:** `figures/fig19_tpm_distances_and_global.pdf` and six CSVs.

# %% [markdown]
# ## What notebook 06 does, restated precisely
#
# For the record, since this notebook departs from it:
#
# 1. Binarize four neurons; pack into a 4-bit integer state series.
# 2. For each stimulus, pool all 24 epochs (8 animals × 3 repeats) and tally
#    every state→state transition at lag τ = 1 sample into a 16 × 16 count
#    matrix. **One TPM per stimulus.**
# 3. Row-normalise with Laplace smoothing, α = 0.5.
# 4. Pick the state with the most transitions out of it — equivalently the
#    most-occupied state; both rules agree, and both give **0000** for all ten
#    stimuli.
# 5. Unfold the Φ-structure at that state and compare structures across stimuli.
#
# So the stimuli differ **only through their TPMs**, since the state is the same
# for all of them. That is worth naming: because *C. elegans* anatomical
# connectivity is static, a per-stimulus TPM is best read as an estimate of
# **dynamic effective connectivity** — how the causal influence among these four
# neurons is reconfigured by the chemical, not how the wiring changes. The
# Φ-structures then differ because the effective causal network differs. This is
# a coherent reading and consistent with the dynamic-effective-connectivity
# literature; it is also the reading under which step 5 makes sense.

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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphi
from pyphi import convert
from scipy.spatial.distance import jensenshannon

import ces_hypergraph as ch

pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False

BLUE, ORANGE, GREY, LIGHT, TEAL = "#1f6fb4", "#c2571a", "#8a8a8a", "#9bb8d4", "#008080"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False})

RECS = list(ch.HERM_DRIVE_IDS)
INTER = ["AIBL", "AVEL", "AVAL", "RIML"]
STIMULI = list(ch.STIMULUS_CLASS)
CLASSES = np.array([ch.STIMULUS_CLASS[s] for s in STIMULI])
COL = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}
short = {s: (s if len(s) <= 12 else s[:12]) for s in STIMULI}
n = len(STIMULI)

FS = ch.SAMPLING_RATE_HZ
WINDOW = round(300 * FS)
EPOCH_N = round(15 * FS)
TAU = 1

for rec in RECS:
    ch.ensure_recording(rec)


def load(rec, neurons):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    binary = [ch.moving_window_binarize(
        d.loc[d.neuron == nm, tc].iloc[0].astype(float).values, WINDOW) for nm in neurons]
    states = ch.combine_states(binary)
    onsets = defaultdict(list)
    for t, lab in ast.literal_eval(d.iloc[0]["stimulus"]):
        onsets[lab].append(int(t))
    return states, onsets


DATA = [load(r, INTER) for r in RECS]


def tpm(C, alpha=0.5):
    return (C + alpha) / (C + alpha).sum(1, keepdims=True)


def counts_occ(stim, neurons_data=None, K=16):
    C = np.zeros((K, K))
    occ = np.zeros(K)
    for states, onsets in (neurons_data or DATA):
        for t0 in onsets.get(stim, []):
            segment = states[t0:t0 + EPOCH_N]
            for s_ in segment:
                occ[s_] += 1
            for a, b in zip(segment[:-TAU], segment[TAU:]):
                C[a, b] += 1
    return C, occ


CNT = {s: counts_occ(s) for s in STIMULI}

agree = all(int(np.argmax(CNT[s][0].sum(1))) == int(np.argmax(CNT[s][1])) for s in STIMULI)
print(f"argmax(row sums) == argmax(occupancy) for every stimulus: {agree}")
print(f"selected state, all stimuli: "
      f"{sorted({format(int(np.argmax(CNT[s][1])), '04b') for s in STIMULI})}")

# %% [markdown]
# ## 1. Do the TPMs themselves separate the classes?
#
# Four distances between transition matrices. `_all` uses all 16 rows (so it
# includes smoothed, data-free rows); `_observed` restricts to rows where
# *both* stimuli actually observed transitions, which is the honest comparison.

# %%
def tpm_distance(Ca, Cb, kind, alpha=0.5):
    Pa, Pb = tpm(Ca, alpha), tpm(Cb, alpha)
    if kind == "L1_all":
        return np.abs(Pa - Pb).sum() / Pa.shape[0]
    if kind == "JS_all":
        return float(np.mean([jensenshannon(Pa[i], Pb[i], base=2) for i in range(Pa.shape[0])]))
    both = np.where((Ca.sum(1) > 0) & (Cb.sum(1) > 0))[0]
    if len(both) == 0:
        return np.nan
    if kind == "L1_observed":
        return np.abs(Pa[both] - Pb[both]).sum() / len(both)
    if kind == "JS_observed":
        return float(np.mean([jensenshannon(Pa[i], Pb[i], base=2) for i in both]))
    raise ValueError(kind)


def class_contrast(M, labels_):
    iu = np.triu_indices(len(labels_), 1)
    a = [M[i, j] for i, j in zip(*iu) if labels_[i] == labels_[j] == "attractant"]
    r = [M[i, j] for i, j in zip(*iu) if labels_[i] == labels_[j] == "repellent"]
    return np.mean(a) - np.mean(r), np.mean(a), np.mean(r)


def permutation_p(M, n_shuffles=20000, seed=0):
    obs = class_contrast(M, CLASSES)[0]
    rng = np.random.default_rng(seed)
    null = np.array([class_contrast(M, rng.permutation(CLASSES))[0] for _ in range(n_shuffles)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_shuffles + 1)


TD = {}
rows = []
for kind in ["L1_all", "JS_all", "L1_observed", "JS_observed"]:
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = tpm_distance(CNT[STIMULI[i]][0], CNT[STIMULI[j]][0], kind)
    TD[kind] = M
    obs, p = permutation_p(M)
    _, mean_a, mean_r = class_contrast(M, CLASSES)
    rows.append(dict(metric=kind, within_attractant=round(mean_a, 4),
                     within_repellent=round(mean_r, 4), diff=round(obs, 4),
                     p_two_sided=round(p, 4)))
tpm_test = pd.DataFrame(rows)
tpm_test.to_csv("results/tpm_level_comparison.csv", index=False)
print(tpm_test.to_string(index=False))
print("\nNo class structure in the TPMs themselves. The null is upstream of Φ:")
print("it is not that the Φ-structure comparison fails to find a difference,")
print("it is that the transition matrices do not differ by class either.")

# %% [markdown]
# ## 2. How much of the TPM is invented?
#
# Laplace smoothing replaces a data-free row with the uniform prior. The
# question is how much of the *whole* matrix is prior rather than data. For a
# K-state system with α = 0.5, row *a* receives αK = K/2 units of prior mass
# against `C[a,:].sum()` units of observed mass.

# %%
def smoothing_audit(neurons, alpha=0.5):
    K = 2 ** len(neurons)
    data = [load(r, neurons) for r in RECS]
    out = []
    for s in STIMULI:
        C, _ = counts_occ(s, data, K)
        invented = (alpha * K) / (C + alpha).sum(1)
        out.append(dict(stimulus=s, n_neurons=len(neurons), n_states=K,
                        unvisited_rows=int((C.sum(1) == 0).sum()),
                        pct_rows_prior_only=round(100 * (C.sum(1) == 0).sum() / K, 1),
                        mean_pct_mass_invented=round(100 * float(invented.mean()), 1),
                        transitions=int(C.sum()),
                        obs_per_param=round(C.sum() / K ** 2, 2)))
    return pd.DataFrame(out)


cov4 = smoothing_audit(INTER)
cov4.to_csv("results/smoothing_audit_4n.csv", index=False)
print(cov4[["stimulus", "unvisited_rows", "pct_rows_prior_only",
            "mean_pct_mass_invented", "obs_per_param"]].to_string(index=False))
print(f"\n4 neurons: mean {cov4.mean_pct_mass_invented.mean():.1f}% of TPM mass is prior, "
      f"{cov4.obs_per_param.mean():.2f} observations per parameter")

for nz in (INTER[:3], INTER[:2]):
    c = smoothing_audit(nz)
    print(f"{len(nz)} neurons ({2**len(nz)} states): "
          f"unvisited rows {c.unvisited_rows.min()}–{c.unvisited_rows.max()}, "
          f"mean invented {c.mean_pct_mass_invented.mean():.1f}%, "
          f"{c.obs_per_param.mean():.1f} obs/param")

# %% [markdown]
# **This is a real violation, not a technicality.** IIT computes cause–effect
# power from the TPM. Where the TPM is prior rather than data, the resulting
# distinctions and relations describe an assumption, not the system. At 4 neurons
# that is ~43% of the matrix. The reduction with substrate size is steep, so a
# smaller substrate is the direct remedy — at the cost of a smaller Φ-structure
# and no ability to detect higher-order relations among the neurons dropped.

# %% [markdown]
# ## 3. Does substrate size change the answer?

# %%
def canonical(ps, neurons):
    name = lambda mech: "·".join(neurons[u] for u in mech)
    return ({name(tuple(d.mechanism)): float(d.phi) for d in ps.distinctions},
            {frozenset(name(tuple(m)) for m in r.mechanisms): float(r.phi)
             for r in ps.relations})


def unit_phi(S):
    total = sum(S[0].values()) + sum(S[1].values())
    if total == 0:
        return S
    return ({k: v / total for k, v in S[0].items()},
            {k: v / total for k, v in S[1].items()})


def distance(A, B):
    return (sum(abs(A[0].get(k, 0.0) - B[0].get(k, 0.0)) for k in set(A[0]) | set(B[0]))
            + sum(abs(A[1].get(k, 0.0) - B[1].get(k, 0.0)) for k in set(A[1]) | set(B[1])))


def unfold_at(C, state_index, neurons, alpha=0.5):
    network = pyphi.Network(convert.state_by_state2state_by_node(tpm(C, alpha)),
                            node_labels=neurons)
    state = tuple((state_index >> i) & 1 for i in range(len(neurons)))
    return pyphi.new_big_phi.phi_structure(pyphi.Subsystem(network, state))


SUB = {}
for nz in (INTER[:2], INTER[:3], INTER):
    K = 2 ** len(nz)
    tag = f"{len(nz)}n"
    data = [load(r, nz) for r in RECS]
    shapes, meta = {}, []
    for s in STIMULI:
        C, _ = counts_occ(s, data, K)
        si = int(np.argmax(C.sum(1)))
        ps = unfold_at(C, si, nz)
        shapes[s] = unit_phi(canonical(ps, nz))
        meta.append(dict(stimulus=s, Phi=round(float(ps.big_phi), 3),
                         n_dist=len(ps.distinctions), n_rel=len(list(ps.relations)),
                         state=format(si, f"0{len(nz)}b")))
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = distance(shapes[STIMULI[i]], shapes[STIMULI[j]])
    obs, p = permutation_p(M)
    _, mean_a, mean_r = class_contrast(M, CLASSES)
    SUB[tag] = dict(M=M, meta=pd.DataFrame(meta), obs=obs, p=p, a=mean_a, r=mean_r,
                    neurons=list(nz))
    print(f"{tag} ({', '.join(nz)}): diff {obs:+.4f}  p = {p:.3f}  |  "
          f"Φ {SUB[tag]['meta'].Phi.min():.2f}–{SUB[tag]['meta'].Phi.max():.2f}  |  "
          f"states {sorted(set(SUB[tag]['meta'].state))}")

pd.DataFrame([{"n_neurons": len(SUB[t]["neurons"]), "neurons": "|".join(SUB[t]["neurons"]),
               "n_states": 2 ** len(SUB[t]["neurons"]),
               "within_attractant": round(SUB[t]["a"], 5),
               "within_repellent": round(SUB[t]["r"], 5),
               "diff": round(SUB[t]["obs"], 5), "p_two_sided": round(SUB[t]["p"], 5),
               "Phi_min": round(SUB[t]["meta"].Phi.min(), 3),
               "Phi_max": round(SUB[t]["meta"].Phi.max(), 3)}
              for t in ["2n", "3n", "4n"]]).to_csv("results/substrate_size.csv", index=False)

# %% [markdown]
# **Both smaller substrates give p ≈ 0.11 with the wrong sign for the
# hypothesis.** So reducing the invented mass from 43% to 2% does not reveal a
# hidden effect — it changes the numbers substantially (Φ falls from ~78–383 to
# ~2.7–3.6 at two neurons) without changing the conclusion.

# %% [markdown]
# ## 4. The global-TPM alternative
#
# If *C. elegans* anatomical connectivity dominates the causal Markov chain and
# is static, then a **single TPM for the whole dataset** is the more defensible
# object, and stimuli should be distinguished by **which state** they drive the
# system into rather than by having their own transition matrix.
#
# Two immediate advantages: the global TPM has ~156 observations per parameter
# and **no** unvisited rows, so the invented mass drops from 43% to 0.3%.
#
# The problem is the state criterion. 0000 dominates every stimulus *and* the
# baseline, so "most frequent" would assign every stimulus the same state and
# every distance would be zero. Instead: pick the state most **enriched during
# the stimulus relative to a non-stimulated baseline**, where baseline is the
# ~45 s between the end of one epoch and the next onset.

# %%
def global_tpm_and_occupancy(neurons):
    K = 2 ** len(neurons)
    data = [load(r, neurons) for r in RECS]
    C_global = np.zeros((K, K))
    occ_stim = {s: np.zeros(K) for s in STIMULI}
    occ_base = np.zeros(K)
    for states, onsets in data:
        for a, b in zip(states[:-TAU], states[TAU:]):
            C_global[a, b] += 1                     # the ENTIRE recording
        marks = sorted((t, lab) for lab, ts in onsets.items() for t in ts)
        for idx, (t0, lab) in enumerate(marks):
            for s_ in states[t0:t0 + EPOCH_N]:
                if lab in occ_stim:
                    occ_stim[lab][s_] += 1
            nxt = marks[idx + 1][0] if idx + 1 < len(marks) else len(states)
            for s_ in states[t0 + EPOCH_N:nxt]:     # inter-stimulus baseline
                occ_base[s_] += 1
    return C_global, occ_stim, occ_base


Cg, OS, OB = global_tpm_and_occupancy(INTER)
pb = OB / OB.sum()
print(f"global TPM: {int(Cg.sum())} transitions, "
      f"{int((Cg.sum(1) == 0).sum())} unvisited rows, {Cg.sum()/256:.0f} obs/param")
print(f"invented mass: {100 * (0.5 * 16) / (Cg + 0.5).sum(1).mean():.2f}%  "
      f"(per-stimulus TPMs: {cov4.mean_pct_mass_invented.mean():.0f}%)")
print("baseline occupancy, top 5:",
      [(format(i, '04b'), round(pb[i], 3)) for i in np.argsort(-pb)[:5]])


def enriched_states(stimulus, k=1, min_frames=10):
    """States most over-represented during the stimulus vs the baseline."""
    ps_ = OS[stimulus] / OS[stimulus].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2((ps_ + 1e-9) / (pb + 1e-9))
    candidates = np.where(OS[stimulus] >= min_frames)[0]
    order = candidates[np.argsort(-log_ratio[candidates])][:k]
    return [(int(i), float(log_ratio[i]), float(ps_[i])) for i in order]


dev = pd.DataFrame([{
    "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "dev_state": format(enriched_states(s)[0][0], "04b"),
    "log2_enrich": round(enriched_states(s)[0][1], 2),
    "stim_frac": round(enriched_states(s)[0][2], 3),
    "base_frac": round(float(pb[enriched_states(s)[0][0]]), 3),
    "n_frames": int(OS[s][enriched_states(s)[0][0]]),
} for s in STIMULI])
dev.to_csv("results/deviant_states.csv", index=False)
print(dev.to_string(index=False))
print(f"\ndistinct states selected: {sorted(set(dev.dev_state))}")
print(f"0000 ever selected? {'0000' in set(dev.dev_state)}   "
      f"1111 ever selected? {'1111' in set(dev.dev_state)}")
print("-> the criterion does what it was designed to: both baseline-dominant")
print("   states are eliminated, and the selected states differ across stimuli.")

# %%
netg = pyphi.Network(convert.state_by_state2state_by_node(tpm(Cg)), node_labels=INTER)
STRUCT_CACHE = {}


def global_structure(state_index):
    if state_index not in STRUCT_CACHE:
        state = tuple((state_index >> i) & 1 for i in range(4))
        ps = pyphi.new_big_phi.phi_structure(pyphi.Subsystem(netg, state))
        STRUCT_CACHE[state_index] = (unit_phi(canonical(ps, INTER)), float(ps.big_phi),
                                     len(ps.distinctions), len(list(ps.relations)))
    return STRUCT_CACHE[state_index]


global_meta = pd.DataFrame([{
    "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "state": dev[dev.stimulus == s].dev_state.iloc[0],
    "Phi": round(global_structure(int(dev[dev.stimulus == s].dev_state.iloc[0], 2))[1], 2),
    "n_dist": global_structure(int(dev[dev.stimulus == s].dev_state.iloc[0], 2))[2],
    "n_rel": global_structure(int(dev[dev.stimulus == s].dev_state.iloc[0], 2))[3],
} for s in STIMULI])
global_meta.to_csv("results/global_tpm_states.csv", index=False)
print(global_meta.to_string(index=False))

# %% [markdown]
# One structural limitation of the single-TPM approach, and it is not marginal:
# because the structure is a pure function of the state, **two stimuli that
# select the same state get distance exactly zero**. The cell below enumerates
# and classifies every such pair. Three stimuli land on `1101`, so there are 5
# zero-distance pairs, **3 of which are attractant/repellent cross-class pairs**
# — meaning 3 of the 24 cross-class pairs are forced to zero and the top-1
# contrast is partly an artefact of that collapse. A top-*k* profile removes the
# degeneracy by giving each stimulus a weighted mixture of its *k* most enriched
# states.

# %%
from collections import defaultdict
from itertools import combinations

by_state = defaultdict(list)
for _, row in global_meta.iterrows():
    by_state[str(row.state).zfill(4)].append((row.stimulus, row["class"]))

zero_pairs = pd.DataFrame([
    dict(state=st, stim_a=a, class_a=ca, stim_b=b, class_b=cb,
         pair_type="-".join(sorted([ca, cb])))
    for st, members in by_state.items()
    for (a, ca), (b, cb) in combinations(members, 2)
])
zero_pairs.to_csv("results/global_tpm_zero_pairs.csv", index=False)
print(zero_pairs.to_string(index=False))
print("\npair types:", zero_pairs.pair_type.value_counts().to_dict())
print(f"cross-class attractant/repellent pairs forced to zero: "
      f"{int((zero_pairs.pair_type == 'attractant-repellent').sum())} of 24")

# %%
def topk_matrix(k, min_frames=10):
    profiles = {s: enriched_states(s, k, min_frames) for s in STIMULI}
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            A_states, B_states = profiles[STIMULI[i]], profiles[STIMULI[j]]
            wa = np.array([w for _, w, _ in A_states]); wa = wa / wa.sum()
            wb = np.array([w for _, w, _ in B_states]); wb = wb / wb.sum()
            total = 0.0
            for (sa, _, _), pa_ in zip(A_states, wa):
                for (sb, _, _), pb_ in zip(B_states, wb):
                    total += pa_ * pb_ * distance(global_structure(sa)[0],
                                                  global_structure(sb)[0])
            M[i, j] = M[j, i] = total
    return M


global_rows = []
for k in (1, 2, 3):
    M = topk_matrix(k)
    obs, p = permutation_p(M)
    _, mean_a, mean_r = class_contrast(M, CLASSES)
    zeros = int(((M < 1e-12).sum() - n) / 2)
    global_rows.append(dict(k=k, within_attractant=round(mean_a, 4),
                            within_repellent=round(mean_r, 4), diff=round(obs, 4),
                            p_two_sided=round(p, 4), zero_distance_pairs=zeros))
    print(f"top-{k}: diff {obs:+.4f}  p = {p:.4f}  |  zero-distance pairs: {zeros}")
    if k == 3:
        Mtop3 = M
        res3 = dict(diff=obs, p=p)
print(f"\nunique states unfolded across all profiles: {len(STRUCT_CACHE)} of 16")

# %% [markdown]
# ## Exact brute force where it is computable
#
# The headline 4-neuron pipeline in `notebooks/06` does **not** minimise over
# bijections — 13–15 distinctions means up to 15! ≈ 1.3 × 10¹² mappings. It uses
# the identity correspondence, which is an **upper bound** on the exact distance.
#
# Three of the pipelines here are small enough to brute-force, so this cell runs
# the real minimisation on them and reports what the shortcut costs.
#
# Note the caching: exact distances are memoised **per state pair**, not per
# stimulus pair. Under the global TPM only ~14 distinct states are ever
# selected, so the same structure pair recurs many times across the top-*k*
# profiles; without caching the top-1 matrix alone takes ~344 s instead of ~94 s.

# %%
import math
import time
from gold_standard import gold_standard_distance


def identity_distance(A, B):
    return (sum(abs(A[0].get(k, 0.0) - B[0].get(k, 0.0)) for k in set(A[0]) | set(B[0]))
            + sum(abs(A[1].get(k, 0.0) - B[1].get(k, 0.0)) for k in set(A[1]) | set(B[1])))


_exact_cache = {}


def exact_between_states(sa, sb):
    """Exact min-over-bijections distance, memoised per STATE pair."""
    key = (min(sa, sb), max(sa, sb))
    if key not in _exact_cache:
        _exact_cache[key] = gold_standard_distance(global_structure(key[0])[0],
                                                   global_structure(key[1])[0])
    return _exact_cache[key]


exact_rows = []

# --- per-stimulus pipelines at 2 and 3 neurons -----------------------------
for nz in (INTER[:2], INTER[:3]):
    K = 2 ** len(nz)
    data = [load(r, nz) for r in RECS]
    shapes = {}
    for s in STIMULI:
        C, _ = counts_occ(s, data, K)
        shapes[s] = unit_phi(canonical(unfold_at(C, int(np.argmax(C.sum(1))), nz), nz))
    n_dist = len(shapes[STIMULI[0]][0])
    M_id = np.zeros((n, n))
    M_ex = np.zeros((n, n))
    t0 = time.perf_counter()
    for i in range(n):
        for j in range(i + 1, n):
            A, B = shapes[STIMULI[i]], shapes[STIMULI[j]]
            M_id[i, j] = M_id[j, i] = identity_distance(A, B)
            M_ex[i, j] = M_ex[j, i] = gold_standard_distance(A, B)
    elapsed = time.perf_counter() - t0
    obs_id, p_id = permutation_p(M_id)
    obs_ex, p_ex = permutation_p(M_ex)
    iu = np.triu_indices(n, 1)
    excess = M_id[iu] - M_ex[iu]
    exact_rows.append(dict(pipeline=f"{len(nz)}n_perstim", n_dist=n_dist,
                           bijections=math.factorial(n_dist),
                           secs_full_matrix=round(elapsed, 2),
                           identity_diff=round(obs_id, 4), identity_p=round(p_id, 4),
                           exact_diff=round(obs_ex, 4), exact_p=round(p_ex, 4),
                           mean_overestimate=round(float(excess.mean()), 4),
                           pct_pairs_identity_suboptimal=round(100 * float((excess > 1e-12).mean()), 1)))
    print(f"{len(nz)}n per-stimulus ({n_dist} distinctions, {math.factorial(n_dist):,} bijections): "
          f"identity {obs_id:+.4f} p={p_id:.3f}  ->  EXACT {obs_ex:+.4f} p={p_ex:.3f}   "
          f"[{elapsed:.1f}s]")

# --- global TPM, top-1 ----------------------------------------------------
profiles = {s: enriched_states(s, 1) for s in STIMULI}
M_id = np.zeros((n, n))
M_ex = np.zeros((n, n))
t0 = time.perf_counter()
for i in range(n):
    for j in range(i + 1, n):
        sa = profiles[STIMULI[i]][0][0]
        sb = profiles[STIMULI[j]][0][0]
        M_id[i, j] = M_id[j, i] = identity_distance(global_structure(sa)[0],
                                                     global_structure(sb)[0])
        M_ex[i, j] = M_ex[j, i] = exact_between_states(sa, sb)
elapsed = time.perf_counter() - t0
obs_id, p_id = permutation_p(M_id)
obs_ex, p_ex = permutation_p(M_ex)
iu = np.triu_indices(n, 1)
excess = M_id[iu] - M_ex[iu]
max_nd = max(global_structure(profiles[s][0][0])[2] for s in STIMULI)
exact_rows.append(dict(pipeline="global_top1", n_dist=max_nd,
                       bijections=math.factorial(max_nd),
                       secs_full_matrix=round(elapsed, 2),
                       identity_diff=round(obs_id, 4), identity_p=round(p_id, 4),
                       exact_diff=round(obs_ex, 4), exact_p=round(p_ex, 4),
                       mean_overestimate=round(float(excess.mean()), 4),
                       pct_pairs_identity_suboptimal=round(100 * float((excess > 1e-12).mean()), 1)))
print(f"global TPM top-1 (max {max_nd} distinctions, {math.factorial(max_nd):,} bijections): "
      f"identity {obs_id:+.4f} p={p_id:.3f}  ->  EXACT {obs_ex:+.4f} p={p_ex:.3f}   [{elapsed:.1f}s]")

exact_vs_identity = pd.DataFrame(exact_rows)
exact_vs_identity.to_csv("results/exact_vs_identity.csv", index=False)
print()
print(exact_vs_identity.to_string(index=False))
print(f"\n4n per-stimulus: 13-15 distinctions -> up to {math.factorial(15):,} bijections; "
      f"not computed, distances there remain upper bounds.")
print("\nIn all three computable cases the exact minimisation moves p FURTHER from")
print("significance and preserves the sign, so the null is not an artefact of")
print("using a single bijection. But the bound is loose: the identity map is")
print(f"suboptimal on {exact_vs_identity.pct_pairs_identity_suboptimal.min():.0f}-"
      f"{exact_vs_identity.pct_pairs_identity_suboptimal.max():.0f}% of pairs.")

# %% [markdown]
# ## All approaches side by side

# %%
approaches = pd.DataFrame([
    dict(approach="per-stimulus TPM, argmax occupancy (4n)",
         invented_mass_pct=round(cov4.mean_pct_mass_invented.mean(), 1),
         diff=round(SUB["4n"]["obs"], 4), p=round(SUB["4n"]["p"], 4)),
    dict(approach="per-stimulus TPM, argmax occupancy (3n)",
         invented_mass_pct=round(smoothing_audit(INTER[:3]).mean_pct_mass_invented.mean(), 1),
         diff=round(SUB["3n"]["obs"], 4), p=round(SUB["3n"]["p"], 4)),
    dict(approach="per-stimulus TPM, argmax occupancy (2n)",
         invented_mass_pct=round(smoothing_audit(INTER[:2]).mean_pct_mass_invented.mean(), 1),
         diff=round(SUB["2n"]["obs"], 4), p=round(SUB["2n"]["p"], 4)),
] + [dict(approach=f"global TPM, top-{r['k']} enriched state(s)",
          invented_mass_pct=round(100 * (0.5 * 16) / (Cg + 0.5).sum(1).mean(), 2),
          diff=r["diff"], p=r["p_two_sided"]) for r in global_rows])
approaches.to_csv("results/approach_comparison.csv", index=False)
print(approaches.to_string(index=False))
print(f"\nSix approaches spanning {approaches.invented_mass_pct.min():.1f}%–"
      f"{approaches.invented_mass_pct.max():.1f}% invented mass. "
      f"Smallest p = {approaches.p.min():.2f}.")
print(f"Sign: {(approaches['diff'] < 0).sum()} negative (the predicted direction), "
      f"{(approaches['diff'] > 0).sum()} positive.")

# %% [markdown]
# ## Figure 19

# %%
oc = np.argsort([{"attractant":0,"repellent":1,"control":2}[c] for c in CLASSES])

fig = plt.figure(figsize=(11.2, 7.6))
g = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.44)

# (a) TPM-level distance matrix -- does class structure appear BEFORE Phi?
ax = fig.add_subplot(g[0,0])
M = TD["JS_observed"]
im = ax.imshow(M[np.ix_(oc,oc)], cmap="magma_r")
ax.set_xticks(range(n)); ax.set_xticklabels([short[STIMULI[i]] for i in oc], rotation=90, fontsize=5.5)
ax.set_yticks(range(n)); ax.set_yticklabels([short[STIMULI[i]] for i in oc], fontsize=5.5)
for b in [3.5,7.5]: ax.axhline(b,color="w",lw=1.3); ax.axvline(b,color="w",lw=1.3)
cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
cb.set_label("Jensen–Shannon (observed rows)", fontsize=6)
r_ = tpm_test[tpm_test.metric=="JS_observed"].iloc[0]
ax.set_title(f"a  TPM distances, no Φ involved\n   diff {r_['diff']:+.3f}, p = {r_['p_two_sided']:.2f}", loc="left")

# (b) how much of each TPM is invented by smoothing
ax = fig.add_subplot(g[0,1])
xs = np.arange(n)
ax.bar(xs, cov4.mean_pct_mass_invented, 0.62, color=[COL[c] for c in CLASSES])
ax.axhline(cov4.mean_pct_mass_invented.mean(), ls="--", lw=1, color="#333")
ax.text(n-0.4, cov4.mean_pct_mass_invented.mean()+2.2, f"mean {cov4.mean_pct_mass_invented.mean():.0f}%",
        fontsize=6, ha="right", va="bottom", color="#333")
ax.set_ylabel("% of TPM mass from the prior", labelpad=7)
ax.set_xticks(xs); ax.set_xticklabels([short[s] for s in STIMULI], rotation=90, fontsize=5.5)
ax.set_ylabel("% of TPM mass from the prior", labelpad=5)
ax.set_ylim(0, 70)
ax.set_title("b  4 neurons: the prior supplies\n   ~43% of the TPM", loc="left")

# (c) substrate size vs invented mass
ax = fig.add_subplot(g[0,2])
sizes=[2,3,4]; inv=[2.0,18.6,43.0]
ax.plot(sizes, inv, "o-", color=ORANGE, lw=1.4, ms=5)
for x_,y_ in zip(sizes,inv):
    ax.annotate(f"{y_:.0f}%", xy=(x_,y_), xytext=(0,7), textcoords="offset points",
                ha="center", fontsize=6.2, color=ORANGE)
ax2 = ax.twinx()
unv=[0,2,5]
ax2.bar(sizes, unv, 0.4, color=LIGHT, alpha=0.75, zorder=0)
ax2.set_ylabel("max unvisited rows", color="#4a6f96", labelpad=4, fontsize=7)
ax2.tick_params(axis="y", colors="#4a6f96", labelsize=6)
ax2.set_ylim(0, 14)
ax.set_xticks(sizes); ax.set_xlabel("neurons in the substrate", labelpad=5)
ax.set_ylabel("% TPM mass invented", labelpad=5, color=ORANGE)
ax.tick_params(axis="y", colors=ORANGE)
ax.set_ylim(0, 58); ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
ax.set_title("c  Fewer neurons, less invention", loc="left")

# (d) enriched-state selection: log2 enrichment per stimulus, with 0000/1111 shown
ax = fig.add_subplot(g[1,0])
xs = np.arange(n)
ax.bar(xs, dev.log2_enrich, 0.62, color=[COL[c] for c in CLASSES])
for xi, st_ in zip(xs, dev.dev_state):
    ax.text(xi, 0.08, st_, rotation=90, ha="center", va="bottom",
            fontsize=5.6, color="w", weight="bold")
ax.axhline(0, color="#333", lw=0.9)
# where 0000 and 1111 would land
e0 = np.array([np.log2(((OS[s]/OS[s].sum())[0]+1e-9)/(pb[0]+1e-9)) for s in STIMULI])
e15 = np.array([np.log2(((OS[s]/OS[s].sum())[15]+1e-9)/(pb[15]+1e-9)) for s in STIMULI])
ax.plot(xs, e0, "x", ms=5, color=TEAL, lw=0)
ax.plot(xs, e15, "+", ms=5, color="#7a4fa3", lw=0)
h=[mpl.lines.Line2D([],[],marker="x",ls="",color=TEAL,label="0000"),
   mpl.lines.Line2D([],[],marker="+",ls="",color="#7a4fa3",label="1111")]
ax.legend(handles=h, frameon=False, loc="lower left", fontsize=6,
          ncol=2, handletextpad=0.1, columnspacing=0.8, borderaxespad=0.15,
          title="where these land", title_fontsize=6)
ax.set_xticks(xs); ax.set_xticklabels([short[s] for s in STIMULI], rotation=90, fontsize=5.5)
ax.set_ylabel("log$_2$ enrichment vs baseline", labelpad=5)
ax.set_ylim(-1.55, 2.9)
ax.set_title("d  Selected state = most enriched vs\n   baseline; 0000 and 1111 never win", loc="left")

# (e) global-TPM distance matrix
ax = fig.add_subplot(g[1,1])
im = ax.imshow(Mtop3[np.ix_(oc,oc)], cmap="magma_r")
ax.set_xticks(range(n)); ax.set_xticklabels([short[STIMULI[i]] for i in oc], rotation=90, fontsize=5.5)
ax.set_yticks(range(n)); ax.set_yticklabels([short[STIMULI[i]] for i in oc], fontsize=5.5)
for b in [3.5,7.5]: ax.axhline(b,color="w",lw=1.3); ax.axvline(b,color="w",lw=1.3)
cb=fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6)
cb.set_label("shape distance", fontsize=6)
ax.set_title(f"e  ONE global TPM, top-3 enriched states\n"
             f"   diff {res3["diff"]:+.3f}, p = {res3["p"]:.2f}", loc="left")

# (f) all six approaches
ax = fig.add_subplot(g[1,2])
lbl = ["4n per-stim","3n per-stim","2n per-stim","global top-1","global top-2","global top-3"]
y = np.arange(len(lbl))
diffs = approaches["diff"].values
ax.barh(y, diffs, 0.58, color=[ORANGE if d<0 else LIGHT for d in diffs])
for yi,(d,p_,iv) in enumerate(zip(diffs, approaches.p.values, approaches.invented_mass_pct.values)):
    ax.text(max(d, 0.0) + 0.015, yi, f"p={p_:.2f}   ({iv:.1f}% invented)",
            va="center", ha="left", fontsize=5.8, color="#333")
ax.axvline(0, color="#333", lw=0.9)
ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(-0.30, 0.86)
for yi, L in zip(y, lbl):
    ax.text(-0.29, yi, L, va="center", ha="left", fontsize=6.2, color="#333")
ax.set_xlabel("mean within-attractant − within-repellent", labelpad=5)
ax.set_title("f  Six approaches, all null.\n   Hypothesis predicts negative", loc="left")

fig.savefig("figures/fig19_tpm_distances_and_global.pdf", bbox_inches="tight")
fig.savefig("figures/fig19_tpm_distances_and_global.png", dpi=200, bbox_inches="tight")
r2=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r2)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:16],b.get_text()[:16]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:8])

print("wrote figures/fig19_tpm_distances_and_global.pdf")

# %% [markdown]
# ## What this settles
#
# 1. **The null is upstream of Φ.** The TPMs themselves do not separate the
#    classes (p = 0.63–0.93 across four TPM distances). Whatever is wrong is not
#    a failure of the Φ-structure distance to see a real difference.
# 2. **Smoothing is a genuine problem at 4 neurons** — 43% of the probability
#    mass is prior, not data. It falls to 19% at 3 neurons and 2% at 2 neurons.
#    Reducing it does not reveal an effect: both smaller substrates give
#    p ≈ 0.11 with the sign *opposite* to the hypothesis.
# 3. **The global-TPM approach is viable and cleaner.** One TPM over the whole
#    dataset has 156 observations per parameter, no unvisited rows, and 0.3%
#    invented mass. With states chosen by enrichment against a non-stimulated
#    baseline, both 0000 and 1111 are correctly eliminated and stimuli receive
#    genuinely different states. It is also null (p = 0.74–0.97).
# 4. **Six approaches, invented mass from 0.3% to 43%, all null**, sign split
#    3–3. Combined with the noise floor in `notebooks/06`, the reading is that
#    at this data volume the measure cannot resolve stimulus identity at all —
#    not that attractants and repellents are known to be indistinguishable.
