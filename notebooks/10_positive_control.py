# %% [markdown]
# # 10 — Does the distance detect anything? Noise floors and a positive control
#
# Three questions, in the order that makes them informative:
#
# 1. **Is the global TPM better?** It is far better *conditioned* (0.3% invented
#    mass against 43%). Does that translate into a better signal-to-noise ratio?
# 2. **Can the distance detect a large, unambiguous contrast?** Not attractant
#    vs repellent, but **chemical present vs chemical absent** — using the 45 s
#    of inter-stimulus baseline the other notebooks discard. This is a *positive
#    control*: if the measure cannot resolve this, no null on the class contrast
#    is interpretable.
# 3. **Is the 3-neuron global TPM the best configuration?** It is the only one
#    that is well-conditioned, IIT-legitimate, *and* exactly computable.
#
# The answers, in short: (1) no — the global TPM is **worse**; (2) **no** — the
# positive control fails; (3) it is the best-conditioned pipeline in the repo and
# also null.
#
# **Outputs:** `figures/fig22_control_and_snr.pdf`,
# `results/noise_floor_global.csv`, `results/noise_floor_all_pipelines.csv`,
# `results/positive_control.csv`, `results/global_3n.csv`

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
import itertools
import math
import time
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphi
from pyphi import convert
from scipy import stats

import ces_hypergraph as ch
from gold_standard import gold_standard_distance

pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False

BLUE, ORANGE, GREY, LIGHT = "#1f6fb4", "#c2571a", "#8a8a8a", "#9bb8d4"
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "pdf.fonttype": 42,
                     "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False})

RECS = list(ch.HERM_DRIVE_IDS)
INTER = ["AIBL", "AVEL", "AVAL", "RIML"]
INTER3 = INTER[:3]
SENS = ["ASEL", "ASER", "AWAL", "AWCL"]
ALL8 = INTER + SENS
STIMULI = list(ch.STIMULUS_CLASS)
CLASSES_OF = ch.STIMULUS_CLASS
CLASS_LIST = ["attractant", "repellent", "control"]
CLASSES = np.array([CLASSES_OF[s] for s in STIMULI])
n = len(STIMULI)

FS = ch.SAMPLING_RATE_HZ
WINDOW = round(300 * FS)
EPOCH_N = round(15 * FS)
TAU = 1

for rec in RECS:
    ch.ensure_recording(rec)


def binary_states(rec, neurons):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    binary = [ch.moving_window_binarize(
        d.loc[d.neuron == nm, tc].iloc[0].astype(float).values, WINDOW) for nm in neurons]
    states = ch.combine_states(binary)
    onsets = defaultdict(list)
    for t, lab in ast.literal_eval(d.iloc[0]["stimulus"]):
        onsets[lab].append(int(t))
    return states, onsets


BS4 = {r: binary_states(r, INTER) for r in RECS}
BS3 = {r: binary_states(r, INTER3) for r in RECS}


def tpm(C, alpha=0.5):
    return (C + alpha) / (C + alpha).sum(1, keepdims=True)


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
    """Identity-correspondence distance (an upper bound; see notebooks/08)."""
    return (sum(abs(A[0].get(k, 0.0) - B[0].get(k, 0.0)) for k in set(A[0]) | set(B[0]))
            + sum(abs(A[1].get(k, 0.0) - B[1].get(k, 0.0)) for k in set(A[1]) | set(B[1])))


def structure_factory(C, neurons):
    """Returns f(state_index) -> unit-Φ structure, memoised."""
    network = pyphi.Network(convert.state_by_state2state_by_node(tpm(C)),
                            node_labels=neurons)
    cache = {}

    def f(state_index):
        if state_index not in cache:
            state = tuple((state_index >> i) & 1 for i in range(len(neurons)))
            ps = pyphi.new_big_phi.phi_structure(pyphi.Subsystem(network, state))
            cache[state_index] = unit_phi(canonical(ps, neurons))
        return cache[state_index]
    return f


def class_contrast(M, labels_):
    iu = np.triu_indices(len(labels_), 1)
    a = [M[i, j] for i, j in zip(*iu) if labels_[i] == labels_[j] == "attractant"]
    r = [M[i, j] for i, j in zip(*iu) if labels_[i] == labels_[j] == "repellent"]
    return np.mean(a) - np.mean(r)


def permutation_p(M, n_shuffles=20000, seed=0):
    obs = class_contrast(M, CLASSES)
    rng = np.random.default_rng(seed)
    null = np.array([class_contrast(M, rng.permutation(CLASSES)) for _ in range(n_shuffles)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_shuffles + 1)


HALVES = [tuple(sorted(c)) for c in itertools.combinations(range(8), 4)][:10]
print(f"{len(HALVES)} disjoint 4-vs-4 animal splits for every split-half estimate")

# %% [markdown]
# ## 1. Global TPM: better conditioned, but is it better?
#
# The global TPM is built from the **entire** recording of each animal, and
# stimuli are distinguished by the state most enriched during their epochs
# relative to the inter-stimulus baseline (see `notebooks/08`). The split-half
# noise floor rebuilds *both* the TPM and the state selection from each half, so
# it captures the full estimation error, not just the state choice.

# %%
def global_parts(recs, neurons, K):
    C_global = np.zeros((K, K))
    occ_stim = {s: np.zeros(K) for s in STIMULI}
    occ_base = np.zeros(K)
    src = BS4 if len(neurons) == 4 else BS3
    for r in recs:
        states, onsets = src[r]
        for a, b in zip(states[:-TAU], states[TAU:]):
            C_global[a, b] += 1
        marks = sorted((t, lab) for lab, ts in onsets.items() for t in ts)
        for idx, (t0, lab) in enumerate(marks):
            for s_ in states[t0:t0 + EPOCH_N]:
                if lab in occ_stim:
                    occ_stim[lab][s_] += 1
            nxt = marks[idx + 1][0] if idx + 1 < len(marks) else len(states)
            for s_ in states[t0 + EPOCH_N:nxt]:
                occ_base[s_] += 1
    return C_global, occ_stim, occ_base


def enriched(occ_stim, occ_base, stimulus, k=1, min_frames=10):
    pb = occ_base / occ_base.sum()
    ps_ = occ_stim[stimulus] / occ_stim[stimulus].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log2((ps_ + 1e-9) / (pb + 1e-9))
    cand = np.where(occ_stim[stimulus] >= min_frames)[0]
    if len(cand) == 0:
        return []
    return [(int(i), float(lr[i])) for i in cand[np.argsort(-lr[cand])][:k]]


t0 = time.perf_counter()
noise_global = {s: [] for s in STIMULI}
for h in HALVES:
    A = [RECS[i] for i in h]
    B = [RECS[i] for i in range(8) if i not in h]
    CA, OSA, OBA = global_parts(A, INTER, 16)
    CB, OSB, OBB = global_parts(B, INTER, 16)
    fA, fB = structure_factory(CA, INTER), structure_factory(CB, INTER)
    for s in STIMULI:
        ea, eb = enriched(OSA, OBA, s), enriched(OSB, OBB, s)
        if ea and eb:
            noise_global[s].append(distance(fA(ea[0][0]), fB(eb[0][0])))
print(f"global-TPM split-half noise floor: {time.perf_counter()-t0:.0f}s")

pd.DataFrame([dict(stimulus=s, cls=CLASSES_OF[s], n_splits=len(noise_global[s]),
                   noise_mean=round(float(np.mean(noise_global[s])), 4),
                   noise_sd=round(float(np.std(noise_global[s])), 4))
              for s in STIMULI]).to_csv("results/noise_floor_global.csv", index=False)

CgF, OSF, OBF = global_parts(RECS, INTER, 16)
fF = structure_factory(CgF, INTER)
sel4 = {s: enriched(OSF, OBF, s)[0][0] for s in STIMULI}
Mg = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        Mg[i, j] = Mg[j, i] = distance(fF(sel4[STIMULI[i]]), fF(sel4[STIMULI[j]]))
iu = np.triu_indices(n, 1)
between_g = Mg[iu]
within_g = np.concatenate([noise_global[s] for s in STIMULI])
print(f"global 4n : between {between_g.mean():.4f}  within {within_g.mean():.4f}  "
      f"ratio {between_g.mean()/within_g.mean():.3f}")
print("per-stim 4n: between 1.2198  within 1.1539  ratio 1.057   (results/noise_floor.csv)")
print("\n=> the better-conditioned pipeline has a LOWER noise floor but its")
print("   between-stimulus distances fall further still. Conditioning does not help.")

# %% [markdown]
# ## Panel-d helper: AWAL class PSTH (see notebooks/09 for the full analysis)

# %%
PRE_N, POST_N = round(5.0 * FS), round(25.0 * FS)
tvec = (np.arange(PRE_N + POST_N) - PRE_N) / FS


def _raw(rec, neurons):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    out = {nm: d.loc[d.neuron == nm, tc].iloc[0].astype(float).values for nm in neurons}
    onsets = defaultdict(list)
    for t, lab in ast.literal_eval(d.iloc[0]["stimulus"]):
        onsets[lab].append(int(t))
    return out, onsets


RAW = {r: _raw(r, ALL8) for r in RECS}


def psth_class(neuron, cls, recs):
    rows = []
    for r in recs:
        trace, onsets = RAW[r][0][neuron], RAW[r][1]
        for s in STIMULI:
            if CLASSES_OF[s] != cls:
                continue
            for t0 in sorted(onsets.get(s, [])):
                a, b = t0 - PRE_N, t0 + POST_N
                if a < 0 or b > len(trace):
                    continue
                seg = trace[a:b].astype(float)
                f0 = np.nanmean(seg[:PRE_N])
                if not np.isfinite(f0) or abs(f0) < 1e-9:
                    continue
                rows.append((seg - f0) / abs(f0))
    M = np.array(rows)
    return dict(mean=np.nanmean(M, 0), sem=np.nanstd(M, 0, ddof=1) / np.sqrt(len(M)), n=len(M))


# %% [markdown]
# ## 2. The positive control
#
# Every other analysis in this repo throws away the ~45 s between the end of one
# stimulus epoch and the next onset — **three times more data than it keeps**.
# That discarded window is a stimulus-absent condition, and comparing it to the
# stimulus-present condition is a far larger contrast than attractant vs
# repellent.
#
# The design is matched so that signal and noise are measured on the same
# footing:
#
# * **signal** = *D*(stimulus, baseline) computed **within** one half of the
#   animals — a real condition difference, estimated from 4 animals;
# * **noise** = *D*(stimulus, stimulus) and *D*(baseline, baseline) **across** the
#   two halves — the same condition, different animals, so pure sampling error.
#
# If signal does not exceed noise, the measure cannot detect the presence of a
# chemical, and no null on a *subtler* contrast can be interpreted.

# %%
def condition_counts(recs, which, neurons, K):
    """which='stim' -> the 15 s epochs; 'base' -> the inter-epoch remainder."""
    C = np.zeros((K, K))
    n_frames = 0
    src = BS4 if len(neurons) == 4 else BS3
    for r in recs:
        states, onsets = src[r]
        marks = sorted((t, lab) for lab, ts in onsets.items() for t in ts)
        for idx, (t0, lab) in enumerate(marks):
            nxt = marks[idx + 1][0] if idx + 1 < len(marks) else len(states)
            seg = states[t0:t0 + EPOCH_N] if which == "stim" else states[t0 + EPOCH_N:nxt]
            if len(seg) <= TAU:
                continue
            for a, b in zip(seg[:-TAU], seg[TAU:]):
                C[a, b] += 1
            n_frames += len(seg)
    return C, n_frames


def condition_structure(C, neurons):
    """Unfold at the most-occupied state of that condition."""
    return structure_factory(C, neurons)(int(np.argmax(C.sum(1))))


Cs, ns_ = condition_counts(RECS, "stim", INTER, 16)
Cb, nb_ = condition_counts(RECS, "base", INTER, 16)
print(f"stimulus windows: {int(Cs.sum())} transitions, {ns_} frames, "
      f"{int((Cs.sum(1)==0).sum())} unvisited rows")
print(f"baseline windows: {int(Cb.sum())} transitions, {nb_} frames, "
      f"{int((Cb.sum(1)==0).sum())} unvisited rows   "
      f"({nb_/ns_:.1f}x more data than the epochs)")
d_full = distance(condition_structure(Cs, INTER), condition_structure(Cb, INTER))
print(f"full-data D(stimulus, baseline) = {d_full:.4f}")

control_rows = []
for neurons, K, tag in [(INTER, 16, "4 neurons"), (INTER3, 8, "3 neurons")]:
    signal, noise = [], []
    for h in HALVES:
        A = [RECS[i] for i in h]
        B = [RECS[i] for i in range(8) if i not in h]
        sA = condition_structure(condition_counts(A, "stim", neurons, K)[0], neurons)
        sB = condition_structure(condition_counts(B, "stim", neurons, K)[0], neurons)
        bA = condition_structure(condition_counts(A, "base", neurons, K)[0], neurons)
        bB = condition_structure(condition_counts(B, "base", neurons, K)[0], neurons)
        signal += [distance(sA, bA), distance(sB, bB)]
        noise += [distance(sA, sB), distance(bA, bB)]
    signal, noise = np.array(signal), np.array(noise)
    u = stats.mannwhitneyu(signal, noise, alternative="greater")
    d = ((signal.mean() - noise.mean())
         / np.sqrt((signal.var(ddof=1) + noise.var(ddof=1)) / 2))
    control_rows.append(dict(substrate=tag,
                             signal_mean=round(float(signal.mean()), 4),
                             signal_sd=round(float(signal.std(ddof=1)), 4),
                             noise_mean=round(float(noise.mean()), 4),
                             noise_sd=round(float(noise.std(ddof=1)), 4),
                             ratio=round(float(signal.mean() / noise.mean()), 3),
                             p_signal_gt_noise=round(float(u.pvalue), 5),
                             cohens_d=round(float(d), 3),
                             full_data_distance=round(d_full, 4) if tag == "4 neurons" else np.nan))
    print(f"{tag}: signal {signal.mean():.4f}  noise {noise.mean():.4f}  "
          f"ratio {signal.mean()/noise.mean():.3f}  p = {u.pvalue:.4f}  d = {d:.2f}")
    if tag == "4 neurons":
        sig4, noise4 = signal, noise
    else:
        sig3, noise3 = signal, noise

pd.DataFrame(control_rows).to_csv("results/positive_control.csv", index=False)
print("\n=> THE POSITIVE CONTROL FAILS on both substrates. Chemical present vs")
print("   chemical absent -- a contrast the raw traces resolve at d = 0.72 in")
print("   AWAL (notebooks/09) -- is invisible to the Phi-structure distance.")

# %% [markdown]
# ## 3. The 3-neuron global TPM
#
# The untried combination: 8 states, 64 parameters, ~620 observations per
# parameter, 0.08% invented mass, and few enough distinctions that the **exact**
# minimisation runs in seconds.

# %%
Cg3, OS3, OB3 = global_parts(RECS, INTER3, 8)
f3 = structure_factory(Cg3, INTER3)
sel3 = {s: enriched(OS3, OB3, s)[0][0] for s in STIMULI}
nd3 = {format(v, "03b"): len(f3(v)[0]) for v in sorted(set(sel3.values()))}
print(f"3n global TPM: {int(Cg3.sum())} transitions, "
      f"{int((Cg3.sum(1)==0).sum())} unvisited rows, {Cg3.sum()/64:.0f} obs/param, "
      f"invented mass {100*(0.5*8)/(Cg3+0.5).sum(1).mean():.2f}%")
print(f"selected states: {sorted(nd3)}  distinctions: {nd3}  "
      f"-> {math.factorial(max(nd3.values()))} bijections at worst")

_exact3 = {}


def exact3(a, b):
    key = (min(a, b), max(a, b))
    if key not in _exact3:
        _exact3[key] = gold_standard_distance(f3(key[0]), f3(key[1]))
    return _exact3[key]


M_id3 = np.zeros((n, n))
M_ex3 = np.zeros((n, n))
t0 = time.perf_counter()
for i in range(n):
    for j in range(i + 1, n):
        M_id3[i, j] = M_id3[j, i] = distance(f3(sel3[STIMULI[i]]), f3(sel3[STIMULI[j]]))
        M_ex3[i, j] = M_ex3[j, i] = exact3(sel3[STIMULI[i]], sel3[STIMULI[j]])
elapsed3 = time.perf_counter() - t0
obs_id3, p_id3 = permutation_p(M_id3)
obs_ex3, p_ex3 = permutation_p(M_ex3)
print(f"3n global: identity {obs_id3:+.4f} p={p_id3:.3f}  |  "
      f"EXACT {obs_ex3:+.4f} p={p_ex3:.3f}   [{elapsed3:.1f}s]")

noise3_pipeline = {s: [] for s in STIMULI}
for h in HALVES:
    A = [RECS[i] for i in h]
    B = [RECS[i] for i in range(8) if i not in h]
    CA, OSA, OBA = global_parts(A, INTER3, 8)
    CB, OSB, OBB = global_parts(B, INTER3, 8)
    fa, fb = structure_factory(CA, INTER3), structure_factory(CB, INTER3)
    for s in STIMULI:
        ea, eb = enriched(OSA, OBA, s), enriched(OSB, OBB, s)
        if ea and eb:
            noise3_pipeline[s].append(distance(fa(ea[0][0]), fb(eb[0][0])))
within3 = np.concatenate([noise3_pipeline[s] for s in STIMULI])
between3 = M_id3[iu]

pd.DataFrame([dict(pipeline="3n_global", n_states=8,
                   obs_per_param=round(float(Cg3.sum() / 64), 1),
                   invented_mass_pct=round(100 * (0.5 * 8) / (Cg3 + 0.5).sum(1).mean(), 2),
                   unvisited_rows=int((Cg3.sum(1) == 0).sum()),
                   max_distinctions=max(nd3.values()),
                   bijections=math.factorial(max(nd3.values())),
                   identity_diff=round(obs_id3, 4), identity_p=round(p_id3, 4),
                   exact_diff=round(obs_ex3, 4), exact_p=round(p_ex3, 4),
                   exact_secs=round(elapsed3, 1))]).to_csv("results/global_3n.csv", index=False)

pipelines = pd.DataFrame([
    dict(pipeline="per-stimulus TPM, 4n", between=1.2198, within=1.1539, ratio=1.057,
         invented_pct=43.0),
    dict(pipeline="global TPM, 4n", between=round(float(between_g.mean()), 4),
         within=round(float(within_g.mean()), 4),
         ratio=round(float(between_g.mean() / within_g.mean()), 3), invented_pct=0.32),
    dict(pipeline="global TPM, 3n", between=round(float(between3.mean()), 4),
         within=round(float(within3.mean()), 4),
         ratio=round(float(between3.mean() / within3.mean()), 3), invented_pct=0.08),
])
pipelines.to_csv("results/noise_floor_all_pipelines.csv", index=False)
print()
print(pipelines.to_string(index=False))
print("\n=> the BETTER-conditioned the pipeline, the WORSE its signal-to-noise.")

# %% [markdown]
# ## Figure 22

# %%
ev = pd.read_csv("results/exact_vs_identity.csv")

fig = plt.figure(figsize=(11.4, 6.8))
g = fig.add_gridspec(2, 3, hspace=0.68, wspace=0.42)

# (a) raw-trace class discrimination per neuron
ax = fig.add_subplot(g[0,0])
aa2 = pd.read_csv("results/psth_class_test.csv")
aa2 = aa2[aa2.scope=="all_animals"].set_index("neuron").reindex(ALL8)
y = np.arange(len(ALL8))
cols = [ORANGE if p<0.05 else LIGHT for p in aa2.p_holm]
ax.barh(y, aa2.cohens_d, 0.62, color=cols)
for yi,(d_,p_,nm) in enumerate(zip(aa2.cohens_d, aa2.p_holm, ALL8)):
    if p_ < 0.05:
        ax.text(d_ + (0.04 if d_>0 else -0.04), yi, "*", va="center",
                ha="left" if d_>0 else "right", fontsize=10, color=ORANGE)
ax.axvline(0, color="#333", lw=0.9)
ax.set_yticks(y); ax.set_yticklabels(ALL8, fontsize=6.4)
for tick, nm in zip(ax.get_yticklabels(), ALL8):
    tick.set_color("#333" if nm in INTER else "#7a4fa3")
ax.invert_yaxis()
ax.set_xlabel("Cohen's d, attractant − repellent\n(mean ΔF/F$_0$ per epoch)", labelpad=5)
ax.set_xlim(-0.65, 0.95)
ax.set_title("a  Raw traces DO discriminate:\n   AWAL d=0.72, AIBL d=−0.41 (Holm *)", loc="left")

# (b) signal-to-noise across the three Phi pipelines
ax = fig.add_subplot(g[0,1])
nfa = pipelines
x = np.arange(len(nfa)); w = 0.36
ax.bar(x-w/2, nfa.between, w, color=ORANGE, label="between-stimulus")
ax.bar(x+w/2, nfa.within, w, color=GREY, label="within-stimulus (noise)")
for xi, r_ in zip(x, nfa.ratio):
    ax.text(xi, max(nfa.between[xi], nfa.within[xi])+0.05, f"ratio\n{r_:.2f}",
            ha="center", va="bottom", fontsize=6, color="#333")
ax.set_xticks(x); ax.set_xticklabels(["per-stim\n4n","global\n4n","global\n3n"], fontsize=6.4)
ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2, borderaxespad=0.1)
ax.set_ylim(0,1.85)
ax.set_title("b  No pipeline clears its own\n   noise floor (all ratios ≤ 1.06)", loc="left")

# (c) the positive control
ax = fig.add_subplot(g[0,2])
rng2 = np.random.default_rng(1)
for i,(s_,nz2,lab) in enumerate([(sig4,noise4,"4 neurons"), (sig3,noise3,"3 neurons")]):
    for j,(vals,col,nm) in enumerate([(s_,ORANGE,"signal"),(nz2,GREY,"noise")]):
        xs = i*1.0 + (j-0.5)*0.34 + rng2.normal(0,0.035,len(vals))
        ax.scatter(xs, vals, s=13, color=col, alpha=0.75, lw=0)
        ax.hlines(vals.mean(), i+ (j-0.5)*0.34 -0.13, i+(j-0.5)*0.34+0.13, color="#222", lw=1.4)
    ax.text(i, 1.74, f"ratio {s_.mean()/nz2.mean():.2f}\np = "
            f"{(stats.mannwhitneyu(s_,nz2,alternative='greater').pvalue):.2f}",
            ha="center", va="top", fontsize=6.2, color="#333")
ax.set_xticks([0,1]); ax.set_xticklabels(["4 neurons","3 neurons"], fontsize=6.6)
ax.set_ylabel("shape distance", labelpad=5); ax.set_ylim(0, 1.78)
h=[mpl.lines.Line2D([],[],marker="o",ls="",color=ORANGE,label="stimulus vs baseline (signal)"),
   mpl.lines.Line2D([],[],marker="o",ls="",color=GREY,label="same condition, other animals (noise)")]
ax.legend(handles=h, frameon=False, fontsize=5.8, loc="upper center",
          bbox_to_anchor=(0.5,-0.16), handletextpad=0.2)
ax.set_title("c  POSITIVE CONTROL FAILS:\n   chemical present vs absent invisible", loc="left")

# (d) AWAL -- the clearest single-neuron effect
ax = fig.add_subplot(g[1,0])
for cls,col in [("attractant",BLUE),("repellent",ORANGE),("control",GREY)]:
    p = psth_class("AWAL", cls, RECS)
    ax.fill_between(tvec, p["mean"]-p["sem"], p["mean"]+p["sem"], color=col, alpha=0.22, lw=0)
    ax.plot(tvec, p["mean"], color=col, lw=1.3, label=f"{cls} (n={p['n']})")
ax.axvspan(0,15,color="#000",alpha=0.05,lw=0,zorder=0)
ax.axhline(0,color="#666",lw=0.6,ls=":"); ax.axvline(0,color="#333",lw=0.7)
ax.set_xlabel("time from onset (s)", labelpad=5); ax.set_ylabel("ΔF/F$_0$", labelpad=5)
ax.legend(frameon=False, fontsize=6, loc="upper right")
ax.set_title("d  AWAL, all 8 animals:\n   clean attractant selectivity", loc="left")

# (e) invented mass vs signal-to-noise -- conditioning does not rescue it
ax = fig.add_subplot(g[1,1])
inv = list(pipelines.invented_pct.values)
ax.scatter(inv, nfa.ratio, s=48, color=ORANGE, zorder=3)
for xi,yi,lab in zip(inv, nfa.ratio, ["per-stim 4n","global 4n","global 3n"]):
    ax.annotate(lab, xy=(xi,yi), xytext=(6,5), textcoords="offset points", fontsize=6, color="#333")
ax.axhline(1.0, ls="--", lw=1, color="#333")
ax.text(0.06, 1.02, "ratio = 1: no signal", fontsize=6, color="#333", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("% of TPM mass invented by the prior", labelpad=5)
ax.set_ylabel("between / within ratio", labelpad=5)
ax.set_ylim(0.2, 1.25); ax.set_xlim(0.04, 90)
ax.set_title("e  Better conditioning makes it\n   WORSE, not better", loc="left")

# (f) what the exact minimisation costs across pipelines
ax = fig.add_subplot(g[1,2])
ev = pd.read_csv("results/exact_vs_identity.csv")
labels = ["per-stim 2n","per-stim 3n","global 4n\ntop-1","global 3n"]
ident = list(ev.identity_p.values) + [p_id3]
exact = list(ev.exact_p.values) + [p_ex3]
xs = np.arange(len(labels)); w=0.36
ax.bar(xs-w/2, ident, w, color=LIGHT, label="identity bijection")
ax.bar(xs+w/2, exact, w, color=ORANGE, label="exact minimisation")
ax.set_ylim(0,1.12)
ax.axhline(0.05, ls="--", lw=1, color="#333")
ax.text(len(labels)-0.4, 0.075, "p = 0.05", fontsize=6, color="#333", ha="right")
ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("permutation p", labelpad=5); ax.set_ylim(0,1.0)
ax.legend(frameon=False, fontsize=6, loc="upper left")
ax.set_title("f  Exact distance moves every p\n   FURTHER from significance", loc="left")

fig.savefig("figures/fig22_control_and_snr.pdf", bbox_inches="tight")
fig.savefig("figures/fig22_control_and_snr.png", dpi=200, bbox_inches="tight")
r_=fig.canvas.get_renderer()
tx=[(t,t.get_window_extent(r_)) for t in fig.findobj(mpl.text.Text) if t.get_text().strip() and t.get_visible()]
tls={a_:set(a_.get_xticklabels()+a_.get_yticklabels()) for a_ in fig.axes}
print("overlaps:",[(a.get_text()[:14],b.get_text()[:14]) for i,(a,ba) in enumerate(tx) for b,bb in tx[i+1:]
                   if ba.overlaps(bb) and not any(a in s2 and b in s2 for s2 in tls.values())][:6])

print("wrote figures/fig22_control_and_snr.pdf")

# %% [markdown]
# ## What this settles
#
# 1. **The global TPM is better conditioned but has worse signal-to-noise.** Its
#    noise floor drops from 1.15 to 0.64, but its between-stimulus distances drop
#    further, from 1.22 to 0.38 — ratio **0.60** against 1.06. The 3-neuron
#    global TPM is worse still at **0.33**. Better conditioning concentrates the
#    structures rather than separating them.
# 2. **The positive control fails.** Chemical present vs chemical absent gives
#    ratio **0.95** on both substrates (p = 0.75 and 0.38). The raw traces resolve
#    the *harder* attractant-vs-repellent contrast at d = 0.72 in AWAL
#    (`notebooks/09`), so this is not an absence of signal in the data — it is an
#    absence of sensitivity in the pipeline.
# 3. **Therefore the class null is uninformative about biology.** An instrument
#    that cannot detect the presence of a chemical cannot be used to conclude
#    that two chemical classes evoke similar Φ-structures. Everything from
#    `notebooks/06` onward should be read as characterising the *measure*, not
#    the worm.
# 4. **The 3-neuron global TPM is the best-conditioned configuration in the
#    repo** — 620 observations per parameter, 0.08% invented mass, exact distance
#    in ~2 s — and it is also null. That is worth knowing: the failure is not
#    fixable by conditioning or by substrate size at this recording length and
#    sampling rate.
#
# ## What would change the answer
#
# The binding constraint is per-structure estimation error, and it is not
# addressed by more stimuli (the design already detects a 14% class difference in
# principle) or by better TPM conditioning (shown above). It requires either much
# longer recordings per condition, a finer state space than binary thresholding
# provides, or a faster indicator. Until a manipulation exists that this distance
# *can* detect, further hypothesis tests on these data are not interpretable.
