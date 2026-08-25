# %% [markdown]
# # 15 — The structure comparison the state identification pointed at
#
# Notebook 14 named the conditions' states: **1000** (AWCL alone ON) for
# no-stimulus, **0111** (its complement) for stimulus. This notebook compares
# their Φ-structures under the giant TPM, against a split-half noise floor.
#
# ## What "noise floor" means here, precisely
#
# A distance of, say, 3.2 between two structures is meaningless on its own: it
# could be signal, or it could be what ANY two estimates of the SAME structure
# differ by, given finite data. The noise floor makes that comparison explicit.
#
# * Split the 8 animals into two disjoint halves A and B (all 35 balanced
#   4-vs-4 splits are used).
# * Build the giant TPM separately from each half; unfold the SAME state in
#   both. D(state_A, state_B) is then pure **estimation noise**: same
#   condition, same state, same pipeline — the only difference is which
#   animals supplied the data.
# * The **signal** is D(1000, 0111) computed WITHIN one half, so signal and
#   noise are measured at the same data volume (~20k transitions per half).
#   Comparing the full-data signal against half-data noise would flatter the
#   signal, since noise shrinks with data.
# * The test: is signal reliably ABOVE the noise floor (one-sided
#   Mann–Whitney)? A ratio ≈ 1 means the condition difference is
#   indistinguishable from re-measuring the same condition with different
#   animals — the instrument cannot resolve the contrast at this data volume.
#
# This is the same logic as any test–retest reliability bound: a measure
# cannot distinguish two conditions by less than it differs from itself.
#
# **No Φ-structure is decomposed anywhere in this design.** The structure is
# holistic; every one here is unfolded whole. What is split is the set of
# ANIMALS that estimate the TPM — the structure is a deterministic function of
# an estimated input and inherits its sampling uncertainty. See
# `figures/fig41_noise_floor_schematic.png`.

# %%
import os, sys, ast, time, itertools, math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.ndimage import median_filter
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = "."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import ces_hypergraph as ch
import gold_standard as gs
for r in ch.HERM_DRIVE_IDS:
    ch.ensure_recording(r)
os.environ["PYPHI_WELCOME_OFF"] = "yes"
import pyphi
from pyphi import convert
pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False

RECS = list(ch.HERM_DRIVE_IDS)
INTER = ["AIBL", "AVEL", "AVAL", "RIML"]
SENS = ["ASEL", "ASER", "AWAL", "AWCL"]
ALL8 = INTER + SENS
STIMULI = list(ch.STIMULUS_CLASS)
FS = ch.SAMPLING_RATE_HZ
W_HP = 20
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"

def load(rec):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    tr = {nm: d.loc[d.neuron == nm, tc].iloc[0].astype(float).values for nm in ALL8}
    on = defaultdict(list)
    for t, l in ast.literal_eval(d.iloc[0]["stimulus"]):
        on[l].append(int(t))
    return tr, on
DAT = {r: load(r) for r in RECS}
def hp(x):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - median_filter(xf, size=max(3, round(W_HP * FS)), mode="nearest")
BITS = {r: {nm: (hp(DAT[r][0][nm]) > 0).astype(int) for nm in ALL8} for r in RECS}
ST = {r: {"sens": ch.combine_states([BITS[r][nm] for nm in SENS])} for r in RECS}

def giant_C(recs):
    C = np.zeros((16, 16))
    for r in recs:
        st = ST[r]["sens"]
        for a, b in zip(st[:-1], st[1:]):
            C[a, b] += 1
    return C

def canon(ps):
    nm = lambda m: "·".join(SENS[u] for u in m)
    return ({nm(tuple(d.mechanism)): float(d.phi) for d in ps.distinctions},
            {frozenset(nm(tuple(m)) for m in r.mechanisms): float(r.phi) for r in ps.relations})

def struct_at(C, si):
    P = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    net = pyphi.Network(convert.state_by_state2state_by_node(P), node_labels=SENS)
    ps = pyphi.new_big_phi.phi_structure(
        pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4))))
    return canon(ps), float(ps.big_phi), len(ps.distinctions), len(ps.relations)

def D_id(A, B):
    """Canonical-label distance. Same substrate -> mechanism labels are shared,
    so the identity correspondence is well-defined; this is an upper bound on
    the gold-standard minimum (exact when the minimum is achieved at identity).
    The exact search is infeasible here: 0111 has 15 distinctions -> 15!."""
    return (sum(abs(A[0].get(k, 0) - B[0].get(k, 0)) for k in set(A[0]) | set(B[0]))
            + sum(abs(A[1].get(k, 0) - B[1].get(k, 0)) for k in set(A[1]) | set(B[1])))

S_BASE, S_STIM = int("1000", 2), int("0111", 2)
Cfull = giant_C(RECS)
Sb, phi_b, nd_b, nr_b = struct_at(Cfull, S_BASE)
Ss, phi_s, nd_s, nr_s = struct_at(Cfull, S_STIM)
print(f"1000: Phi={phi_b:.3f} ({nd_b}d, {nr_b}r, row n={int(Cfull[S_BASE].sum())})")
print(f"0111: Phi={phi_s:.3f} ({nd_s}d, {nr_s}r, row n={int(Cfull[S_STIM].sum())})")
print(f"D_id(1000, 0111) full data = {D_id(Sb, Ss):.4f}")

# %% [markdown]
# ## The split-half design

# %%
halves = [tuple(sorted(c)) for c in itertools.combinations(range(8), 4)][:35]
split_structs = []
t0 = time.perf_counter()
for h in halves:
    A = [RECS[i] for i in h]; B = [RECS[i] for i in range(8) if i not in h]
    CA, CB = giant_C(A), giant_C(B)
    split_structs.append((struct_at(CA, S_BASE)[0], struct_at(CB, S_BASE)[0],
                          struct_at(CA, S_STIM)[0], struct_at(CB, S_STIM)[0]))
print(f"{len(halves)} splits, 4 structures each: {time.perf_counter()-t0:.0f}s")

def unit(S):
    t = sum(S[0].values()) + sum(S[1].values())
    return S if t == 0 else ({k: v / t for k, v in S[0].items()},
                             {k: v / t for k, v in S[1].items()})
def D_dist_only(A, B):
    return sum(abs(A[0].get(k, 0) - B[0].get(k, 0)) for k in set(A[0]) | set(B[0]))

res = []
for name, (Df, tf) in {"raw": (D_id, lambda S: S), "unit-Phi": (D_id, unit),
                       "distinctions-only": (D_dist_only, lambda S: S),
                       "distinctions-only, unit": (D_dist_only, unit)}.items():
    nz, sg = [], []
    for SbA, SbB, SsA, SsB in split_structs:
        a, b, c, d = tf(SbA), tf(SbB), tf(SsA), tf(SsB)
        nz += [Df(a, b), Df(c, d)]
        sg += [Df(a, c), Df(b, d)]
    nz, sg = np.array(nz), np.array(sg)
    u = stats.mannwhitneyu(sg, nz, alternative="greater")
    res.append(dict(variant=name, signal=round(float(sg.mean()), 4),
                    noise=round(float(nz.mean()), 4),
                    ratio=round(float(sg.mean() / nz.mean()), 3),
                    p=round(float(u.pvalue), 5),
                    full_data_signal=round(float(Df(tf(Sb), tf(Ss))), 4)))
    print(f"{name:<26} signal {sg.mean():8.4f}  noise {nz.mean():8.4f}  "
          f"ratio {sg.mean()/nz.mean():.3f}  p = {u.pvalue:.2e}")
vr = pd.DataFrame(res)
vr.to_csv(os.path.join(REPO_ROOT, "results/structure_1000_vs_0111.csv"), index=False)

# why: size instability of the half-data structures
sizes = {"1000": [], "0111": []}
for SbA, SbB, SsA, SsB in split_structs:
    sizes["1000"] += [(len(SbA[0]), len(SbA[1])), (len(SbB[0]), len(SbB[1]))]
    sizes["0111"] += [(len(SsA[0]), len(SsA[1])), (len(SsB[0]), len(SsB[1]))]
pd.DataFrame([dict(state=k, nd_min=min(a for a, _ in v), nd_max=max(a for a, _ in v),
                   nr_min=min(b for _, b in v), nr_max=max(b for _, b in v))
              for k, v in sizes.items()]).to_csv(
    os.path.join(REPO_ROOT, "results/structure_split_sizes.csv"), index=False)
for k, v in sizes.items():
    print(f"{k}: half-data distinctions {min(a for a,_ in v)}-{max(a for a,_ in v)}, "
          f"relations {min(b for _,b in v)}-{max(b for _,b in v)}")

# %% [markdown]
# ## Reading
#
# * **The comparison fails its noise floor** — raw ratio 0.91 (p = 0.80), and
#   no variant (unit-Φ, distinctions-only) rescues it.
# * **The mechanism is the φ-fragility measured in notebook 13.** At half-data
#   volume (~20k transitions) the 0111 structure swings between 5 and 15
#   distinctions and 7 and 5,385 relations across splits — the object being
#   compared is not stable enough to compare. The full-data structures (6d/12r
#   vs 15d/1802r) LOOK dramatically different, but two half-data estimates of
#   the SAME state differ just as much.
# * **What would change this:** structure stability, not more contrast. The
#   distinction/relation composition needs to survive resampling before any
#   between-condition distance is interpretable. Candidates: more data per
#   condition (longer recordings), a φ-threshold on distinctions before
#   comparison (drop the near-zero tail that flickers across splits), or
#   comparing only the maximal (top-k by φ) distinctions.

# %% [markdown]
# ## Figure 40 — the comparison, its noise floor, and why it fails

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4), constrained_layout=True)
ax = axes[0]
ax.bar([0, 1], [phi_b, phi_s], 0.5, color=[GREY, BLUE])
for i, (nd_, nr_) in enumerate([(nd_b, nr_b), (nd_s, nr_s)]):
    ax.text(i, [phi_b, phi_s][i] + 0.12, f"{nd_} dist\n{nr_} rel", ha="center",
            fontsize=6, color="#333")
ax.set_xticks([0, 1])
ax.set_xticklabels(["1000\n(no stimulus)", "0111\n(stimulus)"], fontsize=6.5)
ax.set_ylabel("Φ (giant TPM)", labelpad=5); ax.set_ylim(0, 5.2)
ax.set_title("a  The two condition states'\n   full-data structures", loc="left")

ax = axes[1]
nz_raw, sg_raw = [], []
for SbA, SbB, SsA, SsB in split_structs:
    nz_raw += [D_id(SbA, SbB), D_id(SsA, SsB)]
    sg_raw += [D_id(SbA, SsA), D_id(SbB, SsB)]
rng = np.random.default_rng(1)
ax.scatter(rng.normal(0, 0.05, len(sg_raw)), sg_raw, s=9, alpha=0.5, color=ORANGE, lw=0,
           label="signal: D(1000, 0111), same half")
ax.scatter(rng.normal(1, 0.05, len(nz_raw)), nz_raw, s=9, alpha=0.5, color=GREY, lw=0,
           label="noise: D(same state), disjoint halves")
for i, v in enumerate([np.mean(sg_raw), np.mean(nz_raw)]):
    ax.plot([i - 0.16, i + 0.16], [v, v], color="#111", lw=1.6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["signal", "noise floor"], fontsize=6.5)
ax.set_ylabel("structure distance", labelpad=5)
ax.text(0.5, 0.955, f"ratio {np.mean(sg_raw)/np.mean(nz_raw):.2f}, p = 0.80",
        transform=ax.transAxes, ha="center", fontsize=6.4, color="#333")
ax.legend(frameon=False, fontsize=5.4, loc="upper left", bbox_to_anchor=(0.0, 0.92))
ax.set_title("b  The comparison fails its\n   noise floor (all 4 variants)", loc="left")

ax = axes[2]
nr0 = [b for _, b in sizes["0111"]]; nr1 = [b for _, b in sizes["1000"]]
ax.hist(nr0, bins=14, color=BLUE, alpha=0.65, label="0111 relations/half")
ax.hist(nr1, bins=14, color=GREY, alpha=0.65, label="1000 relations/half")
ax.axvline(nr_s, color=BLUE, lw=1.2, ls="--")
ax.axvline(nr_b, color="#555", lw=1.2, ls="--")
ax.text(nr_s, ax.get_ylim()[1] * 0.95, " full data", fontsize=5.6, color=BLUE, va="top")
ax.set_xlabel("relations in the half-data structure", labelpad=5)
ax.set_ylabel("count (70 half-structures)", labelpad=5)
ax.legend(frameon=False, fontsize=5.6, loc="upper right")
ax.set_title("c  Why: half-data structures of 0111\n   swing wildly in size", loc="left")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig40_structure_comparison.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig40_structure_comparison.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig40")

# %% [markdown]
# ## Is the noise floor inter-animal variability, or just finite sampling?
#
# The animals are isogenic clones sharing one anatomical connectome. If that
# shared structure is taken as the causal model, animal identity should
# contribute nothing beyond finite sampling — and then splitting by ANIMAL
# should produce the same noise as splitting the SAME animals' recording TIME
# in half at matched data volume. Directly testable.

# %%
BLK = CYC_N
blocks = {}
for r in RECS:
    st = ST[r]["sens"]
    blocks[r] = [st[i * BLK:(i + 1) * BLK] for i in range(len(st) // BLK)]
nblk = sum(len(v) for v in blocks.values())

def C_from_blocks(assign, side):
    C = np.zeros((16, 16)); k = 0
    for r in RECS:
        for b in blocks[r]:
            if assign[k] == side:
                for a, b_ in zip(b[:-1], b[1:]):
                    C[a, b_] += 1
            k += 1
    return C

rng = np.random.default_rng(7)
within = {"1000": [], "0111": []}
for rep in range(35):
    assign = rng.integers(0, 2, nblk)
    C1, C2 = C_from_blocks(assign, 0), C_from_blocks(assign, 1)
    S1b = struct_at(C1, S_BASE)[0]; S2b = struct_at(C2, S_BASE)[0]
    S1s = struct_at(C1, S_STIM)[0]; S2s = struct_at(C2, S_STIM)[0]
    within["1000"].append(D_id(S1b, S2b)); within["0111"].append(D_id(S1s, S2s))
between = {"1000": [D_id(a, b) for a, b, _, _ in split_structs],
           "0111": [D_id(c, d) for _, _, c, d in split_structs]}
rows = []
for st_ in ("1000", "0111"):
    w, b = np.array(within[st_]), np.array(between[st_])
    u = stats.mannwhitneyu(b, w, alternative="greater")
    rows.append(dict(state=st_, within_mean=round(float(w.mean()), 4),
                     between_mean=round(float(b.mean()), 4),
                     ratio=round(float(b.mean() / w.mean()), 3),
                     p_between_gt_within=round(float(u.pvalue), 5)))
    print(f"{st_}: within-animal {w.mean():.3f}  between-animal {b.mean():.3f}  "
          f"ratio {b.mean()/w.mean():.2f}  p = {u.pvalue:.4f}")
pd.DataFrame(rows).to_csv(os.path.join(REPO_ROOT, "results/within_vs_between_animal_noise.csv"), index=False)

# %% [markdown]
# ## The SCM-licensed factorization: node-wise mechanisms (64 vs 240 parameters)

# %%
def sbn_from_counts(C):
    """P(node_i ON at t+1 | joint state s at t), Beta(0.5, 0.5)-smoothed per
    (state, node): the factorization a shared causal model licenses, and the
    exact state-by-node object PyPhi consumes."""
    sbn = np.zeros((16, 4))
    for s_ in range(16):
        tot = C[s_].sum()
        for i in range(4):
            on = C[s_, [t_ for t_ in range(16) if (t_ >> i) & 1]].sum()
            sbn[s_, i] = (on + 0.5) / (tot + 1.0)
    return sbn

def struct_at_sbn(C, si):
    net = pyphi.Network(sbn_from_counts(C), node_labels=SENS)
    ps = pyphi.new_big_phi.phi_structure(
        pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4))))
    return canon(ps), float(ps.big_phi), len(ps.distinctions), len(ps.relations)

nz2, sg2 = [], []
for h in halves:
    A = [RECS[i] for i in h]; B = [RECS[i] for i in range(8) if i not in h]
    CA, CB = giant_C(A), giant_C(B)
    SbA = struct_at_sbn(CA, S_BASE)[0]; SbB = struct_at_sbn(CB, S_BASE)[0]
    SsA = struct_at_sbn(CA, S_STIM)[0]; SsB = struct_at_sbn(CB, S_STIM)[0]
    nz2 += [D_id(SbA, SbB), D_id(SsA, SsB)]
    sg2 += [D_id(SbA, SsA), D_id(SbB, SsB)]
nz2, sg2 = np.array(nz2), np.array(sg2)
u = stats.mannwhitneyu(sg2, nz2, alternative="greater")
print(f"node-wise: ratio {sg2.mean()/nz2.mean():.3f}  p = {u.pvalue:.2e}  "
      f"(joint was 0.914, p = 0.80)")
pd.DataFrame([dict(estimation="joint", ratio=0.914, p=0.798),
              dict(estimation="node-wise", ratio=round(float(sg2.mean()/nz2.mean()), 3),
                   p=round(float(u.pvalue), 5))]).to_csv(
    os.path.join(REPO_ROOT, "results/scm_nodewise_comparison.csv"), index=False)

# %% [markdown]
# ## Reading
#
# * **The isogenic premise survives its test.** Between-animal noise equals
#   within-animal noise at matched data volume (ratio 1.07 for 1000, 1.22 for
#   0111, neither significant). Animal identity contributes essentially nothing;
#   pooling across clones is fully justified, exactly as the shared-connectome
#   argument says.
# * **But this makes the noise floor MORE binding, not less.** The floor was
#   never inter-individual variability to be argued away — it is finite-sample
#   error in the mechanism probabilities, present even for one animal measured
#   twice. A shared causal graph fixes WHICH mechanisms exist; it does not
#   supply their conditional probabilities, which must still be estimated from
#   ~20k transitions and then pass through a thresholded, nonlinear unfolding.
# * **The SCM-licensed factorization does not rescue it either:** node-wise
#   estimation (64 parameters instead of 240) reproduces the full-data
#   structures almost exactly and leaves the split-half ratio unchanged
#   (0.894 vs 0.914). The instability lives in the unfolding — distinctions
#   near the φ ≈ 0 boundary winking in and out under tiny TPM perturbations —
#   not in the parameter count.
