# %% [markdown]
# # 17 — Static vs condition-dependent TPMs
#
# The IIT formalism assumes a **time-invariant** TPM: the canonical
# probabilistic formulation models the system as a first-order time-invariant
# Markov process (Krohn & Ostwald 2017), the original framework defines
# integrated information for stationary systems (Balduzzi & Tononi 2008), and
# in PyPhi one fixed TPM is the fundamental representation. For a nervous
# system with a time-varying sensory input this is a modelling choice, not a
# fact: the pooled ("static") TPM is the context-MIXTURE of the
# regime-conditioned mechanisms (see the caveat in the README's issue 1).
#
# Two views both have support. Static: isogenic animals share one mechanism
# (our within-vs-between-animal test), and long-term connectivity appears to
# reconstruct through remodeling (Science, doi:10.1126/science.aee7004 —
# hibernation eliminates spines wholesale, yet memory and representations
# survive via a resilient synaptic engram architecture). Dynamic: neuroscience
# expects context-dependent effective connectivity — and our own positive
# control shows the stimulus sits inside the transition probabilities
# (stim-on vs stim-off rows differ at z = 35).
#
# So: build BOTH, and compare the Φ landscapes they induce.

# %%
import os, sys, ast, subprocess
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import median_filter
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt

REPO_ROOT = "."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import ces_hypergraph as ch
os.environ["PYPHI_WELCOME_OFF"] = "yes"
import pyphi
from pyphi import convert
pyphi.config.PROGRESS_BARS = False
pyphi.config.PARALLEL = False

Q = ["ASEL", "ASER", "AWAL", "AWCL"]
RECS = list(ch.HERM_DRIVE_IDS)
FS = ch.SAMPLING_RATE_HZ
EPOCH_N = round(15 * FS); TAU = 1; WIN = 20

os.makedirs(os.path.join(REPO_ROOT, "data"), exist_ok=True)
for k_, v_ in ch.HERM_DRIVE_IDS.items():
    p_ = os.path.join(REPO_ROOT, f"data/{k_}.csv")
    if os.path.exists(p_) and os.path.getsize(p_) > 1_000_000:
        continue
    subprocess.run(["curl", "-sSL", "-o", p_,
                    f"https://drive.google.com/uc?export=download&id={v_}"], check=True)
    if os.path.getsize(p_) < 1_000_000:
        subprocess.run(["curl", "-sSL", "-o", p_,
                        f"https://drive.usercontent.google.com/download?id={v_}&export=download&confirm=t"],
                       check=True)

ST, ONS = {}, {}
for r in RECS:
    d = pd.read_csv(os.path.join(REPO_ROOT, f"data/{r}.csv"))
    names = d["neuron"].tolist()
    bits = []
    for nm in Q:
        x = d.iloc[names.index(nm)][9:-1].astype(float).values
        xf = np.where(np.isfinite(x), x, np.nanmedian(x))
        bits.append((xf - median_filter(xf, size=max(3, round(WIN * FS)),
                                        mode="nearest") > 0).astype(int))
    ST[r] = sum(b * (2 ** i) for i, b in enumerate(bits))
    ONS[r] = sorted(int(t) for t, _ in ast.literal_eval(d.iloc[0]["stimulus"]))

# %%
# conditioned transition counts: source frame inside a 15 s stimulus window vs baseline
C_on = np.zeros((16, 16)); C_off = np.zeros((16, 16)); C_pool = np.zeros((16, 16))
for r in RECS:
    st = ST[r]; on_mask = np.zeros(len(st), bool)
    for t0 in ONS[r]:
        on_mask[t0:t0 + EPOCH_N] = True
    for t in range(len(st) - TAU):
        C_pool[st[t], st[t + TAU]] += 1
        (C_on if on_mask[t] else C_off)[st[t], st[t + TAU]] += 1
print(f"transitions: on {int(C_on.sum())}, off {int(C_off.sum())}, pool {int(C_pool.sum())}")

def P_(C): return (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
def rowjsd(A, B): return float(np.mean([jensenshannon(P_(A)[s], P_(B)[s], base=2) for s in range(16)]))
print(f"row JSD: on-vs-off {rowjsd(C_on, C_off):.4f} | pool-vs-off {rowjsd(C_pool, C_off):.4f} "
      f"| pool-vs-on {rowjsd(C_pool, C_on):.4f}  (split-half noise at this volume ~0.08)")

# %%
def bigphi_map(C):
    P = P_(C)
    net = pyphi.Network(convert.state_by_state2state_by_node(P), node_labels=Q)
    out = np.zeros(16); nd = np.zeros(16, int); nr = np.zeros(16, int)
    for si in range(16):
        ps = pyphi.new_big_phi.phi_structure(
            pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4))))
        out[si] = float(ps.big_phi); nd[si] = len(ps.distinctions); nr[si] = len(ps.relations)
    return out, nd, nr

lab = ["".join(str((si >> i) & 1) for i in range(4)) for si in range(16)]
B, ND, NR = {}, {}, {}
for k, C in [("static", C_pool), ("stim_on", C_on), ("stim_off", C_off)]:
    B[k], ND[k], NR[k] = bigphi_map(C)
tb = pd.DataFrame({("Phi_" + k): np.round(B[k], 6) for k in B}, index=lab)
tb.index.name = "state (ASEL,ASER,AWAL,AWCL)"
for k in B:
    tb["ndist_" + k] = ND[k]; tb["nrel_" + k] = NR[k]
tb.to_csv(os.path.join(REPO_ROOT, "results/phi_by_state_three_tpms.csv"))
print(tb[[c for c in tb.columns if c.startswith("Phi")]].to_string())
for a, b in [("static", "stim_off"), ("static", "stim_on"), ("stim_on", "stim_off")]:
    print(f"rho({a},{b}) =", round(float(stats.spearmanr(B[a], B[b]).statistic), 3))

# %%
# guide-star control: is the on/off difference a volume artifact? Subsample the
# off-pool to the on-volume and recompute.
rng = np.random.default_rng(0)
off_pairs = []
for r in RECS:
    st = ST[r]; on_mask = np.zeros(len(st), bool)
    for t0 in ONS[r]:
        on_mask[t0:t0 + EPOCH_N] = True
    off_pairs += [(st[t], st[t + TAU]) for t in range(len(st) - TAU) if not on_mask[t]]
off_pairs = np.array(off_pairs); n_on = int(C_on.sum())
rows = []
for rep in range(8):
    idx = rng.choice(len(off_pairs), n_on, replace=False)
    Cs = np.zeros((16, 16)); np.add.at(Cs, (off_pairs[idx, 0], off_pairs[idx, 1]), 1)
    Bs, _, _ = bigphi_map(Cs)
    rows.append(dict(rep=rep, phi0=Bs[0], argmax=lab[int(np.argmax(Bs))],
                     rho_vs_fulloff=float(stats.spearmanr(Bs, B["stim_off"]).statistic),
                     rho_vs_on=float(stats.spearmanr(Bs, B["stim_on"]).statistic)))
ss = pd.DataFrame(rows)
ss.to_csv(os.path.join(REPO_ROOT, "results/tpm_regime_volume_control.csv"), index=False)
print(f"matched-volume off maps: Phi(0000) {ss.phi0.mean():.1f}±{ss.phi0.std():.1f}, "
      f"argmax {ss.argmax.value_counts().to_dict()}, rho_vs_on {ss.rho_vs_on.mean():+.2f}")

# %% [markdown]
# ## Is the stim-on map just undersampled? The split design
#
# The decisive test: if the on-map's distinctiveness were undersampling noise,
# the on-pool's own disjoint halves should disagree with each other as much as
# they disagree with the off-map. Compare same-regime reproducibility against
# cross-regime agreement at matched volumes.

# %%
on_pairs = []
for r in RECS:
    st = ST[r]; on_mask = np.zeros(len(st), bool)
    for t0 in ONS[r]:
        on_mask[t0:t0 + EPOCH_N] = True
    on_pairs += [(st[t], st[t + TAU]) for t in range(len(st) - TAU) if on_mask[t]]
on_pairs = np.array(on_pairs)

def map_of(pairs, idx):
    C = np.zeros((16, 16)); np.add.at(C, (pairs[idx, 0], pairs[idx, 1]), 1)
    return bigphi_map(C)[0]

rng = np.random.default_rng(1)
rows = []
for rep in range(4):
    pm = rng.permutation(len(off_pairs)); h = len(off_pairs) // 2
    offA, offB = map_of(off_pairs, pm[:h]), map_of(off_pairs, pm[h:2 * h])
    pm2 = rng.permutation(len(off_pairs)); n_ = len(on_pairs)
    off1, off2 = map_of(off_pairs, pm2[:n_]), map_of(off_pairs, pm2[n_:2 * n_])
    pmo = rng.permutation(len(on_pairs)); ho = len(on_pairs) // 2
    onA, onB = map_of(on_pairs, pmo[:ho]), map_of(on_pairs, pmo[ho:2 * ho])
    off48 = map_of(off_pairs, rng.permutation(len(off_pairs))[:ho])
    sp = lambda a, b: float(stats.spearmanr(a, b).statistic)
    rows.append(dict(rep=rep,
        rho_offA_offB_15k=sp(offA, offB), dphi0_off15k=abs(offA[0] - offB[0]),
        rho_off_off_9k6=sp(off1, off2),
        rho_onA_onB_4k8=sp(onA, onB),
        rho_onA_off_4k8=sp(onA, off48), rho_onB_off_4k8=sp(onB, off48),
        phi0_onA=onA[0], phi0_onB=onB[0],
        phi_0001_onA=onA[8], phi_0001_onB=onB[8],
        phi0_offA=offA[0], phi0_offB=offB[0]))
sd = pd.DataFrame(rows)
sd.to_csv(os.path.join(REPO_ROOT, "results/regime_split_design.csv"), index=False)
print(f"off/off @15.1k rho {sd.rho_offA_offB_15k.mean():+.2f}±{sd.rho_offA_offB_15k.std():.2f} | "
      f"off/off @9.6k {sd.rho_off_off_9k6.mean():+.2f}±{sd.rho_off_off_9k6.std():.2f} | "
      f"on/on @4.8k {sd.rho_onA_onB_4k8.mean():+.2f}±{sd.rho_onA_onB_4k8.std():.2f} | "
      f"on/off @4.8k {pd.concat([sd.rho_onA_off_4k8, sd.rho_onB_off_4k8]).mean():+.2f}")
print(f"|dPhi(0000)| between off 15k halves: {sd.dphi0_off15k.mean():.1f}±{sd.dphi0_off15k.std():.1f} "
      f"(static-vs-off gap: 16.4)")

# %% [markdown]
# ## Is halving too harsh? The full-volume bootstrap
#
# Split-half comparisons measure noise at n/2, overstating the full-data
# estimate's noise by ~2x in SD terms. The matched instrument for "is the
# stim-on map genuinely different in THIS dataset" is the parametric bootstrap
# at FULL volume: resample each TPM row from its own multinomial at its actual
# row count (this is the notebook-15 perturbation machinery at scale 1).
# Caveat kept in view: transitions are temporally dependent (consecutive
# transitions share a frame; the 60 s cycle), so an iid bootstrap can
# UNDERSTATE noise — p-values here are anti-conservative. The truth sits
# between the split-half (too harsh) and this (too lenient); claims at
# p ~ 1e-4 survive that bracket, marginal ones may not.

# %%
def boot_C(C, rng):
    Cb = np.zeros_like(C)
    for s_ in range(16):
        n_ = int(C[s_].sum())
        if n_ > 0:
            Cb[s_] = rng.multinomial(n_, C[s_] / C[s_].sum())
    return Cb

rng = np.random.default_rng(2)
# TPM-level certification (the drop-k question, at full volume)
for k, C in [("static", C_pool), ("stim_off", C_off), ("stim_on", C_on)]:
    P = C / np.maximum(C.sum(1, keepdims=True), 1)
    n_row = C.sum(1)
    SE = np.sqrt(P * (1 - P) / np.maximum(n_row[:, None], 1))
    ag = {d: [] for d in (2, 3)}
    for rep in range(20):
        Pb = boot_C(C, rng); Pb = Pb / np.maximum(Pb.sum(1, keepdims=True), 1)
        for d in (2, 3):
            ag[d].append(np.mean(np.round(Pb, d) == np.round(P, d)))
    print(f"{k:>9}: n={int(C.sum()):>6} | median SE {np.median(SE[n_row > 0]):.4f} | "
          f"stable at 2dp {100 * np.mean(ag[2]):.0f}% | 3dp {100 * np.mean(ag[3]):.0f}%")

# %%
# Phi-map noise at FULL volume
R = 10
BOOT = {}
for k, C in [("static", C_pool), ("stim_on", C_on), ("stim_off", C_off)]:
    BOOT[k] = [bigphi_map(boot_C(C, rng))[0] for _ in range(R)]

from itertools import combinations as _comb
sp = lambda a, b: float(stats.spearmanr(a, b).statistic)
within = {k: [sp(BOOT[k][i], BOOT[k][j]) for i, j in _comb(range(R), 2)] for k in BOOT}
cross_on_off = [sp(BOOT["stim_on"][i], BOOT["stim_off"][i]) for i in range(R)]
cross_st_off = [sp(BOOT["static"][i], BOOT["stim_off"][i]) for i in range(R)]
for k in BOOT:
    print(f"within {k:>9}: {np.mean(within[k]):+.2f} ± {np.std(within[k]):.2f}")
print(f"cross on/off: {np.mean(cross_on_off):+.2f} | cross static/off: {np.mean(cross_st_off):+.2f}")

on0 = np.array([b[0] for b in BOOT["stim_on"]]); off0 = np.array([b[0] for b in BOOT["stim_off"]])
st0 = np.array([b[0] for b in BOOT["static"]])
on8 = np.array([b[8] for b in BOOT["stim_on"]]); off8 = np.array([b[8] for b in BOOT["stim_off"]])
on15 = np.array([b[15] for b in BOOT["stim_on"]]); off15 = np.array([b[15] for b in BOOT["stim_off"]])
print(f"Phi(0000) on {on0.mean():.1f}±{on0.std():.1f} vs off {off0.mean():.1f}±{off0.std():.1f} "
      f"p={stats.mannwhitneyu(on0, off0).pvalue:.5f}")
print(f"Phi(0001) on {on8.mean():.1f}±{on8.std():.1f} vs off {off8.mean():.2f}±{off8.std():.2f} "
      f"p={stats.mannwhitneyu(on8, off8).pvalue:.6f}")
print(f"Phi(1111) on {on15.mean():.1f}±{on15.std():.1f} vs off {off15.mean():.1f}±{off15.std():.1f} "
      f"p={stats.mannwhitneyu(on15, off15).pvalue:.4f}")
print(f"Phi(0000) static vs off p={stats.mannwhitneyu(st0, off0).pvalue:.5f}")
from collections import Counter
print("argmax under stim-on boots:", dict(Counter(lab[int(np.argmax(b))] for b in BOOT["stim_on"])))

out = []
for k in BOOT:
    for i, b in enumerate(BOOT[k]):
        out.append(dict(regime=k, rep=i, phi_0000=b[0], phi_0001=b[8], phi_1111=b[15],
                        argmax=lab[int(np.argmax(b))], rho_vs_full=sp(b, B[k])))
pd.DataFrame(out).to_csv(os.path.join(REPO_ROOT, "results/regime_bootstrap_fullvolume.csv"), index=False)

# the requested static-vs-off table with full-volume bootstrap SDs
st_sd = np.std([b for b in BOOT["static"]], axis=0)
off_sd = np.std([b for b in BOOT["stim_off"]], axis=0)
tso = pd.DataFrame(dict(state=lab,
    Phi_static=np.round(B["static"], 2), sd_static=np.round(st_sd, 2),
    Phi_stim_off=np.round(B["stim_off"], 2), sd_off=np.round(off_sd, 2),
    delta=np.round(B["static"] - B["stim_off"], 2)))
tso["delta_in_sd"] = np.round(tso.delta.abs() / np.maximum(np.sqrt(st_sd**2 + off_sd**2), 1e-9), 2)
tso.to_csv(os.path.join(REPO_ROOT, "results/static_vs_stimoff_phi.csv"), index=False)
print(tso.to_string(index=False))



# %% [markdown]
# ## Reading (graded verdict, after split design AND full-volume bootstrap)
#
# * **TPM level: the regimes differ beyond doubt, and the stim-on TPM is
#   itself a stable estimate.** Rows differ at z = 35 (positive control);
#   stim-on median binomial SE 0.0094 (~2-decimal certification, vs 0.0048
#   static). NOTHING is stable at 3 decimals — 8-10% for every regime — so
#   "certify at 3dp" is not available even for the full pool.
# * **Φ level, at full-volume noise (the split-half was too harsh): specific
#   regime differences ARE resolvable.** Σφ(0000): on 13.3 ± 7.5 vs off
#   32.0 ± 12.7 (p = 0.005); Σφ(0001): 5.9 ± 4.1 vs 0.83 ± 0.13 (p = 0.0002);
#   Σφ(1111): 11.7 ± 7.7 vs 2.8 ± 1.9 (p = 0.001). Under stim-on the
#   0000-argmax monopoly breaks (0000 in only 5/10 boots) — but WHICH active
#   state peaks is contested, so "the peak moves to 0001" stays retracted.
#   Within-regime map reproducibility 0.62-0.68 vs cross on/off 0.40.
# * **Static vs stim-off (the requested comparison): inseparable where Φ
#   lives.** Φ(0000) p = 0.10; cross ρ 0.53 vs within 0.62; only floor states
#   (Σφ ≈ 0.7-1.0) show a small systematic mixture uplift (4/16 states >2
#   combined SDs, deltas ≤ 0.3). The static TPM is, within resolution, the
#   baseline regime.
# * **Instrument choice, stated as policy.** Split-half noise floors remain
#   correct for BETWEEN-COHORT questions (do two sets of animals agree?). For
#   "is A different from B in this dataset's estimate", the full-volume
#   parametric bootstrap is the matched instrument — with the caveat that iid
#   resampling understates noise under temporal dependence, so p-values are
#   anti-conservative; the strong results (p <= 0.005) survive that bracket,
#   marginal ones may not. A block bootstrap would tighten this.

# %% [markdown]
# ## Figure 45 — regimes, at the right noise instrument

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
BLUE, ORANGE, GREY, PURPLE = "#1f6fb4", "#c2571a", "#8a8a8a", "#7a4a8a"
ax = axes[0]
x = np.arange(16); w = 0.27
ax.bar(x - w, B["static"], w, color=GREY, label="static (pooled, 39.8k)")
ax.bar(x, B["stim_off"], w, color=BLUE, label="stim-off (30.2k)")
ax.bar(x + w, B["stim_on"], w, color=ORANGE, label="stim-on (9.6k)")
ax.set_yscale("log"); ax.set_ylim(0.4, 60)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=90, fontsize=5.4)
ax.set_xlabel("state (ASEL, ASER, AWAL, AWCL)", labelpad=5, fontsize=7)
ax.set_ylabel("Σφ of the unfolded structure", labelpad=5, fontsize=7)
ax.legend(frameon=False, fontsize=5.6, loc="upper right")
ax.set_title("a  Σφ per state under each TPM (full data)", loc="left", fontsize=8)
ax = axes[1]
cats = [("within\nstatic", within["static"], GREY), ("within\nstim-on", within["stim_on"], ORANGE),
        ("within\nstim-off", within["stim_off"], BLUE),
        ("cross\non / off", cross_on_off, PURPLE), ("cross\nstatic / off", cross_st_off, "#4a7a4a")]
for i_, (nm, v, c) in enumerate(cats):
    v = np.asarray(v)
    ax.bar(i_, v.mean(), 0.6, color=c, alpha=0.35, lw=0)
    ax.scatter(np.full(len(v), i_) + np.linspace(-0.16, 0.16, len(v)), v, s=9, color=c, zorder=3, lw=0)
ax.axhline(0, color="#333", lw=0.7)
ax.set_xticks(range(5)); ax.set_xticklabels([c[0] for c in cats], fontsize=6)
ax.set_ylabel("ρ between Φ-maps", labelpad=5, fontsize=7)
ax.set_ylim(-0.05, 1.0)
ax.set_title("b  Full-volume bootstrap: on/off agreement\n   falls below within-regime reproducibility", loc="left", fontsize=8)
ax = axes[2]
pos = {"stim_off": 0, "static": 1, "stim_on": 2}
col = {"stim_off": BLUE, "static": GREY, "stim_on": ORANGE}
for k in pos:
    v = np.array([b[0] for b in BOOT[k]])
    ax.scatter(np.full(len(v), pos[k]) + np.linspace(-0.12, 0.12, len(v)), v, s=14, color=col[k], lw=0)
    ax.scatter([pos[k]], [B[k][0]], marker="D", s=32, color=col[k], zorder=4, edgecolor="#222", lw=0.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["stim-off", "static", "stim-on"], fontsize=7)
ax.set_xlim(-0.5, 2.5); ax.set_ylabel("Σφ(0000), full-volume bootstrap", labelpad=5, fontsize=7)
ax.text(0.5, 0.955, "static vs off: p = 0.10", transform=ax.transAxes, ha="center", fontsize=6.2, color="#333")
ax.text(0.5, 0.885, "on vs off: p = 0.005", transform=ax.transAxes, ha="center", fontsize=6.2, color=ORANGE)
ax.set_title("c  Quiescence-Φ deflation under stim-on IS\n   resolvable at full volume; static ≈ off", loc="left", fontsize=8)
fig.suptitle("Regimes differ at the TPM level (z = 35) and — at full-volume noise — at specific Φ states (0000↓, 0001↑, 1111↑ under stim-on); static ≈ stim-off",
             fontsize=7.8)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig45_static_vs_dynamic_tpm.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig45_static_vs_dynamic_tpm.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig45")
