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
# ## Reading (corrected by the split design)
#
# * **At the TPM level the regimes differ beyond doubt.** Stim-on vs stim-off
#   rows differ at z = 35 under epoch relabelling (the positive control), and
#   row JSD 0.197 vs same-regime noise ~0.08. The static TPM is the majority
#   regime in disguise: 3x closer to off in JSD, rho = +0.71 vs +0.44.
# * **At the Φ-landscape level the difference is NOT resolvable at current
#   volume.** Same-regime reproducibility is as low as cross-regime agreement:
#   off/off halves rho ~ +0.48-0.62, on/on halves +0.47, on/off at matched
#   volume +0.42. And Σφ(0000) scatters by 23 ± 20 between off-halves — which
#   swallows both the apparent on-deflation (36.2 → 6.9) and the static-off
#   gap (16.4). The full-data observations (peak at 0001 under stim-on,
#   rho = +0.19 between regime maps) were carried by margins far inside that
#   scatter (argmax flipped on 6.9 vs 7.5) and cannot be attributed to regime
#   rather than sampling.
# * **This is the stability hierarchy again, not an exception to it.** The
#   regimes demonstrably differ where estimation is stable (TPM rows) and
#   cannot yet be distinguished where it is not (the unfolded Φ landscape).
#   The suggestive pattern — on-halves' Σφ(0000) sitting low — is
#   volume-confounded (4.8k vs 15.1k halves) and stays unclaimed.
# * **Both TPM views retained.** Static-across-animals holds (isogenic
#   design; engram-architecture reading). Static-across-contexts is refuted
#   at the TPM level. The ecological (marginal) TPM remains the declared
#   default; condition-dependent Φ comparisons need per-regime volumes the
#   current recordings do not provide.

# %% [markdown]
# ## Figure 45 — regimes, and what survives the noise check

# %%
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
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
cats = [("off / off\n@15.1k", sd.rho_offA_offB_15k, BLUE),
        ("off / off\n@9.6k", sd.rho_off_off_9k6, BLUE),
        ("on / on\n@4.8k", sd.rho_onA_onB_4k8, ORANGE),
        ("on / off\n@4.8k", pd.concat([sd.rho_onA_off_4k8, sd.rho_onB_off_4k8]), "#7a4a8a")]
for i_, (nm, v, c) in enumerate(cats):
    ax.bar(i_, v.mean(), 0.6, color=c, alpha=0.35, lw=0)
    ax.scatter(np.full(len(v), i_) + np.linspace(-0.13, 0.13, len(v)), v, s=14, color=c, zorder=3, lw=0)
ax.axhline(0, color="#333", lw=0.7)
ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cats], fontsize=6)
ax.set_ylabel("ρ between Φ-maps (disjoint samples)", labelpad=5, fontsize=7)
ax.set_ylim(-0.05, 1.0)
ax.set_title("b  Same-regime reproducibility ≈\n   cross-regime agreement", loc="left", fontsize=8)
ax = axes[2]
offh = np.concatenate([sd.phi0_offA, sd.phi0_offB])
onh = np.concatenate([sd.phi0_onA, sd.phi0_onB])
ax.scatter(np.full(len(offh), 0) + np.linspace(-0.1, 0.1, len(offh)), offh, s=16, color=BLUE, lw=0, label="off halves (15.1k)")
ax.scatter(np.full(len(onh), 1) + np.linspace(-0.1, 0.1, len(onh)), onh, s=16, color=ORANGE, lw=0, label="on halves (4.8k)")
ax.scatter([0.5], [B["static"][0]], marker="D", s=30, color=GREY, zorder=4, label="static full (19.8)")
ax.scatter([0], [B["stim_off"][0]], marker="D", s=30, color=BLUE, zorder=4)
ax.scatter([1], [B["stim_on"][0]], marker="D", s=30, color=ORANGE, zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(["stim-off", "stim-on"], fontsize=7)
ax.set_xlim(-0.5, 1.5)
ax.set_ylabel("Σφ(0000)", labelpad=5, fontsize=7)
ax.legend(frameon=False, fontsize=5.6, loc="upper right")
ax.set_title("c  Σφ(0000): half-sample scatter swallows\n   the regime gaps (dots = halves, ◆ = full)", loc="left", fontsize=8)
fig.suptitle("The regimes differ at the TPM level (z = 35); their Φ landscapes cannot be distinguished above sampling noise at current volume",
             fontsize=8.2)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig45_static_vs_dynamic_tpm.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig45_static_vs_dynamic_tpm.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig45")
