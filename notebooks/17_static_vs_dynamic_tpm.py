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
# ## Reading
#
# * **The Φ landscape is regime-dependent.** Under the stim-off TPM, Φ is
#   concentrated on quiescence (0000: 36.2, everything else ≤ 2.9). Under the
#   stim-on TPM, 0000 deflates to 6.9 and the peak MOVES to active states
#   (0001 = AWCL-only: 7.5; 1111: 6.2). The two regime maps barely correlate
#   (ρ = +0.19, p = 0.49).
# * **The static TPM is not neutral — it is the majority regime in disguise.**
#   It correlates ρ = +0.71 with the off-map vs +0.44 with the on-map, and its
#   JSD sits 3× closer to off than to on — matching the 76% baseline share of
#   transitions. Every static-TPM result in this repo (the Φ(t) traces, the
#   offset dip, the 0000-concentration) is therefore predominantly a
#   baseline-regime result.
# * **Not a volume artifact.** Subsampling the off-pool to the on-volume
#   (9,600 transitions, 8 draws): argmax stays 0000 in 8/8, Phi(0000) = 28 ± 16
#   >> the on-map's 6.9, and the subsampled off-maps still fail to correlate
#   with the on-map (ρ ≈ +0.27).
# * **Both views retained, per the project's position.** Static-across-ANIMALS
#   is supported (isogenic clones, within-vs-between test) and matches the
#   engram-architecture reading of long-term connectivity; static-across-
#   CONTEXTS is refuted in this data (z = 35; JSD 0.197 vs noise 0.08). The
#   ecological (marginal) TPM remains the declared default; the conditioned
#   TPMs are the IIT-stricter objects, and any future condition-assigned
#   structure comparison should use them.

# %% [markdown]
# ## Figure 45 — the two regimes side by side

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.7), constrained_layout=True)
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
ax = axes[0]
x = np.arange(16); w = 0.27
ax.bar(x - w, B["static"], w, color=GREY, label="static (pooled, 39.8k)")
ax.bar(x, B["stim_off"], w, color=BLUE, label="stim-off (30.2k)")
ax.bar(x + w, B["stim_on"], w, color=ORANGE, label="stim-on (9.6k)")
ax.set_yscale("log"); ax.set_ylim(0.4, 60)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=90, fontsize=5.6)
ax.set_xlabel("state (ASEL, ASER, AWAL, AWCL)", labelpad=5, fontsize=7)
ax.set_ylabel("Σφ of the unfolded structure", labelpad=5, fontsize=7)
ax.legend(frameon=False, fontsize=6, loc="upper right")
ax.set_title("a  The Φ landscape is regime-dependent", loc="left", fontsize=8.5)
ax = axes[1]
ax.scatter(B["stim_off"], B["stim_on"], s=22, color="#444", zorder=3, lw=0)
for si in (0, 8, 15):
    ax.annotate(lab[si], (B["stim_off"][si], B["stim_on"][si]),
                xytext=(4, 3), textcoords="offset points", fontsize=6)
lim = max(B["stim_off"].max(), B["stim_on"].max()) * 1.15
ax.plot([0.4, lim], [0.4, lim], ls=":", lw=0.9, color="#888")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.4, lim); ax.set_ylim(0.4, lim)
ax.set_xlabel("Σφ under the stim-off TPM", labelpad=5, fontsize=7)
ax.set_ylabel("Σφ under the stim-on TPM", labelpad=5, fontsize=7)
r_ = stats.spearmanr(B["stim_on"], B["stim_off"])
ax.text(0.05, 0.92, f"ρ = {r_.statistic:+.2f} (p = {r_.pvalue:.2f})",
        transform=ax.transAxes, fontsize=7)
ax.set_title("b  The two regimes barely agree on which\n   states carry Φ", loc="left", fontsize=8.5)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig45_static_vs_dynamic_tpm.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig45_static_vs_dynamic_tpm.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig45")
