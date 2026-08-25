# %% [markdown]
# # 14 — Which state is "stimulus" and which is "no stimulus"?
#
# The Φ(t) work left one question: the offset dip detects the delivery event,
# but the *states* were never formally assigned to conditions. This notebook
# compares the occupancy distribution over the 16 states between the stimulus
# window (0–15 s) and each epoch's own pre-stimulus window (−15–0 s), paired
# per epoch, with medians as well as means, and a rank–frequency view.
#
# Setup identical to notebooks 12–13 (20 s high-pass bits, giant TPM, Φ per state).

# %%
import os, sys, ast
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.ndimage import median_filter
from scipy.spatial.distance import jensenshannon
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = "."
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import ces_hypergraph as ch
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
CLS = ch.STIMULUS_CLASS
FS = ch.SAMPLING_RATE_HZ
EPOCH_N = round(15 * FS)
CYC_N = round(60 * FS)
PRE_N = round(15 * FS)
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
ST = {r: {"core": ch.combine_states([BITS[r][nm] for nm in INTER]),
          "sens": ch.combine_states([BITS[r][nm] for nm in SENS])} for r in RECS}

PHI_OF = {}
for key, neurons in [("core", INTER), ("sens", SENS)]:
    C = np.zeros((16, 16))
    for r in RECS:
        st = ST[r][key]
        for a, b in zip(st[:-1], st[1:]):
            C[a, b] += 1
    P = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    net = pyphi.Network(convert.state_by_state2state_by_node(P), node_labels=neurons)
    PHI_OF[key] = {si: float(pyphi.new_big_phi.phi_structure(
        pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4)))).big_phi)
        for si in range(16)}
print("setup complete")

# %% [markdown]
# ## Per-epoch occupancy histograms, paired stim-vs-pre

# %%
def occ_hists(key):
    H = {"pre": [], "stim": [], "post": []}
    for r in RECS:
        st = ST[r][key]
        for s in STIMULI:
            for t0 in sorted(DAT[r][1].get(s, [])):
                if t0 - PRE_N < 0 or t0 + CYC_N > len(st):
                    continue
                for lbl, seg in [("pre", st[t0 - PRE_N:t0]),
                                 ("stim", st[t0:t0 + EPOCH_N]),
                                 ("post", st[t0 + EPOCH_N:t0 + CYC_N])]:
                    H[lbl].append(np.bincount(seg, minlength=16) / len(seg))
    return {k: np.array(v) for k, v in H.items()}
OCC = {key: occ_hists(key) for key in ("core", "sens")}

rows = []
for key in ("core", "sens"):
    H = OCC[key]
    for si in range(16):
        pre, stim = H["pre"][:, si], H["stim"][:, si]
        d = stim - pre
        nz = d[d != 0]
        rows.append(dict(substrate=key, state=format(si, "04b"),
            mean_pre=round(float(pre.mean()), 4), mean_stim=round(float(stim.mean()), 4),
            median_pre=round(float(np.median(pre)), 4),
            median_stim=round(float(np.median(stim)), 4),
            log2_ratio=round(float(np.log2((stim.mean() + 1e-4) / (pre.mean() + 1e-4))), 3),
            mean_paired_diff=round(float(d.mean()), 4),
            p_wilcoxon=stats.wilcoxon(nz).pvalue if len(nz) >= 10 else np.nan,
            phi=round(PHI_OF[key][si], 2)))
et = pd.DataFrame(rows)
for key in ("core", "sens"):
    m = et.substrate == key
    p = et.loc[m, "p_wilcoxon"].values
    valid = ~np.isnan(p)
    pv = p[valid]; o = np.argsort(pv); h = np.empty(len(pv))
    for rank, idx in enumerate(o):
        h[idx] = min(1.0, (len(pv) - rank) * pv[idx])
    h = np.maximum.accumulate(h[o])[np.argsort(o)]
    hp_ = np.full(len(p), np.nan); hp_[valid] = h
    et.loc[m, "p_holm"] = hp_
et.to_csv(os.path.join(REPO_ROOT, "results/state_enrichment.csv"), index=False)
for key in ("core", "sens"):
    print(f"\n{key.upper()}:")
    print(et[et.substrate == key].sort_values("mean_paired_diff")[
        ["state", "mean_pre", "mean_stim", "median_pre", "median_stim",
         "mean_paired_diff", "log2_ratio", "p_holm", "phi"]].to_string(index=False))

# %% [markdown]
# ## Distribution-level test: JSD(stim, pre) against a within-epoch label shuffle

# %%
def dist_test(key, nperm=5000):
    H = OCC[key]
    pairs = list(zip(H["pre"], H["stim"]))
    obs = jensenshannon(np.mean([p for p, _ in pairs], 0),
                        np.mean([s for _, s in pairs], 0), base=2)
    rg = np.random.default_rng(0)
    null = []
    for _ in range(nperm):
        A, B = [], []
        for p_, s_ in pairs:
            if rg.random() < 0.5:
                A.append(p_); B.append(s_)
            else:
                A.append(s_); B.append(p_)
        null.append(jensenshannon(np.mean(A, 0), np.mean(B, 0), base=2))
    null = np.array(null)
    return float(obs), float(null.mean()), float(null.std()), float((np.sum(null >= obs) + 1) / (nperm + 1))

res = []
for key in ("core", "sens"):
    o, m, sd, p = dist_test(key)
    res.append(dict(substrate=key, jsd_stim_vs_pre=round(o, 4), null_mean=round(m, 4),
                    null_sd=round(sd, 4), z=round((o - m) / sd, 2), p=round(p, 4)))
    print(f"{key}: JSD(stim, pre) = {o:.4f}  null {m:.4f}±{sd:.4f}  z = {(o-m)/sd:+.2f}  p = {p:.4f}")
pd.DataFrame(res).to_csv(os.path.join(REPO_ROOT, "results/state_distribution_test.csv"), index=False)

# %% [markdown]
# ## Reading
#
# * **The distributions differ massively on the sensory substrate** (JSD z = +18)
#   and mildly on the core (z = +2.4) — consistent with everything upstream.
# * **The named states.** On the sensory substrate the baseline-enriched state is
#   **1000 = AWCL alone ON** (occupancy 0.107 pre → 0.047 stim, log₂ = −1.20,
#   Holm p = 3 × 10⁻²²) and the stimulus-enriched state is its exact complement,
#   **0111 = ASEL+ASER+AWAL ON, AWCL OFF** (0.036 → 0.081, log₂ = +1.15,
#   Holm p = 4 × 10⁻¹³). Nine of sixteen sensory states shift significantly
#   after Holm correction; the ladder is bit-interpretable throughout — every baseline-enriched state
#   contains AWCL ON, every stimulus-enriched state has AWCL OFF with ON cells
#   among ASEL/ASER/AWAL. AWCL is an OFF cell (odour removal activates it), so
#   this is chemosensory biology read straight off the state labels.
# * **The high-Φ states carry no condition information.** 0000 does not shift
#   (sensory p = 0.50, core p = 0.33), and core 1111 does not either (p = 0.79).
#   **Every earlier per-condition Φ-structure comparison unfolded at 0000** — the
#   argmax-occupancy state — which this table now shows is exactly the state
#   that says nothing about the condition. The informative states (1000, 0111)
#   have Φ of 0.97 and 4.08 — visible to a structure comparison, invisible to
#   the scalar dominated by 0000.
# * **Medians agree with means** on every significant state (the enrichment
#   survives the median columns), so this is not driven by a few extreme epochs.

# %% [markdown]
# ## Figure 39 — rank–frequency by condition, and the enrichment ladder

# %%
def on_neurons(si, key):
    neurons = INTER if key == "core" else SENS
    return "+".join(neurons[i] for i in range(4) if (si >> i) & 1) or "none"

fig = plt.figure(figsize=(12.8, 7.6))
g = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.24)
for col, key in enumerate(("core", "sens")):
    H = OCC[key]
    pooled = (H["pre"].mean(0) + H["stim"].mean(0)) / 2
    order = np.argsort(pooled)[::-1]
    labels = [format(int(si), "04b") for si in order]

    ax = fig.add_subplot(g[0, col])
    x = np.arange(16); w = 0.38
    ax.bar(x - w / 2, H["pre"].mean(0)[order], w, color=GREY, label="pre (−15–0 s)")
    ax.bar(x + w / 2, H["stim"].mean(0)[order], w, color=BLUE, label="stimulus (0–15 s)")
    sub = et[et.substrate == key].set_index("state")
    for xi, si in enumerate(order):
        ph = sub.loc[format(int(si), "04b"), "p_holm"]
        if not np.isnan(ph) and ph < 0.05:
            ax.text(xi, max(H["pre"].mean(0)[si], H["stim"].mean(0)[si]) + 0.004, "*",
                    ha="center", fontsize=7.5, color="#c2571a", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\nΦ {PHI_OF[key][int(si)]:.1f}" for l, si in zip(labels, order)],
                       fontsize=4.6)
    ax.set_ylabel("occupancy fraction", labelpad=5)
    ax.set_xlabel("state, by decaying overall frequency (Φ below each)", labelpad=4)
    ax.set_title(f"{'ab'[col]}  {key.upper()} — rank–frequency by condition (* = Holm p < 0.05)",
                 loc="left")
    if col == 0:
        ax.legend(frameon=False, fontsize=6, loc="upper right")

    ax = fig.add_subplot(g[1, col])
    sub2 = et[et.substrate == key].copy()
    sub2["si"] = sub2.state.apply(lambda s_: int(s_, 2))
    sub2 = sub2.sort_values("log2_ratio")
    cols = [ORANGE if (not np.isnan(ph) and ph < 0.05) else "#c9c9c9" for ph in sub2.p_holm]
    ax.barh(range(16), sub2.log2_ratio, 0.66, color=cols)
    ax.set_yticks(range(16))
    ax.set_yticklabels([f"{r_.state}  {on_neurons(r_.si, key)}" for _, r_ in sub2.iterrows()],
                       fontsize=4.9, family="monospace")
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("log₂ (stimulus / pre occupancy)", labelpad=5)
    ax.set_title(f"{'cd'[col]}  {key.upper()} — enrichment ladder (orange = Holm p < 0.05)",
                 loc="left")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig39_state_identification.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig39_state_identification.png"), dpi=190, bbox_inches="tight")
print("wrote figures/fig39")
