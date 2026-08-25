# %% [markdown]
# # 12 — Φ as a time series: one giant TPM, Φ per state, Φ(t)
#
# The idea: build **one TPM from the entire dataset** (all 8 animals, all
# ~40,000 transitions), compute the IIT 4.0 Φ for **each of the 16 states** of
# that TPM, and then map every sample of the binarized recording to the Φ of the
# state it is in. That converts the 4-bit state sequence into a **Φ time series**
# that can be epoch-averaged like any PSTH.
#
# Bits use the settled preprocessing: high-pass (rolling median subtraction) with
# the 20 s window chosen in notebook 11, threshold at zero.
#
# Two properties make this attractive. The giant TPM is the best-conditioned
# object in the repository — every row has ≥1,600 observations, no smoothing mass
# to speak of. And Φ(t) needs **no structure distance at all**: it is a scalar
# per sample, so standard PSTH statistics apply.

# %%
import os, sys, ast, subprocess, time
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
W_HP = 20      # settled in notebook 11

def load(rec):
    d = pd.read_csv(ch.recording_path(rec))
    tc = d.columns[9:-1]
    tr = {nm: d.loc[d.neuron == nm, tc].iloc[0].astype(float).values for nm in ALL8}
    on = defaultdict(list)
    for t, l in ast.literal_eval(d.iloc[0]["stimulus"]):
        on[l].append(int(t))
    return tr, on

DAT = {r: load(r) for r in RECS}

def hp(x, w=W_HP):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return xf - median_filter(xf, size=max(3, round(w * FS)), mode="nearest")

HPS = {r: {nm: hp(DAT[r][0][nm]) for nm in ALL8} for r in RECS}
BITS = {r: {nm: (HPS[r][nm] > 0).astype(int) for nm in ALL8} for r in RECS}
ST = {r: {"core": ch.combine_states([BITS[r][nm] for nm in INTER]),
          "sens": ch.combine_states([BITS[r][nm] for nm in SENS])} for r in RECS}
print("bits at", W_HP, "s window;", len(ST[RECS[0]]["core"]), "samples per recording")

# %% [markdown]
# ## One giant TPM per substrate; Φ for every state

# %%
def giant_tpm(key, K=16, tau=1):
    C = np.zeros((K, K))
    for r in RECS:
        st = ST[r][key]
        for a, b in zip(st[:-tau], st[tau:]):
            C[a, b] += 1
    return C

PHI_OF, STRUCT_OF = {}, {}
for key, neurons in [("core", INTER), ("sens", SENS)]:
    C = giant_tpm(key)
    print(f"{key}: {int(C.sum())} transitions | min row count {int(C.sum(1).min())}")
    P = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    net = pyphi.Network(convert.state_by_state2state_by_node(P), node_labels=neurons)
    phis, structs = {}, {}
    for si in range(16):
        ps = pyphi.new_big_phi.phi_structure(
            pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4))))
        phis[si] = float(ps.big_phi)
        structs[si] = (len(ps.distinctions), len(ps.relations))
    PHI_OF[key], STRUCT_OF[key] = phis, structs

phi_tab = pd.DataFrame([dict(substrate=key, state=format(si, "04b"),
                             phi=round(PHI_OF[key][si], 4),
                             n_dist=STRUCT_OF[key][si][0], n_rel=STRUCT_OF[key][si][1])
                        for key in ("core", "sens") for si in range(16)])
phi_tab.to_csv(os.path.join(REPO_ROOT, "results/phi_by_state_giant_tpm.csv"), index=False)
print(phi_tab.pivot(index="state", columns="substrate", values="phi").to_string())

# %% [markdown]
# ## Φ(t): map each sample's state to its Φ

# %%
PHI_T = {r: {key: np.array([PHI_OF[key][s] for s in ST[r][key]])
             for key in ("core", "sens")} for r in RECS}
print("Phi(t) built:", {k: PHI_T[RECS[0]][k].shape for k in ("core", "sens")})

# %% [markdown]
# ## What does Φ(t) actually measure here?
#
# Decomposition first, statistics second. If one or two states carry almost all
# the Φ, then Φ(t) is (nearly) an **indicator function** of those states, and its
# epoch average is just their occupancy profile.

# %%
rows = []
for key in ("core", "sens"):
    hi = sorted(PHI_OF[key], key=PHI_OF[key].get)[-2:]
    allphi = np.concatenate([PHI_T[r][key] for r in RECS])
    allst = np.concatenate([ST[r][key] for r in RECS])
    ind = np.isin(allst, hi).astype(float)
    rows.append(dict(substrate=key,
                     top2_states=",".join(format(s, "04b") for s in hi),
                     top2_phi=",".join(f"{PHI_OF[key][s]:.1f}" for s in hi),
                     frac_time_top2=round(float(ind.mean()), 3),
                     corr_phi_top2indicator=round(float(np.corrcoef(allphi, ind)[0, 1]), 4)))
dec = pd.DataFrame(rows)
dec.to_csv(os.path.join(REPO_ROOT, "results/phi_timeseries_decomposition.csv"), index=False)
print(dec.to_string(index=False))

# %% [markdown]
# ## Is Φ(t) stimulus-locked? Epoch-label permutation, within animal

# %%
def phi_windows(key):
    out = {}
    for r in RECS:
        pt = PHI_T[r][key]
        marks = sorted((t, l) for l, ts in DAT[r][1].items() for t in ts)
        segs = []
        for i, (t0, lab) in enumerate(marks):
            nxt = marks[i + 1][0] if i + 1 < len(marks) else len(pt)
            s_, b_ = pt[t0:t0 + EPOCH_N], pt[t0 + EPOCH_N:nxt]
            if len(s_) > 1:
                segs.append(("stim", float(np.mean(s_)), CLS[lab]))
            if len(b_) > 1:
                segs.append(("base", float(np.mean(b_)), CLS[lab]))
        out[r] = segs
    return out

res = []
for key in ("core", "sens"):
    W = phi_windows(key)
    obs = (np.mean([v for r in RECS for l, v, _ in W[r] if l == "stim"])
           - np.mean([v for r in RECS for l, v, _ in W[r] if l == "base"]))
    rg = np.random.default_rng(0)
    null = []
    for _ in range(5000):
        A, B = [], []
        for r in RECS:
            labs = np.array([l for l, _, _ in W[r]])
            vals = [v for _, v, _ in W[r]]
            for l, v in zip(rg.permutation(labs), vals):
                (A if l == "stim" else B).append(v)
        null.append(np.mean(A) - np.mean(B))
    null = np.array(null)
    res.append(dict(substrate=key, contrast="stim - base", diff=round(obs, 4),
                    z=round(float((obs - null.mean()) / null.std()), 2),
                    p=round(float((np.sum(np.abs(null) >= abs(obs)) + 1) / 5001), 4)))
    sa = [v for r in RECS for l, v, c in W[r] if l == "stim" and c == "attractant"]
    sr = [v for r in RECS for l, v, c in W[r] if l == "stim" and c == "repellent"]
    u = stats.mannwhitneyu(sa, sr)
    d = (np.mean(sa) - np.mean(sr)) / np.sqrt((np.var(sa, ddof=1) + np.var(sr, ddof=1)) / 2)
    res.append(dict(substrate=key, contrast="att - rep (stim Phi)",
                    diff=round(np.mean(sa) - np.mean(sr), 4),
                    z=round(float(d), 2), p=round(float(u.pvalue), 4)))
pr = pd.DataFrame(res)
pr.to_csv(os.path.join(REPO_ROOT, "results/phi_timeseries_tests.csv"), index=False)
print(pr.to_string(index=False))
print("\n-> neither stimulus presence nor class is significant in mean Phi(t);")
print("   the apparent 13 s ramp in the sensory grand average is state-occupancy")
print("   noise passed through a 24x Phi spread, not a reliable Phi effect.")

# %% [markdown]
# ## Figure 31 — the binarized responses at the settled window

# %%
apply = None  # style applied at import in Colab-compatible runs
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
CCOL = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}
REC0 = RECS[0]
EX_STIM = "100mM NaCl"
t_ex = sorted(DAT[REC0][1][EX_STIM])[0]
tcyc = np.arange(CYC_N) / FS

fig, axes = plt.subplots(3, 8, figsize=(15.6, 6.4), sharex=True)
for j, nm in enumerate(ALL8):
    ax = axes[0, j]
    y = HPS[REC0][nm][t_ex:t_ex + CYC_N]
    b = BITS[REC0][nm][t_ex:t_ex + CYC_N]
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    ys = 0.5 + 0.44 * y / max(np.max(np.abs(y)), 1e-9)
    ax.plot(tcyc, ys, color="#999", lw=0.55, zorder=1)
    ax.axhline(0.5, color="#c00", lw=0.7, ls="--", zorder=1)
    ax.step(tcyc, b, where="mid", color="#111", lw=1.0, zorder=3)
    ax.fill_between(tcyc, 0, b, step="mid", color="#111", alpha=0.20, lw=0, zorder=2)
    ax.set_ylim(-0.08, 1.3); ax.set_yticks([0, 1]); ax.tick_params(labelsize=5.6)
    ax.set_title(f"{nm}\n({'core' if nm in INTER else 'sensory'})", fontsize=6.8,
                 color="#333" if nm in INTER else "#7a4fa3")
    if j == 0:
        ax.set_ylabel("single trial\nbit", fontsize=6.2, labelpad=5)
    ax.text(7.5, 1.16, f"{b[:EPOCH_N].mean():.2f}", fontsize=5.6, color="#333", ha="center")

for j, nm in enumerate(ALL8):
    ax = axes[1, j]
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    Ms = [BITS[r][nm][t0:t0 + CYC_N] for r in RECS for s in STIMULI
          for t0 in sorted(DAT[r][1].get(s, [])) if t0 + CYC_N <= len(BITS[r][nm])]
    M = np.array(Ms, float)
    mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
    ax.fill_between(tcyc, mu - se, mu + se, color="#555", alpha=0.25, lw=0)
    ax.plot(tcyc, mu, color="#111", lw=1.0)
    ax.axhline(0.5, color="#666", lw=0.6, ls=":")
    ax.set_ylim(0.15, 0.85); ax.tick_params(labelsize=5.6)
    if j == 0:
        ax.set_ylabel("grand average\nP(bit = 1)", fontsize=6.2, labelpad=5)

for j, nm in enumerate(ALL8):
    ax = axes[2, j]
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    for cls in ["attractant", "repellent", "control"]:
        Ms = [BITS[r][nm][t0:t0 + CYC_N] for r in RECS for s in STIMULI if CLS[s] == cls
              for t0 in sorted(DAT[r][1].get(s, [])) if t0 + CYC_N <= len(BITS[r][nm])]
        M = np.array(Ms, float)
        mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        ax.fill_between(tcyc, mu - se, mu + se, color=CCOL[cls], alpha=0.16, lw=0)
        ax.plot(tcyc, mu, color=CCOL[cls], lw=0.95)
    ax.axhline(0.5, color="#666", lw=0.6, ls=":")
    ax.set_ylim(0.15, 0.85); ax.tick_params(labelsize=5.6)
    ax.set_xlabel("time from onset (s)", labelpad=3, fontsize=6.4)
    if j == 0:
        ax.set_ylabel("by class\nP(bit = 1)", fontsize=6.2, labelpad=5)

fig.suptitle(f"Binarized responses at the 20 s high-pass window — single trial and grand averages",
             fontsize=8.2, y=1.00)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig31_binarized_20s.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig31_binarized_20s.png"), dpi=185, bbox_inches="tight")
print("wrote figures/fig31_binarized_20s.pdf")

# %% [markdown]
# ## Figure 32 — Φ(t), single trial and epoch averages

# %%
fig, axes = plt.subplots(2, 3, figsize=(12.8, 5.6))
for row, key in [(0, "core"), (1, "sens")]:
    ax = axes[row, 0]
    pt = PHI_T[REC0][key][t_ex:t_ex + CYC_N]
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    ax.semilogy(tcyc, pt, color="#111", lw=0.8, drawstyle="steps-mid")
    ax.set_ylabel(f"{key.upper()}\nΦ(t)", labelpad=5)
    if row == 0:
        ax.set_title("a  single trial\n(100 mM NaCl, herm_2)", loc="left")
    if row == 1:
        ax.set_xlabel("time from onset (s)", labelpad=4)

    ax = axes[row, 1]
    Ms = [PHI_T[r][key][t0:t0 + CYC_N] for r in RECS for s in STIMULI
          for t0 in sorted(DAT[r][1].get(s, [])) if t0 + CYC_N <= len(PHI_T[r][key])]
    M = np.array(Ms)
    mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    ax.fill_between(tcyc, mu - se, mu + se, color="#555", alpha=0.25, lw=0)
    ax.plot(tcyc, mu, color="#111", lw=1.0)
    if row == 0:
        ax.set_title("b  grand average, 232 epochs", loc="left")
    if row == 1:
        ax.set_xlabel("time from onset (s)", labelpad=4)

    ax = axes[row, 2]
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    for cls in ["attractant", "repellent", "control"]:
        Ms = [PHI_T[r][key][t0:t0 + CYC_N] for r in RECS for s in STIMULI if CLS[s] == cls
              for t0 in sorted(DAT[r][1].get(s, [])) if t0 + CYC_N <= len(PHI_T[r][key])]
        M = np.array(Ms)
        mu, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M))
        ax.fill_between(tcyc, mu - se, mu + se, color=CCOL[cls], alpha=0.16, lw=0)
        ax.plot(tcyc, mu, color=CCOL[cls], lw=1.0, label=cls if row == 0 else None)
    if row == 0:
        ax.set_title("c  by class", loc="left")
        ax.legend(frameon=False, fontsize=6, loc="upper right")
    if row == 1:
        ax.set_xlabel("time from onset (s)", labelpad=4)

fig.suptitle("Φ(t): every sample's 4-bit state mapped to its Φ under one giant TPM per substrate",
             fontsize=8.4, y=1.00)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig32_phi_timeseries.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig32_phi_timeseries.png"), dpi=190, bbox_inches="tight")
print("wrote figures/fig32_phi_timeseries.pdf")

# %% [markdown]
# ## Reading
#
# * **The single-trial Φ(t) is a spike train between a low floor and the two
#   high-Φ states.** Φ(t) correlates at 0.97 (core) / 0.87 (sensory) with a
#   simple indicator of being in the top-2 states, which occupy 16–21% of time.
#   So Φ(t) here is close to "am I in 0000 (or its complement)?" — the states
#   whose TPM rows are best estimated and most self-predictive.
# * **The apparent stimulus-locking in the grand averages does not survive
#   testing.** Mean Φ(t) in stimulus vs baseline windows, permuted within animal:
#   core z = +0.91 (p = 0.36), sensory z = +0.60 (p = 0.56). The attractant vs
#   repellent contrast on stimulus-window Φ is also null (p = 0.30 / 0.33).
# * The Φ(t) idea itself remains attractive — it needs no structure distance and
#   admits PSTH statistics — but with Φ concentrated on one or two states it
#   inherits all the noise of their occupancy. A substrate or preprocessing in
#   which Φ varies more smoothly across states would make it informative.

# %% [markdown]
# ## Φ(t) with the pre-stimulus baseline, and state rasters
#
# "Doing nothing" is causally as potent as doing something under IIT, so the
# pre-stimulus window is the reference everything must be judged against. The
# rasters above each trace show the system state directly: black = bit ON, so
# 1111 is a solid black column and 0000 a white one. If a Φ peak is a run of
# 0000 samples, it is visible as white directly above the peak.
#
# Two grand averages, deliberately different: (A) average the STATES first
# (P(bit = 1) per neuron), take the majority state per sample, assign its Φ;
# (B) average the individual-epoch Φ traces. A is a description of the average
# animal; B is the average of the actual Φ experience of each epoch.

# %%
PRE_S = 15
PRE_N = round(PRE_S * FS)
text = np.arange(-PRE_N, CYC_N) / FS

def epochs_with_pre(key_or_nm, kind):
    out = []
    for r in RECS:
        src = ST[r][key_or_nm] if kind == "state" else BITS[r][key_or_nm]
        for s in STIMULI:
            for t0 in sorted(DAT[r][1].get(s, [])):
                if t0 - PRE_N >= 0 and t0 + CYC_N <= len(src):
                    out.append(src[t0 - PRE_N:t0 + CYC_N])
    return np.array(out)

AVG_BIT, AVG_STATE, PHI_OF_AVG = {}, {}, {}
for key, neurons in [("core", INTER), ("sens", SENS)]:
    mb = np.array([epochs_with_pre(nm, "bit").mean(0) for nm in neurons])
    AVG_BIT[key] = mb
    st_avg = sum((mb[i] > 0.5).astype(int) * (2 ** i) for i in range(4))
    AVG_STATE[key] = st_avg
    PHI_OF_AVG[key] = np.array([PHI_OF[key][s] for s in st_avg])
print("epochs with full pre-window:", len(epochs_with_pre("core", "state")))

# %%
REC0 = RECS[0]; EX_STIM = "100mM NaCl"
t_ex0 = sorted(DAT[REC0][1][EX_STIM])[0]
fig = plt.figure(figsize=(12.8, 9.6))
g = fig.add_gridspec(9, 2, hspace=0.16, wspace=0.24,
                     height_ratios=[0.55, 1.0, 0.34, 0.55, 1.0, 0.34, 0.55, 1.0, 0.10])

def raster(ax, M, key, title=None):
    neurons = INTER if key == "core" else SENS
    ax.imshow(1 - M, aspect="auto", cmap="gray", vmin=0, vmax=1,
              interpolation="nearest", extent=[text[0], text[-1], -0.5, 3.5])
    ax.set_yticks(range(4)); ax.set_yticklabels(neurons[::-1], fontsize=5.0)
    ax.set_xlim(text[0], text[-1]); ax.set_xticks([])
    ax.axvline(0, color="#c00", lw=0.8); ax.axvline(15, color="#c00", lw=0.8, ls=":")
    if title:
        ax.set_title(title, loc="left", fontsize=7.2, pad=3)

def phiax(ax, y, log=True, band=None):
    ax.axvspan(0, 15, color="#000", alpha=0.06, lw=0)
    if band is not None:
        ax.fill_between(text, band[0], band[1], color="#555", alpha=0.25, lw=0)
    (ax.semilogy if log else ax.plot)(text, y, color="#111", lw=0.9,
        drawstyle="steps-mid" if log else "default")
    ax.set_xlim(text[0], text[-1]); ax.axvline(0, color="#c00", lw=0.7)
    ax.tick_params(labelsize=5.6)

for col, (key, neurons) in enumerate([("core", INTER), ("sens", SENS)]):
    ax = fig.add_subplot(g[0, col])
    Mtr = np.array([BITS[REC0][nm][t_ex0 - PRE_N:t_ex0 + CYC_N] for nm in neurons])[::-1]
    raster(ax, Mtr, key, title=f"{key.upper()} — single trial ({EX_STIM}, {REC0})")
    ax = fig.add_subplot(g[1, col])
    phiax(ax, PHI_T[REC0][key][t_ex0 - PRE_N:t_ex0 + CYC_N])
    ax.set_ylabel("Φ(t)", labelpad=4); ax.set_xticklabels([])

    ax = fig.add_subplot(g[3, col])
    raster(ax, AVG_BIT[key][::-1], key,
           title="grand average of STATES (P(bit=1)) → majority state → Φ")
    ax = fig.add_subplot(g[4, col])
    phiax(ax, PHI_OF_AVG[key])
    ax.set_ylabel("Φ(majority\nstate)", labelpad=4); ax.set_xticklabels([])

    ax = fig.add_subplot(g[6, col])
    raster(ax, AVG_BIT[key][::-1], key, title="grand average of Φ TRACES (same raster)")
    ax = fig.add_subplot(g[7, col])
    M = epochs_with_pre(key, "state")
    PM = np.array([[PHI_OF[key][s] for s in row] for row in M])
    mu, se = PM.mean(0), PM.std(0, ddof=1) / np.sqrt(len(PM))
    phiax(ax, mu, log=False, band=(mu - se, mu + se))
    ax.set_ylabel("mean Φ(t)", labelpad=4)
    ax.set_xlabel("time from onset (s)", labelpad=4)

fig.suptitle("Φ(t) with pre-stimulus baseline; state rasters above each trace (black = ON)",
             fontsize=8.4, y=0.925)
fig.savefig(os.path.join(REPO_ROOT, "figures/fig33_phi_with_rasters.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig33_phi_with_rasters.png"), dpi=185, bbox_inches="tight")
print("wrote figures/fig33_phi_with_rasters.pdf")

# %% [markdown]
# ## Against the pre-stimulus baseline: a significant OFFSET effect

# %%
win = lambda a, b: slice(round((a + PRE_S) * FS), round((b + PRE_S) * FS))
segs = {"pre (-15-0)": win(-15, 0), "stim (0-15)": win(0, 15),
        "post (16-31)": win(16, 31), "late (35-55)": win(35, 55)}
rows = []
for key in ("sens", "core"):
    M = epochs_with_pre(key, "state")
    P = np.array([[PHI_OF[key][s] for s in row] for row in M])
    m = {k: P[:, v].mean(1) for k, v in segs.items()}
    for k in segs:
        rows.append(dict(substrate=key, window=k, mean_phi=round(float(m[k].mean()), 3),
                         sem=round(float(m[k].std(ddof=1) / np.sqrt(len(P))), 3)))
    for lbl, dd in [("stim - pre", m["stim (0-15)"] - m["pre (-15-0)"]),
                    ("post - pre", m["post (16-31)"] - m["pre (-15-0)"])]:
        w = stats.wilcoxon(dd)
        print(f"{key}: {lbl} {dd.mean():+.3f}  (paired Wilcoxon p = {w.pvalue:.5f}, n = {len(P)})")
wt = pd.DataFrame(rows)
wt.to_csv(os.path.join(REPO_ROOT, "results/phi_windows_with_pre.csv"), index=False)

# per-animal and per-class consistency of the sensory offset dip
rows = []
for r in RECS:
    P = []
    for s in STIMULI:
        for t0 in sorted(DAT[r][1].get(s, [])):
            if t0 - PRE_N >= 0 and t0 + CYC_N <= len(ST[r]["sens"]):
                seg = ST[r]["sens"][t0 - PRE_N:t0 + CYC_N]
                P.append([PHI_OF["sens"][x] for x in seg])
    P = np.array(P)
    d = P[:, segs["post (16-31)"]].mean(1) - P[:, segs["pre (-15-0)"]].mean(1)
    rows.append(dict(animal=r, n_epochs=len(P), post_minus_pre=round(float(d.mean()), 3),
                     p_wilcoxon=round(float(stats.wilcoxon(d).pvalue), 4)))
pa = pd.DataFrame(rows)
pa.to_csv(os.path.join(REPO_ROOT, "results/phi_offset_dip_per_animal.csv"), index=False)
print(pa.to_string(index=False))
print(f"negative in {int((pa.post_minus_pre < 0).sum())} of 8 animals")

# %% [markdown]
# ## Reading
#
# * **Stimulus window vs pre: null on both substrates** (sensory −0.04 p = 0.66;
#   core +0.67 p = 0.22). The pre-stimulus baseline confirms the onset-locked
#   ramp was not a Φ effect.
# * **But the post-offset window is significant on the sensory substrate:**
#   Φ drops 0.32 below the pre-stimulus baseline in the 15 s after offset
#   (paired Wilcoxon p = 1e-5, n = 232), negative in 6 of 8 animals, and present
#   in every class *including control* — so it tracks the delivery event, not
#   the chemical identity.
# * The raster explains the mechanism: after offset the OFF-responding sensory
#   neurons (visible as the dark ASEL band at 16–30 s) push the system OUT of
#   the high-Φ rest state 0000. Sensory activity here *lowers* Φ, because Φ is
#   concentrated on the quiescent state.
# * This is the first Φ-level quantity in the project that distinguishes a
#   stimulus event from baseline. It is an OFF-event signature: "stimulus ON vs
#   OFF" is detected at the transition, in the direction opposite to intuition.

# %% [markdown]
# ## Does the offset dip carry chemical identity?
#
# The dip is the first Φ-level event this pipeline detects, so the natural next
# question is whether its **magnitude** differs between attractants and
# repellents. Per-epoch dip = mean Φ(post 16–31 s) − mean Φ(pre −15–0 s).

# %%
from itertools import combinations

def dips(key):
    rows = []
    for r in RECS:
        for s in STIMULI:
            for k, t0 in enumerate(sorted(DAT[r][1].get(s, []))):
                if t0 - PRE_N >= 0 and t0 + CYC_N <= len(ST[r][key]):
                    seg = ST[r][key][t0 - PRE_N:t0 + CYC_N]
                    pp = np.array([PHI_OF[key][x] for x in seg])
                    rows.append(dict(animal=r, stimulus=s, cls=CLS[s], rep=k,
                        dip=float(pp[segs["post (16-31)"]].mean() - pp[segs["pre (-15-0)"]].mean())))
    return pd.DataFrame(rows)

D_sens = dips("sens")
D_sens.to_csv(os.path.join(REPO_ROOT, "results/phi_dip_epochs.csv"), index=False)
print(D_sens.groupby("cls").dip.agg(["mean", "sem", "count"]).round(3).to_string())

# stimulus-level permutation: class labels live on stimuli, so that is the
# exchangeable unit (the project's standard design)
stim_means = D_sens.groupby("stimulus").dip.mean()
stims_nc = [s for s in STIMULI if CLS[s] != "control"]
obs = (stim_means[[s for s in stims_nc if CLS[s] == "attractant"]].mean()
       - stim_means[[s for s in stims_nc if CLS[s] == "repellent"]].mean())
labs = np.array([CLS[s] for s in stims_nc])
vals = stim_means[stims_nc].values
rg = np.random.default_rng(0)
null = []
for _ in range(20000):
    pl = rg.permutation(labs)
    null.append(vals[pl == "attractant"].mean() - vals[pl == "repellent"].mean())
null = np.array(null)
p_stim = (np.sum(np.abs(null) >= abs(obs)) + 1) / 20001
print(f"\natt - rep dip contrast: {obs:+.4f}   stimulus-level permutation p = {p_stim:.4f}")
pd.DataFrame([dict(contrast="att - rep (dip)", diff=round(float(obs), 4),
                   p=round(float(p_stim), 4))]).to_csv(
    os.path.join(REPO_ROOT, "results/phi_dip_class_test.csv"), index=False)

# %% [markdown]
# ## Mechanism: the dip is the loss of 0000 occupancy, almost identically

# %%
rows = []
for r in RECS:
    for s in STIMULI:
        for t0 in sorted(DAT[r][1].get(s, [])):
            if t0 - PRE_N >= 0 and t0 + CYC_N <= len(ST[r]["sens"]):
                seg = ST[r]["sens"][t0 - PRE_N:t0 + CYC_N]
                pp = np.array([PHI_OF["sens"][x] for x in seg])
                dip = pp[segs["post (16-31)"]].mean() - pp[segs["pre (-15-0)"]].mean()
                off = np.mean([HPS[r][nm][t0 + round(16 * FS):t0 + round(31 * FS)].mean()
                               - HPS[r][nm][t0 - PRE_N:t0].mean() for nm in SENS])
                z0 = (np.mean(seg[segs["post (16-31)"]] == 0)
                      - np.mean(seg[segs["pre (-15-0)"]] == 0))
                rows.append(dict(dip=dip, off_amp=off, d_occ0=z0))
M = pd.DataFrame(rows)
M.to_csv(os.path.join(REPO_ROOT, "results/phi_dip_mechanism.csv"), index=False)
r1 = stats.spearmanr(M.off_amp, M.dip)
r2 = stats.spearmanr(M.d_occ0, M.dip)
print(f"dip vs OFF-transient amplitude : rho = {r1.statistic:+.3f}  p = {r1.pvalue:.2e}")
print(f"dip vs Delta occupancy of 0000 : rho = {r2.statistic:+.3f}  p = {r2.pvalue:.2e}")

# %% [markdown]
# ## Reading
#
# * **The dip carries no chemical identity.** Class means differ numerically
#   (attractant −0.44, repellent −0.08, control −0.60) but the stimulus-level
#   permutation gives p = 0.31, and the control dip is the largest of the three
#   — whatever drives it is present in the vehicle deliveries too.
# * **The mechanism is fully resolved.** The per-epoch dip correlates at
#   ρ = +0.99 with the change in 0000 occupancy: Φ(t) analysis here IS occupancy
#   analysis of the rest state, passed through one large constant. The OFF
#   amplitude of the sensory quartet predicts the dip (ρ = −0.28): a larger
#   OFF-transient means less time in 0000 and hence lower Φ.
# * **Where this leaves the thread.** Φ(t) detects the delivery event (offset,
#   p = 1e-5) but not the chemical. The bottleneck is now clear and quantified:
#   Φ collapses the 16-state repertoire to essentially one informative bit
#   (in-0000 vs not). Any route to chemical identity through IIT will need the
#   per-state STRUCTURE (which distinctions and relations exist in each state),
#   not the scalar.

# %% [markdown]
# ## Figure 34 — the offset dip by stimulus, class, and mechanism

# %%
CCOL = {"attractant": "#1f6fb4", "repellent": "#c2571a", "control": "#8a8a8a"}
fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), constrained_layout=True)
ax = axes[0]
bs = D_sens.groupby(["cls", "stimulus"]).dip.agg(["mean", "sem"]).reset_index()
bs = bs.loc[sorted(bs.index, key=lambda i: (bs.loc[i, "cls"], bs.loc[i, "mean"]))].reset_index(drop=True)
ax.barh(range(len(bs)), bs["mean"], 0.62, xerr=bs["sem"],
        color=[CCOL[c] for c in bs.cls], error_kw=dict(lw=0.7))
ax.set_yticks(range(len(bs)))
ax.set_yticklabels([s if len(s) <= 13 else s[:13] for s in bs.stimulus], fontsize=5.6)
ax.axvline(0, color="#333", lw=0.8)
ax.set_xlabel("offset dip in Φ (post − pre)", labelpad=5)
ax.set_title("a  The dip is universal:\n   every stimulus class shows it", loc="left")

ax = axes[1]
cm = D_sens.groupby("cls").dip.agg(["mean", "sem"]).reindex(["attractant", "repellent", "control"])
ax.bar(range(3), cm["mean"], 0.6, yerr=cm["sem"], color=[CCOL[c] for c in cm.index],
       error_kw=dict(lw=0.8))
ax.set_xticks(range(3)); ax.set_xticklabels(cm.index, fontsize=6)
ax.axhline(0, color="#333", lw=0.8)
ax.set_ylabel("offset dip in Φ", labelpad=5)
ax.text(0.5, 0.06, f"att − rep: p = {p_stim:.2f} (stimulus-level permutation)",
        transform=ax.transAxes, fontsize=5.8, ha="center", color="#333")
ax.set_title("b  ...and carries no\n   chemical identity", loc="left")

ax = axes[2]
ax.scatter(M.d_occ0, M.dip, s=7, alpha=0.45, color="#555", lw=0)
ax.set_xlabel("Δ occupancy of state 0000\n(post − pre)", labelpad=4)
ax.set_ylabel("offset dip in Φ", labelpad=5)
ax.text(0.03, 0.95, f"Spearman ρ = {r2.statistic:+.2f}\n(OFF amplitude: ρ = {r1.statistic:+.2f})",
        transform=ax.transAxes, fontsize=6.2, va="top", color="#333")
ax.axhline(0, color="#333", lw=0.6, ls=":"); ax.axvline(0, color="#333", lw=0.6, ls=":")
ax.set_title("c  Mechanism: the dip IS the\n   loss of 0000 occupancy", loc="left")

fig.savefig(os.path.join(REPO_ROOT, "figures/fig34_offset_dip_by_class.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig34_offset_dip_by_class.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig34_offset_dip_by_class.pdf")
