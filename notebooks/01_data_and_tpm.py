# %% [markdown]
# # 01 — Data, binarization, and the TPM
#
# **What this notebook does:** loads the *C. elegans* whole-brain chemosensory
# imaging data, reproduces the original binarization and transition-probability
# matrix (TPM) exactly, and then measures how much data is actually available
# **per stimulus** — which turns out to be the factor that limits the whole
# project.
#
# Runs as-is in Google Colab (`Runtime > Run all`) or locally.
#
# **Data source:** [chemosensory-data.worm.world](https://chemosensory-data.worm.world/index.html)
# — whole-brain NeuroPAL recordings, 189 identified neurons, 4979 frames at
# ~2.667 Hz (31 min), with 10 chemical stimuli each delivered 3 times.
#
# **Outputs written:** `figures/fig01_traces_and_tpm.pdf`,
# `figures/fig02_epoch_budget.pdf`, `results/epoch_budget.csv`,
# `results/tpm_20220327_herm_2.npy`

# %%
# --- Environment setup: works in Colab and locally -------------------------
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    # Clone the repo so `src/` and the figure style are available.
    if not os.path.exists("iit4-celegans-phi-structure"):
        subprocess.run(
            ["git", "clone", "--quiet",
             "https://github.com/maierav/iit4-celegans-phi-structure.git"],
            check=True,
        )
    os.chdir("iit4-celegans-phi-structure")
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "gdown"], check=True)

REPO_ROOT = os.path.abspath(os.getcwd())
if os.path.basename(REPO_ROOT) == "notebooks":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
    os.chdir(REPO_ROOT)

sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("data", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx

import ces_hypergraph as ch

print("repo root:", REPO_ROOT)
print("sampling rate: %.4f Hz" % ch.SAMPLING_RATE_HZ)

# %% [markdown]
# ## Figure style
#
# One place to set fonts/colors so every figure in the repo matches. Figures are
# saved as **PDF (vector)** so they stay sharp at any zoom.

# %%
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
CLASS_COLOR = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}

mpl.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlelocation": "left",
    "pdf.fonttype": 42,   # editable text in vector output
    "ps.fonttype": 42,
})

# %% [markdown]
# ## 1. Download the recordings
#
# Eight hermaphrodite recordings, by Google Drive id (taken from the original
# notebook 5). Each file is ~4.6 MB.

# %%
def fetch_recording(key, drive_id, outdir="data"):
    """Download one recording CSV if not already present.

    Two paths are tried. `curl` against the Drive download endpoint follows the
    303 to `drive.usercontent.google.com` and works in restricted-network
    environments; `gdown` is the fallback and is what usually runs in Colab.
    """
    path = os.path.join(outdir, f"{key}.csv")
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path

    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    try:
        subprocess.run(["curl", "-sSL", "-o", path, url], check=True, timeout=300)
        if os.path.getsize(path) > 1_000_000:
            return path
    except Exception:
        pass

    import gdown
    gdown.download(url, path, quiet=True)
    return path


paths = {k: fetch_recording(k, v) for k, v in ch.HERM_DRIVE_IDS.items()}
sizes = {k: round(os.path.getsize(p) / 1e6, 1) for k, p in paths.items()}
print("downloaded (MB):", sizes)

# %% [markdown]
# ## 2. What is in one recording

# %%
PRIMARY = "20220327_herm_2"
df, time_cols = ch.load_recording(paths[PRIMARY])

print("shape:", df.shape)
print("neurons identified:", df["neuron"].nunique())
print("time points:", len(time_cols))
print("neuron groups:", sorted(df["neuron_group"].unique()))

epochs = ch.stimulus_epochs(df)
print("\nstimuli (label: onsets in samples)")
for label, onsets in epochs.items():
    print(f"  {label:16s} {onsets}  [{ch.STIMULUS_CLASS[label]}]")

# %% [markdown]
# ### Neuron selection: a substantive caveat
#
# The original notebooks analyze **AIBL, AVEL, AVAL, RIML**. These are an
# interneuron (AIB) and premotor/ring interneurons (AVE, AVA, RIM) of the
# locomotor command circuit — *not* sensory neurons. The stated aim of the
# project is the "main sensory neurons".
#
# The canonical amphid chemosensory neurons are all present in these files, at
# full confidence. We keep the original four for continuity below, but this is
# a scientific choice worth revisiting.

# %%
avail = df[["neuron", "neuron_group", "conf"]].set_index("neuron")
check = ch.NOTEBOOK_NEURONS + ch.SENSORY_NEURONS
rows = []
for n in check:
    if n in avail.index:
        rows.append({
            "neuron": n,
            "group": avail.loc[n, "neuron_group"],
            "conf": avail.loc[n, "conf"],
            "used_in_notebooks": n in ch.NOTEBOOK_NEURONS,
        })
neuron_table = pd.DataFrame(rows)
print(neuron_table.to_string(index=False))

# %% [markdown]
# ## 3. Binarize and build the TPM (reproducing the original pipeline)
#
# The original recipe: threshold each trace at the mid-range of a 300 s window
# centred on each sample, combine the four bits into a state 0–15, then count
# transitions at a lag of tau = 3 s.
#
# Both steps are kept **exactly** as published so the numbers match. Two
# properties to be aware of: the window is *non-causal* (it uses future
# samples), and a mid-range threshold is set by the two most extreme values in
# the window.

# %%
WINDOW_S, TAU_S = 300, 3
window = round(WINDOW_S * ch.SAMPLING_RATE_HZ)
tau = round(TAU_S * ch.SAMPLING_RATE_HZ)
print(f"window = {window} samples, tau = {tau} samples")

binary = [
    ch.moving_window_binarize(ch.neuron_trace(df, time_cols, n), window)
    for n in ch.NOTEBOOK_NEURONS
]
states = ch.combine_states(binary)
tpm, row_counts = ch.build_tpm(states, tau, n_units=4)

np.save(f"results/tpm_{PRIMARY}.npy", tpm)
print("state occupancy:", dict(zip(*[a.tolist() for a in np.unique(states, return_counts=True)])))
print("TPM rows with >=1 observation:", int((row_counts > 0).sum()), "/ 16")

# %% [markdown]
# ## Figure 1 — traces, binarization, and the resulting TPM

# %%
fig1 = plt.figure(figsize=(11, 5.4))
gs = fig1.add_gridspec(3, 2, width_ratios=[2.1, 1], height_ratios=[1, 1, 1],
                       hspace=0.55, wspace=0.22)

t_min = np.arange(len(time_cols)) / ch.SAMPLING_RATE_HZ / 60
neuron_colors = dict(zip(ch.NOTEBOOK_NEURONS, [BLUE, ORANGE, "#3c8a3c", "#7a4fa3"]))

# (a) raw traces with stimulus bands
ax_a = fig1.add_subplot(gs[0, 0])
for n in ch.NOTEBOOK_NEURONS:
    ax_a.plot(t_min, ch.neuron_trace(df, time_cols, n), lw=0.5,
              color=neuron_colors[n], label=n)
for label, onsets in epochs.items():
    cls = ch.STIMULUS_CLASS[label]
    for o in onsets:
        ax_a.axvspan(o / ch.SAMPLING_RATE_HZ / 60,
                     (o + 40) / ch.SAMPLING_RATE_HZ / 60,
                     color=CLASS_COLOR[cls], alpha=0.13, lw=0)
ax_a.set_ylabel("fluorescence")
ax_a.set_xlim(0, t_min[-1])
ax_a.legend(frameon=False, ncol=4, loc="upper right")
ax_a.set_title("a  Whole-recording traces; shaded bands = stimulus deliveries")

# (b) binarized state series
ax_b = fig1.add_subplot(gs[1, 0], sharex=ax_a)
ax_b.step(t_min, states, where="mid", lw=0.5, color="#333")
ax_b.set_ylabel("state (0–15)")
ax_b.set_yticks([0, 5, 10, 15])
ax_b.set_title("b  After binarization: one integer state per frame")

# (c) occupancy
ax_c = fig1.add_subplot(gs[2, 0])
occ = np.array([(states == s).sum() for s in range(16)])
ax_c.bar(np.arange(16), occ, color=BLUE, width=0.72)
ax_c.set_yscale("symlog", linthresh=1)
ax_c.set_xticks(np.arange(16))
ax_c.set_xticklabels([f"{i:04b}" for i in range(16)], rotation=90, fontsize=5)
ax_c.set_xlabel("state (bits: RIML AVAL AVEL AIBL)", labelpad=7)
ax_c.set_ylabel("frames")
frac = 100 * occ[[0, 15]].sum() / occ.sum()
ax_c.set_title(f"c  {frac:.0f}% of frames sit in all-off or all-on")

# (d) TPM
ax_d = fig1.add_subplot(gs[:, 1])
im = ax_d.imshow(tpm, cmap="viridis", vmin=0, vmax=1)
ax_d.set_xticks(np.arange(16))
ax_d.set_yticks(np.arange(16))
ax_d.set_xticklabels([f"{i:04b}" for i in range(16)], rotation=90, fontsize=5)
ax_d.set_yticklabels([f"{i:04b}" for i in range(16)], fontsize=5)
ax_d.set_xlabel("next state", labelpad=7)
ax_d.set_ylabel("current state")
ax_d.set_title(f"d  TPM, full recording (tau = {TAU_S} s)")
cb = fig1.colorbar(im, ax=ax_d, fraction=0.046, pad=0.03)
cb.set_label("transition probability", fontsize=6)
cb.ax.tick_params(labelsize=5)

fig1.savefig("figures/fig01_traces_and_tpm.pdf")
fig1.savefig("figures/fig01_traces_and_tpm.png", dpi=200)
print("wrote figures/fig01_traces_and_tpm.pdf")

# %% [markdown]
# ## 4. The binding constraint: per-stimulus data budget
#
# The project needs a Φ-structure **per stimulus**, so the TPM must be built
# from that stimulus's frames only. A 4-neuron TPM has 16 x 16 = 256 free
# parameters. One stimulus provides 3 repeats x ~40 frames (15 s) = **120
# frames**.
#
# Below we measure, for every recording and every stimulus, how many of the 16
# states are actually visited.

# %%
EPOCH_LEN = 40  # ~15 s response window

records = []
for key, path in paths.items():
    d_i, tc_i = ch.load_recording(path)
    present = [n for n in ch.NOTEBOOK_NEURONS if n in set(d_i["neuron"])]
    if len(present) < 4:
        print(f"  {key}: missing {set(ch.NOTEBOOK_NEURONS) - set(present)}")
        continue
    bin_i = [
        ch.moving_window_binarize(ch.neuron_trace(d_i, tc_i, n), window)
        for n in ch.NOTEBOOK_NEURONS
    ]
    st_i = ch.combine_states(bin_i)
    for label, onsets in ch.stimulus_epochs(d_i).items():
        seg = np.concatenate([st_i[o:o + EPOCH_LEN] for o in onsets])
        records.append({
            "recording": key,
            "stimulus": label,
            "class": ch.STIMULUS_CLASS[label],
            "n_frames": len(seg),
            "distinct_states": len(set(seg.tolist())),
        })

budget = pd.DataFrame(records)
budget.to_csv("results/epoch_budget.csv", index=False)

print("\nrecordings analyzed:", budget["recording"].nunique())
print("\ndistinct states visited per stimulus epoch (of 16):")
print(budget.groupby("class")["distinct_states"]
      .agg(["count", "mean", "min", "max"]).round(2).to_string())

# %% [markdown]
# ### Pooling across recordings
#
# If epochs of the same stimulus **class** are pooled across all recordings,
# the budget improves by roughly an order of magnitude. This is the basis of
# the proposed fix.

# %%
pooled = (budget.groupby("class")
          .agg(epochs=("n_frames", "size"), total_frames=("n_frames", "sum"))
          .assign(frames_per_tpm_param=lambda x: (x.total_frames / 256).round(1)))
print(pooled.to_string())

# %% [markdown]
# ## Figure 2 — the sampling problem

# %%
fig2, axes2 = plt.subplots(1, 3, figsize=(11, 3.3))
order = ["attractant", "repellent", "control"]
rng = np.random.default_rng(0)

# (a) coverage per epoch
ax = axes2[0]
for i, cls in enumerate(order):
    v = budget.loc[budget["class"] == cls, "distinct_states"].values
    ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=13,
               color=CLASS_COLOR[cls], alpha=0.75,
               linewidths=0.4, edgecolors="white", zorder=3)
    ax.hlines(np.median(v), i - 0.29, i + 0.29,
              color=CLASS_COLOR[cls], lw=2.3, zorder=4)
ax.axhline(16, ls=":", lw=1, color="#444")
ax.text(2.45, 15.4, "all 16 states\nneeded", ha="right", va="top",
        fontsize=6, color="#444")
ax.set_xticks(range(3))
ax.set_xticklabels(order)
ax.set_ylim(0, 17.5)
ax.set_xlim(-0.5, 2.5)
ax.set_ylabel("distinct states visited", labelpad=7)
ax.set_title("a  One epoch never samples the state space")

# (b) pooled vs single-epoch occupancy
ax = axes2[1]
single = np.zeros(16)
for o in epochs["100mM NaCl"]:
    for s in range(16):
        single[s] += (states[o:o + EPOCH_LEN] == s).sum()
xs = np.arange(16)
ax.bar(xs - 0.2, occ, width=0.4, color=BLUE, label="full recording (n=4979)")
ax.bar(xs + 0.2, single, width=0.4, color=ORANGE,
       label="one stimulus, 3 reps (n=120)")
ax.set_yscale("symlog", linthresh=1)
ax.set_xticks(xs)
ax.set_xticklabels([f"{i:04b}" for i in xs], rotation=90, fontsize=5)
ax.set_xlabel("state (bits: RIML AVAL AVEL AIBL)", labelpad=7)
ax.set_ylabel("frames", labelpad=7)
ax.legend(frameon=False, loc="upper center")
ax.set_title("b  A single epoch leaves most rows empty")

# (c) per-stimulus means
ax = axes2[2]
per_stim = (budget.groupby(["class", "stimulus"])["distinct_states"]
            .mean().reset_index().sort_values("distinct_states"))
ypos = np.arange(len(per_stim))
ax.hlines(ypos, 0, per_stim["distinct_states"],
          color=[CLASS_COLOR[c] for c in per_stim["class"]], lw=1.1, alpha=0.6)
ax.scatter(per_stim["distinct_states"], ypos, s=26,
           color=[CLASS_COLOR[c] for c in per_stim["class"]], zorder=3)
ax.set_yticks(ypos)
ax.set_yticklabels(per_stim["stimulus"], fontsize=6)
ax.set_xlabel("mean distinct states per epoch", labelpad=7)
ax.set_xlim(0, 16)
ctrl_mean = budget.loc[budget["class"] == "control", "distinct_states"].mean()
att_mean = budget.loc[budget["class"] == "attractant", "distinct_states"].mean()
ax.set_title("c  Controls score above attractants")
ax.text(0.97, 0.05,
        f"control {ctrl_mean:.1f} > attractant {att_mean:.1f}\n(a noise signature)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6, color="#444")

fig2.tight_layout()
fig2.savefig("figures/fig02_epoch_budget.pdf")
fig2.savefig("figures/fig02_epoch_budget.png", dpi=200)
print("wrote figures/fig02_epoch_budget.pdf")

# %% [markdown]
# ## Summary
#
# 1. The pipeline reproduces: 8 recordings, all four original neurons present.
# 2. State occupancy is dominated by two states (all-off / all-on).
# 3. **Per stimulus, a mean of ~5 of 16 states are visited** — and controls
#    score *higher* than attractants, which is what noise looks like, not
#    biology.
# 4. Pooling by stimulus class across recordings raises the budget by ~10x and
#    is the recommended path forward.
#
# Next: `02_phi_structure.ipynb` unfolds the Φ-structure and shows why the
# published connectivity matrix returns Φ = 0.
