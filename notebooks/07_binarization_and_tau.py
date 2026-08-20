# %% [markdown]
# # 07 — Binarization check and the choice of τ
#
# Two things this notebook settles, both raised by the reference notebook that
# marks the project's state of the art and by the *Applying IIT to Your Data*
# slide deck (Maier & Ikeda):
#
# 1. **Binarization.** The repo's `moving_window_binarize` is verified
#    **bit-for-bit identical** to the reference implementation, on all 8
#    recordings × 4 neurons plus the combined state series.
# 2. **The transition lag τ.** The deck prescribes choosing τ by
#    `argmax φ_s` rather than fixing it. Implemented here — and found **not
#    identifiable** at this data volume.
#
# **Outputs:** `figures/fig15_binarization_and_tau.pdf`,
# `results/tau_sweep_15s.csv`, `results/tau_selection.csv`

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

RECORDINGS = ["20220327_herm_2", "20220327_herm_4", "20220403_herm_2", "20220403_herm_3",
              "20220427_herm_2", "20220427_herm_3", "20220427_herm_4", "20220427_herm_5"]
NEURONS = ["AIBL", "AVEL", "AVAL", "RIML"]
STIMULI = list(ch.STIMULUS_CLASS)
CLASSES = np.array([ch.STIMULUS_CLASS[s] for s in STIMULI])

for rec in RECORDINGS:
    ch.ensure_recording(rec)

# %% [markdown]
# ## 1. Binarization, verified against the reference
#
# The reference cell is transcribed **verbatim** below, including its use of
# `.iloc` on a pandas Series, so the comparison is against the actual code
# rather than a paraphrase of it. Both the sampling rate and the window length
# are recomputed the way the reference does.

# %%
def reference_binarization(series, window_size):
    """Transcribed verbatim from the reference notebook."""
    binarized_series = np.zeros_like(series)
    for i in range(len(series)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(series), i + window_size // 2 + 1)
        window = series.iloc[start_index:end_index]
        threshold = (np.nanmax(window) + np.nanmin(window)) / 2
        binarized_series[i] = 1 if series.iloc[i] >= threshold else 0
    return binarized_series


# constants, as the reference computes them
longest_time_point_seconds = 1866.5
num_data_points = 4979
reference_fs = num_data_points / longest_time_point_seconds

FS = ch.SAMPLING_RATE_HZ
WINDOW = round(300 * FS)
print(f"sampling rate  reference {reference_fs!r}")
print(f"               repo      {FS!r}")
print(f"               identical {reference_fs == FS}")
print(f"window  reference {round(300 * reference_fs)}   repo {WINDOW}")
print(f"τ (3 s) reference {round(3 * reference_fs)}   repo {round(3 * FS)}")

# %%
all_identical = True
for rec in RECORDINGS:
    data = pd.read_csv(ch.recording_path(rec))
    names = data["neuron"].tolist()
    per_neuron = []
    ref_bits, repo_bits = [], []
    for neuron in NEURONS:
        trace = data.iloc[names.index(neuron)][9:-1].astype(float)
        ref = np.asarray(reference_binarization(trace, WINDOW), dtype=int)
        repo = ch.moving_window_binarize(trace.values, WINDOW)
        per_neuron.append(np.array_equal(ref, repo))
        ref_bits.append(ref)
        repo_bits.append(repo)
    combined_ref = sum(b * (2 ** i) for i, b in enumerate(ref_bits)).astype(int)
    combined_repo = ch.combine_states(repo_bits)
    same = all(per_neuron) and np.array_equal(combined_ref, combined_repo)
    all_identical &= same
    print(f"  {rec}: 4 traces + combined state series identical = {same}")

print(f"\nALL 8 RECORDINGS BIT-IDENTICAL: {all_identical}")
assert all_identical, "binarization diverged from the reference"

# %% [markdown]
# Two properties of this binarization worth stating plainly, since they are
# inherited rather than chosen here:
#
# * The window is **centred on each sample**, so it uses future samples — the
#   binarization is non-causal.
# * The threshold is the **mid-range** (max+min)/2 of the window, set by the two
#   most extreme values, so it is sensitive to transients rather than to the
#   bulk of the distribution.
#
# Both are kept for exact agreement with the reference.

# %% [markdown]
# ## 2. The transition lag τ
#
# The slide deck's position (its slides 29–32) is that τ is not a free parameter
# to fix by convention but should be **chosen by `argmax φ_s`** — the lag at
# which the system is most irreducible. The repo has been using a fixed τ = 3 s.
# This section implements the prescription and tests whether it is usable here.

# %%
def load_states_and_onsets(rec):
    data = pd.read_csv(ch.recording_path(rec))
    tcols = data.columns[9:-1]
    binary = [ch.moving_window_binarize(
        data.loc[data.neuron == n, tcols].iloc[0].astype(float).values, WINDOW)
        for n in NEURONS]
    states = ch.combine_states(binary)
    onsets = {}
    for onset, label in ast.literal_eval(data.iloc[0]["stimulus"]):
        onsets.setdefault(label, []).append(int(onset))
    return states, onsets


states_by_rec, onsets_by_rec = zip(*[load_states_and_onsets(r) for r in RECORDINGS])


def pooled_counts_at_tau(stimulus, tau_samples, epoch_samples):
    counts = np.zeros((16, 16))
    for states, onsets in zip(states_by_rec, onsets_by_rec):
        for t0 in onsets.get(stimulus, []):
            segment = states[t0:t0 + epoch_samples]
            if len(segment) <= tau_samples:
                continue
            for a, b in zip(segment[:-tau_samples], segment[tau_samples:]):
                counts[a, b] += 1
    return counts


def phi_s_of(counts, alpha=0.5):
    """System-level irreducibility of the pooled TPM at its most-occupied state."""
    smoothed = counts + alpha
    tpm_sbs = smoothed / smoothed.sum(1, keepdims=True)
    state_index = int(np.argmax(counts.sum(1)))
    network = pyphi.Network(convert.state_by_state2state_by_node(tpm_sbs),
                            node_labels=NEURONS)
    state = tuple((state_index >> i) & 1 for i in range(len(NEURONS)))
    return float(pyphi.new_big_phi.sia(pyphi.Subsystem(network, state)).phi)


def sweep_tau(epoch_seconds, tau_seconds=range(1, 31)):
    epoch_samples = round(epoch_seconds * FS)
    out = {}
    for stimulus in STIMULI:
        curve, best = [], (None, -np.inf)
        for ts in tau_seconds:
            tau_samples = max(1, round(ts * FS))
            if tau_samples >= epoch_samples:
                curve.append(np.nan)
                continue
            counts = pooled_counts_at_tau(stimulus, tau_samples, epoch_samples)
            if counts.sum() == 0:
                curve.append(np.nan)
                continue
            try:
                value = phi_s_of(counts)
            except Exception:
                curve.append(np.nan)
                continue
            curve.append(value)
            if value > best[1]:
                best = (ts, value)
        out[stimulus] = {"curve": curve, "tau_star": best[0], "phi_s_max": best[1]}
    return out


sweep15 = sweep_tau(15)
TAU_SECONDS = list(range(1, 31))
tau_table = pd.DataFrame([{
    "stimulus": s, "class": ch.STIMULUS_CLASS[s],
    "tau_star": sweep15[s]["tau_star"],
    "phi_s_max": round(sweep15[s]["phi_s_max"], 5),
    "phi_s_at_3s": round(sweep15[s]["curve"][2], 5),
} for s in STIMULI])
tau_table.to_csv("results/tau_sweep_15s.csv", index=False)
print(tau_table.to_string(index=False))
print(f"\nτ* range: {tau_table.tau_star.min()}–{tau_table.tau_star.max()} s")
print(f"τ* equals the fixed 3 s for {(tau_table.tau_star == 3).sum()} of 10 stimuli")
print(f"mean φ_s at τ*: {tau_table.phi_s_max.mean():.5f}  "
      f"at τ = 3 s: {tau_table.phi_s_at_3s.mean():.5f}")

# %% [markdown]
# φ_s is about twice as large at τ* as at 3 s, and no stimulus peaks at 3 s. On
# its face that argues for adopting the prescription. But note where the peaks
# sit: a 15 s epoch admits τ < 15 s, and **4 of 10 stimuli peak at 14 s** — the
# last value the window allows. That is a censored maximum, not a located one.
#
# The test is whether τ* is stable when the epoch is widened.

# %%
sweep30 = sweep_tau(30)
comparison = pd.DataFrame([{
    "stimulus": s,
    "tau_star_15s_epoch": sweep15[s]["tau_star"],
    "tau_star_30s_epoch": sweep30[s]["tau_star"],
} for s in STIMULI])
comparison["shift"] = comparison.tau_star_30s_epoch - comparison.tau_star_15s_epoch
comparison.to_csv("results/tau_selection.csv", index=False)
shift = comparison["shift"]
n_agree = int((shift == 0).sum())
print(comparison.to_string(index=False))
print(f"\nstimuli where the two epoch lengths agree: {n_agree} of {len(comparison)}")
print(f"mean |shift|: {shift.abs().mean():.1f} s  "
      f"(range {shift.min()} to {shift.max()} s)")

# %% [markdown]
# **τ* is not identifiable at this data volume.** Zero of ten stimuli agree
# between the two epoch lengths, and the mean shift is 11.2 s. Six stimuli
# roughly *double* their τ* when the epoch doubles (ratio 1.86–2.15) — exactly
# what a quantity tracking the window would do — and the other four collapse to
# τ* = 1 s.
#
# The prescription is right in principle: τ should be set by the data rather
# than by convention. But locating `argmax φ_s` needs a φ_s curve with a real
# interior maximum, and at 960 pooled frames per stimulus the curve is too flat
# and too noisy to provide one. Adopting τ* here would be fitting the epoch
# length.
#
# **Decision: keep τ = 3 s** for the reported analysis, and record τ* as
# something to revisit with more samples per epoch. For completeness the next
# cell shows what the headline test does under τ*.

# %%
def canonical(ps):
    name = lambda mech: "·".join(NEURONS[u] for u in mech)
    return ({name(tuple(d.mechanism)): float(d.phi) for d in ps.distinctions},
            {frozenset(name(tuple(m)) for m in r.mechanisms): float(r.phi)
             for r in ps.relations})


def unit_phi(S):
    total = sum(S[0].values()) + sum(S[1].values())
    return ({k: v / total for k, v in S[0].items()},
            {k: v / total for k, v in S[1].items()})


def identity_distance(A, B):
    return (sum(abs(A[0].get(k, 0.0) - B[0].get(k, 0.0)) for k in set(A[0]) | set(B[0]))
            + sum(abs(A[1].get(k, 0.0) - B[1].get(k, 0.0)) for k in set(A[1]) | set(B[1])))


def class_contrast(D, labels):
    iu = np.triu_indices(len(labels), 1)
    a = [D[i, j] for i, j in zip(*iu) if labels[i] == labels[j] == "attractant"]
    r = [D[i, j] for i, j in zip(*iu) if labels[i] == labels[j] == "repellent"]
    return np.mean(a) - np.mean(r), np.mean(a), np.mean(r)


epoch_samples = round(15 * FS)
structures_taustar = {}
for s in STIMULI:
    tau_samples = max(1, round(sweep15[s]["tau_star"] * FS))
    counts = pooled_counts_at_tau(s, tau_samples, epoch_samples)
    smoothed = counts + 0.5
    tpm_sbs = smoothed / smoothed.sum(1, keepdims=True)
    state_index = int(np.argmax(counts.sum(1)))
    network = pyphi.Network(convert.state_by_state2state_by_node(tpm_sbs),
                            node_labels=NEURONS)
    state = tuple((state_index >> i) & 1 for i in range(len(NEURONS)))
    ps = pyphi.new_big_phi.phi_structure(pyphi.Subsystem(network, state))
    structures_taustar[s] = unit_phi(canonical(ps))

n = len(STIMULI)
D_taustar = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        D_taustar[i, j] = D_taustar[j, i] = identity_distance(
            structures_taustar[STIMULI[i]], structures_taustar[STIMULI[j]])

rng = np.random.default_rng(0)
obs, mean_a, mean_r = class_contrast(D_taustar, CLASSES)
null = np.array([class_contrast(D_taustar, rng.permutation(CLASSES))[0]
                 for _ in range(20000)])
p_value = (np.sum(np.abs(null) >= abs(obs)) + 1) / 20001
print(f"under τ* : within-attractant {mean_a:.4f}  within-repellent {mean_r:.4f}")
print(f"           difference {obs:+.4f}   p = {p_value:.4f}")
print(f"under 3 s: within-attractant 1.3351  within-repellent 1.1667")
print(f"           difference +0.1684   p = 0.3757")
print(f"\nThe sign flips, and both are far from significance. The conclusion in")
print(f"notebook 06 does not depend on the choice of τ.")

# %% [markdown]
# ## Figure 15

# %%
short = {s: (s if len(s) <= 12 else s[:12]) for s in STIMULI}
COL = {"attractant": BLUE, "repellent": ORANGE, "control": GREY}

fig15 = plt.figure(figsize=(11.0, 6.6))
gs15 = fig15.add_gridspec(2, 3, hspace=0.68, wspace=0.42)

# (a) the binarization itself
ax = fig15.add_subplot(gs15[0, :2])
data0 = pd.read_csv(ch.recording_path(RECORDINGS[0]))
trace = data0.loc[data0.neuron == "AIBL", data0.columns[9:-1]].iloc[0].astype(float).values
window = slice(600, 1200)
t_axis = np.arange(len(trace))[window] / FS
half = WINDOW // 2
threshold = np.array([
    (np.nanmax(trace[max(0, i - half):min(len(trace), i + half + 1)])
     + np.nanmin(trace[max(0, i - half):min(len(trace), i + half + 1)])) / 2
    for i in range(len(trace))])
bits = ch.moving_window_binarize(trace, WINDOW)
ax.plot(t_axis, trace[window], lw=0.8, color=BLUE, label="AIBL ΔF/F")
ax.plot(t_axis, threshold[window], lw=1.0, color=ORANGE, ls="--",
        label="mid-range threshold\n(800-sample centred window)")
lo, hi = np.nanmin(trace[window]), np.nanmax(trace[window])
ax.fill_between(t_axis, lo, lo + 0.12 * (hi - lo), where=bits[window] == 1,
                color=GREY, alpha=0.45, step="mid", lw=0, label="binarized = 1")
ax.set_ylim(top=hi * 1.42)
ax.set_xlabel("time (s)", labelpad=6)
ax.set_ylabel("ΔF/F", labelpad=6)
ax.legend(frameon=False, loc="upper left", ncol=3, handlelength=1.3,
          columnspacing=1.1, borderaxespad=0.3)
ax.set_title("a  Binarization reproduces the reference notebook bit-for-bit\n"
             "(8 recordings × 4 neurons + combined state series, all identical)", loc="left")

# (b) phi_s vs tau
ax = fig15.add_subplot(gs15[0, 2])
for s in STIMULI:
    ax.plot(TAU_SECONDS, sweep15[s]["curve"], lw=0.9,
            color=COL[ch.STIMULUS_CLASS[s]], alpha=0.75)
    ax.plot([sweep15[s]["tau_star"]], [sweep15[s]["phi_s_max"]], "o", ms=3.5,
            color=COL[ch.STIMULUS_CLASS[s]], mec="none")
ax.set_xlim(1, 15)
ax.axvline(3, ls=":", lw=1.1, color="#333")
ax.axvline(14.5, ls="-", lw=1.2, color="#777")
ax.text(3.4, ax.get_ylim()[1] * 0.99, "fixed τ = 3 s", fontsize=6, color="#333", va="top")
ax.text(14.3, ax.get_ylim()[1] * 0.99, "epoch\nlimit", fontsize=6, color="#777",
        va="top", ha="right")
ax.set_xlabel("τ (s)", labelpad=6)
ax.set_ylabel("φ$_s$", labelpad=6)
ax.set_title("b  φ$_s$ rises toward the epoch limit", loc="left")

# (c) tau* under two epoch lengths
ax = fig15.add_subplot(gs15[1, 0])
y = np.arange(n)
t15 = [sweep15[s]["tau_star"] for s in STIMULI]
t30 = [sweep30[s]["tau_star"] for s in STIMULI]
for yi, (u, v) in enumerate(zip(t15, t30)):
    ax.plot([u, v], [yi, yi], color="#bbb", lw=1.0, zorder=1)
ax.scatter(t15, y, s=16, color=GREY, zorder=3, label="15 s epoch")
ax.scatter(t30, y, s=16, color=ORANGE, zorder=3, label="30 s epoch")
ax.axvline(3, ls=":", lw=1.1, color="#333")
ax.set_yticks(y)
ax.set_yticklabels([short[s] for s in STIMULI], fontsize=6)
ax.invert_yaxis()
ax.set_xlabel("τ* = argmax φ$_s$  (s)", labelpad=6)
ax.legend(frameon=False, loc="lower right", handletextpad=0.3)
ax.set_title(f"c  τ* moves with the epoch:\n"
             f"{n_agree} of {len(comparison)} stimuli agree", loc="left")

# (d) phi_s gained
ax = fig15.add_subplot(gs15[1, 1])
x = np.arange(n)
width = 0.38
ax.bar(x - width / 2, tau_table.phi_s_at_3s, width, color=GREY, label="τ = 3 s")
ax.bar(x + width / 2, tau_table.phi_s_max, width, color=ORANGE, label="τ = τ*")
ax.set_xticks(x)
ax.set_xticklabels([short[s] for s in STIMULI], rotation=90, fontsize=5.5)
ax.set_ylabel("φ$_s$", labelpad=6)
ax.legend(frameon=False, loc="upper left")
ax.set_title(f"d  φ$_s$ at τ* vs at τ = 3 s\n"
             f"(mean {tau_table.phi_s_max.mean():.3f} vs "
             f"{tau_table.phi_s_at_3s.mean():.3f})", loc="left")

# (e) the contrast under both
ax = fig15.add_subplot(gs15[1, 2])
ax.hist(null, bins=45, color=LIGHT, edgecolor="none")
ax.axvline(obs, color=ORANGE, lw=2)
ax.axvline(0.1684, color=GREY, lw=2, ls="--")
ax.text(obs - 0.04, ax.get_ylim()[1] * 0.95, f"τ*: {obs:+.3f}\np = {p_value:.2f}",
        fontsize=6.5, color=ORANGE, va="top", ha="right")
ax.text(0.1684 + 0.04, ax.get_ylim()[1] * 0.62, "τ=3s: +0.168\np = 0.38",
        fontsize=6.5, color="#555", va="top", ha="left")
ax.set_xlabel("mean within-attractant − within-repellent", labelpad=6)
ax.set_ylabel("shuffles", labelpad=6)
ax.set_title("e  Sign flips again; still null", loc="left")

fig15.savefig("figures/fig15_binarization_and_tau.pdf")
fig15.savefig("figures/fig15_binarization_and_tau.png", dpi=200)
print("wrote figures/fig15_binarization_and_tau.pdf")

# %% [markdown]
# ## Summary
#
# 1. **Binarization matches the reference exactly** — same sampling rate, same
#    800-sample centred window, same mid-range threshold, and bit-identical
#    output on all 8 recordings including the combined state series.
# 2. **τ chosen by argmax φ_s is not identifiable here.** φ_s roughly doubles at
#    τ* versus 3 s, which looks like a reason to adopt it, but τ* shifts by a
#    mean of 11.2 s when the epoch is widened from 15 s to 30 s and **no**
#    stimulus gives the same answer under both. It is tracking the window.
# 3. **The reported analysis keeps τ = 3 s**, and the conclusion does not depend
#    on that choice: under τ* the class contrast changes sign and remains far
#    from significance (p = 0.62 versus 0.38).
