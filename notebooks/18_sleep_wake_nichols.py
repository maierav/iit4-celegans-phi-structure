# %% [markdown]
# # 18 — Stage 1: sleep vs wake (Nichols et al. 2017)
#
# First stage of the staged plan: test everything this project has learned
# about building a plausible TPM on an **independent dataset**, against the
# contrast where IIT stakes its clearest directional prediction — **Φ should
# drop in sleep (lethargus)**.
#
# Data: Nichols, Eichler, Latham & Zimmer 2017, *Science* 356:eaam6851
# (OSF `kbf38`): whole-brain Ca²⁺ imaging in immobilised L4 larvae, four
# conditions — N2 prelethargus (11), N2 lethargus (12), *npr-1* prelethargus
# (10), *npr-1* lethargus (11). Identified by position/markers (pre-NeuroPAL).
#
# This notebook is self-contained: it fetches the four .mat files (~285 MB)
# if absent, inventories neuron identities per animal, extracts the core
# quartet (AIB/AVE/AVA/RIM — same classes as the project's interneuron
# quartet), runs the standing pipeline (20 s high-pass → binarize → joint
# 4-bit states → conditioned TPMs), and tests the contrast at the TPM level
# and the Φ level.

# %%
import os, sys, json, urllib.request
import numpy as np
import pandas as pd

REPO_ROOT = "."
DATA = os.path.join(REPO_ROOT, "data_nichols")
os.makedirs(DATA, exist_ok=True)

OSF_FILES = {   # name -> osfstorage path id (from api.osf.io/v2/nodes/kbf38/files/osfstorage/)
    "n2_prelet.mat":   "5d24867945253a001939ac2c",
    "n2_let.mat":      "5d248679a26b340016064333",
    "npr1_prelet.mat": "5d24866f1c5b4a001b9bde6e",
    "npr1_let.mat":    "5d248670114a420019020fd1",
    "readme_Nichols2017.txt": "5d24879645253a001b3994a8",
}
WB = "https://files.de-1.osf.io/v1/resources/kbf38/providers/osfstorage"

def fetch(name, pid):
    p = os.path.join(DATA, name)
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    url = f"{WB}/{pid}"
    try:                      # normal path (Colab): follow redirects
        urllib.request.urlretrieve(url, p)
    except Exception:         # sandboxed environments: follow the 302 by hand,
        import urllib.error   # rewriting the GCS URL to its bucket-qualified host
        req = urllib.request.Request(url, method="GET")
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k): return None
        op = urllib.request.build_opener(NoRedirect)
        try: op.open(req)
        except urllib.error.HTTPError as e:
            loc = e.headers["Location"]
        loc = loc.replace("https://storage.googleapis.com/cos-osf-prod-files-de-1/",
                          "https://cos-osf-prod-files-de-1.storage.googleapis.com/")
        urllib.request.urlretrieve(loc, p)
    return p

for name, pid in OSF_FILES.items():
    fetch(name, pid)
print("data files:", {n: os.path.getsize(os.path.join(DATA, n)) // 10**6 for n in OSF_FILES})

# %% [markdown]
# ## Inventory: which animals carry the core quartet?

# %%
import h5py

FILES = {"n2_prelet": "n2_prelet.mat", "n2_let": "n2_let.mat",
         "npr1_prelet": "npr1_prelet.mat", "npr1_let": "npr1_let.mat"}
CORE_CLASSES = {"AIB": ["AIBL", "AIBR"], "AVE": ["AVEL", "AVER"],
                "AVA": ["AVAL", "AVAR"], "RIM": ["RIML", "RIMR"]}

def animal_ids(f, ids_ref):
    out = []
    for r in f[ids_ref][:].ravel():
        v = f[r]
        if v.dtype == np.uint16 and v.size > 0:
            out.append("".join(chr(c) for c in v[:].ravel()))
        elif v.dtype == np.object_:
            inner = [ "".join(chr(c) for c in f[r2][:].ravel())
                      for r2 in v[:].ravel() if f[r2].dtype == np.uint16 and f[r2].size > 0 ]
            out.append(inner[0] if inner else "")
        else:
            out.append("")
    return [x.strip() for x in out]

rows, TRACES, PICK = [], {}, {}
for cond, fn in FILES.items():
    fh = h5py.File(os.path.join(DATA, fn), "r")
    root = fh[cond]
    for a in range(root["IDs"].shape[0]):
        ids = animal_ids(fh, root["IDs"][a, 0])
        ids_set = {i for i in ids if i}
        tr = np.array(fh[root["traces"][a, 0]])
        if tr.shape[0] > tr.shape[1]:
            tr = tr.T
        fps = float(np.array(fh[root["fps"][a, 0]]).ravel()[0])
        row = dict(condition=cond, animal=a, n_neurons_recorded=tr.shape[0],
                   T=tr.shape[1], fps=round(fps, 3), n_identified=len(ids_set))
        idx = {}
        for cl, names in CORE_CLASSES.items():
            found = next(((c, ids.index(c)) for c in [names[0], names[1], cl] if c in ids), None)
            idx[cl] = found
            row[cl] = found[0] if found else ""
        rows.append(row)
        if all(v is not None for v in idx.values()):
            TRACES[(cond, a)] = {cl: tr[i_, :].astype(float) for cl, (nm, i_) in idx.items()}
            TRACES[(cond, a)]["fps"] = fps
            PICK[(cond, a)] = {cl: nm for cl, (nm, i_) in idx.items()}
    fh.close()

inv = pd.DataFrame(rows)
inv.to_csv(os.path.join(REPO_ROOT, "results/nichols_coverage.csv"), index=False)
print("animals per condition:", inv.groupby("condition").size().to_dict())
print("usable (full quartet, any side):", len(TRACES), "of", len(inv))
for cl in CORE_CLASSES:
    print(f"  {cl}: identified in {int(inv[cl].str.len().gt(0).sum())}/{len(inv)}")

# %% [markdown]
# **GO**: 39/44 animals carry all four classes (26 with the exact left-cell
# quartet; where a left cell is missing the right homolog substitutes —
# recorded per animal in the coverage CSV). The sensory quartet is absent
# (pre-NeuroPAL head-ganglia IDs), so stage 1 runs on the core quartet only.

# %% [markdown]
# ## Pipeline: 20 s high-pass → binarize → joint states → conditioned TPMs
#
# Identical to notebooks 11–17, with the window in *seconds* and each
# animal's own fps setting the kernel size (fps varies 2.8–4.6 Hz here).
# τ = 1 native sample (220–360 ms — same regime as the project's 375 ms).

# %%
from scipy.ndimage import median_filter
from scipy.spatial.distance import jensenshannon
from scipy import stats

W_HP = 20.0
def hp_bits(x, fps):
    xf = np.where(np.isfinite(x), x, np.nanmedian(x))
    return (xf - median_filter(xf, size=max(3, round(W_HP * fps)), mode="nearest") > 0).astype(int)

STATES = {}
for k, v in TRACES.items():
    bits = [hp_bits(v[cl], v["fps"]) for cl in CORE_CLASSES]  # order AIB,AVE,AVA,RIM
    STATES[k] = sum(b * (2 ** i) for i, b in enumerate(bits))

def counts_of(keys):
    C = np.zeros((16, 16))
    for k in keys:
        np.add.at(C, (STATES[k][:-1], STATES[k][1:]), 1)
    return C
CC = {c: counts_of([k for k in STATES if k[0] == c]) for c in FILES}
print("transitions:", {c: int(v.sum()) for c, v in CC.items()},
      "| total:", int(sum(v.sum() for v in CC.values())))

# bit behaviour per condition — the notebook-11 lesson check
bs = []
for cond in FILES:
    ks = [k for k in STATES if k[0] == cond]
    bs.append(dict(condition=cond, n=len(ks),
        state_change_frac=round(float(np.mean([np.mean(STATES[k][:-1] != STATES[k][1:]) for k in ks])), 3),
        states_visited=round(float(np.mean([len(set(STATES[k].tolist())) for k in ks])), 1)))
bitstats = pd.DataFrame(bs)
bitstats.to_csv(os.path.join(REPO_ROOT, "results/nichols_bit_stats.csv"), index=False)
print(bitstats.to_string(index=False))

# %% [markdown]
# **Caveat carried forward:** the joint state *changes faster* in lethargus
# (0.88 vs 0.72 per sample). Quiescent traces are flat, so a
# median-subtraction bit sits near threshold and may be noise-dominated
# during sleep. Everything below must be read with that in mind; the
# full-volume bootstrap of the φ-maps is the next check this file gets.

# %% [markdown]
# ## Test 1 — TPM level, animal-label permutation

# %%
def P_(C): return (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
def row_jsd(A, B): return float(np.mean([jensenshannon(P_(A)[s], P_(B)[s], base=2) for s in range(16)]))

rng = np.random.default_rng(0)
res_tpm = []
for strain, (cw, cs) in [("N2", ("n2_prelet", "n2_let")), ("npr1", ("npr1_prelet", "npr1_let"))]:
    wake = [k for k in STATES if k[0] == cw]; slp = [k for k in STATES if k[0] == cs]
    obs = row_jsd(counts_of(wake), counts_of(slp))
    pool = wake + slp; nw = len(wake); null = []
    for _ in range(300):
        pm = rng.permutation(len(pool))
        null.append(row_jsd(counts_of([pool[i] for i in pm[:nw]]),
                            counts_of([pool[i] for i in pm[nw:]])))
    null = np.array(null)
    res_tpm.append(dict(strain=strain, jsd_obs=round(obs, 4),
                        null_mean=round(float(null.mean()), 4), null_sd=round(float(null.std()), 4),
                        z=round(float((obs - null.mean()) / null.std()), 2),
                        p=round(float((np.sum(null >= obs) + 1) / 301), 4)))
    print(res_tpm[-1])
pd.DataFrame(res_tpm).to_csv(os.path.join(REPO_ROOT, "results/nichols_tpm_contrast.csv"), index=False)

# %% [markdown]
# ## Test 2 — IIT's prediction: time-averaged φ_s drops in sleep
#
# φ-map = φ_s at each of the 16 states of a conditioned TPM. Per-animal
# time-averaged φ = occupancy · φ-map. Two map choices: the **shared strain
# map** (pooled wake+sleep; isolates occupancy differences — the cleaner
# test) and each condition's **own map** (couples map and occupancy
# changes). Unit of replication: the animal; one-sided Mann-Whitney.

# %%
os.environ["PYPHI_WELCOME_OFF"] = "yes"
import pyphi
from pyphi import convert
pyphi.config.PROGRESS_BARS = False; pyphi.config.PARALLEL = False

def phi_map(C):
    net = pyphi.Network(convert.state_by_state2state_by_node(P_(C)),
                        node_labels=["AIB", "AVE", "AVA", "RIM"])
    out = np.zeros(16)
    for si in range(16):
        out[si] = float(pyphi.new_big_phi.sia(
            pyphi.Subsystem(net, tuple((si >> i) & 1 for i in range(4)))).phi)
    return out

PM = {c: phi_map(CC[c]) for c in FILES}
PM_STRAIN = {"N2": phi_map(counts_of([k for k in STATES if k[0].startswith("n2")])),
             "npr1": phi_map(counts_of([k for k in STATES if k[0].startswith("npr1")]))}
pd.DataFrame({("phi_" + c): PM[c] for c in FILES},
             index=[format(i, "04b") for i in range(16)]).rename_axis(
             "state (bits: RIM,AVA,AVE,AIB)").to_csv(
             os.path.join(REPO_ROOT, "results/nichols_phi_maps.csv"))

rows = []
for (cond, a), st in STATES.items():
    strain = "N2" if cond.startswith("n2") else "npr1"
    occ = np.bincount(st, minlength=16) / len(st)
    rows.append(dict(strain=strain, state="sleep" if cond.endswith("_let") else "wake",
                     cond=cond, animal=a,
                     phi_own=float(occ @ np.nan_to_num(PM[cond])),
                     phi_shared=float(occ @ np.nan_to_num(PM_STRAIN[strain])),
                     occ0=float(occ[0])))
pa = pd.DataFrame(rows)
pa.to_csv(os.path.join(REPO_ROOT, "results/nichols_phi_per_animal.csv"), index=False)
for strain in ("N2", "npr1"):
    for col in ("phi_shared", "phi_own"):
        w = pa[(pa.strain == strain) & (pa.state == "wake")][col]
        s_ = pa[(pa.strain == strain) & (pa.state == "sleep")][col]
        u = stats.mannwhitneyu(s_, w, alternative="less")
        print(f"{strain} {col}: sleep {s_.mean():+.4f} vs wake {w.mean():+.4f}  p = {u.pvalue:.4f}")

# %% [markdown]
# ## Reading
#
# * **Gate 1 passed — the pipeline transfers.** On a dataset from a different
#   lab, indicator-era, life stage, and fps, the conditioned TPMs separate
#   sleep from wake far beyond the animal-shuffle null in both strains
#   (z = +5.8 N2, +4.4 *npr-1*).
# * **Gate 2 passed with one dissent — Φ drops in sleep.** The shared-map
#   test (occupancy differences only) gives sleep < wake in both strains
#   (p = 0.0002, 0.0003); the own-map test agrees in *npr-1* (p = 0.0001)
#   but not N2 (p = 0.31). 3 of 4 tests, and the cleaner isolation passes in
#   both.
# * **Scale caveat.** φ_s values here are ~100× smaller than the project's
#   chemosensory TPMs (max 0.13 vs 36): these bits flip fast (0.7–0.9 per
#   sample), so determinism is low everywhere. The *contrast* is what
#   passes, not any absolute Φ level — and the faster flipping in lethargus
#   means part of the drop could reflect noisier bits in quiescence rather
#   than lower integration. The full-volume bootstrap of these φ-maps, and a
#   binarization-sensitivity sweep on the lethargus files, are the two
#   checks this result must survive before it is called a positive control.

# %% [markdown]
# ## Figure 48

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
BLUE, ORANGE, GREY = "#1f6fb4", "#c2571a", "#8a8a8a"
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.6), constrained_layout=True)
ax = axes[0]
labels = ["AIB", "AVE", "AVA", "RIM"]
cov = [int(inv[cl].str.len().gt(0).sum()) for cl in labels]
exact_left = int(sum(all(PICK[k][cl] == CORE_CLASSES[cl][0] for cl in CORE_CLASSES) for k in PICK))
ax.bar(range(4), cov, 0.6, color=BLUE)
ax.axhline(44, ls=":", lw=0.9, color="#333")
ax.text(3.45, 44.8, "44 animals", fontsize=6, ha="right", color="#333")
ax.bar(4.2, exact_left, 0.6, color=ORANGE)
ax.bar(5.0, len(TRACES), 0.6, color="#4a7a4a")
ax.set_xticks(list(range(4)) + [4.2, 5.0])
ax.set_xticklabels(labels + ["exact\nL-quartet", "any-side\nquartet"], fontsize=6)
ax.set_ylabel("animals with the neuron identified", labelpad=5)
ax.set_ylim(0, 50)
ax.set_title("a  Quartet coverage in Nichols 2017: GO\n   (39/44 usable, 151.6k transitions)", loc="left")
ax = axes[1]
for i, r_ in enumerate(res_tpm):
    ax.bar(i - 0.17, r_["jsd_obs"], 0.3, color=ORANGE, label="sleep vs wake" if i == 0 else None)
    ax.bar(i + 0.17, r_["null_mean"], 0.3, color=GREY, yerr=r_["null_sd"],
           label="animal-shuffle null" if i == 0 else None, error_kw=dict(lw=0.8))
    ax.text(i - 0.17, r_["jsd_obs"] + 0.004, f"z = {r_['z']:+.1f}", ha="center", fontsize=6.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["N2", "npr-1"])
ax.set_ylabel("TPM row JSD", labelpad=5)
ax.legend(frameon=False, fontsize=6)
ax.set_ylim(0, 0.215)
ax.set_title("b  Sleep vs wake separates at the TPM\n   level in both strains", loc="left")
ax = axes[2]
posmap = {("N2", "wake"): 0, ("N2", "sleep"): 1, ("npr1", "wake"): 2.4, ("npr1", "sleep"): 3.4}
for (strain, state), x0 in posmap.items():
    v = pa[(pa.strain == strain) & (pa.state == state)].phi_shared.values
    col = BLUE if state == "wake" else "#7a4a8a"
    ax.scatter(np.full(len(v), x0) + np.linspace(-0.12, 0.12, len(v)), v, s=16, color=col, lw=0, zorder=3)
    ax.plot([x0 - 0.2, x0 + 0.2], [v.mean()] * 2, color=col, lw=1.6)
ax.axhline(0, color="#ccc", lw=0.6)
ax.set_xticks(list(posmap.values())); ax.set_xticklabels(["wake", "sleep", "wake", "sleep"], fontsize=7)
ax.text(0.5, -0.17, "N2", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5)
ax.text(2.9, -0.17, "npr-1", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.5)
for strain, xm in [("N2", 0.5), ("npr1", 2.9)]:
    w = pa[(pa.strain == strain) & (pa.state == "wake")].phi_shared
    s_ = pa[(pa.strain == strain) & (pa.state == "sleep")].phi_shared
    ax.text(xm, 0.0335, f"p = {stats.mannwhitneyu(s_, w, alternative='less').pvalue:.4f}",
            ha="center", fontsize=6.5)
ax.set_ylim(-0.018, 0.037)
ax.set_ylabel("time-averaged φ_s per animal\n(shared strain φ-map × occupancy)", labelpad=5)
ax.set_title("c  IIT's prediction — φ drops in sleep:\n   holds in both strains (one-sided MW)", loc="left")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig48_sleep_wake.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(REPO_ROOT, "figures/fig48_sleep_wake.png"), dpi=200, bbox_inches="tight")
print("wrote figures/fig48")
