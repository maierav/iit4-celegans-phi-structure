"""
ces_hypergraph.py — shared helpers for IIT 4.0 Phi-structure analysis of
C. elegans chemosensory imaging data.

Everything in this module is plain NumPy/pandas/PyPhi so it runs identically in
Google Colab and in a local conda environment.

Contents
--------
Data layer
    load_recording          read one chemosensory-data CSV
    neuron_trace            pull a single neuron's fluorescence trace
    moving_window_binarize  the notebooks' original binarization (kept for
                            exact reproduction, including its non-causal window)
    combine_states          per-neuron binary traces -> integer state series
    build_tpm               state-by-state TPM from a state series
    stimulus_epochs         parse the stimulus column into onsets per label

IIT layer
    make_subsystem          TPM -> pyphi.Subsystem (with optional connectivity)
    phi_structure           unfold the IIT 4.0 Phi-structure
    ces_hypergraph          PhiStructure -> weighted hypergraph
                            (distinctions = nodes, relation FACES = hyperedges)

Comparison layer
    pairwise_projection     the lossy nb1/nb7 representation (pairwise only)
    ces_distance_pairwise   brute-force bijection distance (nb1/nb7 algorithm)
    ces_distance_hypergraph an earlier assignment-based prototype, kept so
                            notebook 03 can reproduce the comparison tests.
                            The distance in use is src/gold_standard.py.

Stimulus labels follow the worm.world chemosensory panel.
"""

from __future__ import annotations

import ast
import itertools
import math
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

#: The four neurons used throughout the original notebooks. NOTE: these are
#: interneurons / premotor interneurons of the locomotor command circuit, NOT
#: amphid sensory neurons. See README "Neuron selection".
NOTEBOOK_NEURONS = ["AIBL", "AVEL", "AVAL", "RIML"]

#: Canonical amphid chemosensory neurons, all present in these recordings.
SENSORY_NEURONS = ["ASEL", "ASER", "AWAL", "AWCL", "ASHL", "ADLL", "AWBL", "ASKL"]

ATTRACTANTS = ["100mM NaCl", "e-2 IAA", "e-6 IAA", "OP50"]
REPELLENTS = ["450mM NaCl", "1uM ascr#3", "10mM CuSO4", "800mM Sorbitol"]
CONTROLS = ["Control", "Fluorescein"]

STIMULUS_CLASS = {
    **{s: "attractant" for s in ATTRACTANTS},
    **{s: "repellent" for s in REPELLENTS},
    **{s: "control" for s in CONTROLS},
}

#: Acquisition constants stated in the original notebooks.
N_TIMEPOINTS = 4979
DURATION_S = 1866.5
SAMPLING_RATE_HZ = N_TIMEPOINTS / DURATION_S  # ~2.667 Hz

#: Google Drive file ids for the hermaphrodite recordings (from notebook 5).
HERM_DRIVE_IDS = {
    "20220327_herm_2": "19iZqFwVzBQxEwCdeBSvtkhJ5zTkJLMBd",
    "20220327_herm_4": "1xGTPUeWc8i6x5WvfArBavLHssGDkcbny",
    "20220403_herm_2": "1kPgPvx2d5HKGNvRnilBK1gmMuk2ZejzZ",
    "20220403_herm_3": "1MFyyBcR40wiDnZ-wcNYjlZlUUlWt0Ol3",
    "20220427_herm_2": "1HOVm1p4IwAWXgZol0EuRwp_fdjEGt6c-",
    "20220427_herm_3": "1MlqcRaJgLwPdk5w-keNKi1Xtdfpnpe21",
    "20220427_herm_4": "1wUI0qeWQd5ltrx_IsDsQId3rH1-b9jWj",
    "20220427_herm_5": "16r91GeaWZ6a2eifPJ5VHbQg4H2E7etob",
}

#: The binary connectivity matrix used in notebooks 3/5/6, verbatim.
#: Row 0 (AIBL) is all zeros -> the graph is NOT strongly connected, which
#: makes PyPhi short-circuit to NullSystemIrreducibilityAnalysis.
CM_NOTEBOOKS = np.array(
    [
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ]
)

#: Weighted functional connectivity as stated in the notebooks' own markdown
#: table (funconn.princeton.edu), as (source, target, weight).
FUNCONN_EDGES = [
    ("AVEL", "AIBL", 28),
    ("AVEL", "AVAL", 22),
    ("AVAL", "AVEL", 35),
    ("RIML", "AVEL", 10),
]


def funconn_binary_cm(neurons=None, edges=None):
    """Binary CM built from FUNCONN_EDGES with cm[source, target] = 1.

    Provided so the direction convention is explicit and testable. Note that
    the binary CM used in the original notebooks (CM_NOTEBOOKS) additionally
    contains an AVEL -> RIML edge that FUNCONN_EDGES does not.
    """
    neurons = list(neurons or NOTEBOOK_NEURONS)
    edges = edges or FUNCONN_EDGES
    idx = {n: i for i, n in enumerate(neurons)}
    cm = np.zeros((len(neurons), len(neurons)), dtype=int)
    for src, dst, _w in edges:
        if src in idx and dst in idx:
            cm[idx[src], idx[dst]] = 1
    return cm


def is_strongly_connected(cm):
    """True if the directed graph given by ``cm`` is strongly connected.

    PyPhi requires this; otherwise it returns a
    NullSystemIrreducibilityAnalysis with NO_STRONG_CONNECTIVITY and phi_s = 0.
    """
    import networkx as nx

    g = nx.from_numpy_array(np.asarray(cm), create_using=nx.DiGraph)
    return nx.is_strongly_connected(g)


# ----------------------------------------------------------------------------
# Data layer
# ----------------------------------------------------------------------------

def recording_path(key, outdir="data"):
    """Local path for one recording CSV."""
    return os.path.join(outdir, f"{key}.csv")


def ensure_recording(key, outdir="data"):
    """Download one recording CSV if it is not already present locally.

    Two paths are tried. `curl` against the Drive download endpoint follows the
    303 to `drive.usercontent.google.com` and works in restricted-network
    environments; `gdown` is the fallback and is what usually runs in Colab.
    """
    path = recording_path(key, outdir)
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(outdir, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={HERM_DRIVE_IDS[key]}"
    try:
        subprocess.run(["curl", "-sSL", "-o", path, url], check=True, timeout=300)
        if os.path.getsize(path) > 1_000_000:
            return path
    except Exception:
        pass
    import gdown

    gdown.download(url, path, quiet=True)
    return path


def load_recording(path):
    """Load one chemosensory-data CSV.

    Returns (df, time_columns). The file layout is 9 metadata columns, then one
    column per frame, then a trailing 'key' column naming the recording.
    """
    df = pd.read_csv(path)
    time_cols = df.columns[9:-1]
    return df, time_cols


def neuron_trace(df, time_cols, neuron):
    """Fluorescence trace for one neuron as a float array."""
    row = df.loc[df["neuron"] == neuron]
    if len(row) == 0:
        raise KeyError(f"neuron {neuron!r} not in this recording")
    return row[time_cols].iloc[0].astype(float).values


def moving_window_binarize(x, window_size):
    """Binarize with a mid-range threshold in a window CENTRED on each sample.

    This reproduces the original notebooks exactly. Two properties worth
    knowing: the window is non-causal (uses future samples), and the mid-range
    (max+min)/2 threshold is set by the two most extreme values in the window,
    so it is sensitive to transients.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x), dtype=int)
    half = window_size // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        win = x[lo:hi]
        thr = (np.nanmax(win) + np.nanmin(win)) / 2
        out[i] = 1 if x[i] >= thr else 0
    return out


def combine_states(binary_rows):
    """Combine per-neuron binary traces into one integer state series.

    Bit i (weight 2**i) is neuron i, matching both the original notebooks and
    PyPhi's LOLI convention (node 0 is the low-order bit).
    """
    return sum(np.asarray(b) * (2 ** i) for i, b in enumerate(binary_rows)).astype(int)


def build_tpm(state_series, tau_samples, n_units):
    """Row-normalized state-by-state TPM at lag ``tau_samples``.

    Returns (tpm, row_counts). ``row_counts`` is what tells you whether a row
    was estimated from data or left empty.
    """
    n_states = 2 ** n_units
    counts = np.zeros((n_states, n_states))
    for i in range(len(state_series) - tau_samples):
        counts[state_series[i], state_series[i + tau_samples]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    tpm = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums != 0)
    return tpm, row_sums.squeeze()


def stimulus_epochs(df):
    """Parse the 'stimulus' column into {label: [onset_sample, ...]}."""
    raw = ast.literal_eval(df.iloc[0]["stimulus"])
    epochs = defaultdict(list)
    for onset, label in raw:
        epochs[label].append(int(float(onset)))
    return dict(epochs)


# ----------------------------------------------------------------------------
# IIT layer
# ----------------------------------------------------------------------------

def make_subsystem(tpm_sbs, state, node_labels, cm=None):
    """State-by-state TPM -> pyphi.Subsystem."""
    import pyphi
    from pyphi import convert

    sbn = convert.sbs2sbn(tpm_sbs)
    network = pyphi.Network(sbn, cm=cm, node_labels=tuple(node_labels))
    return pyphi.Subsystem(network, tuple(state))


def phi_structure(subsystem):
    """Unfold the IIT 4.0 Phi-structure (congruence-resolved by PyPhi)."""
    import pyphi

    return pyphi.new_big_phi.phi_structure(subsystem)


def ces_hypergraph(ps):
    """PhiStructure -> weighted hypergraph.

    Returns (nodes, faces).

    nodes : dict
        {(mechanism, cause_purview, effect_purview): phi_d}
    faces : list of dict
        One entry per relation FACE, with keys:
          degree  : number of relata joined by this face (>= 2)
          purview : the congruent overlap
          phi     : phi_r of the face
          relata  : sorted tuple of (mechanism, direction) pairs

    Faces are the right unit of analysis: one PyPhi ``Relation`` can carry
    several faces of different degree, and a degree-k face for k > 2 is an
    irreducibly higher-order object with no pairwise equivalent.
    """
    nodes = {}
    for concept in ps.distinctions:
        key = (
            tuple(concept.mechanism),
            tuple(concept.cause_purview),
            tuple(concept.effect_purview),
        )
        nodes[key] = float(concept.phi)

    faces = []
    for relation in ps.relations:
        for face in relation.faces:
            relata = tuple(
                sorted((tuple(m.mechanism), str(m.direction)) for m in face)
            )
            faces.append(
                {
                    "degree": len(face),
                    "purview": tuple(face.purview),
                    "phi": float(face.phi),
                    "relata": relata,
                }
            )
    return nodes, faces


def face_degree_distribution(faces):
    """{degree: count} over relation faces."""
    return {
        k: sum(1 for f in faces if f["degree"] == k)
        for k in sorted({f["degree"] for f in faces})
    }


# ----------------------------------------------------------------------------
# Comparison layer
# ----------------------------------------------------------------------------

def pairwise_projection(nodes, faces):
    """The representation used in notebooks 1 and 7.

    A Phi-structure is flattened to (phi_d, phi_r) where phi_r is keyed by
    ORDERED PAIRS of distinctions. Faces of degree > 2 have no slot and are
    dropped. Returned for side-by-side comparison, not as a recommendation.

    Returns (phi_d, phi_r, n_dropped).
    """
    phi_d = {str(k): float(v) for k, v in nodes.items()}
    phi_r = {}
    dropped = 0
    collided = 0
    for f in faces:
        mechs = [str(m) for m, _ in f["relata"]]
        if len(mechs) == 2:
            key = (mechs[0], mechs[1])
            if key in phi_r:
                # Two degree-2 faces on the same MECHANISM pair (they differ only
                # in which cause/effect sides they join) land on one dict key.
                # This is itself part of what the pairwise representation loses.
                collided += 1
            phi_r[key] = f["phi"]
        else:
            dropped += 1
    # `dropped` counts faces with no pairwise slot; `collided` counts degree-2
    # faces silently overwritten. kept + dropped + collided == len(faces).
    return phi_d, phi_r, dropped + collided


def ces_distance_pairwise(phi_d1, phi_r1, phi_d2, phi_r2, max_permutations=40320):
    """Brute-force minimum-cost bijection distance (the nb1/nb7 algorithm).

    Kept close to the original so the repo can demonstrate its two limits: it
    is O(n! n^2) — the default guard blocks n > 8 — and it can only see
    pairwise relation structure.
    """
    keys1, keys2 = sorted(phi_d1), sorted(phi_d2)
    n = max(len(keys1), len(keys2))
    if math.factorial(n) > max_permutations:
        raise ValueError(
            f"{n}! = {math.factorial(n):,} permutations exceeds "
            f"max_permutations={max_permutations:,}"
        )

    keys1 = keys1 + [f"__pad1_{i}" for i in range(n - len(keys1))]
    keys2 = keys2 + [f"__pad2_{i}" for i in range(n - len(keys2))]
    idx1 = {k: i for i, k in enumerate(keys1)}
    idx2 = {k: i for i, k in enumerate(keys2)}

    d1 = np.array([phi_d1.get(k, 0.0) for k in keys1])
    d2 = np.array([phi_d2.get(k, 0.0) for k in keys2])
    R1 = np.zeros((n, n))
    R2 = np.zeros((n, n))
    for (u, v), phi in phi_r1.items():
        if u in idx1 and v in idx1:
            R1[idx1[u], idx1[v]] = phi
    for (u, v), phi in phi_r2.items():
        if u in idx2 and v in idx2:
            R2[idx2[u], idx2[v]] = phi

    best, best_perm = float("inf"), None
    for perm in itertools.permutations(range(n)):
        cost = np.abs(d1 - d2[list(perm)]).sum()
        if cost >= best:
            continue
        cost += np.abs(R1 - R2[np.ix_(perm, perm)]).sum()
        if cost < best:
            best, best_perm = float(cost), perm
    return best, best_perm


def ces_distance_hypergraph(A, B, degree_weight=None):
    """Degree-graded distance between two Phi-structure hypergraphs.

    Parameters
    ----------
    A, B : (nodes, faces) tuples from :func:`ces_hypergraph`.
    degree_weight : callable, optional
        Maps face degree -> cost weight. Default is the identity, so a
        mismatch in a degree-4 face costs twice a mismatch in a degree-2 face.

    Returns
    -------
    (total, distinction_term, relation_term)

    Design
    ------
    * Distinctions are matched by optimal assignment (Hungarian, O(n^3))
      instead of brute-force permutation, with a structural penalty for
      matching distinctions of different mechanism order or purview size.
    * Relation faces are bucketed by (degree, purview size) and compared
      within bucket as sorted phi spectra. Higher-order faces therefore
      contribute on their own terms and are never projected onto pairs.

    This is a working prototype for discussion, NOT a metric with proven
    axioms: the structural penalty and the degree weighting are free
    parameters, and the bucketed spectrum comparison ignores which specific
    distinctions a face joins.
    """
    from scipy.optimize import linear_sum_assignment

    if degree_weight is None:
        degree_weight = lambda k: float(k)

    nodes_a, faces_a = A
    nodes_b, faces_b = B

    # --- distinction term -----------------------------------------------
    ka, kb = list(nodes_a), list(nodes_b)
    n = max(len(ka), len(kb))
    if n == 0:
        distinction_term = 0.0
    else:
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                pa = nodes_a[ka[i]] if i < len(ka) else 0.0
                pb = nodes_b[kb[j]] if j < len(kb) else 0.0
                penalty = 0.0
                if i < len(ka) and j < len(kb):
                    ma, ca, ea = ka[i]
                    mb, cb, eb = kb[j]
                    penalty = 0.5 * (
                        abs(len(ma) - len(mb))
                        + abs(len(ca) - len(cb))
                        + abs(len(ea) - len(eb))
                    )
                cost[i, j] = abs(pa - pb) + penalty * (pa + pb) / 2
        rows, cols = linear_sum_assignment(cost)
        distinction_term = float(cost[rows, cols].sum())

    # --- relation term --------------------------------------------------
    def buckets(faces):
        out = defaultdict(list)
        for f in faces:
            out[(f["degree"], len(f["purview"]))].append(f["phi"])
        return {k: sorted(v, reverse=True) for k, v in out.items()}

    ba, bb = buckets(faces_a), buckets(faces_b)
    relation_term = 0.0
    for key in set(ba) | set(bb):
        la, lb = ba.get(key, []), bb.get(key, [])
        length = max(len(la), len(lb))
        la = la + [0.0] * (length - len(la))
        lb = lb + [0.0] * (length - len(lb))
        relation_term += degree_weight(key[0]) * sum(
            abs(x - y) for x, y in zip(la, lb)
        )

    return distinction_term + relation_term, distinction_term, float(relation_term)
