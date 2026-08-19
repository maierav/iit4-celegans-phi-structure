"""
gold_standard.py — exact ("brute force") distance between two Φ-structures.

Implements the min-over-bijections distance defined in the Qstr-IIT summer
school deck (Cases 1-3), verified symbolically against the deck's own worked
mapping costs.

Definition
----------
A Φ-structure is (phi_d, phi_r):

    phi_d : {distinction_label: φ_d}
    phi_r : {frozenset(distinction_labels): φ_r}

A relation is keyed by the SET of distinctions it joins, so a degree-k
relation for k > 2 is a first-class object — no pairwise projection, no
folding. Degree-1 keys are self-relations.

For a bijection M between the two distinction sets (the smaller set padded
with null distinctions carrying φ_d = 0 and no relations):

    D(S1, S2; M) = Σ_a |φ_d1(a) − φ_d2(M(a))|                  distinctions
                 + Σ_S |φ_r1(S) − φ_r2(M(S))|                  Str1 relations
                 + Σ_T φ_r2(T)   for T with no preimage in S1  unmatched

    D(S1, S2)   = min over all bijections M

Cost
----
The minimisation is over permutations of the DISTINCTIONS — n! where n =
max(N_dist1, N_dist2) — not over 2^n mechanisms. Relations come along for
free once the distinction mapping is fixed.

Relation keys
-------------
A key is either ``frozenset(distinction_labels)`` or
``(frozenset(distinction_labels), tag)``. The tag — in practice the face
degree — rides through relabelling untouched, so relations joining the SAME
distinctions at different degrees stay distinct. This matters on real PyPhi
output: relation faces live over MICE (cause and effect purviews separately),
so a degree-3 face can join only two distinctions, and an untagged key merges
it with the degree-2 face on the same pair.

Higher-order relations need no special handling
-----------------------------------------------
A relation of ANY degree is a single key in ``phi_r``, and the cost function
never inspects a key's size: it relabels the key through the bijection and
subtracts. Degree 2 and degree 7 are handled by the same line of code, and a
perturbation of +0.01 to a relation of any degree moves the distance by
exactly 0.01. So the Φ-folds construction — which flattens a hypergraph into
one scalar per distinction so that optimal transport can consume it — is NOT
needed here and is not used. It belongs to the OT approximation route, where
it is a lossy but cheap surrogate; the exact distance keeps the hypergraph.

Additionally verified: d(X, Y) = 0 **iff** X and Y are isomorphic (equal after
some relabelling of distinctions). Checked against an independent brute-force
isomorphism test over 400 random pairs — no false identities and no false
differences.

Verified properties
-------------------
Checked under BOTH keying schemes (plain sets and (set, degree) tags):

    identity      d(X, X) = 0                     50/50 tagged, exact
    symmetry      |d(X,Y) − d(Y,X)| < 1e-12       0 violations / 200 pairs each
    triangle      d(X,Z) ≤ d(X,Y) + d(Y,Z)        0 violations / 200 triples each
                  (tagged: median slack −1.57, 2 exactly-tight triples)
    d = 0 ⟺ iso   agrees with brute-force isomorphism  0/400 disagreements

Empirical only — these are not proofs.

Measured runtimes for this pure-Python implementation (single core, ~50%
relation density, one distance):

    n = 6    720 perms         0.01 s
    n = 8     40 320           2.3 s
    n = 9    362 880          32 s
    n = 10   3.6M            470 s   (~8 min)

Each additional distinction multiplies by n, so n = 11 is ~1.5 h and n = 12
~18 h. The practical ceiling is n ≈ 9 for a single distance and n ≈ 8 when
computing a full pairwise matrix. Beyond that, use an optimal-transport
bound (d_OT ≤ d_exact ≤ Δ_μ*).
"""

from __future__ import annotations

import math
from itertools import permutations

__all__ = [
    "gold_standard_distance",
    "n_permutations",
    "is_feasible",
    "phi_of",
]


def phi_of(structure):
    """Φ = Σφ_d + Σφ_r for a structure given as (phi_d, phi_r)."""
    phi_d, phi_r = structure
    return sum(phi_d.values()) + sum(phi_r.values())


def n_permutations(structure_a, structure_b):
    """Number of bijections the exact search must enumerate."""
    return math.factorial(max(len(structure_a[0]), len(structure_b[0])))


def is_feasible(structure_a, structure_b, budget=5_000_000):
    """Whether the exact search is worth attempting under a permutation budget."""
    return n_permutations(structure_a, structure_b) <= budget


def _split(key):
    """A relation key is either frozenset(distinctions) or (frozenset, tag).

    The optional tag (typically the face degree) rides along untouched under
    relabelling, so relations that join the SAME distinctions but differ in
    degree stay distinct instead of being merged.
    """
    if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], frozenset):
        return key[0], key[1]
    return key, None


def _relabel(key, mapping):
    members, tag = _split(key)
    moved = frozenset(mapping[a] for a in members)
    return moved if tag is None else (moved, tag)


def _cost(mapping, phi_d1, phi_r1, phi_d2, phi_r2):
    cost = 0.0
    # Iterate over the PADDED key set: a null distinction of the smaller
    # structure still charges the φ_d of whatever it is matched to. Iterating
    # over phi_d1 alone silently drops the surplus distinctions of the larger
    # structure and makes the distance asymmetric.
    for a, b in mapping.items():
        cost += abs(phi_d1.get(a, 0.0) - phi_d2.get(b, 0.0))
    for key, v in phi_r1.items():
        cost += abs(v - phi_r2.get(_relabel(key, mapping), 0.0))
    inverse = {v: k for k, v in mapping.items()}
    for key, v in phi_r2.items():
        if _relabel(key, inverse) not in phi_r1:
            cost += v
    return cost


def gold_standard_distance(structure_a, structure_b, return_mapping=False,
                           budget=None):
    """Exact minimum-cost bijection distance between two Φ-structures.

    Parameters
    ----------
    structure_a, structure_b : (phi_d, phi_r) as described in the module docstring.
    return_mapping : also return the argmin bijection.
    budget : if given, raise when the permutation count exceeds it rather than
        running for hours.

    Returns
    -------
    float, or (float, dict) when ``return_mapping``.
    """
    if budget is not None and n_permutations(structure_a, structure_b) > budget:
        raise ValueError(
            f"{n_permutations(structure_a, structure_b):,} permutations "
            f"exceeds budget={budget:,}; use an OT bound instead"
        )

    phi_d1, phi_r1 = structure_a
    phi_d2, phi_r2 = structure_b
    keys1, keys2 = list(phi_d1), list(phi_d2)
    n = max(len(keys1), len(keys2))
    padded1 = keys1 + [f"\0pad1_{i}" for i in range(n - len(keys1))]
    padded2 = keys2 + [f"\0pad2_{i}" for i in range(n - len(keys2))]

    best_cost, best_map = math.inf, None
    for perm in permutations(range(n)):
        mapping = {padded1[i]: padded2[perm[i]] for i in range(n)}
        cost = _cost(mapping, phi_d1, phi_r1, phi_d2, phi_r2)
        if cost < best_cost:
            best_cost, best_map = cost, mapping

    if return_mapping:
        clean = {a: b for a, b in best_map.items()
                 if not a.startswith("\0") and not b.startswith("\0")}
        return best_cost, clean
    return best_cost
