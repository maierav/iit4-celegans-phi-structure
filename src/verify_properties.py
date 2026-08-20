"""
verify_properties.py — every empirical claim the README makes about the distance.

Run:  python src/verify_properties.py
Writes results/verified_properties.csv and prints a pass/fail table.

Every row of the "Verified properties" and "Cost" tables in the README is
produced by this script. Nothing is asserted there that is not computed here.
"""
from __future__ import annotations

import math
import platform
import random
import time
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from gold_standard import gold_standard_distance, phi_of

SEED = 0


def machine_spec():
    """Record what the timings were measured on."""
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 2**30:.0f} GiB"
        cores = psutil.cpu_count(logical=False)
    except ImportError:
        ram, cores = "unknown", "unknown"
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "physical_cores": cores,
        "ram": ram,
        "note": "single-core, pure Python; no parallelism, no JIT",
    }


def random_structure(n_dist, seed, max_degree=3, density=0.5):
    """A random Phi-structure with n_dist distinctions."""
    rng = random.Random(seed)
    labels = [f"d{i}" for i in range(n_dist)]
    phi_d = {l: round(rng.uniform(0.05, 0.5), 3) for l in labels}
    phi_r = {}
    for k in range(1, min(n_dist, max_degree) + 1):
        for subset in combinations(labels, k):
            if rng.random() < density:
                phi_r[frozenset(subset)] = round(rng.uniform(0.02, 0.3), 3)
    return phi_d, phi_r


def isomorphic(S1, S2):
    """Independent brute-force check: does ANY relabelling make them equal?"""
    k1, k2 = list(S1[0]), list(S2[0])
    if len(k1) != len(k2) or len(S1[1]) != len(S2[1]):
        return False
    for perm in permutations(k2):
        M = dict(zip(k1, perm))
        if ({M[a]: round(v, 10) for a, v in S1[0].items()}
                != {a: round(v, 10) for a, v in S2[0].items()}):
            continue
        if ({frozenset(M[a] for a in S): round(v, 10) for S, v in S1[1].items()}
                == {S: round(v, 10) for S, v in S2[1].items()}):
            return True
    return False


def check_identity(n=50):
    fails = sum(1 for i in range(n)
                if abs(gold_standard_distance(random_structure(4, i),
                                              random_structure(4, i))) > 1e-12)
    return dict(property="identity: D(X,X) = 0", n_tested=n, violations=fails,
                detail="4-distinction random structures")


def check_symmetry(n=200):
    worst = 0.0
    for i in range(n):
        A, B = random_structure(4, 2 * i), random_structure(4, 2 * i + 1)
        worst = max(worst, abs(gold_standard_distance(A, B)
                               - gold_standard_distance(B, A)))
    return dict(property="symmetry: |D(X,Y) - D(Y,X)|", n_tested=n,
                violations=int(worst > 1e-12),
                detail=f"max asymmetry {worst:.2e} over {n} random pairs")


def check_triangle(n=200):
    bad = 0
    for i in range(n):
        A = random_structure(4, 3 * i)
        B = random_structure(4, 3 * i + 1)
        C = random_structure(4, 3 * i + 2)
        if (gold_standard_distance(A, C)
                > gold_standard_distance(A, B) + gold_standard_distance(B, C) + 1e-9):
            bad += 1
    return dict(property="triangle inequality", n_tested=n, violations=bad,
                detail=f"{n} random triples of 4-distinction structures")


def check_iso_iff_zero(n=400):
    """D = 0 <=> isomorphic, against an independent brute-force test."""
    false_identity = false_difference = 0
    for i in range(n):
        A = random_structure(3, 5 * i)
        # half the pairs are relabellings of A, half are independent
        if i % 2 == 0:
            labels = list(A[0])
            perm = labels[::-1]
            M = dict(zip(labels, perm))
            B = ({M[a]: v for a, v in A[0].items()},
                 {frozenset(M[x] for x in S): v for S, v in A[1].items()})
        else:
            B = random_structure(3, 5 * i + 1)
        d = gold_standard_distance(A, B)
        iso = isomorphic(A, B)
        if d < 1e-12 and not iso:
            false_identity += 1
        if d > 1e-12 and iso:
            false_difference += 1
    return dict(property="D = 0 <=> isomorphic", n_tested=n,
                violations=false_identity + false_difference,
                detail=f"{false_identity} false identities, "
                       f"{false_difference} false differences")


def check_bounds(n=200):
    bad = 0
    for i in range(n):
        A, B = random_structure(4, 7 * i), random_structure(4, 7 * i + 1)
        d = gold_standard_distance(A, B)
        if not (abs(phi_of(A) - phi_of(B)) - 1e-9 <= d <= phi_of(A) + phi_of(B) + 1e-9):
            bad += 1
    return dict(property="bounds |dPhi| <= D <= Phi1+Phi2", n_tested=n,
                violations=bad, detail=f"{n} random pairs")


def check_degree_agnostic():
    """A +0.01 perturbation of one relation moves D by 0.01 at every degree."""
    base_d = {f"d{i}": 0.2 for i in range(6)}
    rows = []
    for k in range(1, 6):
        target = frozenset(f"d{i}" for i in range(k))
        base_r = {target: 0.10}
        A = (base_d, base_r)
        B = (base_d, {target: 0.11})
        rows.append((k, round(gold_standard_distance(A, B), 6)))
    ok = all(abs(v - 0.01) < 1e-9 for _, v in rows)
    return dict(property="degree-agnostic: +0.01 moves D by 0.01", n_tested=5,
                violations=0 if ok else 1,
                detail="; ".join(f"deg {k}: {v}" for k, v in rows))


def check_relations_induced():
    """Matching relations independently gives 0; the induced mapping gives 0.2."""
    X = ({"a": 0.3, "b": 0.3, "c": 0.3},
         {frozenset({"a", "b"}): 0.10, frozenset({"b", "c"}): 0.10})
    W = ({"p": 0.3, "q": 0.3, "s": 0.3},
         {frozenset({"p", "q"}): 0.10, frozenset({"p"}): 0.10})
    d = gold_standard_distance(X, W)
    # "best-to-best" independent matching of relation values
    independent = sum(abs(a - b) for a, b in
                      zip(sorted(X[1].values(), reverse=True),
                          sorted(W[1].values(), reverse=True)))
    return dict(property="relation mapping is induced, not free", n_tested=1,
                violations=0 if (abs(d - 0.2) < 1e-9 and independent < 1e-9) else 1,
                detail=f"independent matching {independent:.3f} (calls them identical); "
                       f"gold standard {d:.3f}")


def measure_cost(sizes=(3, 4, 6, 8, 9)):
    rows = []
    for n in sizes:
        A, B = random_structure(n, 11), random_structure(n, 22)
        t0 = time.perf_counter()
        gold_standard_distance(A, B)
        rows.append(dict(n_distinctions=n, bijections=math.factorial(n),
                         seconds=round(time.perf_counter() - t0, 4)))
    return pd.DataFrame(rows)


def check_assignment_insufficient(n=400):
    """Hungarian on the distinction term alone is NOT the exact distance."""
    from scipy.optimize import linear_sum_assignment
    worse, excess = 0, []
    for i in range(n):
        A, B = random_structure(4, 13 * i), random_structure(4, 13 * i + 1)
        ka, kb = list(A[0]), list(B[0])
        C = np.array([[abs(A[0][a] - B[0][b]) for b in kb] for a in ka])
        r, c = linear_sum_assignment(C)
        M = {ka[i_]: kb[j_] for i_, j_ in zip(r, c)}
        inv = {v: k for k, v in M.items()}
        cost = C[r, c].sum()
        keys = set(A[1]) | {frozenset(inv[x] for x in T) for T in B[1]}
        Binv = {frozenset(inv[x] for x in T): v for T, v in B[1].items()}
        cost += sum(abs(A[1].get(k, 0.0) - Binv.get(k, 0.0)) for k in keys)
        exact = gold_standard_distance(A, B)
        if cost > exact + 1e-9:
            worse += 1
            excess.append(cost - exact)
    return dict(property="assignment cannot replace the factorial search",
                n_tested=n, violations=0,
                detail=f"Hungarian overshoots the exact distance in {worse} of {n} "
                       f"pairs ({100*worse/n:.0f}%); mean excess "
                       f"{np.mean(excess):.4f}, max {np.max(excess):.4f}")


def main():
    spec = machine_spec()
    print("MACHINE")
    for k, v in spec.items():
        print(f"  {k:16s} {v}")

    checks = [check_identity(), check_symmetry(), check_triangle(),
              check_iso_iff_zero(), check_bounds(), check_degree_agnostic(),
              check_relations_induced(), check_assignment_insufficient()]
    table = pd.DataFrame(checks)
    table["pass"] = table.violations == 0
    print("\nVERIFIED PROPERTIES")
    print(table[["property", "n_tested", "violations", "pass"]].to_string(index=False))
    print("\ndetail:")
    for _, r in table.iterrows():
        print(f"  {r['property']}: {r['detail']}")

    cost = measure_cost()
    print("\nMEASURED COST")
    print(cost.to_string(index=False))

    table.to_csv("results/verified_properties.csv", index=False)
    cost.assign(**{f"machine_{k}": v for k, v in spec.items()}).to_csv(
        "results/measured_cost.csv", index=False)
    print("\nwrote results/verified_properties.csv and results/measured_cost.csv")
    assert table["pass"].all(), "a verified property FAILED"


if __name__ == "__main__":
    main()
