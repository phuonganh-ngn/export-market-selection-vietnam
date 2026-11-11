from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import warnings
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, hstack, vstack

TOL = 1e-8
FEASIBILITY_TOL = 2e-6


@dataclass
class SolveResult:
    selected: np.ndarray
    objective: float
    status: str
    gap: float
    absolute_gap: float
    max_constraint_violation: float
    runtime_seconds: float
    scenarios: int


def topk_set(score: np.ndarray, candidate_ids: list[str], k: int) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    if len(score) != len(candidate_ids) or k <= 0:
        raise ValueError("Invalid score vector or K")
    order = np.lexsort((np.asarray(candidate_ids, dtype=str), -score))
    return order[:min(int(k), len(score))]


def scenario_matrix(objective1: np.ndarray, objective2: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    a = np.asarray(objective1, dtype=float)
    b = np.asarray(objective2, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Objectives must be same-shaped S×N matrices")
    return np.vstack([w * a + (1.0 - w) * b for w in weights for _ in range(a.shape[0])])


def _solve_auxiliary(c, integ, lb, ub, rows, row_lb, row_ub, objective_idx, maximize=False) -> SolveResult:
    start = time.perf_counter()
    nvar = len(c)
    A = csc_matrix(np.asarray(rows, dtype=float))
    constraints = LinearConstraint(A, np.asarray(row_lb, dtype=float), np.asarray(row_ub, dtype=float))
    # HiGHS 1.12 can return an internal Solve error on otherwise feasible finite
    # regret instances under default settings. Retry supported SciPy/HiGHS options
    # only, then independently validate feasibility, integrality, and the absolute
    # primal-dual objective gap. Relative gaps can look large when the regret optimum
    # is close to zero, despite a negligible absolute error in normalized utility.
    options = [
        {"presolve": True, "mip_rel_gap": 1e-12},
        {"presolve": False, "mip_rel_gap": 1e-12},
        {"presolve": True, "mip_rel_gap": 1e-14},
    ]
    res = None
    last_message = ""
    accepted_violation = np.inf
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        for opt in options:
            try:
                candidate = milp(np.asarray(c, dtype=float), integrality=np.asarray(integ, dtype=np.int8),
                                 bounds=Bounds(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)),
                                 constraints=constraints,
                                 options={"time_limit": 60.0, "mip_rel_gap": 1e-12, **opt})
            except Exception as exc:
                last_message = f"HiGHS/SciPy raised {type(exc).__name__}: {exc} under options={opt}"
                continue
            last_message = str(candidate.message)
            if candidate.success and candidate.x is not None:
                x = np.asarray(candidate.x, float)
                activity = np.asarray(A @ x).ravel()
                violation = max(float(np.maximum(np.asarray(row_lb) - activity, 0).max(initial=0)),
                                float(np.maximum(activity - np.asarray(row_ub), 0).max(initial=0)))
                integral = np.max(np.abs(x[np.asarray(integ, dtype=bool)] -
                                         np.round(x[np.asarray(integ, dtype=bool)])), initial=0.0)
                mip_gap = float(getattr(candidate, "mip_gap", np.inf))
                dual_bound = float(getattr(candidate, "mip_dual_bound", np.nan))
                absolute_gap = abs(float(candidate.fun) - dual_bound) if np.isfinite(dual_bound) else np.inf
                if violation <= FEASIBILITY_TOL and integral <= 1e-7 and absolute_gap <= 1e-6:
                    res = candidate
                    accepted_violation = violation
                    break
                last_message = (f"returned solution failed independent check: violation={violation}, "
                                f"integrality={integral}, relative_mip_gap={mip_gap}, "
                                f"absolute_mip_gap={absolute_gap}")
    if res is None:
        raise RuntimeError(f"MILP failed after numerical retries: {last_message}")
    elapsed = time.perf_counter() - start
    if not res.success or res.x is None:
        raise RuntimeError(f"MILP failed: status={res.status}, message={res.message}")
    selected = np.flatnonzero(res.x[:objective_idx] > 0.5)
    if not selected.size and objective_idx:
        raise RuntimeError("MILP returned an empty selected set")
    dual_bound = float(getattr(res, "mip_dual_bound", np.nan))
    absolute_gap = abs(float(res.fun) - dual_bound) if np.isfinite(dual_bound) else np.nan
    return SolveResult(selected=selected, objective=float(res.fun), status=str(res.message),
                       gap=float(getattr(res, "mip_gap", 0.0) or 0.0), absolute_gap=absolute_gap,
                       max_constraint_violation=accepted_violation,
                       runtime_seconds=elapsed,
                       scenarios=len(rows))


def solve_maximin(utilities: np.ndarray, k: int) -> SolveResult:
    """Exact finite-scenario maximin utility MILP."""
    u = np.asarray(utilities, dtype=float)
    if u.ndim != 2 or not np.isfinite(u).all():
        raise ValueError("Utilities must be finite S×N")
    s, n = u.shape
    kk = min(int(k), n)
    # min -z; each scenario utility >= z; cardinality exactly K.
    rows = np.column_stack([u, -np.ones(s)])
    lows = np.zeros(s)
    ups = np.full(s, np.inf)
    card = np.r_[np.ones(n), 0.0][None, :]
    rows = np.vstack([rows, card])
    lows = np.r_[lows, kk]
    ups = np.r_[ups, kk]
    c = np.r_[np.zeros(n), -1.0]
    integ = np.r_[np.ones(n), 0]
    lb = np.r_[np.zeros(n), -np.inf]
    ub = np.r_[np.ones(n), np.inf]
    result = _solve_auxiliary(c, integ, lb, ub, rows, lows, ups, n, maximize=True)
    if len(result.selected) != kk:
        raise AssertionError(f"Maximin cardinality violation: expected {kk}, got {len(result.selected)}")
    result.objective = float(np.min(u[:, result.selected].sum(axis=1)))
    result.scenarios = s
    return result


def solve_minimax_regret(utilities: np.ndarray, k: int) -> SolveResult:
    """Exact finite-scenario minimax absolute regret for a cardinality-K set."""
    u = np.asarray(utilities, dtype=float)
    if u.ndim != 2 or not np.isfinite(u).all():
        raise ValueError("Utilities must be finite S×N")
    s, n = u.shape
    kk = min(int(k), n)
    oracle = np.sort(u, axis=1)[:, ::-1][:, :kk].sum(axis=1)
    # U(S,s)+t >= OPT_s; minimize t. t is constrained nonnegative.
    rows = np.column_stack([u, np.ones(s)])
    lows = oracle
    ups = np.full(s, np.inf)
    rows = np.vstack([rows, np.r_[np.ones(n), 0.0]])
    lows = np.r_[lows, kk]
    ups = np.r_[ups, kk]
    c = np.r_[np.zeros(n), 1.0]
    integ = np.r_[np.ones(n), 0]
    lb = np.r_[np.zeros(n), 0.0]
    ub = np.r_[np.ones(n), np.inf]
    result = _solve_auxiliary(c, integ, lb, ub, rows, lows, ups, n)
    if len(result.selected) != kk:
        raise AssertionError(f"Minimax-regret cardinality violation: expected {kk}, got {len(result.selected)}")
    result.objective = float(np.max(oracle - u[:, result.selected].sum(axis=1)))
    result.scenarios = s
    return result


def solve_relative_robust(objective1: np.ndarray, objective2: np.ndarray,
                          weight_vertices: tuple[float, ...], k: int) -> SolveResult:
    """Weber-style maximin relative performance over the interval endpoints.

    For each endpoint w, the constraint compares the portfolio's linear utility
    with the ex-post best cardinality-K utility at that same w. For a fixed
    portfolio, its numerator is affine in w. The oracle denominator is the maximum
    of affine portfolio utilities and is therefore convex. A convex denominator is
    no larger than the chord joining its endpoint values; with a nonnegative
    numerator, the ratio is bounded below by the ratio with that affine chord.
    The latter is fractional-linear and attains its minimum at an endpoint.
    Therefore endpoint checks are sufficient for two objectives when the oracle
    denominator remains strictly positive on the interval.
    """
    a = np.asarray(objective1, dtype=float)
    b = np.asarray(objective2, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Objective vectors must be finite and same-shaped")
    n = len(a)
    kk = min(int(k), n)
    rows = []
    for w in weight_vertices:
        u = w * a + (1.0 - w) * b
        opt = float(np.sort(u)[::-1][:kk].sum())
        if opt <= 1e-12:
            continue
        rows.append(np.r_[u / opt, -1.0])  # portfolio ratio - t >= 0
    if not rows:
        raise ValueError("Relative-robust denominator is zero for every preference endpoint")
    mat = np.asarray(rows, dtype=float)
    mat = np.vstack([mat, np.r_[np.ones(n), 0.0]])
    lows = np.r_[np.zeros(len(rows)), kk]
    ups = np.r_[np.full(len(rows), np.inf), kk]
    c = np.r_[np.zeros(n), -1.0]
    integ = np.r_[np.ones(n), 0]
    lb = np.r_[np.zeros(n), -np.inf]
    ub = np.r_[np.ones(n), np.inf]
    result = _solve_auxiliary(c, integ, lb, ub, mat, lows, ups, n, maximize=True)
    if len(result.selected) != kk:
        raise AssertionError(f"Relative-robust cardinality violation: expected {kk}, got {len(result.selected)}")
    ratios = [float((w*a[result.selected] + (1-w)*b[result.selected]).sum() /
                    np.sort(w*a + (1-w)*b)[::-1][:kk].sum())
              for w in weight_vertices if np.sort(w*a + (1-w)*b)[::-1][:kk].sum() > 1e-12]
    result.objective = min(ratios)
    result.scenarios = len(rows)
    return result


def brute_force_regret(utilities: np.ndarray, k: int) -> tuple[float, tuple[int, ...]]:
    from itertools import combinations
    u = np.asarray(utilities, dtype=float)
    s, n = u.shape
    kk = min(k, n)
    oracle = np.sort(u, axis=1)[:, ::-1][:, :kk].sum(axis=1)
    best, best_set = np.inf, None
    for idx in combinations(range(n), kk):
        regret = float(np.max(oracle - u[:, idx].sum(axis=1)))
        if regret < best - 1e-12:
            best, best_set = regret, idx
    return best, tuple(best_set or ())


def verify_solver_small_instances() -> list[dict]:
    examples = [
        np.array([[1.0, 0.2, 0.7, 0.1], [0.0, 1.0, 0.2, 0.6]]),
        np.array([[0.1, 0.3, 0.7, 0.9], [0.8, 0.6, 0.2, 0.1], [0.4, 0.5, 0.6, 0.2]]),
    ]
    rows = []
    for j, u in enumerate(examples):
        for k in [1, 2]:
            exact, _ = brute_force_regret(u, k)
            solved = solve_minimax_regret(u, k)
            feasible = len(solved.selected) == min(k, u.shape[1])
            ok = feasible and abs(solved.objective - exact) <= 1e-7
            rows.append({"case": j, "k": k, "solver": "scipy.optimize.milp/HiGHS",
                         "solver_status": solved.status, "mip_gap": solved.gap,
                         "absolute_mip_gap": solved.absolute_gap,
                         "max_constraint_violation": solved.max_constraint_violation,
                         "enumerated_objective": exact, "milp_objective": solved.objective,
                         "feasible": feasible, "matches_enumeration": ok,
                         "runtime_seconds": solved.runtime_seconds})
            if not ok:
                raise AssertionError("MILP disagrees with exhaustive minimax-regret enumeration")
    return rows
