"""Constrained mean-variance portfolio optimization (see §9 of the documentation)

Two formulations are supported via the `objective` parameter of function `optimize_weights)`:

- "max_return": maximize expected portfolio return subject to a variance cap.
- "min_variance": minimize portfolio variance subject to a return lower bound.

Both share the same exposure constraints (beta neutrality, net exposure cap, gross exposure cap) 
and an optional per-asset weight box.

Public API
----------

* function `align_universe(...)`
        intersect tickers across optimizer inputs (mu, sigma, betas); returns a sorted index.

* function `optimize_weights(...)`
        solve the constrained mean-variance problem for either objective.

* function `optimize_min_variance_with_feasibility_recovery(...)`
        min-variance solve that relaxes the return lower bound by halving until feasible.

* class `OptimizerResult`  
        dataclass output of `optimize_weights`: optimal weights + diagnostics dict.

"""
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.linalg as la


logger = logging.getLogger(__name__)


Objective = Literal["max_return", "min_variance"]


@dataclass
class OptimizerResult:
    """Output of function `optimize_weights`.

    Attributes
    ----------
    optimal_weights
        Series of weights indexed by ticker (subset of input tickers
        after universe alignment).
    metrics
        Diagnostic dict with keys:
        - "objective": the objective used ("max_return" or "min_variance")
        - "status": cvxpy solver status
        - "objective_value": optimal objective value (or None)
        - "solver_stats": {"solver_name", "solve_time", "num_iters"}
        - "risk": {"volatility", "variance", "expected_return", ...}
            plus "volatility_cap"/"variance_cap" (max_return) or
            "return_lower_bound" (min_variance)
        - "exposures": {"net", "net_cap", "gross", "gross_cap"}
            plus "beta"/"beta_cap" when betas are provided
        - "residuals": realized-minus-cap for each active constraint
        - "tickers": aligned ticker list (row order of `optimal_weights`)
        - "solvers_tried": solvers attempted, in order
    """
    optimal_weights: pd.Series
    metrics: dict[str, Any]


def align_universe(
    tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    betas: pd.Series | None,
) -> pd.Index:
    """Intersect all tickers across optimizer inputs.

    Parameters
    ----------
    tickers
        Initial candidate universe.
    mu
        estimated expected returns for the next inter-rebalance period of ticker (Series indexed by ticker).
    sigma
        estimated covariance matrix of tickers' returns for the next inter-rebalance period (DataFrame indexed and columned by ticker).
    betas
        estimated CAPM beta of tickers (Series indexed by ticker, or None (skipped from intersection)).
    """
    idx = (
        pd.Index(tickers)
        .intersection(sigma.index)
        .intersection(sigma.columns)
        .intersection(mu.index)
    )
    if betas is not None:
        idx = idx.intersection(betas.index)

    if len(idx) == 0:
        raise ValueError("empty intersection of tickers across optimizer inputs")

    return idx.sort_values()


# internal helper
def _cholesky_or_eigh(S: np.ndarray) -> np.ndarray:
    """First attempts a Cholesky decomposition (valid when `S` is positive
    definite). If Cholesky fails, falls back to an eigendecomposition with a
    small eigenvalue floor to guard against tiny negative eigenvalues caused by
    floating-point error.

    The fallback is not expected to trigger when `S` is a Ledoit-Wolf shrunk
    covariance matrix, since such matrices are positive definite by
    construction, although floating-point arithmetic may still produce tiny
    negative eigenvalues.
    """
    try:
        return la.cholesky(S, lower=False)
    except la.LinAlgError:
        logger.warning(
            "sigma not numerically PD; falling back to eigendecomposition "
            "with negative eigenvalues clipped to 1e-12."
        )
        eigvals, eigvecs = np.linalg.eigh(S)
        eigvals = np.maximum(eigvals, 1e-12)
        return np.diag(np.sqrt(eigvals)) @ eigvecs.T


def optimize_weights(
    tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    tickers_betas: pd.Series | None,
    *,
    objective: Objective = "max_return",
    ptf_variance_cap: float | None = None,
    ptf_return_lower_bound: float | None = None,
    ptf_beta_cap: float = 0.01,
    ptf_net_exposure_cap: float = 0.001,
    ptf_gross_exposure_cap: float = 4.0,
    w_min: float | None = None,
    w_max: float | None = None,
    solver: str = "ECOS",
    fallback_solvers: tuple[str, ...] = ("SCS",),
    verbose: bool = False,
) -> OptimizerResult:
    """Solve the constrained mean-variance portfolio optimization.

    Parameters
    ----------
    tickers
        Candidate universe; intersected with the indices of `mu`, `sigma`,
        `tickers_betas`.
    mu
        Expected returns per ticker.
    sigma
        Return covariance matrix (must be symmetric PSD).
    tickers_betas
        Per-ticker betas to a market index, or None to drop the
        beta-neutrality constraint.
    objective
        set to "max_return" or "min_variance".
    ptf_variance_cap
        Variance cap (required when `objective` set to "max_return").
    ptf_return_lower_bound
        Return floor (required when `objective` set to "min_variance").
    ptf_beta_cap
        Upper bound on |portfolio beta|.
    ptf_net_exposure_cap
        Upper bound on |net exposure|.
    ptf_gross_exposure_cap
        Upper bound on gross exposure.
    w_min, w_max
        Optional per-ticker weight bounds (hard constraints applied uniformely for all assets
        during optimization).
    solver, fallback_solvers, verbose
        Solver configuration.

    Returns
    -------
    OptimizerResult
    """
    # --- Validate objective and required constraints
    if objective == "max_return":
        if ptf_variance_cap is None:
            raise ValueError(
                "ptf_variance_cap is required when objective='max_return'"
            )
        if ptf_variance_cap < 0:
            raise ValueError("ptf_variance_cap must be >= 0")
    elif objective == "min_variance":
        if ptf_return_lower_bound is None:
            raise ValueError(
                "ptf_return_lower_bound is required when objective='min_variance'"
            )
    else:
        raise ValueError(
            f"objective must be 'max_return' or 'min_variance', got {objective!r}"
        )

    # --- Align universe 
    aligned_tickers = align_universe(tickers, mu, sigma, tickers_betas)
    if set(aligned_tickers) != set(tickers):
        dropped = sorted(set(tickers) - set(aligned_tickers))
        preview = ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else "")
        logger.warning(
            "investable universe reduced from %d to %d tickers due to missing "
            "optimizer inputs (mu, sigma, or betas). Dropped: %s",
            len(tickers), len(aligned_tickers), preview,
        )

    # --- Formulate with numpy arrays 
    mu_vec = mu.loc[aligned_tickers].to_numpy()
    if np.isnan(mu_vec).any() or np.isinf(mu_vec).any():
        raise ValueError("mu contains NaN/Inf for one or more aligned tickers")

    S = sigma.loc[aligned_tickers, aligned_tickers].to_numpy()
    if np.isnan(S).any() or np.isinf(S).any():
        raise ValueError("sigma contains NaN/Inf for one or more aligned tickers")
    S = 0.5 * (S + S.T)  # ensure symmetry, guarding against floating-point arithmetic errors,
                        # even if theoretically S is already symmetric (cvxpy is strict about symmetry)

    betas_vec: np.ndarray | None = None
    if tickers_betas is not None:
        betas_vec = tickers_betas.loc[aligned_tickers].to_numpy()
        if np.isnan(betas_vec).any() or np.isinf(betas_vec).any():
            raise ValueError("tickers_betas contains NaN/Inf for one or more aligned tickers")
        
    n = len(aligned_tickers)
    eps_beta = float(ptf_beta_cap)
    eps_net = float(ptf_net_exposure_cap)
    G_max = float(ptf_gross_exposure_cap)

    # ---  Build optimization problem 
    w = cp.Variable(n)
    constraints = [
        cp.abs(cp.sum(w)) <= eps_net,  # net exposure
        cp.norm(w, 1) <= G_max,     # gross exposure
    ]
    if betas_vec is not None:
        constraints.append(cp.abs(betas_vec @ w) <= eps_beta)
    if w_min is not None:
        constraints.append(w >= float(w_min))
    if w_max is not None:
        constraints.append(w <= float(w_max))

    if objective == "max_return":
        L = _cholesky_or_eigh(S)
        sigma2 = float(ptf_variance_cap)
        constraints.append(cp.norm(L @ w, 2) <= np.sqrt(sigma2))
        prob = cp.Problem(cp.Maximize(mu_vec @ w), constraints)
    else:  # min_variance
        R = float(ptf_return_lower_bound)
        constraints.append(mu_vec @ w >= R)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, cp.psd_wrap(S))), constraints  # we use function `psd_wrap` to signals to cvxpy that S is PSD,
                                                                       # in order to skip the time-consuming PSD check 
        )

    # --- Solve with fallback chain 
    def _try_solve(slv: str) -> float | None:
        if slv.upper() == "SCS":
            return prob.solve(solver=slv, verbose=verbose, max_iters=20000, eps=1e-5)
        if slv.upper() == "ECOS":
            return prob.solve(
                solver=slv, verbose=verbose,
                abstol=1e-8, reltol=1e-8, feastol=1e-8,
            )
        return prob.solve(solver=slv, verbose=verbose)

    tried: list[str] = []
    last_err: Exception | None = None
    for slv in (solver, *fallback_solvers):
        try:
            tried.append(slv)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                _try_solve(slv)
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
        except Exception as e:
            last_err = e
            logger.debug("solver %s raised: %s", slv, e)
            continue

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        msg = (
            f"optimization failed (objective={objective}). "
            f"status={prob.status}, solvers tried={tried}"
        )
        if last_err is not None:
            msg += f", last error={last_err!r}"
        raise RuntimeError(msg)

    w_star = np.asarray(w.value).reshape(-1)

    # --- Diagnostics
    risk_var = float(w_star @ S @ w_star)
    risk_vol = float(np.sqrt(max(risk_var, 0.0)))
    expected_return = float(mu_vec @ w_star)
    net_expo = float(np.sum(w_star))
    gross_expo = float(np.sum(np.abs(w_star)))

    exposures: dict[str, float] = {
        "net": net_expo, "net_cap": eps_net,
        "gross": gross_expo, "gross_cap": G_max,
    }
    residuals: dict[str, float] = {
        "net_minus_cap": net_expo - eps_net,
        "gross_minus_cap": gross_expo - G_max,
    }
    if betas_vec is not None:
        ptf_beta = float(betas_vec @ w_star)
        exposures["beta"] = ptf_beta
        exposures["beta_cap"] = eps_beta
        residuals["beta_minus_epsilon"] = ptf_beta - eps_beta

    risk: dict[str, float] = {
        "volatility": risk_vol,
        "variance": risk_var,
        "expected_return": expected_return,
    }
    if objective == "max_return":
        sigma_cap = float(np.sqrt(ptf_variance_cap))
        risk["volatility_cap"] = sigma_cap
        risk["variance_cap"] = float(ptf_variance_cap)
        residuals["variance_minus_cap"] = risk_var - ptf_variance_cap
    else:
        risk["return_lower_bound"] = float(ptf_return_lower_bound)
        residuals["return_minus_lower_bound"] = expected_return - ptf_return_lower_bound

    metrics: dict[str, Any] = {
        "objective": objective,
        "status": prob.status,
        "objective_value": float(prob.value) if prob.value is not None else None,
        "solver_stats": {
            "solver_name": getattr(prob.solver_stats, "solver_name", None),
            "solve_time": getattr(prob.solver_stats, "solve_time", None),
            "num_iters": getattr(prob.solver_stats, "num_iters", None),
        },
        "risk": risk,
        "exposures": exposures,
        "residuals": residuals,
        "tickers": list(aligned_tickers),
        "solvers_tried": tried,
    }

    optimal_weights = pd.Series(w_star, index=aligned_tickers, name="optimal_weights")
    return OptimizerResult(optimal_weights=optimal_weights, metrics=metrics)


def optimize_min_variance_with_feasibility_recovery(
    tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    tickers_betas: pd.Series | None,
    *,
    initial_return_lower_bound: float,
    min_return_lower_bound: float = 0.0,
    halving_steps: int = 6,
    **kwargs: Any,
) -> tuple[OptimizerResult, float]:
    """ 
    Minimum-variance solve with automatic relaxation of the return lower bound.

    Calls `optimize_weights` with `objective` set to "min_variance"` and
    `ptf_return_lower_bound` set to `initial_return_lower_bound`. If a solve fails (raises
    `RuntimeError` (e.g. the problem is infeasible at that lower bound), the return lower_bound 
    is halved and the solve retried, up to `halving_steps` times. 
    
    The lower bound is never reduced below `min_return_lower_bound`.

    Parameters
    ----------
    tickers, mu, sigma, tickers_betas
        Passed through to `optimize_weights`; see its docstring.
    initial_return_lower_bound
        Return floor used on the first attempt.
    min_return_lower_bound
        Lower limit on the return lower bound: a halved lower bound is never taken
        below this value. Default 0.0.
    halving_steps
        Maximum number of halvings after the initial attempt, so at most
        `halving_steps` + 1 solves are attempted. The smallest lower bound
        actually tried is
            max(`initial_return_lower_bound`/ 2 ** `halving_steps`, `min_return_lower_bound`)
    **kwargs
        Additional keyword arguments forwarded to `optimize_weights`
        (exposure caps, `w_min`/`w_max`, `solver`, etc.). Do not pass
        `objective` or `ptf_return_lower_bound`; both are set internally.

    Returns
    -------
    (OptimizerResult, used_return_lower_bound)
        The successful result and the return lower bound at which it was
        obtained (below `initial_return_lower_bound` if any relaxation
        occurred).

    Raises
    ------
    RuntimeError
        If none of the attempted lower bounds yields a feasible solution.

    Notes
    -----
    `min_return_lower bound` is a lower bound on the halving sequence, not a value
    that is guaranteed to be attempted. In particular, with the default
    `min_return_lower_bound=0.0`, repeated halving of a positive lower bound stays
    strictly positive, so a return lower bound of exactly zero (where the
    all-zero portfolio is trivially feasible) is never reached.
    """
    R = float(initial_return_lower_bound)
    floor = float(min_return_lower_bound)

    for step in range(halving_steps + 1):
        try:
            result = optimize_weights(
                tickers, mu, sigma, tickers_betas,
                objective = "min_variance",
                ptf_return_lower_bound = R,
                **kwargs,
            )
            if step > 0:
                logger.warning(
                    "min-variance feasibility recovered after %d halving step(s); "
                    "used ptf_return_lower_bound = %.6f instead of requested %.6f",
                    step, R, initial_return_lower_bound,
                )
            return result, R
        except RuntimeError as e:
            if R <= floor:
                raise
            R = max(R / 2.0, floor)
            logger.info("min-variance infeasible, halving return lower bound to %.6f", R)
            continue

    raise RuntimeError(f"min-variance infeasible even at return lower bound {floor};")