from typing import List

import numpy as np


def is_value_unique(
    arr: np.ndarray, is_sorted: bool = False, atol: float = 1e-5, rtol: float = 1e-8
) -> np.ndarray:
    if is_sorted:
        _arr = arr
    else:
        sort_idxs = np.argsort(arr)
        _arr = arr[sort_idxs]
        unsort_idxs = np.empty_like(sort_idxs)
        unsort_idxs[sort_idxs] = np.arange(sort_idxs.size)

    # Find degenerate values up to `abs_tol` and `rel_tol`
    degenerate = np.zeros_like(_arr, dtype=bool)
    degenerate[1:] = np.isclose(_arr[1:], _arr[:-1], atol=atol, rtol=rtol)
    degenerate[0] = degenerate[1]
    return ~degenerate if is_sorted else ~degenerate[unsort_idxs]


def count_degenerate_eigenvalues(evals: np.ndarray, atol: float = 1e-5, rtol: float = 1e-8) -> int:
    # Find degenerate values up to `abs_tol` and `rel_tol`
    is_degenerate = ~is_value_unique(evals, is_sorted=True, atol=atol, rtol=rtol)
    # Count islands of degenerate values
    return np.sum(is_degenerate)


def count_degenerate_eigenspaces(evals: np.ndarray, atol: float = 1e-5, rtol: float = 1e-8) -> int:
    # Find degenerate values up to `abs_tol` and `rel_tol`
    is_degenerate = ~is_value_unique(evals, is_sorted=True, atol=atol, rtol=rtol)
    # Count islands of degenerate values
    return np.sum(np.diff(is_degenerate.astype(int)) == -1) + is_degenerate[-1]


def get_degenerate_eigenspaces(evals: np.ndarray) -> List[slice]:
    is_eval_degenerate = ~is_value_unique(evals, is_sorted=True)
    degernerate_islands = []
    if is_eval_degenerate[0]:
        degernerate_islands.append(0)
    for curr, next in zip(range(len(evals) - 1), range(1, len(evals))):
        if is_eval_degenerate[curr] and not is_eval_degenerate[next]:  # end of island
            degernerate_islands.append(next)
        elif not is_eval_degenerate[curr] and is_eval_degenerate[next]:  # start of island
            degernerate_islands.append(curr)
    if is_eval_degenerate[-1]:
        degernerate_islands.append(len(evals))
    return [slice(*degernerate_islands[i : i + 2]) for i in range(0, len(degernerate_islands), 2)]