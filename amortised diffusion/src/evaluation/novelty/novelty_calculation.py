import numpy as np
from numba import jit, njit

from .novelty_calculation_cython import kabsch_alignment, rmsd


@njit
def tm_score(P: np.ndarray, Q: np.ndarray):
    """
    Calculate TM-score between P and Q
    """
    d0 = 1.24 * np.cbrt(P.shape[0] - 15) - 1.8
    d = np.sqrt(np.sum((P - Q) ** 2, axis=1))
    score = np.mean(1 / (1 + (d / d0) ** 2))
    return score


@jit(nopython=False, forceobj=True)
def gdt_score(p1: np.ndarray, p2: np.ndarray):
    """
    Calculate GDT score between p1 and p2.
    """
    thresholds = [1.0, 2.0, 4.0, 8.0]
    # calculate distance matrix
    diff = p1[:, np.newaxis, :] - p2[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=-1))

    gdt_scores = []
    for t in thresholds:
        # for each threshold, calculate GDT
        gdt = (dist_mat < t).max(axis=1).mean()
        gdt_scores.append(gdt)

    # GDT_TS is average of GDT scores
    gdt_ts = np.mean(gdt_scores)

    return gdt_ts


@jit(nopython=False, forceobj=True)
def find_closest_structure(p1: np.ndarray, samples: dict[str, np.ndarray]):
    """
    Find the closest structure to p1 in samples
    """
    ids = {
        "rmsd": None,
        "tm_score": None,
        "gdt_score": None,
    }
    metrics = {
        "rmsd": np.inf,
        "tm_score": -np.inf,
        "gdt_score": -np.inf,
    }

    for sample_id, sample in samples.items():
        if len(p1) == len(sample):
            p1_aligned = kabsch_alignment(p1, sample)

            # Check RMSD:
            rmsd_score = rmsd(p1_aligned, sample)
            if rmsd_score < metrics["rmsd"]:
                metrics["rmsd"] = rmsd_score
                ids["rmsd"] = sample_id
            # Check TM-score:
            tm_score_score = tm_score(p1_aligned, sample)
            if tm_score_score > metrics["tm_score"]:
                metrics["tm_score"] = tm_score_score
                ids["tm_score"] = sample_id
            # Check GDT-score:
            gdt_score_score = gdt_score(p1_aligned, sample)
            if gdt_score_score > metrics["gdt_score"]:
                metrics["gdt_score"] = gdt_score_score
                ids["gdt_score"] = sample_id

        elif len(p1) < len(sample):
            for i in range(len(sample) - len(p1)):
                _sample = sample[i : i + len(p1)]
                p1_aligned = kabsch_alignment(p1, _sample)

                # Check RMSD:
                rmsd_score = rmsd(p1_aligned, _sample)
                if rmsd_score < metrics["rmsd"]:
                    metrics["rmsd"] = rmsd_score
                    ids["rmsd"] = sample_id + f"_{i}"
                # Check TM-score:
                tm_score_score = tm_score(p1_aligned, _sample)
                if tm_score_score > metrics["tm_score"]:
                    metrics["tm_score"] = tm_score_score
                    ids["tm_score"] = sample_id + f"_{i}"
                # Check GDT-score:
                gdt_score_score = gdt_score(p1_aligned, _sample)
                if gdt_score_score > metrics["gdt_score"]:
                    metrics["gdt_score"] = gdt_score_score
                    ids["gdt_score"] = sample_id + f"_{i}"

        elif len(p1) > len(sample):
            for i in range(len(p1) - len(sample)):
                _p1 = p1[i : i + len(sample)]
                p1_aligned = kabsch_alignment(_p1, sample)

                # Check RMSD:
                rmsd_score = rmsd(p1_aligned, sample)
                if rmsd_score < metrics["rmsd"]:
                    metrics["rmsd"] = rmsd_score
                    ids["rmsd"] = sample_id + f"^{i}"
                # Check TM-score:
                tm_score_score = tm_score(p1_aligned, sample)
                if tm_score_score > metrics["tm_score"]:
                    metrics["tm_score"] = tm_score_score
                    ids["tm_score"] = sample_id + f"^{i}"
                # Check GDT-score:
                gdt_score_score = gdt_score(p1_aligned, sample)
                if gdt_score_score > metrics["gdt_score"]:
                    metrics["gdt_score"] = gdt_score_score
                    ids["gdt_score"] = sample_id + f"^{i}"

    # Combine ids and metrics
    out = {key + "_match": val for key, val in ids.items()}
    out.update(metrics)
    return out