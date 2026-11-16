"""Selection and estimation for the fixed-worker S2D experiment.

History maps immutable candidate IDs to {validation_index: success_rate}.
No missing observation is imputed as a failed execution.
"""

import hashlib
import json
import math
import random
from itertools import combinations


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def priorities(history, size):
    weights, anchors = {}, {}
    for index in range(size):
        outcomes = [row[index] for row in history.values() if index in row]
        if not outcomes:
            raise ValueError(f"Validation example {index} has no initialization evidence")
        p = sum(outcomes) / len(outcomes)
        floor = 0.5 if p == 0 else 0.0
        weights[index] = max(4 * p * (1 - p), floor) + 0.1 / math.sqrt(len(outcomes))
        anchors[index] = p
    return weights, anchors


def relationships(history, size, min_overlap, shrinkage):
    pairs, supported = {}, 0
    overlaps = []
    for left, right in combinations(range(size), 2):
        shared = [row for row in history.values() if left in row and right in row]
        n = len(shared)
        overlaps.append(n)
        if n < min_overlap:
            continue
        xs, ys = [r[left] for r in shared], [r[right] for r in shared]
        mx, my = sum(xs) / n, sum(ys) / n
        vx, vy = sum((x - mx) ** 2 for x in xs), sum((y - my) ** 2 for y in ys)
        if vx == 0 or vy == 0:
            continue
        supported += 1
        corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)
        redundancy = max(0.0, min(1.0, corr)) * n / (n + shrinkage)
        if redundancy > 0:
            pairs[(left, right)] = redundancy
    return pairs, {
        "total_pairs": len(overlaps), "supported_nonconstant_pairs": supported,
        "positive_pairs": len(pairs),
        "mean_joint_candidates": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "max_joint_candidates": max(overlaps, default=0),
    }


def draw(weights, pairs, count, alpha, exploration, rng, trace=False):
    remaining = sorted(weights)
    selected, steps = [], []
    while remaining and len(selected) < count:
        redundancy = {
            item: max((pairs.get(tuple(sorted((item, other))), 0.0)
                       for other in selected), default=0.0)
            for item in remaining
        }
        adjusted = {i: weights[i] / (1 + alpha * redundancy[i]) for i in remaining}
        total = sum(adjusted.values())
        probabilities = {
            i: ((1 - exploration) * adjusted[i] / total + exploration / len(remaining)
                if total > 0 else 1 / len(remaining)) for i in remaining
        }
        threshold, cumulative = rng.random(), 0.0
        chosen = remaining[-1]
        for item in remaining:
            cumulative += probabilities[item]
            if threshold < cumulative:
                chosen = item
                break
        if trace:
            unpenalized_total = sum(weights[i] for i in remaining)
            control = {
                i: ((1 - exploration) * weights[i] / unpenalized_total
                    + exploration / len(remaining)) if unpenalized_total else 1 / len(remaining)
                for i in remaining
            }
            steps.append({
                "step": len(selected) + 1, "chosen": chosen,
                "probabilities": probabilities, "redundancy": redundancy,
                "adjusted_priorities": adjusted,
                "penalty_changes_probabilities": any(
                    abs(probabilities[i] - control[i]) > 1e-12 for i in remaining),
            })
        selected.append(chosen)
        remaining.remove(chosen)
    return selected, steps


def make_plan(history, size, candidate_id, settings):
    weights, anchors = priorities(history, size)
    pairs, coverage = relationships(
        history, size, settings["min_overlap"], settings["shrinkage"])
    mode = settings["policy"]
    count = size if mode == "full" else max(1, int(size * settings["fraction"]))
    alpha = settings["alpha"] if mode == "adaptive-redundancy" else 0.0
    exploration = 0.0 if mode == "task-coevolve" else settings["exploration"]
    seed_key = f"{settings['seed']}:{candidate_id}"
    selected, steps = draw(weights, pairs, count, alpha, exploration,
                           random.Random(f"selection:{seed_key}"), trace=True)
    control, _ = draw(weights, pairs, count, 0.0, exploration,
                      random.Random(f"selection:{seed_key}"))
    counts = dict.fromkeys(weights, 0)
    reps = settings["probability_draws"]
    if count == size:
        probabilities = dict.fromkeys(weights, 1.0)
        zero_hits = 0
    else:
        for repetition in range(reps):
            sampled, _ = draw(weights, pairs, count, alpha, exploration,
                              random.Random(f"inclusion:{seed_key}:{repetition}"))
            for item in sampled:
                counts[item] += 1
        zero_hits = sum(n == 0 for n in counts.values())
        probabilities = {i: max(n, 0.5) / reps for i, n in counts.items()}
    return {
        "candidate_id": candidate_id, "history_hash": digest(history),
        "history_candidates": len(history), "pool_size": size,
        "selected": sorted(selected), "selection_order": selected,
        "priorities": weights, "anchors": anchors,
        "inclusion_probabilities": probabilities, "coverage": coverage,
        "positive_relationships": [
            {"left": a, "right": b, "redundancy": value}
            for (a, b), value in sorted(pairs.items())],
        "steps": steps, "probability_draws": reps, "zero_mc_hits": zero_hits,
        "min_inclusion_probability": min(probabilities.values()),
        "max_inclusion_mc_se": max(math.sqrt(p * (1 - p) / reps)
                                    for p in probabilities.values()),
        "selection_changed_vs_alpha_zero": sorted(selected) != sorted(control),
        "penalized_steps": sum(s["penalty_changes_probabilities"] for s in steps),
        "alpha_zero_selected": sorted(control), "settings": settings,
    }


def estimate(outcomes, plan, estimator):
    chosen = plan["selected"]
    if set(outcomes) != set(chosen):
        raise ValueError("Observed examples do not match the saved selection plan")
    raw = sum(outcomes.values()) / len(chosen)
    if len(chosen) == plan["pool_size"]:
        return {"subset_accuracy": raw, "estimated_accuracy": raw,
                "unclipped_estimate": raw, "approximate_se": 0.0, "measured_full": True}
    probabilities = {int(i): p for i, p in plan["inclusion_probabilities"].items()}
    if estimator == "hajek":
        weights = {i: 1 / probabilities[i] for i in chosen}
        total = sum(weights.values())
        value = sum(weights[i] * outcomes[i] for i in chosen) / total
        se = math.sqrt(sum((weights[i] * (outcomes[i] - value)) ** 2
                           for i in chosen)) / total
    elif estimator == "difference":
        anchors = {int(i): p for i, p in plan["anchors"].items()}
        terms = [(outcomes[i] - anchors[i]) / probabilities[i] for i in chosen]
        value = (sum(anchors.values()) + sum(terms)) / plan["pool_size"]
        se = math.sqrt(sum(t * t for t in terms)) / plan["pool_size"]
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    return {"subset_accuracy": raw, "estimated_accuracy": max(0.0, min(1.0, value)),
            "unclipped_estimate": value, "approximate_se": se,
            "measured_full": False}
