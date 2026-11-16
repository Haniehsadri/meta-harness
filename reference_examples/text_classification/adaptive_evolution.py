"""Auditable, resumable fixed-worker S2D evolution, opt-in via --task-selector."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from adaptive_selection import digest, estimate, make_plan


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(root, stage, **details):
    row = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, **details}
    with (root / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[adaptive] {stage}: {json.dumps(details, sort_keys=True)}", flush=True)


@contextmanager
def run_lock(root):
    path = root / "running.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run is locked: {path}. Check the recorded PID before removing a stale lock.")
    os.write(descriptor, json.dumps({"pid": os.getpid(), "host": os.environ.get(
        "COMPUTERNAME", os.environ.get("HOSTNAME", "unknown"))}).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def pool_fingerprint(code):
    files = [code / "config.yaml"] + sorted((code / "data").rglob("*.jsonl"))
    return digest({str(p.relative_to(code)): file_hash(p) for p in files})


def protected_files(code, baselines):
    files = list(code.glob("*.py")) + list((code / "data").rglob("*.py"))
    files += list((code / ".claude").rglob("SKILL.md")) + [code / "config.yaml"]
    files += list((code / "data").rglob("*.jsonl"))
    files += [code / "agents" / f"{name}.py" for name in baselines]
    return {str(p.relative_to(code)): file_hash(p) for p in files}


def validate_settings(args, cfg):
    if cfg["datasets"] != ["Symptom2Disease"]:
        raise ValueError("Stage 1 requires datasets: [Symptom2Disease]")
    if len(cfg["benchmark"]["seeds"]) != 1:
        raise ValueError("Stage 1 uses one fixed evaluation seed")
    if cfg["inner_loop"].get("mode") != "offline" or cfg["inner_loop"].get("num_epochs") != 1:
        raise ValueError("Stage 1 uses offline mode and exactly one training epoch")
    if len(cfg["models"]) != 1 or cfg["models"][0]["model"] != args.model:
        raise ValueError("Stage 1 requires exactly one configured fixed worker matching --model")
    if args.fresh or args.skip_baseline:
        raise ValueError("Use a new run name; adaptive runs initialize with full baselines")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name or ""):
        raise ValueError("Adaptive runs require a simple --run-name (letters, digits, _ or -)")
    if not math.isfinite(args.selection_fraction) or not 0 < args.selection_fraction <= 1:
        raise ValueError("selection-fraction must be in (0, 1]")
    if not math.isfinite(args.selection_exploration) or not 0 < args.selection_exploration < 1:
        raise ValueError("selection-exploration must be in (0, 1)")
    if not math.isfinite(args.redundancy_alpha) or args.redundancy_alpha < 0:
        raise ValueError("redundancy-alpha must be finite and nonnegative")
    if not math.isfinite(args.redundancy_shrinkage) or args.redundancy_shrinkage <= 0:
        raise ValueError("redundancy-shrinkage must be positive")
    if args.redundancy_min_overlap < 3 or args.inclusion_draws < 100:
        raise ValueError("Use min-overlap >= 3 and inclusion-draws >= 100")
    if args.iterations < 0 or args.evaluation_timeout <= 0:
        raise ValueError("iterations must be nonnegative and evaluation-timeout positive")


def outcome_map(result):
    rows = result.get("per_sample") or []
    indices = result.get("selected_val_indices")
    if indices is None or len(rows) != len(indices):
        raise ValueError("Result lacks complete per-example evidence")
    if {row["example_index"] for row in rows} != set(indices):
        raise ValueError("Result indices do not match per-example evidence")
    if any(type(row["was_correct"]) is not bool for row in rows):
        raise ValueError("Result correctness must contain observed booleans")
    return {row["example_index"]: float(row["was_correct"]) for row in rows}


def candidate_history(state):
    # Repeated measurements of identical source count as one harness version.
    grouped = {}
    for row in state["records"]:
        if row["status"] != "complete":
            continue
        by_index = grouped.setdefault(row["source_hash"], {})
        for index, value in row["outcomes"].items():
            by_index.setdefault(int(index), []).append(value)
    return {source: {i: sum(values) / len(values) for i, values in indices.items()}
            for source, indices in grouped.items()}


def verify_candidate_sources(code, records):
    for record in records:
        if file_hash(code / "agents" / f"{record['name']}.py") != record["source_hash"]:
            raise ValueError(f"Previously evaluated candidate changed: {record['name']}")


def write_frontier(root, state):
    complete = [r for r in state["records"] if r["status"] == "complete"]
    if not complete:
        return
    best = max(complete, key=lambda r: r["score"]["estimated_accuracy"])
    value = 100 * best["score"]["estimated_accuracy"]
    save_json(root / "frontier_val.json", {
        "Symptom2Disease": {"best_system": best["name"], "accuracy": value,
                            "score_kind": "estimated_full_pool"},
        "_pareto": [{"system": best["name"], "val_accuracy": value}],
        "_selection_estimator": state["manifest"]["estimator"],
    })
    rows = [{"iteration": r["iteration"], "system": r["name"],
             "candidate_id": r["candidate_id"], "status": r["status"],
             "hypothesis": r.get("hypothesis", ""), **r.get("score", {}),
             "avg_val": 100 * r["score"]["subset_accuracy"] if r.get("score") else None,
             "est_full_val": 100 * r["score"]["estimated_accuracy"] if r.get("score") else None,
             "est_full_se": 100 * r["score"]["approximate_se"] if r.get("score") else None}
            for r in state["records"] if r["iteration"] > 0]
    temporary = root / "evolution_summary.tmp"
    temporary.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    temporary.replace(root / "evolution_summary.jsonl")


def evaluate(root, code, cfg, args, name, indices, phase, candidate_id, load_memory=None):
    model = cfg["models"][0]
    sizes = {**{k: cfg["dataset"][k] for k in ("num_train", "num_val", "num_test")},
             **cfg["dataset"].get("overrides", {}).get("Symptom2Disease", {})}
    seed = cfg["benchmark"]["seeds"][0]
    identity = {"candidate_id": candidate_id, "indices": indices, "phase": phase,
                "config": cfg, "pool": pool_fingerprint(code)}
    if load_memory:
        identity["memory_hash"] = file_hash(load_memory)
    evaluation_id = digest(identity)
    directory = root / "evaluations" / phase / name
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ("test.json" if phase == "test" else "val.json")
    memory = load_memory or directory / "memory.json"
    if output.exists():
        result = read_json(output)
        if result.get("evaluation_id") != evaluation_id:
            raise ValueError(f"Incompatible cached result: {output}")
        if result.get("memory_sha256") != file_hash(memory):
            raise ValueError(f"Memory changed since evaluation: {memory}")
        event(root, "evaluation_cached", name=name, phase=phase, result=str(output))
        return result, memory
    command = [sys.executable, "-m", "text_classification.inner_loop", "--memory", name,
               "--dataset", "Symptom2Disease", "--model", model["model"],
               "--seed", str(seed), "--mode", "offline", "--num-epochs", "1",
               "--single-validation-pass", "--evaluation-id", evaluation_id,
               "--log", str(directory / "trace.jsonl"), "--force"]
    for key, value in sizes.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if model.get("api_base"):
        command.extend(["--api-base", model["api_base"]])
    temperature = cfg["inner_loop"].get("temperature")
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if load_memory:
        command.extend(["--load-memory", str(load_memory)])
    else:
        command.extend(["--save-memory", str(memory)])
    command.extend(["--test-output" if phase == "test" else "--val-output", str(output)])
    if phase != "test":
        command.extend(["--val-indices-json", json.dumps(indices)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code.parent)
    env["PYTHONUNBUFFERED"] = "1"
    attempt = len(list(directory.glob("attempt-*.log"))) + 1
    log = directory / f"attempt-{attempt:03d}.log"
    event(root, "evaluation_started", name=name, phase=phase, selected=indices,
          evaluation_id=evaluation_id, attempt=attempt, log=str(log),
          memory_reused=bool(load_memory))
    started = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(command) + "\n")
            handle.flush()
            process = subprocess.run(command, cwd=code.parent, env=env, stdout=handle,
                                     stderr=subprocess.STDOUT, timeout=args.evaluation_timeout)
        if process.returncode != 0:
            raise RuntimeError(f"Evaluation exit {process.returncode}; see {log}")
    except (subprocess.TimeoutExpired, RuntimeError) as error:
        event(root, "evaluation_failed", name=name, phase=phase, status="execution_error",
              elapsed_seconds=time.monotonic() - started, error=str(error), log=str(log))
        raise
    result = read_json(output)
    if result.get("evaluation_id") != evaluation_id or result.get("total") != (
            sizes["num_test"] if phase == "test" else len(indices)):
        raise ValueError(f"Incomplete or mismatched evaluation: {output}")
    event(root, "evaluation_completed", name=name, phase=phase,
          elapsed_seconds=time.monotonic() - started, accuracy=result["accuracy"],
          examples=result["total"], llm_calls=result.get("llm_calls"),
          input_tokens=result.get("llm_input_tokens"), output_tokens=result.get("llm_output_tokens"),
          estimated_cost_usd=result.get("estimated_cost_usd"),
          result=str(output), trace=str(directory / "trace.jsonl"))
    return result, memory


def run_adaptive_evolution(args, harness):
    import yaml
    cfg = yaml.safe_load(harness.CONFIG_PATH.read_text(encoding="utf-8-sig"))
    validate_settings(args, cfg)
    settings = {"policy": args.task_selector, "fraction": args.selection_fraction,
                "alpha": args.redundancy_alpha, "exploration": args.selection_exploration,
                "min_overlap": args.redundancy_min_overlap,
                "shrinkage": args.redundancy_shrinkage,
                "probability_draws": args.inclusion_draws, "seed": cfg["benchmark"]["seeds"][0]}
    root = harness.EVOLVE_DIR / "logs" / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"config": cfg, "settings": settings, "estimator": args.selection_estimator,
                "proposer_model": os.environ.get("METAHARNESS_PROPOSER_MODEL") or
                os.environ.get("ANTHROPIC_MODEL", "opus"),
                "proposer_endpoint": os.environ.get("ANTHROPIC_BASE_URL"),
                "proposer_output_limit": os.environ.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS"),
                "source_pool_hash": pool_fingerprint(harness.EVOLVE_DIR),
                "implementation_hash": digest({p.name: file_hash(p)
                    for p in harness.EVOLVE_DIR.glob("*.py")})}
    with run_lock(root):
        state_path = root / "state.json"
        if state_path.exists():
            state = read_json(state_path)
            if state["manifest"] != manifest:
                raise ValueError("Run configuration changed; use a new run name")
        else:
            state = {"manifest": manifest, "records": [], "completed_iteration": 0,
                     "active_iteration": None}
            save_json(state_path, state)
        if (root / "finalized.json").exists() and not args.test:
            raise ValueError("Final confirmation has started; this run cannot evolve further")
        code = root / "code" / "text_classification"
        if not code.exists():
            staging = root / "code" / "staging"
            if staging.exists():
                raise RuntimeError(f"Incomplete snapshot at {staging}; inspect before resuming")
            shutil.copytree(harness.EVOLVE_DIR, staging, ignore=shutil.ignore_patterns(
                "logs", "results", ".venv", "__pycache__", ".git", ".env", ".env.*",
                ".claude-cli", ".empty_plugins", "tests"))
            code.parent.mkdir(exist_ok=True)
            staging.rename(code)
        baselines = cfg["memory_systems"]["baselines"]
        state.setdefault("protected_agents", sorted(p.stem for p in (code / "agents").glob("*.py")))
        protected = protected_files(code, state["protected_agents"])
        if "protected" in state and protected != state["protected"]:
            raise ValueError("Evaluation snapshot has changed")
        state["protected"] = protected
        save_json(state_path, state)
        size = cfg["dataset"].get("overrides", {}).get("Symptom2Disease", {}).get(
            "num_val", cfg["dataset"]["num_val"])
        if not 2 <= size <= 50:
            raise ValueError("S2D validation pool size must be between 2 and 50")
        full_indices = list(range(size))
        event(root, "run_started", resumed=bool(state["records"]), settings=settings,
              validation_pool=size, estimator=args.selection_estimator, snapshot=str(code))

        for name in baselines:
            if any(r["name"] == name for r in state["records"]):
                continue
            source_hash = file_hash(code / "agents" / f"{name}.py")
            candidate_id = f"baseline:{name}:{source_hash}"
            result, memory = evaluate(root, code, cfg, args, name, full_indices,
                                      "baseline", candidate_id)
            outcomes = outcome_map(result)
            if set(outcomes) != set(full_indices):
                raise ValueError("Initialization requires complete validation evidence")
            state["records"].append({"candidate_id": candidate_id, "name": name,
                "source_hash": source_hash, "iteration": 0, "status": "complete",
                "outcomes": outcomes, "memory": str(memory),
                "score": {"subset_accuracy": result["accuracy"],
                          "estimated_accuracy": result["accuracy"], "approximate_se": 0.0,
                          "measured_full": True}})
            save_json(state_path, state)
        verify_candidate_sources(code, state["records"])
        write_frontier(root, state)

        if args.test:
            if state["active_iteration"] is not None:
                raise ValueError("Finish the interrupted search iteration before final testing")
            finalize(root, code, cfg, args, state)
            return
        # Proposer reads and writes inside the run's source snapshot.
        harness.EVOLVE_DIR = code
        harness.LOGS_DIR = root
        harness.PENDING_EVAL = root / "pending_eval.json"
        harness.FRONTIER_VAL = root / "frontier_val.json"
        harness.EVOLUTION_SUMMARY = root / "evolution_summary.jsonl"
        for _ in range(args.iterations):
            if harness._interrupted:
                break
            iteration = state["active_iteration"] or state["completed_iteration"] + 1
            state["active_iteration"] = iteration
            save_json(state_path, state)
            proposal_path = root / f"iteration-{iteration:04d}.json"
            if not proposal_path.exists():
                harness.PENDING_EVAL.unlink(missing_ok=True)
                event(root, "proposal_started", iteration=iteration)
                prompt = harness.render_task_prompt(iteration, 1) + (
                    "\nValidation uses sampled examples. Compare candidates with est_full_val, "
                    "not raw subset accuracy. Only edit NEW agents/<name>.py files. "
                    "Evaluator/config/data and existing candidates must remain unchanged.")
                if not harness.propose_claude(prompt, iteration, timeout=args.propose_timeout):
                    event(root, "proposal_failed", iteration=iteration)
                    raise RuntimeError("Proposer produced no usable pending_eval.json; inspect claude_sessions")
                if protected_files(code, state["protected_agents"]) != protected:
                    raise ValueError("Proposer modified protected evaluation files")
                verify_candidate_sources(code, state["records"])
                proposal = read_json(harness.PENDING_EVAL)
                candidates = proposal.get("candidates") or []
                if not candidates:
                    raise ValueError("Proposer returned an empty candidate list")
                names = [c["name"] for c in candidates]
                known = {r["name"] for r in state["records"]}
                if len(set(names)) != len(names) or known.intersection(names):
                    raise ValueError("Candidate names must be new and unique")
                for candidate in candidates:
                    name = candidate["name"]
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                        raise ValueError(f"Invalid candidate module name: {name}")
                    candidate["source_hash"] = file_hash(code / "agents" / f"{name}.py")
                    candidate["candidate_id"] = f"iter{iteration}:{name}:{candidate['source_hash']}"
                save_json(proposal_path, proposal)
                event(root, "proposal_completed", iteration=iteration, candidates=names)
            for candidate in read_json(proposal_path)["candidates"]:
                name, candidate_id = candidate["name"], candidate["candidate_id"]
                if any(r["candidate_id"] == candidate_id for r in state["records"]):
                    continue
                if harness._interrupted:
                    return
                if protected_files(code, state["protected_agents"]) != protected:
                    raise ValueError("Evaluation snapshot was modified")
                if file_hash(code / "agents" / f"{name}.py") != candidate["source_hash"]:
                    raise ValueError(f"Candidate source changed: {name}")
                plan_path = root / "selections" / f"{iteration:04d}-{name}.json"
                history = candidate_history(state)
                if plan_path.exists():
                    plan = read_json(plan_path)
                    if plan["history_hash"] != digest(history):
                        raise ValueError("Saved plan history differs from committed history")
                else:
                    event(root, "selection_started", iteration=iteration, name=name,
                          history_hash=digest(history), history_candidates=len(history),
                          monte_carlo_draws=settings["probability_draws"])
                    selection_started = time.monotonic()
                    plan = make_plan(history, size, candidate_id, settings)
                    plan["selection_seconds"] = time.monotonic() - selection_started
                    save_json(plan_path, plan)
                if settings["policy"] == "adaptive-redundancy" and settings["alpha"] > 0:
                    if plan["coverage"]["positive_pairs"] == 0:
                        event(root, "no_supported_positive_relationships", name=name,
                              min_overlap=settings["min_overlap"], coverage=plan["coverage"],
                              message="No redundancy penalty is active for this candidate")
                if plan["zero_mc_hits"]:
                    event(root, "inclusion_probability_warning", name=name,
                          zero_mc_hits=plan["zero_mc_hits"],
                          message="Some inclusion probabilities used a Monte Carlo floor")
                event(root, "selection_frozen", iteration=iteration, name=name,
                      selection_seconds=plan.get("selection_seconds"),
                      selected=plan["selected"], selected_count=len(plan["selected"]),
                      history_candidates=plan["history_candidates"], coverage=plan["coverage"],
                      penalized_steps=plan["penalized_steps"],
                      selection_changed_vs_alpha_zero=plan["selection_changed_vs_alpha_zero"],
                      zero_mc_hits=plan["zero_mc_hits"],
                      min_inclusion_probability=plan["min_inclusion_probability"],
                      max_inclusion_mc_se=plan["max_inclusion_mc_se"], plan=str(plan_path))
                result, memory = evaluate(root, code, cfg, args, name, plan["selected"],
                                          "search", candidate_id)
                if protected_files(code, state["protected_agents"]) != protected:
                    raise ValueError("Candidate modified protected evaluation files")
                verify_candidate_sources(code, state["records"])
                if file_hash(code / "agents" / f"{name}.py") != candidate["source_hash"]:
                    raise ValueError("Candidate source changed during execution")
                outcomes = outcome_map(result)
                score = estimate(outcomes, plan, args.selection_estimator)
                state["records"].append({**candidate, "iteration": iteration,
                    "status": "complete", "outcomes": outcomes, "score": score,
                    "memory": str(memory), "selection_plan": str(plan_path)})
                save_json(state_path, state)
                write_frontier(root, state)
                event(root, "history_updated", name=name, score=score,
                      candidates=len(state["records"]))
            state["completed_iteration"] = iteration
            state["active_iteration"] = None
            save_json(state_path, state)
            event(root, "iteration_completed", iteration=iteration)
        event(root, "search_stopped", completed_iteration=state["completed_iteration"],
              next_step="Review logs; use the same selector arguments plus --test for final confirmation")


def finalize(root, code, cfg, args, state):
    marker = root / "finalized.json"
    if marker.exists() and read_json(marker).get("status") == "complete":
        print(json.dumps(read_json(marker), indent=2))
        return
    # All policies use the same declared top-three full-validation confirmation.
    if not marker.exists():
        ranked = sorted(state["records"], key=lambda r: -r["score"]["estimated_accuracy"])
        save_json(marker, {"status": "confirming", "shortlist": [r["name"] for r in ranked[:3]]})
    final = read_json(marker)
    records = {r["name"]: r for r in state["records"]}
    confirmations = []
    size = cfg["dataset"].get("overrides", {}).get("Symptom2Disease", {}).get(
        "num_val", cfg["dataset"]["num_val"])
    for name in final["shortlist"]:
        record = records[name]
        if file_hash(code / "agents" / f"{name}.py") != record["source_hash"]:
            raise ValueError(f"Candidate source changed before confirmation: {name}")
        # A full-pool result already measures the confirmation objective.
        if record["score"]["measured_full"]:
            confirmations.append((name, record["score"]["estimated_accuracy"]))
        else:
            result, _ = evaluate(root, code, cfg, args, name, list(range(size)),
                                 "confirmation", record["candidate_id"], Path(record["memory"]))
            confirmations.append((name, result["accuracy"]))
    winner, accuracy = max(confirmations, key=lambda pair: pair[1])
    event(root, "winner_selected", winner=winner, full_validation_accuracy=accuracy,
          confirmations=confirmations)
    record = records[winner]
    result, _ = evaluate(root, code, cfg, args, winner, [], "test", record["candidate_id"],
                         Path(record["memory"]))
    final.update(status="complete", winner=winner, confirmation_scores=confirmations,
                 test_accuracy=result["accuracy"], test_examples=result["total"])
    save_json(marker, final)
    event(root, "finalization_completed", **final)
