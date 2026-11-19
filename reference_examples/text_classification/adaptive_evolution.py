"""Auditable, resumable fixed-worker classification evolution, opt-in via --task-selector."""

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
from proposal_workspace import (accept_proposal, archive_source, install_prepared_sources,
                                preserve_sources, proposal_workspace)


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
    from runpy import run_path
    registry = run_path(str(Path(__file__).parent / "data" / "constants.py"))
    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets or any(
            not isinstance(d, str) or d not in registry["ALL_TASKS"] for d in datasets):
        raise ValueError("datasets must be a nonempty list of supported classification tasks")
    if len(set(datasets)) != len(datasets):
        raise ValueError("datasets must not contain duplicates")
    for dataset in datasets:
        sizes = split_sizes(cfg, dataset)
        if any(type(v) is not int or v <= 0 for v in sizes.values()) or sizes["num_val"] < 2:
            raise ValueError(f"{dataset}: split sizes must be positive integers; num_val >= 2")
        official = registry["MCE_DATA_SPLITS"].get(dataset)
        if official and any(sizes["num_" + split] > limit for split, limit in official.items()):
            raise ValueError(f"{dataset}: requested sizes {sizes} exceed official splits {official}")
    if not cfg["memory_systems"]["baselines"]:
        raise ValueError("Adaptive selection requires at least one full-validation baseline")
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


def candidate_history(state, dataset="Symptom2Disease"):
    # Repeated measurements of identical source count as one harness version.
    grouped = {}
    for row in state["records"]:
        if row["status"] != "complete" or row.get("dataset", "Symptom2Disease") != dataset:
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


def split_sizes(cfg, dataset):
    """Use the same default + per-dataset override precedence as benchmark.py."""
    return {**{k: cfg["dataset"][k] for k in ("num_train", "num_val", "num_test")},
            **{k: v for k, v in cfg["dataset"].get("overrides", {}).get(dataset, {}).items()
               if k in ("num_train", "num_val", "num_test")}}


def record_dataset(record):
    # Old S2D records did not carry this explicit field.
    return record.get("dataset", "Symptom2Disease")


def completed_record(state, candidate_id, dataset):
    return next((r for r in state["records"] if r["candidate_id"] == candidate_id
                 and record_dataset(r) == dataset and r["status"] == "complete"), None)


def aggregate_records(state):
    """Rank a harness only after all configured datasets have committed evidence."""
    datasets = state["manifest"]["config"]["datasets"]
    grouped = {}
    for record in state["records"]:
        if record["status"] == "complete":
            grouped.setdefault(record["candidate_id"], {})[record_dataset(record)] = record
    result = []
    for records in grouped.values():
        if not all(d in records for d in datasets):
            continue
        first = records[datasets[0]]
        scores = [records[d]["score"] for d in datasets]
        score = {
            "subset_accuracy": sum(s["subset_accuracy"] for s in scores) / len(scores),
            "estimated_accuracy": sum(s["estimated_accuracy"] for s in scores) / len(scores),
            # Approximation assuming independent sampling across dataset pools.
            "approximate_se": math.sqrt(sum(s["approximate_se"] ** 2 for s in scores)) / len(scores),
            "measured_full": all(s["measured_full"] for s in scores),
        }
        result.append({**first, "score": score, "dataset_records": records})
    return result


def write_frontier(root, state):
    complete = aggregate_records(state)
    if not complete:
        return
    datasets = state["manifest"]["config"]["datasets"]
    best = max(complete, key=lambda r: r["score"]["estimated_accuracy"])
    frontier = {}
    for dataset in datasets:
        leader = max(complete, key=lambda r: r["dataset_records"][dataset]["score"]["estimated_accuracy"])
        score = leader["dataset_records"][dataset]["score"]
        frontier[dataset] = {"best_system": leader["name"],
                             "accuracy": 100 * score["estimated_accuracy"],
                             "score_kind": "measured_full_pool" if score["measured_full"] else "estimated_full_pool"}
    frontier.update({"_pareto": [{"system": best["name"],
                                 "val_accuracy": 100 * best["score"]["estimated_accuracy"]}],
                     "_selection_estimator": state["manifest"]["estimator"],
                     "_aggregation": "unweighted_mean_across_datasets"})
    save_json(root / "frontier_val.json", frontier)
    rows = [{"iteration": r["iteration"], "system": r["name"],
             "candidate_id": r["candidate_id"], "status": r["status"],
             "hypothesis": r.get("hypothesis", ""), **r["score"],
             "avg_val": 100 * r["score"]["subset_accuracy"],
             "est_full_val": 100 * r["score"]["estimated_accuracy"],
             "est_full_se": 100 * r["score"]["approximate_se"],
             "per_dataset": {d: {**record["score"], "selected_count": len(record["outcomes"]),
                                  "memory": record["memory"]}
                             for d, record in r["dataset_records"].items()}}
            for r in complete if r["iteration"] > 0]
    temporary = root / "evolution_summary.tmp"
    temporary.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    temporary.replace(root / "evolution_summary.jsonl")


def evaluate_dataset(root, code, cfg, args, name, indices, phase, candidate_id,
                     dataset, load_memory=None):
    # Preserve old single-S2D paths and evaluator call compatibility.
    kwargs = {} if cfg["datasets"] == ["Symptom2Disease"] else {"dataset": dataset}
    return evaluate(root, code, cfg, args, name, indices, phase, candidate_id,
                    load_memory=load_memory, **kwargs)

def evaluate(root, code, cfg, args, name, indices, phase, candidate_id, load_memory=None, dataset=None):
    model = cfg["models"][0]
    dataset = dataset or cfg["datasets"][0]
    sizes = split_sizes(cfg, dataset)
    seed = cfg["benchmark"]["seeds"][0]
    identity = {"candidate_id": candidate_id, "indices": indices, "phase": phase,
                "config": cfg, "pool": pool_fingerprint(code)}
    if cfg["datasets"] != ["Symptom2Disease"]:
        identity["dataset"] = dataset
    if load_memory:
        identity["memory_hash"] = file_hash(load_memory)
    evaluation_id = digest(identity)
    directory = root / "evaluations" / phase
    if cfg["datasets"] != ["Symptom2Disease"]:
        directory /= dataset
    directory /= name
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / ("test.json" if phase == "test" else "val.json")
    memory = load_memory or directory / "memory.json"
    if output.exists():
        result = read_json(output)
        if result.get("evaluation_id") != evaluation_id:
            raise ValueError(f"Incompatible cached result: {output}")
        if result.get("memory_sha256") != file_hash(memory):
            raise ValueError(f"Memory changed since evaluation: {memory}")
        event(root, "evaluation_cached", name=name, dataset=dataset, phase=phase, result=str(output))
        return result, memory
    command = [sys.executable, "-m", "text_classification.inner_loop", "--memory", name,
               "--dataset", dataset, "--model", model["model"],
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
    event(root, "evaluation_started", name=name, dataset=dataset, phase=phase, selected=indices,
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
        event(root, "evaluation_failed", name=name, dataset=dataset, phase=phase, status="execution_error",
              elapsed_seconds=time.monotonic() - started, error=str(error), log=str(log))
        raise
    result = read_json(output)
    if result.get("dataset") != dataset:
        raise ValueError(f"Result dataset mismatch: {output}")
    if result.get("evaluation_id") != evaluation_id or result.get("total") != (
            sizes["num_test"] if phase == "test" else len(indices)):
        raise ValueError(f"Incomplete or mismatched evaluation: {output}")
    event(root, "evaluation_completed", name=name, dataset=dataset, phase=phase,
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
        preserve_sources(root, code, state["records"], file_hash, event)
        baselines = cfg["memory_systems"]["baselines"]
        state.setdefault("protected_agents", sorted(p.stem for p in (code / "agents").glob("*.py")))
        protected = protected_files(code, state["protected_agents"])
        if "protected" in state and protected != state["protected"]:
            raise ValueError("Evaluation snapshot has changed")
        state["protected"] = protected
        save_json(state_path, state)
        datasets = cfg["datasets"]
        pools = {dataset: split_sizes(cfg, dataset)["num_val"] for dataset in datasets}
        event(root, "run_started", resumed=bool(state["records"]), settings=settings,
              datasets=datasets, validation_pools=pools,
              validation_pool=pools[datasets[0]] if len(datasets) == 1 else None,
              aggregation="unweighted_mean_across_datasets",
              estimator=args.selection_estimator, snapshot=str(code))

        for name in baselines:
            source_hash = file_hash(code / "agents" / f"{name}.py")
            archive_source(root, name, source_hash, (code / "agents" / f"{name}.py").read_bytes())
            candidate_id = f"baseline:{name}:{source_hash}"
            for dataset in datasets:
                if completed_record(state, candidate_id, dataset):
                    continue
                full_indices = list(range(pools[dataset]))
                result, memory = evaluate_dataset(root, code, cfg, args, name, full_indices,
                                                 "baseline", candidate_id, dataset)
                outcomes = outcome_map(result)
                if set(outcomes) != set(full_indices):
                    raise ValueError(f"{dataset}: initialization requires complete validation evidence")
                state["records"].append({"candidate_id": candidate_id, "name": name,
                    "dataset": dataset, "source_hash": source_hash, "iteration": 0,
                    "status": "complete", "outcomes": outcomes, "memory": str(memory),
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
        # Canonical paths are used only by the evaluator; proposals get disposable copies.
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
                event(root, "proposal_started", iteration=iteration)
                try:
                    with proposal_workspace(root, code, harness, iteration, datasets,
                                            file_hash, event) as (workspace, work_code, existing):
                        prompt = harness.render_task_prompt(iteration, len(datasets)) + (
                            "\nValidation uses sampled examples. Compare candidates with est_full_val, "
                            "not raw subset accuracy. Each dataset has its own pool/history; overall "
                            "scores are unweighted dataset means. This is a disposable proposal workspace. "
                            "Only NEW agents/<name>.py files will be imported. Use names prefixed with "
                            f"iter{iteration:04d}_ and never reuse or delete an existing filename. "
                            "Existing agent/evaluator/config/data files must remain unchanged. "
                            "Put temporary check scripts outside the code directory, within this workspace. "
                            "Filesystem writes outside the workspace are blocked, including from Bash.")
                        if not harness.propose_claude(prompt, iteration, timeout=args.propose_timeout):
                            raise RuntimeError("Proposer produced no usable pending_eval.json; inspect proposal workspace logs")
                        if protected_files(code, state["protected_agents"]) != protected:
                            raise ValueError("Canonical protected files changed during proposal")
                        verify_candidate_sources(code, state["records"])
                        proposal = accept_proposal(root, code, work_code, existing,
                            harness.PENDING_EVAL, iteration, file_hash, save_json)
                except Exception as error:
                    event(root, "proposal_rejected", iteration=iteration, error=str(error),
                          message="No unvalidated agent files imported; prior results retained")
                    raise
                event(root, "proposal_completed", iteration=iteration,
                      candidates=[c["name"] for c in proposal["candidates"]])
            install_prepared_sources(root, code, read_json(proposal_path)["candidates"], file_hash, event)
            for candidate in read_json(proposal_path)["candidates"]:
                name, candidate_id = candidate["name"], candidate["candidate_id"]
                if harness._interrupted:
                    return
                if protected_files(code, state["protected_agents"]) != protected:
                    raise ValueError("Evaluation snapshot was modified")
                if file_hash(code / "agents" / f"{name}.py") != candidate["source_hash"]:
                    raise ValueError(f"Candidate source changed: {name}")
                for dataset in datasets:
                    if completed_record(state, candidate_id, dataset):
                        continue
                    if harness._interrupted:
                        return
                    plan_directory = root / "selections"
                    if datasets != ["Symptom2Disease"]:
                        plan_directory /= dataset
                    plan_path = plan_directory / f"{iteration:04d}-{name}.json"
                    history = candidate_history(state, dataset)
                    if plan_path.exists():
                        plan = read_json(plan_path)
                        if plan["history_hash"] != digest(history):
                            raise ValueError("Saved plan history differs from committed history")
                    else:
                        event(root, "selection_started", iteration=iteration, name=name, dataset=dataset,
                              history_hash=digest(history), history_candidates=len(history),
                              monte_carlo_draws=settings["probability_draws"])
                        selection_started = time.monotonic()
                        # Namespace RNG streams across datasets; preserve single-S2D draws.
                        selection_id = candidate_id if datasets == ["Symptom2Disease"] else f"{dataset}:{candidate_id}"
                        plan = make_plan(history, pools[dataset], selection_id, settings)
                        plan["dataset"] = dataset
                        plan["selection_seconds"] = time.monotonic() - selection_started
                        save_json(plan_path, plan)
                    if plan.get("dataset", "Symptom2Disease") != dataset:
                        raise ValueError("Saved selection plan belongs to another dataset")
                    if settings["policy"] == "adaptive-redundancy" and settings["alpha"] > 0:
                        if plan["coverage"]["positive_pairs"] == 0:
                            event(root, "no_supported_positive_relationships", name=name, dataset=dataset,
                                  min_overlap=settings["min_overlap"], coverage=plan["coverage"],
                                  message="No redundancy penalty is active for this candidate")
                    if plan["zero_mc_hits"]:
                        event(root, "inclusion_probability_warning", name=name, dataset=dataset,
                              zero_mc_hits=plan["zero_mc_hits"],
                              message="Some inclusion probabilities used a Monte Carlo floor")
                    event(root, "selection_frozen", iteration=iteration, name=name, dataset=dataset,
                          validation_pool=pools[dataset], selection_seconds=plan.get("selection_seconds"),
                          selected=plan["selected"], selected_count=len(plan["selected"]),
                          history_candidates=plan["history_candidates"], coverage=plan["coverage"],
                          penalized_steps=plan["penalized_steps"],
                          selection_changed_vs_alpha_zero=plan["selection_changed_vs_alpha_zero"],
                          zero_mc_hits=plan["zero_mc_hits"],
                          min_inclusion_probability=plan["min_inclusion_probability"],
                          max_inclusion_mc_se=plan["max_inclusion_mc_se"], plan=str(plan_path))
                    result, memory = evaluate_dataset(root, code, cfg, args, name, plan["selected"],
                                                     "search", candidate_id, dataset)
                    if protected_files(code, state["protected_agents"]) != protected:
                        raise ValueError("Candidate modified protected evaluation files")
                    verify_candidate_sources(code, state["records"])
                    if file_hash(code / "agents" / f"{name}.py") != candidate["source_hash"]:
                        raise ValueError("Candidate source changed during execution")
                    outcomes = outcome_map(result)
                    score = estimate(outcomes, plan, args.selection_estimator)
                    state["records"].append({**candidate, "iteration": iteration, "dataset": dataset,
                        "status": "complete", "outcomes": outcomes, "score": score,
                        "memory": str(memory), "selection_plan": str(plan_path)})
                    save_json(state_path, state)
                    write_frontier(root, state)
                    event(root, "history_updated", name=name, dataset=dataset, score=score,
                          candidates=len(candidate_history(state, dataset)))
                aggregate = next(r for r in aggregate_records(state) if r["candidate_id"] == candidate_id)
                event(root, "candidate_completed", name=name, datasets=datasets,
                      score=aggregate["score"],
                      per_dataset={d: r["score"] for d, r in aggregate["dataset_records"].items()})
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
    datasets = cfg["datasets"]
    ranked = sorted(aggregate_records(state), key=lambda r: -r["score"]["estimated_accuracy"])
    if not ranked:
        raise ValueError("No harness has complete results on all configured datasets")
    if not marker.exists():
        save_json(marker, {"status": "confirming", "datasets": datasets,
                           "shortlist": [r["name"] for r in ranked[:3]]})
    final = read_json(marker)
    records = {r["name"]: r for r in ranked}
    confirmations, per_dataset = [], {}
    for name in final["shortlist"]:
        candidate = records[name]
        per_dataset[name] = {}
        for dataset in datasets:
            record = candidate["dataset_records"][dataset]
            if file_hash(code / "agents" / f"{name}.py") != record["source_hash"]:
                raise ValueError(f"Candidate source changed before confirmation: {name}")
            if record["score"]["measured_full"]:
                accuracy = record["score"]["estimated_accuracy"]
                event(root, "confirmation_reused", name=name, dataset=dataset, accuracy=accuracy)
            else:
                size = split_sizes(cfg, dataset)["num_val"]
                result, _ = evaluate_dataset(root, code, cfg, args, name, list(range(size)),
                    "confirmation", record["candidate_id"], dataset, Path(record["memory"]))
                accuracy = result["accuracy"]
            per_dataset[name][dataset] = accuracy
        confirmations.append((name, sum(per_dataset[name].values()) / len(datasets)))
    winner, accuracy = max(confirmations, key=lambda pair: pair[1])
    event(root, "winner_selected", winner=winner, full_validation_accuracy=accuracy,
          confirmations=confirmations, per_dataset=per_dataset,
          aggregation="unweighted_mean_across_datasets")
    tests = {}
    for dataset in datasets:
        record = records[winner]["dataset_records"][dataset]
        result, _ = evaluate_dataset(root, code, cfg, args, winner, [], "test",
            record["candidate_id"], dataset, Path(record["memory"]))
        tests[dataset] = {"accuracy": result["accuracy"], "examples": result["total"]}
    final.update(status="complete", winner=winner, confirmation_scores=confirmations,
                 confirmation_per_dataset=per_dataset, test_per_dataset=tests,
                 aggregation="unweighted_mean_across_datasets",
                 test_accuracy=sum(r["accuracy"] for r in tests.values()) / len(datasets),
                 test_examples=sum(r["examples"] for r in tests.values()))
    save_json(marker, final)
    event(root, "finalization_completed", **final)
