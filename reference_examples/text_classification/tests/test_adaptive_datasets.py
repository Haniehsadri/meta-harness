"""Dataset isolation, aggregation, configuration, and partial-resume regressions."""
import argparse
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml
import adaptive_evolution as e


def config():
    return {"datasets": ["LawBench", "USPTO"],
            "dataset": {"num_train": 4, "num_val": 10, "num_test": 4,
                        "overrides": {"LawBench": {"num_val": 6}, "USPTO": {"num_val": 8}}},
            "inner_loop": {"mode": "offline", "num_epochs": 1},
            "benchmark": {"seeds": [42]}, "models": [{"model": "fake"}],
            "memory_systems": {"baselines": ["no_memory", "fewshot_all"]}}


def arguments():
    return argparse.Namespace(task_selector="adaptive-redundancy", model="fake",
        run_name="multi", iterations=1, fresh=False, skip_baseline=False, test=False,
        selection_fraction=0.5, redundancy_alpha=1.0, selection_exploration=0.1,
        redundancy_min_overlap=5, redundancy_shrinkage=5.0, inclusion_draws=100,
        selection_estimator="hajek", evaluation_timeout=30, propose_timeout=30)


class DatasetTests(unittest.TestCase):
    def test_registry_all_tasks_and_official_split_validation(self):
        from runpy import run_path
        registry = run_path(str(Path(e.__file__).parent / "data/constants.py"))
        for dataset in registry["ALL_TASKS"]:
            cfg = config()
            cfg["datasets"] = [dataset]
            e.validate_settings(arguments(), cfg)
        for datasets in ([], ["LawBench", "LawBench"], ["unknown"], "LawBench"):
            cfg = config()
            cfg["datasets"] = datasets
            with self.assertRaises(ValueError):
                e.validate_settings(arguments(), cfg)
        cfg = config()
        cfg["dataset"]["overrides"]["USPTO"]["num_val"] = 31
        with self.assertRaisesRegex(ValueError, "official splits"):
            e.validate_settings(arguments(), cfg)
        cfg["datasets"] = ["FiNER"]
        cfg["dataset"]["overrides"]["FiNER"] = {"num_val": 100}
        e.validate_settings(arguments(), cfg)  # No old S2D-only cap of 50.

    def test_history_does_not_mix_same_source_and_index_between_datasets(self):
        state = {"records": [
            {"dataset": "LawBench", "source_hash": "same", "status": "complete", "outcomes": {"0": 1}},
            {"dataset": "USPTO", "source_hash": "same", "status": "complete", "outcomes": {"0": 0}},
        ]}
        self.assertEqual(e.candidate_history(state, "LawBench"), {"same": {0: 1}})
        self.assertEqual(e.candidate_history(state, "USPTO"), {"same": {0: 0}})

    def test_macro_average_and_incomplete_candidate_exclusion(self):
        state = {"manifest": {"config": config()}, "records": []}
        for candidate, scores in (("a", [1, 0]), ("b", [0.6, 0.6])):
            for dataset, score in zip(config()["datasets"], scores):
                state["records"].append({"candidate_id": candidate, "name": candidate,
                    "dataset": dataset, "status": "complete", "score": {
                    "subset_accuracy": score, "estimated_accuracy": score,
                    "approximate_se": 0.1, "measured_full": False}})
        state["records"].append({**state["records"][0], "candidate_id": "partial"})
        ranked = sorted(e.aggregate_records(state), key=lambda r: -r["score"]["estimated_accuracy"])
        self.assertEqual([r["name"] for r in ranked], ["b", "a"])
        self.assertAlmostEqual(ranked[0]["score"]["estimated_accuracy"], 0.6)

    def test_partial_multidataset_candidate_resumes_and_finalizes_with_saved_memories(self):
        cfg, args = config(), arguments()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "text_classification"
            (project / "agents").mkdir(parents=True)
            for name in cfg["memory_systems"]["baselines"]:
                (project / "agents" / f"{name}.py").write_text(f"# {name}")
            (project / "config.yaml").write_text(yaml.safe_dump(cfg))
            calls, proposals = [], []
            failed = [False]

            def harness():
                h = types.SimpleNamespace(EVOLVE_DIR=project, CONFIG_PATH=project / "config.yaml", _interrupted=False)
                def render(iteration, count):
                    self.assertEqual(count, 2)
                    return "test"
                h.render_task_prompt = render
                def propose(prompt, iteration, timeout):
                    proposals.append(iteration)
                    (h.EVOLVE_DIR / "agents/candidate.py").write_text("# candidate")
                    e.save_json(h.PROPOSAL_WORKSPACE / "context/pending_eval.json",
                                {"iteration": iteration, "candidates": [{"name": "candidate"}]})
                    return True
                h.propose_claude = propose
                return h

            def evaluator(root, code, config, options, name, indices, phase, cid,
                          load_memory=None, dataset=None):
                calls.append((name, dataset, phase, tuple(indices), load_memory))
                if phase == "search" and dataset == "USPTO" and not failed[0]:
                    failed[0] = True
                    raise RuntimeError("interrupt second dataset")
                if phase in ("confirmation", "test"):
                    self.assertIsNotNone(load_memory)
                    self.assertIn(dataset, str(load_memory))
                memory = root / f"{dataset}-{name}-memory.json"
                if load_memory is None:
                    memory.write_text("{}")
                accuracy = 0.25 if dataset == "LawBench" else 0.75
                count = e.split_sizes(config, dataset)["num_test"] if phase == "test" else len(indices)
                return {"accuracy": accuracy, "total": count, "selected_val_indices": indices,
                        "per_sample": [{"example_index": i, "was_correct": bool(i % 2)} for i in indices]}, memory

            with patch.object(e, "evaluate", side_effect=evaluator), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "interrupt second"):
                    e.run_adaptive_evolution(args, harness())
                root = project / "logs/multi"
                plans = {p: p.read_bytes() for p in (root / "selections").rglob("*.json")}
                e.run_adaptive_evolution(args, harness())
                self.assertEqual(proposals, [1])
                self.assertTrue(all(p.read_bytes() == value for p, value in plans.items()))
                self.assertEqual(sum(n == "candidate" and d == "LawBench" and phase == "search"
                                     for n, d, phase, _, _ in calls), 1)
                state = e.read_json(root / "state.json")
                self.assertEqual(state["completed_iteration"], 1)
                self.assertEqual(len(state["records"]), 6)
                frontier = e.read_json(root / "frontier_val.json")
                self.assertIn("LawBench", frontier)
                self.assertIn("USPTO", frontier)
                args.test = True
                e.run_adaptive_evolution(args, harness())
                final = e.read_json(root / "finalized.json")
                self.assertEqual(final["status"], "complete")
                self.assertAlmostEqual(final["test_accuracy"], 0.5)
                self.assertEqual(final["test_examples"], 8)
                self.assertEqual(set(final["test_per_dataset"]), {"LawBench", "USPTO"})
                call_count = len(calls)
                e.run_adaptive_evolution(args, harness())
                self.assertEqual(len(calls), call_count)
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["stage"] == "proposal_manifest_recovered" for row in events))
            for row in events:
                if row["stage"] in ("selection_started", "selection_frozen", "history_updated"):
                    self.assertIn(row["dataset"], cfg["datasets"])
            self.assertFalse((root / "running.lock").exists())
