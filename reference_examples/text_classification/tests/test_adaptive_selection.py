"""Regression tests for aligned evidence, progressive sampling, and S2D runs."""
import argparse
import io
import json
import random
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

import adaptive_evolution as evolution
from adaptive_selection import draw, estimate, make_plan, relationships
from text_classification.inner_loop import select_validation_examples


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.history = {str(i): {0: x, 1: x, 2: y, 3: 1}
                        for i, (x, y) in enumerate([(0, 0), (0, 1), (1, 0),
                                                    (1, 1), (0, 0), (1, 1)])}
        self.settings = {"policy": "adaptive-redundancy", "fraction": 0.5,
                         "alpha": 2.0, "exploration": 0.1, "min_overlap": 3,
                         "shrinkage": 5.0, "probability_draws": 500, "seed": 42}

    def test_redundancy_requires_shared_candidates_and_nonconstant_vectors(self):
        pairs, coverage = relationships(self.history, 4, 3, 5)
        self.assertAlmostEqual(pairs[(0, 1)], 6 / 11)
        self.assertNotIn((0, 3), pairs)
        self.assertGreater(coverage["positive_pairs"], 0)
        sparse = {"a": {0: 0}, "b": {1: 1}, "c": {0: 1, 1: 0}}
        self.assertEqual(relationships(sparse, 2, 3, 5)[0], {})

    def test_progressive_probabilities_penalize_duplicate_evidence(self):
        selected, steps = draw({0: 1, 1: 1, 2: 1}, {(0, 1): 0.8}, 2,
                               2, 0.1, random.Random(1), trace=True)
        self.assertEqual(selected[0], 0)
        self.assertLess(steps[1]["probabilities"][1], steps[1]["probabilities"][2])
        self.assertTrue(steps[1]["penalty_changes_probabilities"])
        for step in steps:
            self.assertAlmostEqual(sum(step["probabilities"].values()), 1)
            self.assertTrue(all(p > 0 for p in step["probabilities"].values()))

    def test_zero_alpha_control_and_saved_json_plan(self):
        settings = {**self.settings, "alpha": 0}
        plan = make_plan(self.history, 4, "candidate", settings)
        self.assertFalse(plan["selection_changed_vs_alpha_zero"])
        self.assertEqual(plan["penalized_steps"], 0)
        self.assertEqual(len(plan["selected"]), 2)
        recovered = json.loads(json.dumps(plan))
        result = estimate({i: 1.0 for i in plan["selected"]}, recovered, "hajek")
        self.assertAlmostEqual(result["estimated_accuracy"], 1)
        with self.assertRaises(ValueError):
            estimate({}, recovered, "hajek")

    def test_inclusion_probabilities_match_independent_draws(self):
        plan = make_plan(self.history, 4, "candidate", self.settings)
        pairs, _ = relationships(self.history, 4, 3, 5)
        counts = dict.fromkeys(range(4), 0)
        for repetition in range(2000):
            selected, _ = draw(plan["priorities"], pairs, 2, 2.0, 0.1,
                               random.Random(f"audit:{repetition}"))
            for index in selected:
                counts[index] += 1
        for index, count in counts.items():
            self.assertAlmostEqual(plan["inclusion_probabilities"][index], count / 2000,
                                   delta=0.08)
        self.assertAlmostEqual(sum(plan["inclusion_probabilities"].values()), 2, delta=0.01)

    def test_no_redundancy_below_minimum_overlap(self):
        plan = make_plan(self.history, 4, "candidate", {**self.settings, "min_overlap": 9})
        self.assertEqual(plan["coverage"]["positive_pairs"], 0)
        self.assertFalse(plan["selection_changed_vs_alpha_zero"])

    def test_duplicate_source_versions_are_not_independent_harnesses(self):
        state = {"records": [
            {"source_hash": "same", "status": "complete", "outcomes": {"0": 0}},
            {"source_hash": "same", "status": "complete", "outcomes": {"0": 1, "1": 1}},
        ]}
        self.assertEqual(evolution.candidate_history(state), {"same": {0: 0.5, 1: 1}})

    def test_original_example_indices_survive_selection(self):
        pool = [{"input": str(i)} for i in range(6)]
        chosen, indices = select_validation_examples(pool, [4, 1])
        self.assertEqual(indices, [4, 1])
        self.assertEqual([r["input"] for r in chosen], ["4", "1"])
        for invalid in ([], [1, 1], [6], [-1], [True]):
            with self.assertRaises(ValueError):
                select_validation_examples(pool, invalid)


class SinglePassEvaluatorTests(unittest.TestCase):
    def test_real_s2d_loader_evaluates_only_requested_examples_once(self):
        class FakeLLM:
            def __init__(self, **kwargs):
                self.total_calls = self.total_input_tokens = self.total_output_tokens = 0
                instances.append(self)

            def __call__(self, prompt):
                self.total_calls += 1
                return '{"final_answer": "allergy"}'

            def get_usage(self):
                return {"calls": self.total_calls, "input_tokens": 0,
                        "output_tokens": 0, "total_tokens": 0}

        instances = []
        with tempfile.TemporaryDirectory() as directory:
            output, memory, log = [Path(directory) / name for name in
                                   ("val.json", "memory.json", "trace.jsonl")]
            argv = ["inner_loop", "--dataset", "Symptom2Disease", "--memory", "no_memory",
                    "--model", "fake", "--num-train", "4", "--num-val", "50",
                    "--num-test", "212", "--mode", "offline", "--num-epochs", "1",
                    "--val-indices-json", "[3, 17]", "--single-validation-pass",
                    "--evaluation-id", "test-evaluation", "--val-output", str(output),
                    "--save-memory", str(memory), "--log", str(log)]
            with patch.object(sys, "argv", argv), patch(
                    "text_classification.llm.LLM", FakeLLM), redirect_stdout(io.StringIO()):
                # Execute the real command-line evaluator with the actual S2D data.
                code = Path(__file__).parents[1] / "inner_loop.py"
                namespace = {"__name__": "__main__", "__package__": "text_classification",
                             "__file__": str(code)}
                exec(compile(code.read_text(encoding="utf-8"), str(code), "exec"), namespace)
            result = evolution.read_json(output)
            self.assertEqual(instances[0].total_calls, 2)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["selected_val_indices"], [3, 17])
            self.assertEqual(result["validation_pool_size"], 50)
            self.assertEqual(result["evaluation_id"], "test-evaluation")
            self.assertEqual(result["memory_sha256"], evolution.file_hash(memory))
            self.assertTrue(all(row["example_id"] for row in result["per_sample"]))
            events = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(sum(row["type"] == "validation_prediction" for row in events), 2)
            self.assertFalse(any(row["type"] == "val_epoch" for row in events))


class EvolutionIntegrationTests(unittest.TestCase):
    def test_propose_select_evaluate_resume_and_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "text_classification"
            (project / "agents").mkdir(parents=True)
            for name in ("no_memory", "fewshot_all"):
                (project / "agents" / f"{name}.py").write_text(f"# {name}\n")
            cfg = {"datasets": ["Symptom2Disease"],
                   "dataset": {"num_train": 200, "num_val": 50, "num_test": 212},
                   "inner_loop": {"mode": "offline", "num_epochs": 1},
                   "benchmark": {"seeds": [42]}, "models": [{"model": "fake"}],
                   "memory_systems": {"baselines": ["no_memory", "fewshot_all"]}}
            (project / "config.yaml").write_text(yaml.safe_dump(cfg))
            args = argparse.Namespace(task_selector="adaptive-redundancy", model="fake",
                run_name="smoke", iterations=1, fresh=False, skip_baseline=False, test=False,
                selection_fraction=0.2, redundancy_alpha=1.0, selection_exploration=0.1,
                redundancy_min_overlap=5, redundancy_shrinkage=5.0, inclusion_draws=100,
                selection_estimator="hajek", evaluation_timeout=30, propose_timeout=30)
            calls = []
            fail_once = [True]

            def harness():
                h = types.SimpleNamespace(EVOLVE_DIR=project, CONFIG_PATH=project / "config.yaml",
                                          _interrupted=False)
                h.render_task_prompt = lambda iteration, datasets: "test prompt"

                def propose(prompt, iteration, timeout):
                    name = f"candidate_{iteration}"
                    (h.EVOLVE_DIR / "agents" / f"{name}.py").write_text(f"# {name}\n")
                    evolution.save_json(h.PENDING_EVAL, {"candidates": [{"name": name}]})
                    return True
                h.propose_claude = propose
                return h

            def evaluate(root, code, config, options, name, indices, phase, candidate_id,
                         load_memory=None):
                calls.append((name, indices, phase))
                if phase == "search" and fail_once[0]:
                    fail_once[0] = False
                    raise RuntimeError("simulated interrupted evaluation")
                memory = root / f"{name}-memory.json"
                memory.write_text("{}")
                return {"accuracy": 0.5, "total": len(indices),
                        "selected_val_indices": indices,
                        "per_sample": [{"example_index": i, "was_correct": bool(i % 2)}
                                       for i in indices]}, memory

            with patch.object(evolution, "evaluate", side_effect=evaluate), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                    evolution.run_adaptive_evolution(args, harness())
                root = project / "logs" / "smoke"
                plan_path = root / "selections" / "0001-candidate_1.json"
                original_plan = plan_path.read_bytes()
                evolution.run_adaptive_evolution(args, harness())
                self.assertEqual(plan_path.read_bytes(), original_plan)
                state = evolution.read_json(root / "state.json")
                self.assertEqual(state["completed_iteration"], 1)
                self.assertEqual(len(state["records"]), 3)
                self.assertEqual([len(i) for _, i, p in calls if p == "baseline"], [50, 50])
                self.assertTrue(all(len(i) == 10 for _, i, p in calls if p == "search"))
                args.test = True
                evolution.run_adaptive_evolution(args, harness())
                self.assertEqual(evolution.read_json(root / "finalized.json")["status"], "complete")
                args.test = False
                with self.assertRaisesRegex(ValueError, "cannot evolve"):
                    evolution.run_adaptive_evolution(args, harness())
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertTrue(any(e["stage"] == "selection_frozen" for e in events))
            self.assertTrue(any(e["stage"] == "finalization_completed" for e in events))
            self.assertFalse((root / "running.lock").exists())


class CommandLinePipelineTests(unittest.TestCase):
    def test_cli_runs_real_evaluator_subprocesses_with_fake_models(self):
        self._run_cli(["Symptom2Disease"])

    def test_cli_runs_single_lawbench(self):
        self._run_cli(["LawBench"])

    def test_cli_runs_multiple_datasets_with_separate_memories(self):
        self._run_cli(["Symptom2Disease", "LawBench", "USPTO"])

    def _run_cli(self, datasets):
        source = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "text_classification"
            shutil.copytree(source, project, ignore=shutil.ignore_patterns(
                "logs", "results", ".venv", "__pycache__", "tests"))
            config_path = project / "config.yaml"
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
            cfg["datasets"] = datasets
            cfg["models"] = [{"model": "fake"}]
            cfg["benchmark"]["seeds"] = [42]
            cfg["inner_loop"]["mode"] = "offline"
            cfg["inner_loop"]["num_epochs"] = 1
            pool_sizes = {d: 10 if d == "Symptom2Disease" else 6 if d == "LawBench" else 8 for d in datasets}
            for dataset in datasets:
                cfg["dataset"]["overrides"][dataset] = {
                    "num_train": 4, "num_val": pool_sizes[dataset], "num_test": 4}
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            (project / "llm.py").write_text(
                "from typing import Callable\n"
                "LLMCallable = Callable\n"
                "class LLM:\n"
                "    def __init__(self, **kwargs):\n"
                "        self.total_calls = 0\n"
                "        self.total_input_tokens = self.total_output_tokens = 0\n"
                "    def __call__(self, prompt):\n"
                "        self.total_calls += 1\n"
                "        return __import__('json').dumps(dict(final_answer='allergy'))\n"
                "    def get_usage(self):\n"
                "        return dict(calls=self.total_calls, input_tokens=0, "
                "output_tokens=0, total_tokens=0)\n", encoding="utf-8")
            launcher = Path(directory) / "launch.py"
            launcher.write_text(
                "import json\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
                "import meta_harness as harness\n"
                "def propose(**kwargs):\n"
                "    code = Path(kwargs['cwd'])\n"
                "    src = (code / 'agents/no_memory.py').read_text()\n"
                "    (code / 'agents/pilot_candidate.py').write_text('# pilot' + chr(10) + src)\n"
                "    harness.PENDING_EVAL.write_text(json.dumps({'candidates': "
                "[{'name': 'pilot_candidate', 'hypothesis': 'test plumbing'}]}))\n"
                "    return SimpleNamespace(exit_code=0, show=lambda: None)\n"
                "harness.claude_wrapper.run = propose\n"
                "harness.main()\n", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(project)
            env["PYTHONUNBUFFERED"] = "1"
            command = [sys.executable, str(launcher), "--model", "fake", "--run-name", "cli",
                       "--task-selector", "adaptive-redundancy", "--inclusion-draws", "100",
                       "--selection-fraction", "0.2"]
            process = subprocess.run(command + ["--iterations", "1"], env=env, cwd=project,
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            root = project / "logs" / "cli"
            results = list((root / "evaluations" / "search").rglob("val.json"))
            self.assertEqual(len(results), len(datasets))
            identities = []
            for path in results:
                result = evolution.read_json(path)
                dataset = result["dataset"]
                count = max(1, int(pool_sizes[dataset] * 0.2))
                self.assertEqual(result["total"], count)
                self.assertEqual(result["llm_calls"], count)
                self.assertEqual(result["validation_pool_size"], pool_sizes[dataset])
                self.assertEqual(len(result["per_sample"]), count)
                identities.append(result["evaluation_id"])
                self.assertTrue((path.parent / "memory.json").exists())
            self.assertEqual(len(set(identities)), len(datasets))
            records = evolution.read_json(root / "state.json")["records"]
            self.assertEqual(len(records), 3 * len(datasets))
            self.assertTrue((root / "iteration-0001.json").exists())
            process = subprocess.run(command + ["--test"], env=env, cwd=project,
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            final = evolution.read_json(root / "finalized.json")
            self.assertEqual(final["status"], "complete")
            self.assertEqual(final["test_examples"], 4 * len(datasets))
            self.assertEqual(set(final["test_per_dataset"]), set(datasets))


if __name__ == "__main__":
    unittest.main()
