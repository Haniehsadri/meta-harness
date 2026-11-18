---
name: meta-harness
description: Propose one bounded iteration of memory-system evolution.
---

# Meta-Harness proposer

Deliver exactly THREE new candidate memory systems and pending_eval.json for this iteration. The outer loop evaluates them. Work in this session; do not delegate. Prefer a small, correct mechanism change to an elaborate unfinished design. These instructions also apply when Claude Code is connected to a local model.

## Execution budget and completion

- Target at most 30 tool calls in total: roughly 6 for inspection, then implementation and checks. This is a planning target, not an enforced limit.
- Spend at most one quarter of the session inspecting history. Start implementing immediately afterward. If the context already contains the needed information, do not read it again.
- Use the compact history in the task prompt first. Read at most one recent report and, only if necessary, at most three failure/prediction records. Never print complete log.jsonl, memory.json, prompts, checkpoints, or all reports. They can contain enormous repeated text.
- Do not search the whole repository, inspect unrelated runs, research external methods, run benchmarks, call a real model to test prototypes, install packages, or make parameter sweeps.
- Produce one implementation per hypothesis. Do one small offline correctness check per candidate instead of trying 2-3 algorithm variants. Repair a failing check with a targeted edit; simplify the mechanism if needed.
- As soon as a candidate passes its check, add it to the pending_eval.json manifest. Update the manifest as the other candidates pass. Finish with exactly three checked entries and exit promptly; do not continue exploring after submission.
- Do not claim improvement before the outer evaluator measures it. Never stop because the frontier seems optimal.

## Focused workflow

1. Use the exact paths in the task prompt. Read config.yaml, memory_system.py, agents/fewshot_memory.py, and ONE useful frontier implementation if needed. Missing optional history is normal; do not repeatedly search for it.
2. Write three short falsifiable hypotheses covering different mechanisms. Include exploitation of a successful mechanism and exploration of a different mechanism. Record the hypothesis and base_system in the manifest; keep explanations brief.
3. Create each candidate as a new agents/<snake_case_name>.py file. Reuse working imports, answer extraction, serialization, and prompt boundaries. Change only the mechanism needed for the hypothesis.
4. Run the bounded correctness check below. Write/update the manifest immediately after a pass.
5. Finish with CANDIDATES: <name1>, <name2>, <name3>.

## Candidate constraints

- Exactly three candidates; do not submit three numeric variants of the same algorithm. A change to retrieval logic, memory representation, prompt organization, or learning logic is appropriate. Combining proven mechanisms is allowed.
- General-purpose code only: no hardcoded dataset names, language-specific legal/medical knowledge, label lists, or validation-answer lookup tables. Learn any label-dependent information from the training examples.
- Only create NEW agent modules and a small temporary check script. Do not edit baselines, existing agents, evaluator, config, data, dependencies, or past results. No registration edits are needed.
- Never access held-out test data/results. Validation history may guide the hypothesis; never embed its examples or answers in candidate code.
- Favor deterministic local computation and one worker call per predict(). Do not add per-example model calls in training, repeated summaries of all labels on every update, voting, or unbounded refinement loops in this budgeted run. These choices change the search space and must be documented when comparing experiments.
- Respect the existing character cap and preserve the complete query/instructions. A request for a short response does not enforce a generation-token limit.

## Interface and known pitfalls

Extend MemorySystem from ..memory_system. Import LLMCallable from ..llm and extract_json_field from ..memory_system. Constructor must work with only llm. Use self.call_llm(prompt), not the client directly. predict(input) returns (answer_string, metadata_dict), using extract_json_field(response, "final_answer").

- Cold-start predict() must work before learning.
- learn_from_batch receives input, prediction, ground_truth, was_correct; metadata and raw_question are OPTIONAL. Ground truth is already visible in offline mode; prediction may equal ground_truth. Do not infer actual model errors from these synthetic offline records.
- The training loop calls learn_from_batch repeatedly, with batches of size 1 or larger. Retain prior examples. Store index = len(self.examples) BEFORE appending. Never use start_idx + len(self.examples): the existing length already includes earlier batches.
- Use hashlib.sha256, NOT hashlib.sha_256. Preserve deterministic per-input seeds and avoid global random.seed().
- Validation calls predict() concurrently on the SAME instance (up to 32 threads). Keep learned memory read-only during prediction and avoid shared mutable per-query buffers.
- get_state() returns a JSON string. set_state() must rebuild all retrieval indexes/caches needed by predict(), so a restored agent works without retraining. Serialize sorted lists instead of sets.
- Avoid broad exception handling that silently hides programming errors.

## Required small correctness prototype (no real model calls)

Use the active interpreter and package root given in the task prompt. Do not repeatedly guess uv project paths or package imports. Import candidate classes from text_classification.agents.<name>; modules use relative imports and cannot be executed directly.

Create one temporary Python script outside agents/ and reuse it for all three candidates. Instantiate each class with this stub:

    def fake_llm(prompt):
        return '{"reasoning":"smoke check","final_answer":"label_a"}'

Use six tiny synthetic training records across two labels. Each record contains input, prediction, ground_truth, was_correct; omit metadata/raw_question on some records. Check:

1. Cold-start predict().
2. Train on TWO consecutive batches (three records each), then predict(). This catches indexes that double-count history.
3. Repeat on a fresh instance with six singleton batches, then predict().
4. Parse get_state() with json.loads; restore into a fresh instance using set_state(); predict again. Check predictions have a string answer and dictionary metadata.
5. Use ThreadPoolExecutor(max_workers=2) for two predictions on that restored instance; verify learned state is unchanged.

These checks establish basic runtime compatibility, not accuracy. Keep the script small; delete only your own temporary script afterward. If a check fails, inspect its short traceback and repair the NEW candidate, not the evaluator.

## Submission schema

Write valid JSON at the EXACT pending_eval path from the task prompt:

```json
{
  "iteration": 1,
  "candidates": [
    {
      "name": "new_mechanism",
      "file": "agents/new_mechanism.py",
      "hypothesis": "A short measurable claim relative to the base",
      "axis": "exploitation",
      "base_system": "fewshot_all",
      "components": ["mechanism_tag"]
    }
  ]
}
```

Use the actual iteration number. The final candidates array must contain exactly THREE entries with unique new names. Incomplete manifests are not a completed proposal. Do not run evaluation yourself; exit after submission.
