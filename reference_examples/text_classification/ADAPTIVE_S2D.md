# Adaptive classification task selection

This optional experiment adds the selector from `ADAPTIVE_TASK_SELECTION_README.md`
to the existing text-classification Meta-Harness entry point. It uses one fixed
worker and evaluates a selected subset of each configured classification validation pool.
The file keeps its original name for existing links; S2D is now just one example.

## Configuration

Edit the existing sections of `config.yaml` before starting a new run:

```yaml
datasets: [Symptom2Disease]

dataset:
  num_train: 200
  num_val: 50
  num_test: 212
  overrides:
    Symptom2Disease: {num_train: 200, num_val: 50, num_test: 212}

inner_loop:
  mode: offline
  num_epochs: 1
  temperature: 0.0
  eval_interval: 0
  batch_size: 1
  seed: 42

models:
  - model: Qwen3.6-27B
    api_base: http://localhost:8000/v1

benchmark:
  seeds: [42]
  concurrency: 1
```

Keep the existing `memory_systems` section. Configure the local Claude proposer
and worker credentials in your shell as in your existing local-model setup.
For local vLLM the worker uses `OPENAI_API_KEY=dummy`. The proposer uses
`ANTHROPIC_BASE_URL=http://localhost:8000`, `ANTHROPIC_AUTH_TOKEN=dummy`,
`METAHARNESS_PROPOSER_MODEL=Qwen3.6-27B`, and the Claude context/output settings.
Do not set `ANTHROPIC_BASE_URL` to a URL ending in `/v1`.

Run from `reference_examples/text_classification`:

```bash
uv sync
export LITELLM_LOCAL_MODEL_COST_MAP=True
uv run python meta_harness.py \
  --model Qwen3.6-27B \
  --run-name s2d-adaptive-pilot \
  --task-selector adaptive-redundancy \
  --selection-fraction 0.2 \
  --redundancy-alpha 1.0 \
  --selection-exploration 0.1 \
  --redundancy-min-overlap 5 \
  --redundancy-shrinkage 5 \
  --inclusion-draws 4000 \
  --selection-estimator hajek \
  --iterations 1
```

This first run is a pipeline check. Each baseline evaluates all 50 validation
examples. Each generated candidate trains on all 200 training examples and
then evaluates 10 validation examples once. Training may itself invoke the
model depending on the candidate's learning method. Sampling and Monte Carlo
probability simulations make no worker calls. The selector updates after each
candidate, including candidates from the same proposer iteration.

## Selecting datasets

Use the same `datasets` list as the original Meta-Harness. A single entry is supported:

```yaml
datasets: [LawBench]
```

Or select several datasets in one run:

```yaml
datasets: [Symptom2Disease, LawBench, USPTO]
dataset:
  num_train: 200
  num_val: 50
  num_test: 100
  overrides:
    Symptom2Disease: {num_train: 200, num_val: 50, num_test: 212}
    LawBench: {num_train: 200, num_val: 50, num_test: 100}
    USPTO: {num_train: 50, num_val: 30, num_test: 100}
```

Only listed datasets run; an override does not enable a dataset. Defaults are merged
with the selected dataset's overrides, just as in `benchmark.py`. Registry-supported
classification tasks are accepted, including the transfer tasks supported by the
existing loader. Data availability/download requirements are unchanged. Official
MCE split limits and positive sizes are checked before the run; validation needs at
least two examples. Larger official pools (e.g. FiNER's 100) are supported.

The adaptive backbone is unchanged: the same priorities, redundancy computation,
exploration, Monte Carlo inclusion probabilities, and estimators run independently
inside each dataset. Example index 0 from LawBench never shares outcomes or redundancy
history with example index 0 from USPTO. Candidate implementations are shared across
datasets, but each gets a separate training memory. Selection fraction is applied
PER dataset (0.2 gives 10/50 for LawBench and 6/30 for USPTO).
The existing rounding rule is unchanged: max(1, int(pool_size * fraction)).

A harness is ranked only when all selected datasets have committed results. Overall
scores use the unweighted mean of dataset accuracies, matching the original average
rather than weighting by pool size. `evolution_summary.jsonl` also reports per-dataset
scores. Aggregate standard errors assume independent dataset sampling and remain
approximate. Datasets and candidates are evaluated sequentially; per-example prediction
concurrency inside the evaluator is unchanged.

For runs other than single-S2D, artifact paths are dataset-qualified:

- `selections/<dataset>/<iteration>-<candidate>.json`
- `evaluations/<phase>/<dataset>/<candidate>/` (attempt logs, results, memory and trace)

Single-S2D keeps its previous paths. Selection/evaluation/history events identify the
dataset, and `run_started` lists validation pool sizes. An interrupted candidate resumes
only the missing dataset evaluations and keeps already-frozen plans.

Finalization with `--test` confirms the same top-three overall shortlist on ALL configured
validation pools, chooses one overall winner by mean full-validation accuracy, then tests
that winner on every selected dataset, reusing each dataset's saved training memory.
`finalized.json` records `confirmation_per_dataset` and `test_per_dataset`; `test_accuracy`
is the macro average and `test_examples` is the sum of test counts. Already-measured full
validation results are reused. This remains a separate finalization command, not an
automatic stage appended to evolution.

Use a NEW run name after this implementation update. The existing implementation-hash
check deliberately rejects resuming old runs with changed evaluator code. Old artifacts
are not deleted; finalize an old run using its matching code/configuration version.

## Defaults and controls

The new mode is enabled by `--task-selector`. Omitting that flag preserves the
original evolution entry point. The three policies are:

- `full`: every candidate evaluates all configured validation examples.
- `task-coevolve`: Task-CoEvolve's individual priorities and weighted sampling,
  without the added uniform exploration mixture or redundancy penalty.
- `adaptive-redundancy`: the proposed progressive sampler, uniform exploration,
  and support-shrunk positive-correlation penalty.

For the matched zero-penalty control use `adaptive-redundancy` with
`--redundancy-alpha 0`. Keep all other selection settings identical and choose
a different run name. Repeated search runs are needed for policy comparisons.

The priorities follow `max(4*p*(1-p), floor) + 0.1/sqrt(n)` with floor 0.5 for
never-solved examples and zero otherwise. Correlation uses only jointly observed
source versions. Repeated measurements of identical candidate source are averaged
within that source version. Unknown outcomes never become failures.

Defaults are fraction 0.2, alpha 1, exploration 0.1, minimum overlap 5, shrinkage 5,
4000 Monte Carlo draws, and the Hajek estimator. These are pilot settings, not
validated hyperparameters. Two baseline source versions cannot satisfy overlap 5.
The penalty is zero for unsupported pairs and constant outcome vectors. A short
run may collect no usable relationships. Inspect the coverage logs before drawing
conclusions about the redundancy mechanism.

## Logs and verification

All experiment artifacts are under `logs/<run-name>/`:

| Artifact | What to inspect |
| --- | --- |
| `events.jsonl` | Timestamped stages, scores, timings, model usage, failures and paths |
| `state.json` | Frozen configuration, completed/active iteration and committed observations |
| `selections/<iteration>-<candidate>.json` | Frozen history hash, weights, redundancy pairs, step probabilities, selected original indices, anchors and inclusion probabilities |
| `evaluations/baseline/<name>/` | Full baseline results, training memory, worker traces and attempt logs |
| `evaluations/search/<name>/` | Candidate subset results and individual correctness with stable example IDs |
| `frontier_val.json` | Current best by measured or estimated full-pool accuracy |
| `evolution_summary.jsonl` | Candidate subset accuracy, estimated full accuracy, approximate error and hypothesis |
| `claude_sessions/` | Proposer interactions |
| `code/text_classification/agents/` | Candidate files generated inside this run's snapshot |

Useful commands:

```bash
tail -f logs/s2d-adaptive-pilot/events.jsonl
cat logs/s2d-adaptive-pilot/evolution_summary.jsonl
ls -lt logs/s2d-adaptive-pilot/code/text_classification/agents/
```

Look for these stages in order:

1. `run_started` records the policy and validation pool.
2. `evaluation_completed` for each baseline reports `examples=50`.
3. `proposal_completed` lists new candidate names.
4. `selection_frozen` reports 10 selected indices, history coverage, probability
   diagnostics, and the full plan path.
5. `evaluation_completed` for each search candidate reports `examples=10`.
6. `history_updated` records measured subset accuracy, estimated pool accuracy,
   and an approximate standard error.
7. `iteration_completed` confirms all proposed candidates have committed results.

The selection plan logs whether the penalty altered probabilities at each step,
and whether the selected subset differs from an alpha-zero draw using common
random numbers. Probabilities are logged for every remaining example at every
step. The full inclusion probabilities are estimated by independent simulations
of the same progressive sampler. Rare items with no Monte Carlo hits use a
0.5/draws floor and are explicitly reported through `zero_mc_hits`.

Hajek is a ratio estimator and has finite-sample bias. Monte Carlo probabilities
add approximation error. The difference option uses pre-evaluation historical
anchors and records both its unclipped and clipped estimate. Reported standard
errors are approximations and can underestimate uncertainty for tiny or unanimous
subsets. They do not include every source of model or probability-estimation noise.

Worker traces include `validation_prediction` records. The subset evaluator skips
full epoch validation, preventing the old duplicate validation execution.
`llm_calls` and token usage include the worker wrapper's accounting, which can
include cached responses. These are not guaranteed billed-dollar costs. Each
attempt has a separate log and wall time. Failed/incomplete evaluations abort
search and contribute no fabricated correctness observations. Partial provider
usage during a failed process may only be recoverable from its logs or server.
Proposer logs retain their existing usage information.

## Resume

Repeat the same command with the same settings. A partially completed iteration
resumes from its saved candidate list, selection plan and frozen history. If the
previous invocation completed, `--iterations 1` adds one further iteration.
Completed evaluations are reused only when their evaluation identity and memory
hash match. Successful results are written atomically before history is updated.

The run snapshots evaluator code, data, baseline source and config. Protected
snapshot files and evaluated candidate hashes are checked. New run configuration
or implementation changes require a new run name. Leave completed artifacts in
place to allow validation and recovery.

Concurrent execution of the same run is rejected using `running.lock`. After a
hard kill, inspect the PID/host recorded in that file and confirm the old process
has exited before manually removing the stale lock. `--fresh` and `--skip-baseline`
are rejected for this opt-in mode to avoid accidentally discarding initialization.

## Final confirmation and test

After candidate selection, repeat the same command and settings with `--test`
instead of `--iterations 1`. The fixed protocol confirms the top three available
harnesses (including eligible baselines) on full validation, reusing their saved
training memories. It selects the best confirmed harness and evaluates that
winner on the configured test split. Full-validation measurements already obtained
are reused. Confirmation and test costs are logged separately.

A `finalized.json` marker blocks further evolution as soon as confirmation starts.
A failed finalization can be resumed with `--test`. Test data exists in this public
repository, so separation is operational, as in the original project.

## Tests

```bash
PYTHONPATH=.. uv run python -m unittest discover -s tests -v
```

Tests cover sampling probabilities, supported/unsupported correlations, missing
observations, identical-source grouping, inclusion-probability simulations, JSON
plan persistence, the real S2D loader with a fake worker, single-LawBench and multi-dataset CLI runs with actual local data and fake models,
dataset-history isolation, macro averaging, official split limits, and interrupted
multi-dataset resume/finalization flows. A real local-Qwen search
must still be run in your GPU container.
