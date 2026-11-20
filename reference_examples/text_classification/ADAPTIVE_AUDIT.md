# Adaptive task selection: implementation audit

## Scope

This audit addresses the repeated file/submission failures in the LawBench runs
and checks the selection formulas against ADAPTIVE_TASK_SELECTION_README.md.
It does not establish improved accuracy or guarantee validity of every generated
program. No sampling, estimator, evaluator, ranking, training, or candidate-count
formula is changed by these infrastructure fixes. One finalization correction retains already-measured full-validation baselines when comparing the confirmed shortlist; it does not add worker calls.

## Observed failures and handling

| Failure | Handling |
|---|---|
| Proposer overwrites/deletes an evaluated agent | Isolated writable workspace; canonical run is outside the write boundary; verify old files; keep content-addressed originals |
| pending_eval saved in a different directory | Working-directory submission plus bounded, validated alternate-path resolution |
| Stale, conflicting, or symlinked manifest | Reject before publication |
| Missing iterNNNN_ prefix in manifest name but present in filename | Normalize only a unique, current-prefix-related source; preserve raw metadata; no source edits |
| Two possible files or arbitrary name/file mismatch | Reject instead of guessing |
| Same source accepted under duplicate normalized names | Reject before publication |
| Noisy shortlist estimates exclude a better measured baseline | Compare confirmed candidates against all already-measured full-validation records; reuse scores without extra calls |
| Missing dependency, import failure, or hanging import | Sandboxed preflight before worker evaluation, with per-check timeout and process-group cleanup |
| Import modifies source | Reject before publication |
| Interrupted source import | Durable prepared manifest and hash-verified archives; atomic non-overwriting import |
| Missing evaluated source with an available verified archive | Restore exact bytes; never regenerate an approximation |

Failures still stop explicitly if a proposal is ambiguous or invalid. Completed
results are preserved. A stopped invocation is not the same as lost history.

## Core equations checked

- Task priorities: max(4p(1-p), unsolved floor) + 0.1/sqrt(n), with floor 0.5 only
  when p=0. This matches the local Task-CoEvolve default weights; environment-tuned
  Task-CoEvolve runs can use different settings.
- Per-example histories use observed outcomes only. Missing observations are not
  failures. Repeated source hashes are grouped; datasets have isolated histories.
- Redundancy: positive Pearson correlation on shared distinct source versions,
  zero below the support threshold or for constant vectors, multiplied by n/(n+tau).
- Progressive weights: w/(1+alpha*maximum supported redundancy to selected items).
- Selection: normalized adjusted weights mixed with uniform exploration, without
  replacement. The alpha=0 control retains the same exploration mixture.
- Inclusion probabilities: simulate complete subsets under frozen history using
  a separate deterministic random stream. These are not per-step probabilities.
- Hajek: sum(y/pi)/sum(1/pi). Full-pool evaluation returns the measured mean exactly.
- Difference estimator: frozen mean anchors plus inclusion-weighted residuals;
  keep the unclipped value as well as the clipped reported score.
- Freeze selection plans before worker execution; commit observed outcomes only
  after complete evidence; use the committed history for the next candidate.

## Independent numerical checks

The test suite includes hand-calculated priority and correlation cases, exact
enumeration of all ordered two-item selections from a four-item population,
comparison of Monte Carlo inclusion probabilities against that exact reference,
known-value estimator calculations, deterministic replay, full-pool behavior,
negative/unsupported correlation cases, and the matched zero-penalty control.
Other regressions exercise CLI evaluation, dataset isolation, resume, finalization,
actual Linux write confinement, and the exact iteration-4 metadata mismatch.

These checks are performed with fake workers in temporary directories. They do
not require another expensive Qwen evolution run.

## Comparison with the local Task-CoEvolve checkout

Task-CoEvolve has candidate-file rescue, per-run archives, and import checks.
It can skip invalid candidates/iterations. The adaptive runner originally stopped
at protocol violations to preserve immutable evidence. A successful Task-CoEvolve
run does not show that its proposer cannot make these mistakes, and its import
checks do not repair a mismatched metadata name automatically. The adaptive fix
preserves integrity while recovering only unambiguous submissions.

## Remaining scientific and operational limits

- A ten-example subset can produce noisy estimates and incorrect rankings.
- Hajek is a finite-sample ratio estimator, not generally exactly unbiased.
  Monte Carlo probabilities and their zero-hit floor add approximation error.
- Reported standard errors are approximate and can be misleading on tiny or
  unanimous subsets; they are not guaranteed confidence intervals.
- Sparse shared history can leave the redundancy penalty inactive for many pairs.
- Positive correlation is a redundancy heuristic, not proof that discarding a
  task improves candidate ranking or final accuracy.
- Offline training currently marks training records correct using their supplied
  ground truth. Error-driven agents need genuine feedback to learn actual errors.
- An import preflight does not test all learning, serialization, prediction, or
  concurrent execution paths. Generated programs can still contain runtime bugs.
- The filesystem boundary protects content and directory-entry mutations, not all
  metadata/process/network operations. Evaluator execution retains its prior trust
  model. Linux kernel support is required; protection fails closed if unavailable.
- Persisted response caches can avoid new worker requests. Pin model/configuration
  when comparing methods, and account for cache behavior in cost measurements.

Keep historical experiments unchanged. Old damaged sources without verified
archives need exact recovery. New infrastructure versions require new run names
under the existing implementation-hash protocol.
