# Hyperparameter selection procedure

The hyperparameters reported in the paper's tables are the outcome of a
validation-guided iterative search rather than a pre-committed grid or random
sweep. This procedure is applied **identically to TraXion and to every retuned
baseline** so that comparisons stay balanced.

## Protocol

For each `(method, dataset, task)` cell:

1. **Initialize** from the published defaults. If a baseline ships no defaults
   for the cohort in question, initialize from sensible literature priors.

2. **Iterate**. Run successive small batches of configurations (typically 4–8
   runs each) that vary one or two hyperparameters at a time. Each batch is
   scored by the same validation criterion used for early stopping:
   - BCE validation loss for anomaly and ICU-mortality fine-tunes,
   - total fine-tune loss for next-POI / next-visit / social-link,
   - EMA-smoothed total validation loss for pre-training.

3. **Choose** the next batch's settings from the run-by-run trajectory of the
   preceding batch — make an informed decision about what to vary next,
   *instead of random or grid search*.

4. **Stop** when validation no longer improves. The reported configuration is
   the highest-validation point reached under this procedure. **The test set is
   never consulted during selection.**

5. **Log** the best hyperparameters and validation/test metrics together with
   the corresponding W&B run id (when applicable), at the root of the code
   repo, so that the chosen configuration is recoverable.

## Notes for the operator (or LLM agent) running the loop

- Short training sessions tend to favor higher learning rates, which is often
  not a long-term win — take that into consideration when choosing the
  learning rate.
- You may change anything reasonable: `model_dim`, `num_layers`, batch size,
  weight decay, learning-rate schedule, and so on.
- You may try a set of changes concurrently if the changes are independent;
  document the change and its observed effect.

## Agent-orchestrated variant

The selection loop is orchestrated by an LLM coding agent
([Anthropic Claude Code](https://www.anthropic.com/claude-code), in the spirit of
[Karpathy 2025](https://x.com/karpathy/status/1947667575656698076)): the agent
reads the prior batch's logs and validation curves, proposes the next batch's
hyperparameter settings under the criteria above, and launches the runs via
the same training scripts a human would use. This replaces the bookkeeping of
a sequential coordinate-descent search with an automated proposer but does not
change the search space, the validation criterion, or the protocol's
outcome — every reported number comes from the unmodified training and
evaluation code in this repository.

The agent operates under the following standing instructions, used identically
for TraXion and every retuned baseline:

> 1. Tune hyperparameters with the validation set.
> 2. Try altering something from the default; document the change and its
>    effect (you may try a set of changes concurrently); and make an informed
>    decision about what to try next, *instead of random or grid search*.
> 3. After tuning, train and test the model.
> 4. Report results and log the best hyperparameters and results to (i) the
>    root of the code repo and (ii) the run record (including the W&B run id
>    where applicable).
>
> **Notes.** Short sessions tend to favor higher learning rates, which may not
> be a long-term win — take that into consideration when choosing the learning
> rate. You may change anything reasonable, such as `model_dim`, `num_layers`,
> and so on.
