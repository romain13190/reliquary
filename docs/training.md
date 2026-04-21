# GRPO Training in Reliquary v2.1

The validator runs one GRPO step per sealed window. This doc describes
the loss formulation, the reference model, and the hyperparameters.

## Loss

For each group of M=8 rollouts on a prompt:

  advantage_i = (reward_i − mean(rewards)) / std(rewards)

For each completion token t in each rollout:

  ratio_t = exp(log π_new(t) − log π_old(t))        # π_old from miner GRAIL commit
  ppo_t = −min(ratio_t × adv, clip(ratio, 1±ε) × adv)
  kl_t ≈ exp(ref − new) − 1 − (ref − new)            # Schulman k3 estimator

Total loss = mean_token( ppo + β × kl )

## Reference model

A frozen deep-copy of the starting checkpoint. Used only for the KL
term. Never updated — keeps π_new anchored to the base model across
many training steps.

## Hyperparameters

See `reliquary/constants.py`:

- LEARNING_RATE = 5e-7
- PPO_CLIP_EPSILON = 0.2
- KL_BETA = 0.04
- GRAD_CLIP_NORM = 1.0
- LR_WARMUP_WINDOWS = 10
- LR_COSINE_MAX_WINDOWS = 10000

## Known gaps (v2.1)

- Optimiser state + scheduler step count are not persisted across
  validator restarts. Restart means a fresh AdamW momentum and step=0
  for the scheduler → a few windows of instability until warmup
  completes again. Follow-up PR.
- Reference model is the boot-time checkpoint. Swapping it mid-run
  requires a restart.
