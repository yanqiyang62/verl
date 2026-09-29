# Qwen3.8-27B session DAPO

This single-B300 launcher adapts DAPO to the existing multi-turn session recipe.
It uses the same SFT model, recorded tool environment, session judge and LoRA
settings as `tasks/q38_27B_session.sh`.

```bash
# Export the existing judge credential in this shell if it is not already set.
export ARK_API_KEY='your-existing-key' # alternatively JUDGE_API_KEY
bash tasks/q38_27B_session_dapo.sh

# Command preview only; no Ray, GPU work or judge requests:
DRY_RUN=1 bash tasks/q38_27B_session_dapo.sh
```

The new task reads credentials from the environment; it does not copy credentials
embedded in another task script. `JUDGE_API_KEY` takes precedence over `ARK_API_KEY`.

| Setting | Default |
| --- | --- |
| Training session batch / rollouts per session | 2 / 2 |
| Effective accepted trajectories per update | 4 |
| PPO mini batch / micro batch per GPU | 2 / 1 |
| LoRA rank / alpha | 32 / 64 |
| Learning rate | 3e-6 |
| Advantage | GRPO with within-group standard-deviation normalization |
| Clipping lower / upper epsilon | 0.2 / 0.28 |
| Policy loss / aggregation | vanilla clipped PPO / token-mean |
| KL loss / KL in reward / entropy coefficient | disabled / disabled / 0 |
| Equal-score group filtering | enabled, final session `score` |
| Sampling limit | initial batch + 16 refills; 900-second sampler deadline |
| vLLM async scheduling | disabled |
| Checkpoint cadence / epochs | 20 optimizer steps / 1 |
| SwanLab project | DAPO-Qwen3.8 |
| Experiment name | Session-DAPO with Beijing-time timestamp |

Only complete, scorable groups are checked for equal scores. Every trajectory in
an equal-score group is removed, including all-zero, all-one and equal intermediate
scores. Unequal-score groups are retained while replacement sessions are generated.
Validation bypasses equal-score filtering. Missing/non-finite scores are errors;
unscorable trajectories are never converted into training examples with zero reward.

SwanLab receives `training/filter_groups/equal_reward_groups` (group count),
`training/session/unscorable_groups`, `training/session/refill_rounds` and
`training/session/group_coverage` (accepted / accepted plus rejected groups).
The console also shows the reward values of discarded equal-score groups.
More rejections mean more generation and judge calls per optimizer step.
The sampler fails with explicit counts when it cannot assemble a batch within
its limits; it does not silently train an incomplete batch.

This is a session adaptation: the existing judge reward and truncation caps remain
in use. It does not add the math DAPO recipe's soft overlong reward shaping.
Original GRPO runs keep their current KL settings and retain equal-score groups.

The companion recipe at `$RECIPE_ROOT` (default
`/shared/users/yangyq/data/recipe_v3_session_agentloop`) must include the updated
`tf_rl/configuration.py` and `tf_rl/sampler.py`. The former connects
`algorithm.filter_groups` to the custom sampler; the latter performs filtering
and bounded refill. Copying only the shell launcher to an older recipe is insufficient.

Overrides:

```bash
# At most 32 replacement rounds (33 attempts including the initial batch).
DAPO_MAX_NUM_GEN_BATCHES=33 bash tasks/q38_27B_session_dapo.sh

# Save every 10 optimizer steps.
SAVE_FREQ=10 bash tasks/q38_27B_session_dapo.sh
```

Positive `algorithm.filter_groups.max_num_gen_batches` sets the attempt cap;
non-positive values retain the recipe's finite default. The recipe's 900-second
deadline still applies. Recipe-owned context limits and rollout count are shared
with the GRPO launcher. Shell batch-size overrides and other trailing Hydra
overrides work as in the original session launcher.
