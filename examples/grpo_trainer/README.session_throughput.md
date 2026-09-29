# Session GRPO throughput preset

Run with the same model, judge credentials and recipe as the existing session task:

```bash
bash tasks/q38_27B_session_throughput.sh
```

The preset was selected from local logs for SwanLab run `h70s2qpj`, launched at
2026-09-21 15:13:32 Beijing time. Steps 2-4 averaged 169.9 seconds, of which
137.1 seconds (80.7%) were in generation, tool auditing and refill. They used only
four concurrent trajectories. The first 22 recorded trajectories were 4,686-7,521
tokens including prompt and rollout context. Actor peak allocated memory reached
76.7 GiB; this is process allocator memory, not total GPU usage including vLLM.
The web chart itself was unavailable, and no hardware utilization percentage was
inferred from the actor memory counter.

| Parameter | Existing task | Throughput task |
| --- | --- | --- |
| Session batch | 2 | 4 |
| Rollouts per session | 2 | 4 |
| Maximum vLLM sequences | 4 | 16 |
| Agent workers (same GPU engine) | 1 | 4 |
| PPO mini batch, in sessions | 2 | 4 |
| Actor micro batch, in trajectories | 1 | 2 |
| Log-prob micro batch, in trajectories | 1 | 2 |
| vLLM memory utilization budget | 0.50 | 0.55 |
| Max batched prefill tokens | 8192 | 2048 |
| GDN prefill backend | auto | triton |
| Free rollout cache during training | true | true |
| vLLM async scheduling | false | false |
| Checkpoint interval, trainer steps | 20 | 5 |

More sessions can overlap remote judge waits with other sessions' GPU generation.
Micro batch 2 increases work per actor/log-prob forward pass. The slightly larger
vLLM budget provides KV capacity for the increased sequence count; allocating
more KV cache alone cannot speed up a remote judge. Full 64K trajectory peaks
have not been validated with micro batch 2. Keep gradient checkpointing, eager
execution and the existing FA4 compatibility settings.

For long trajectories, keep the increased generation concurrency while reverting
actor and log-prob micro batches:

```bash
PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
  bash tasks/q38_27B_session_throughput.sh
```

The effective optimizer mini batch increases from four to sixteen trajectories;
each trainer step contains one PPO mini batch. This changes optimization
semantics. LR remains 3e-6. Four rollouts provide more within-session reward
comparisons, while using fewer distinct sessions per step than the earlier 8x2
preset. Compare reward, accepted session
coverage and accepted sessions per second after warmup. The checkpoint cadence
remains 20 **trainer steps**; epoch step counts decrease with the larger batch.
The run name includes `Session-Throughput` and a Beijing-time timestamp.

Judge failures and unsupported tool replay branches still require separate fixes.
This preset leaves the running job alone and takes effect on the next launch.
Shell syntax and Hydra/recipe composition are checked; GPU speedup and peak memory
remain to be measured in an actual run.

## Timeout mitigation after the first throughput run

The run `20260921_075429Z_mOunvi` completed one trainer step (actor peak allocation
100.1 GiB), then failed with `RPC call to execute_model timed out` during rollout.
The dumped scheduler batch contained two prefills and one decode, totaling 8192
tokens. Async scheduling was already disabled. The log contains no OOM and no
new JIT warning immediately preceding the failure; it cannot establish the exact
stalled kernel. The sampler's later 900-second timeout is a downstream symptom.

The checkpoint's actual architecture is Qwen3.5 hybrid full/linear attention.
The updated preset explicitly sets `gdn_prefill_backend=triton`, following the
mitigation reported in [vLLM issue #38916](https://github.com/vllm-project/vllm/issues/38916),
and reduces the per-iteration prefill budget from 8192 to 2048. The current
preset uses session batch 4, n=4, max sequences 16, the actor's FA4, and the 64K context limit.
The GDN switch affects linear-attention prefill, not ordinary full attention.

vLLM logging is set to INFO in Ray's runtime environment so the next launch
reports its actual GDN backend. The expected log is `Using Triton/FLA GDN prefill
kernel`. `GDN_PREFILL_BACKEND` and `MAX_NUM_BATCHED_TOKENS` remain overridable.
The vLLM RPC timeout is unchanged. This is an unverified runtime
mitigation, not evidence that the original kernel hang has been fixed.

`ROLLOUT_N` is passed as `recipe_rollout_n` and sets both `rollout.n` and the
sampler's expected sibling count. The GRPO and DAPO base launchers still default
to the recipe's n=2 unless this selector is supplied. All four siblings must be
scorable for a group to survive; more siblings can therefore also increase
whole-group rejection when tool replay coverage is incomplete. This GRPO preset
does not enable DAPO same-score filtering.

The later `20260921_165309` run completed nine steps, then exhausted the recipe's
900-second total sampling budget while refilling groups. No vLLM engine error
was recorded in that run. The recipe now treats `wait_timeout_seconds: 900` as
an inactivity limit: newly written trajectories or newly terminal groups reset
it; polling and pending/running status updates do not. `max_wait_seconds: 3600`
is a separate hard limit for the entire sampling call. The 16-refill bound and
whole-group scoreability checks remain in force. These settings live in
`data/recipe_v3_session_agentloop/recipe.yaml` under the user's data directory.
Waiting logs report elapsed/idle time and pending/running groups every minute.

Replay matching now treats omitted and empty `departure` as equivalent only
for `ai_assistant_receive_input` when the schema makes it optional without a
default. Schema validation and LLM action auditing still run; other arguments
are not relaxed. Traces retain the original call and label applied normalization.

The throughput task saves every five steps by default (`SAVE_FREQ` overrides it).
These changes take effect on a new launch. A run that exited before its first
checkpoint cannot resume its unsaved optimizer/model updates.

## Doubao query-tool simulation

The throughput task now defaults to `TOOL_MOCK_MODE=hybrid`, using
`TOOL_MOCK_MODEL=ep-20260513201344-xl7ph` and
`TOOL_MOCK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3`. The scoring judge
keeps its independent `JUDGE_MODEL` and endpoint. Mock credentials resolve in
order: `TOOL_MOCK_API_KEY`, `ARK_API_KEY`, `JUDGE_API_KEY`. No new key is required
when the existing Ark key can access the mock endpoint. All `TOOL_MOCK_*`
settings are forwarded into Ray workers by the external recipe entry point.

Supported tools are `amap_place_around`, `tool_amap_place_text_batch`,
`tool_tavily_web_search`, and `tool_web_search_all`. Schema-valid exact recorded
results are reused; otherwise Doubao generates a simulated response based on
the actual arguments, visible user/system/tool history and curated return
contracts adapted from `sand_box/工具输出示例/工具输出示例`. Unseen source answers
and future observations are not sent. These query tools bypass the action
judge's execution gate; task compliance still affects final reward. Other tools
keep their original replay and audit behavior. No side-effect tool is simulated.

POI responses must have matching counts and valid fields. Observed unambiguous
named POIs cannot change address. Search responses remain text, as in the source
examples. These checks do not prove geographic or factual correctness: outputs
are synthetic training observations, not verified live API results.

Validated results are cached under the recipe's `tool_mock_cache/` (override
`TOOL_MOCK_CACHE_DIR`) with a versioned endpoint/request hash and a bounded
cross-process lock. Request identity excludes sampled assistant prose and tool
call IDs, so siblings with the same visible state and arguments share a result.
Committed identical simulated calls within a trajectory reuse their observation.
Different queries can still produce semantic inconsistencies; no fully populated
shared POI database is claimed. Recorded follow-ups after a synthetic result
require the continuation judge to verify compatibility before being appended.

Traces include `tool_mock_mode`, per-event `source=recorded|llm_mock`, model,
cache provenance and request hash. Reward extras count `tool_mock_events` and
`tool_recorded_events`. The final judge is explicitly told which observations
are simulated. Tool messages remain excluded from policy loss.

Defaults are 60 seconds per API attempt, 3 attempts and 180 seconds total,
including cache lock wait. They are configurable with
`TOOL_MOCK_TIMEOUT_SECONDS`, `TOOL_MOCK_MAX_ATTEMPTS`, and
`TOOL_MOCK_RETRY_BUDGET_SECONDS`. Invalid/truncated responses and API failures
are unscorable environment failures, never generic success or policy zero.

Run normally to enable hybrid simulation, or restore strict replay:

```bash
TOOL_MOCK_MODE=replay bash tasks/q38_27B_session_throughput.sh
```

Only new workers pick up changes; an already running training job is unaffected.

## Business labels in rollout history

New `rollouts/<step>.jsonl` rows and `session_traces/*.json` include `session_id`,
`source_line`, `business_agent`, `business_agent_status`,
`business_agent_observed_role`, and `business_agent_reason`. Labels come from
the source dataset; inferred or `待复核` labels retain their classification status.
They are inspection metadata, not added to prompts, reward calculations or loss.
The rollout writer resolves labels using each row's session ID after sorting.

Existing completed files can be annotated using the recipe's
`scripts/annotate_rollouts.py RUN_DIR`. It preserves original content fields,
timestamps and permissions, and skips files written within the last minute or
changed during processing. A running old worker still writes its old format;
new launches automatically emit the labels.
