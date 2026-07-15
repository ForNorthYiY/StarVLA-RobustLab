# Replanning Interval Study

## Why eight open-loop steps can accumulate error

At replanning time, QwenOFT observes the current images and predicts an
eight-action chunk. Actions 1 through 7 are predictions about future states,
not reactions to newly observed states. If the first action moves the robot or
an object differently from the model's expectation, later actions are executed
from an off-prediction state.

Typical sources of mismatch include contact dynamics, friction, object pose
uncertainty, camera noise, controller tracking error, and small errors in the
predicted rotations. Closed-loop replanning observes the resulting state and
can correct these errors; open-loop cache execution cannot.

This does not mean interval 1 is always best. More frequent replanning costs
more inference time and may produce action discontinuities because consecutive
chunks can disagree. Longer chunks are cheaper and can preserve a smooth
short-horizon motion. The interval is therefore a control-performance versus
compute tradeoff.

## Implementation

The model output remains `[1, 8, 7]`. `replan_interval` changes only how much of
that chunk is consumed before it is discarded:

```python
cache_index = step % replan_interval
refresh = cache_index == 0
```

- Interval 1 consumes only action 0 from every predicted chunk.
- Interval 2 consumes actions 0 and 1.
- Interval 4 consumes actions 0 through 3.
- Interval 8 consumes the complete chunk.

The implementation rejects intervals outside `[1, action_chunk_size]` and logs
per-episode inference count, cumulative inference time, and mean request time.
The reproducible upstream patch is `patches/starvla_action_cache_trace.patch`.

## Controlled comparison protocol

Keep checkpoint, suite, task, initial state, seed, rendering setup, GPU, and
attention implementation fixed. Change only `replan_interval`. For a meaningful
success-rate comparison, expand beyond the smoke setup to multiple tasks,
initial states, and seeds.

For the 122 policy-controlled steps observed in the first real trace:

| Interval | Expected calls | Relative inference work | Smoke success | Measured latency |
|---:|---:|---:|---:|---:|
| 1 | 122 | 8.0x | TBD | TBD |
| 2 | 61 | 4.0x | TBD | TBD |
| 4 | 31 | 1.94x | TBD | TBD |
| 8 | 16 | 1.0x | 1/1 | not instrumented in the original run |

The 1/1 interval-8 result is an integration observation, not a benchmark
estimate. The full sweep was not started on 2026-07-15 because every server GPU
was occupied by another user's jobs. `scripts/remote/run_replan_sweep.sh`
refuses to start when the selected GPU already uses more than 100 MiB.

## Metrics to interpret together

Success rate alone is insufficient. Record:

1. episode success and number of policy-controlled steps;
2. inference request count;
3. cumulative and mean inference latency;
4. wall-clock episode time;
5. action discontinuity at replanning boundaries;
6. task and initial-state identity.

The request-count identity is `ceil(policy_steps / replan_interval)`. If the
measured count differs, the cache scheduler or step counter is wrong.
