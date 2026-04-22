# Experiment Notes

## Stage 1 Search

Environment:
- EC2 instance: `i-0d8f8200cb495c848` (`g5.xlarge`, `us-east-1a`)
- GPU image in use for successful smoke/search path: fixed `ENTRYPOINT ["python"]` image family
- Representative folds: `fold_2`, `fold_4`
- Search space: `lstm,tcn` × `raw,raw_plus_derived` × `zscore,robust` × batch sizes `32,64,128` × learning rates `3e-4,1e-3`
- Trials per architecture: `8`

Observed Stage 1 leaderboard highlights:
- `lstm-raw_plus_derived-zscore-bs64-lr1e-03-seed7`: rank 1, detection `1.0`, median TTD `0.18`, clean alerts/min `0.0`
- `lstm-raw_plus_derived-zscore-bs64-lr3e-04-seed7`: rank 2, identical top-line metrics
- `tcn-raw-zscore-bs64-lr1e-03-seed7`: rank 3, identical top-line metrics
- `tcn-raw-zscore-bs128-lr1e-03-seed7`: rank 4, identical top-line metrics

Interpretation:
- Stage 1 is saturated at the top cluster.
- Many configs tie on the primary metrics.
- Stage 2 should compare diverse finalists, not redundant ties.

Chosen Stage 2 finalists:
- `lstm` / `raw_plus_derived` / `zscore` / `bs64` / `lr1e-3`
- `lstm` / `raw_plus_derived` / `robust` / `bs64` / `lr1e-3`
- `tcn` / `raw` / `zscore` / `bs64` / `lr1e-3`
- `tcn` / `raw_plus_derived` / `zscore` / `bs64` / `lr1e-3`

## Stage 2 Execution Notes

First remote orchestration attempt:
- Used AWS SSM `AWS-RunShellScript`
- Failed before launching the container
- Root cause: command used `set -o pipefail`, but `AWS-RunShellScript` executed under `/bin/sh`, which does not support that option
- Status: no useful Stage 2 training/evaluation work performed in that failed attempt

Next action:
- rerun Stage 2 under `bash -lc` on the EC2 host
- write outputs to `/opt/rad/modeling_stage2`
- review full 4-fold metrics before any final training

Second remote orchestration attempt:
- Used a heredoc-heavy `AWS-RunShellScript` payload
- Failed before launching the containerized run
- Root cause: the generated shell payload collapsed newlines and heredoc delimiters, which produced a malformed `/bin/bash -c` command and emptied a few injected env vars
- Status: no useful Stage 2 training/evaluation work performed in that failed attempt

Successful Stage 2 run:
- Used AWS SSM `AWS-RunShellScript` with a simple command list
- Wrote the Stage 2 runner script to `/opt/rad/modeling_stage2/stage2_finalists.py` via base64 decode on-host
- Pulled `628161515461.dkr.ecr.us-east-1.amazonaws.com/robot-anomaly-detector:pt26-entrypointfix-399abf8`
- Ran the full 4-fold finalist sweep inside Docker on `i-0d8f8200cb495c848`
- Output root: `/opt/rad/modeling_stage2`

Observed Stage 2 leaderboard:
- `tcn-raw-zscore-bs64-lr1e-03-seed7`: rank 1, folds `4`, detection `1.0`, median TTD `0.18`, clean alerts/min `0.0`, group attribution top1 `1.0`
- `lstm-raw_plus_derived-zscore-bs64-lr1e-03-seed7`: rank 2, folds `4`, detection `1.0`, median TTD `0.18`, clean alerts/min `0.0`, group attribution top1 `0.9642857142857143`
- `tcn-raw_plus_derived-zscore-bs64-lr1e-03-seed7`: rank 3, folds `4`, detection `1.0`, median TTD `0.18`, clean alerts/min `0.0`, group attribution top1 `0.9642857142857143`
- `lstm-raw_plus_derived-robust-bs64-lr1e-03-seed7`: rank 4, folds `4`, detection `0.9375`, median TTD `0.18`, clean alerts/min `0.0`, group attribution top1 `1.0`

Stage 2 interpretation:
- `tcn/raw/zscore/bs64/lr1e-3` is the clear leader after the full 4-fold sweep.
- `lstm/raw_plus_derived/zscore/bs64/lr1e-3` remains competitive, but it is slightly weaker on attribution.
- `robust` normalization underperforms for the `lstm` finalist and does not advance.
- Next step before final training: Stage 3 seed-stability runs on the top 1–2 Stage 2 configs.

## Stage 3 Seed Stability

Scope:
- Compared the top 2 Stage 2 configs:
  - `tcn` / `raw` / `zscore` / `bs64` / `lr1e-3`
  - `lstm` / `raw_plus_derived` / `zscore` / `bs64` / `lr1e-3`
- Folds: `fold_1`, `fold_2`, `fold_3`, `fold_4`
- Seeds: `7`, `17`, `27`
- Output root: `/opt/rad/modeling_stage3`

Execution:
- Used AWS SSM `AWS-RunShellScript`
- Wrote `/opt/rad/modeling_stage3/stage3_finalists.py` on-host via base64 decode
- Ran the seed-stability sweep in Docker on `i-0d8f8200cb495c848`
- Result: successful end-to-end completion

Observed Stage 3 leaderboard:
- `stage3-lstm-raw_plus_derived-zscore-bs64-lr1e-03-seeds7-17-27`: rank 1, folds `4`, seeds `3`, detection `1.0`, median TTD `0.18`, clean alerts/min `0.0`, group attribution top1 `0.9642857142857143`
- `stage3-tcn-raw-zscore-bs64-lr1e-03-seeds7-17-27`: rank 2, folds `4`, seeds `3`, detection `0.9895833333333334`, median TTD `0.18`, clean alerts/min `0.0`, group attribution top1 `0.9761904761904763`

Per-seed breakdown:
- `lstm/raw_plus_derived/zscore/bs64/lr1e-3`
  - seed `7`: detection `1.0`, attribution top1 `0.9642857142857143`
  - seed `17`: detection `1.0`, attribution top1 `0.9642857142857143`
  - seed `27`: detection `1.0`, attribution top1 `0.9642857142857143`
- `tcn/raw/zscore/bs64/lr1e-3`
  - seed `7`: detection `1.0`, attribution top1 `1.0`
  - seed `17`: detection `0.96875`, attribution top1 `1.0`
  - seed `27`: detection `1.0`, attribution top1 `0.9285714285714286`

Stage 3 interpretation:
- `lstm/raw_plus_derived/zscore/bs64/lr1e-3` is the more stable config across seeds.
- `tcn/raw/zscore/bs64/lr1e-3` remains strong, but it shows measurable seed variance in both detection and attribution.
- Recommendation before final training: select `lstm/raw_plus_derived/zscore/bs64/lr1e-3` as the final-training candidate unless we decide that the slightly higher mean attribution of the `tcn` family is worth the seed instability tradeoff.

## Final Training

Selected final config:
- `lstm` / `raw_plus_derived` / `zscore` / `bs64` / `lr1e-3` / `seed7`

Execution:
- Used AWS SSM `AWS-RunShellScript`
- Ran the full-dev training path in Docker on `i-0d8f8200cb495c848`
- Command target: `python -m src.modeling.train_anomaly_model ... --final-train`
- Result: successful completion

Produced artifacts on EC2:
- `/opt/rad/modeling_final/lstm-raw_plus_derived-zscore-bs64-lr1e-03-seed7/full_dev/config.json`
- `/opt/rad/modeling_final/lstm-raw_plus_derived-zscore-bs64-lr1e-03-seed7/full_dev/final_checkpoint.pt`
- `/opt/rad/modeling_final/lstm-raw_plus_derived-zscore-bs64-lr1e-03-seed7/full_dev/train_history.csv`
- `/opt/rad/modeling_final/final_train.log`

Artifact sync:
- Synced final-training outputs to:
  - `s3://robot-anomaly-detector-628161515461-us-east-1-20260421/artifacts/modeling/final`

Current status:
- Model selection is complete.
- Final training is complete.
- Remaining work is demo-oriented:
  - held-out/demo replay validation
  - lightweight browser-based visualization

Replay dashboard direction:
- Build a replay-first localhost dashboard against local artifacts
- Keep it final-model-only and intentionally plain
- Recompute replay scores in memory rather than exporting a dedicated replay timeline file
- Include a simple play mode for recorded demo walkthroughs

## Replay Demo Preparation

Real replay artifact source:
- Real clean and feature artifacts live in `s3://robot-anomaly-detector-data/workspaces/surabaya-v1/artifacts/...`
- The previous dashboard was accidentally pointed at the fixture bucket `robot-anomaly-detector-628161515461-us-east-1-20260421`, which only contains `seq_a`-`seq_d` 4-second fixtures

Held-out run:
- Held-out/demo run is `final_challenge_ugv2`
- Real dev feature tables restored locally under `artifacts/replay_features_real`
- Held-out clean overlap pair restored locally under `artifacts/demo_clean/overlap`

Demo replay plan:
- Build a held-out feature table for `final_challenge_ugv2` into the real replay feature root
- Fit replay normalization from the 4 dev sequences listed in the real `split_manifest.json`, excluding the held-out run
- Use two replay modes:
  - `clean`
  - `demo` with a fixed multi-anomaly schedule
- Demo anomaly schedule is intentionally non-overlapping and group-attributable:
  - `gyro_bias_drift`
  - `accel_freeze`
  - `clipping`
- Playback should advance at the model cadence: `10 Hz` (`100 ms` per update)

Held-out replay stats:
- Held-out source sequence: `final_challenge_ugv2`
- Local replay feature root: `artifacts/replay_features_real`
- Held-out feature table row count: `171228`
- Held-out replay duration: `3424.54 s` (~`57.1 min`)
- Model score timeline length at `10 Hz` cadence: `34216` updates

Replay precompute behavior:
- Full clean timeline on the held-out run is now viable with batched window inference
- Clean + demo full-run smoke completed successfully in `1:14.30` total on local CPU
- That implies roughly `~37 s` first-load cost per replay mode before Streamlit cache reuse

Selected LSTM final-model anomaly performance (Stage 3, 4 folds × 3 seeds = 12 evals/type):
- `gyro_bias_drift`: detection `1.0`, attribution `0.8333`, median TTD `0.58 s`
- `accel_freeze`: detection `1.0`, attribution `0.9167`, median TTD `0.58 s`
- `clipping`: detection `1.0`, attribution `1.0`, median TTD `0.18 s`
- `angular_rate_burst`: detection `1.0`, attribution `1.0`, median TTD `0.18 s`
- `gyro_bias_step`: detection `1.0`, attribution `1.0`, median TTD `0.08 s`
- `impact_pulse`: detection `1.0`, attribution `1.0`, median TTD `0.18 s`
- `vibration_burst`: detection `1.0`, attribution `1.0`, median TTD `0.18 s`
- `noise_burst`: detection `1.0`, attribution is intentionally undefined (`mixed` target group), median TTD `0.18 s`

Interpretation for the demo:
- The model does handle non-spiky anomalies well enough for the replay demo.
- Drift and freeze are detected reliably, but they are slower and slightly weaker on attribution than the obvious pulse/burst cases.
- Current attribution remains group-level (`quaternion`, `gyro`, `accel`) from reconstruction residual energy, not per-channel blame.

Metric semantics:
- These metrics come from the synthetic injection benchmark, not naturally labeled incidents in the real dataset.
- `detection_rate` means the model raised an alert within the injected event interval plus the evaluator grace period.
- `attribution` means the predicted top residual group on the first alerting window matched the injected target group.
- `median_ttd` means time from injected anomaly start to the first alerting window that satisfied the persistence rule.
