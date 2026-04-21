# Production Extension Notes

These are the main ways to extend the current IMU anomaly demo into a more production-grade robotics anomaly detection system. This file is intentionally lightweight and focused on items we discussed but deferred for the MVP.

## 1. Add real production telemetry streams

The demo is IMU-first. A production system should expand beyond IMU so anomaly attribution can move from:

- "the inertial behavior looks wrong"

toward:

- "the likely source is comms / power / drivetrain / control / sensing"

Highest-value additional streams:

- wheel odometry / encoder data
- commanded velocities / control inputs
- motor current / torque / actuator load
- battery voltage / current / thermal telemetry
- Wi-Fi / network metrics:
  - RSSI
  - packet loss
  - tx retry rate
  - reconnect count
  - heartbeat latency

These streams are the real path toward diagnosing issues like:

- stuck or slipping robot
- actuator overload
- battery / power instability
- degraded network link or router problems

## 2. Evolve the architecture beyond IMU-only

The current demo design uses:

- one shared encoder
- one shared decoder
- per-channel residuals
- grouped post-hoc attribution

That is the right bias for IMU-only.

For production multimodal telemetry, the architecture should evolve toward:

- shared temporal encoder
- subsystem-specific decoder heads
  - IMU
  - odometry
  - power
  - network
  - control

Why this was deferred:

- the current dataset does not include the telemetry needed to justify subsystem heads
- single-decoder reconstruction is simpler and more defensible for IMU-only

Possible future objective extensions:

- masked reconstruction
- future prediction / forecasting
- masked future prediction
- multimodal predictive objectives

### Decoder note from the IMU-only baseline

For the current LSTM autoencoder baseline, the decoder should use:

- latent-initialized hidden/cell state
- learned **relative position embeddings** scoped only to the current window

It should explicitly **not** use:

- absolute mission/run timestamps
- autoregressive previous-output decoding in the baseline

Reasoning:

- relative position helps the decoder reconstruct ordered local dynamics
- absolute run time risks learning run chronology instead of local sensor behavior
- autoregressive decoding makes residuals less local and less interpretable because reconstruction errors can propagate forward through the window

This is the right tradeoff for a reconstruction-based anomaly detector where the residual itself is the product.

### TCN baseline note

For the current TCN baseline, the implementation was revised away from a same-length pass-through stack toward a real compressed autoencoder:

- two explicit temporal downsampling stages
- compressed bottleneck sequence used for reconstruction
- mirrored upsampling decoder
- weight-normalized causal convolutions
- ReLU kept as the baseline activation

Reasoning:

- the original same-length version did not enforce a strong enough bottleneck
- temporal compression preserves coarse sequence structure better than collapsing everything to a single pooled vector
- weight normalization is a better fit than batch normalization for this small time-series setup
- mean-pooled latent is kept only as a summary signal, not as the main reconstruction path

This makes the TCN a more credible comparison against the LSTM autoencoder for anomaly detection.

These become much more compelling once there are multiple real subsystem streams and cross-stream dependencies to learn.

## 3. Move from synthetic evaluation toward real incident supervision

For the demo, synthetic anomaly injection is the right evaluation tool because the dataset does not contain trusted anomaly labels.

A production system should add:

- incident review workflow
- operator labels
- maintenance logs
- root-cause taxonomy
- hard-negative review

That would allow:

- validating which alert types correspond to real failures
- tuning thresholds around operational false-positive budgets
- building semi-supervised or supervised anomaly classifiers on top of the unsupervised detector

## 4. Add stronger simulation only when robot metadata exists

We discussed simulation-based anomaly generation (e.g. Isaac Sim) and deferred it for good reason.

What would be needed for credible simulation:

- robot geometry / URDF / CAD
- mass / inertia parameters
- wheelbase / drivetrain details
- controller / command interface
- actuator model
- terrain / contact assumptions
- accurate sensor mounting and timing

What we have today is not enough for a faithful run recreation.

A future production-oriented simulation path could support:

- broader anomaly libraries
- scenario generation
- safety and stress testing
- multimodal synthetic data

But it should only be treated as a real benchmark once the robot and control metadata are available.

## 5. Tighten the online deployment path

The demo already assumes causal trailing-window scoring, which is the correct production bias.

Production extensions:

- streaming inference service with ring buffer
- fixed-latency online resampling
- model warm-up behavior before the first full window is available
- alert smoothing / debounce policy
- watchdogs around missing or stale telemetry
- model fallback behavior if a stream drops out

If latency becomes critical, add a two-tier stack:

- fast heuristic sentinel for abrupt spikes
- slower model score for richer anomaly detection and attribution

## 6. Treat contamination and drift as first-class concerns

The current dataset looks mostly normal, but not perfectly clean.

Production systems should add:

- automated outlier screening in training data
- retraining policies with exclusion/downweighting of suspicious windows
- feature drift monitoring
- alert-rate drift monitoring
- schema / telemetry contract validation on every ingest

This is especially important when deployments vary by:

- robot platform
- site
- route
- terrain
- operator behavior

## 7. Production thresholding should be operational, not just statistical

The demo uses percentile-style thresholding and persistence rules.

A production system should calibrate thresholds against:

- acceptable alert volume
- false-positive budget per hour/day
- severity-specific actions
- robot/site-specific operating regimes

Potential production thresholding extensions:

- per-robot thresholds
- per-site thresholds
- time-of-day / mission-mode thresholds
- risk-tiered alert levels

## 8. Improve the evaluation protocol

The current evaluation plan is good for an MVP:

- synthetic quantitative benchmark
- real suspicious segments for qualitative review

Production extensions:

- replay against real incident timelines
- per-incident root-cause attribution scoring
- human-in-the-loop review tooling
- regression suite of known failures
- continuous benchmarking across model versions

## 9. Add experiment tracking and model governance

The demo can tolerate lightweight experiment management.

Production should add:

- experiment registry
- model registry
- dataset versioning
- feature-schema versioning
- training/eval artifact retention
- reproducible deployment packaging

This should include the full chain:

- raw data version
- feature config
- anomaly injection config
- training config
- threshold config
- deployed model version

## 10. Package the pipeline for real reuse

Before production adoption, the current scripts should be turned into a cleaner product surface:

- pinned dependencies
- CLI entrypoints
- Docker image
- CI checks
- environment-based configuration
- repeatable end-to-end runs without manual script orchestration

## 11. Improve the operator-facing interface

The current plan already includes:

- global anomaly score
- grouped attribution
- top contributing channels

A stronger production UI should add:

- alert timeline
- raw vs reconstructed traces
- subsystem drill-down
- event notes / operator feedback
- links to logs, commands, and telemetry around the alert window

## 12. Recommended production progression

The cleanest path from this demo to a more serious system is:

1. Finish the IMU-first anomaly detector and evaluation stack
2. Add odometry / command telemetry
3. Add power telemetry
4. Add network telemetry
5. Move to subsystem-head architectures
6. Add incident labeling and feedback loops
7. Add simulation only when robot metadata is sufficient

This preserves the current work and extends it rather than replacing it.
