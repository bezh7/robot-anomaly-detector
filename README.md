# robot-anomaly-detector v0.1

## Project Reflection

This project was my attempt to create an anomaly detection and attribution layer for deployed robots. Coming from an academic background, one of my biggest frustrations has always been the gap between working technology and deployable technology. Telemetry and debugging is a critical bottleneck for real robotic deployments and this was my attempt at contributing to the problem.

The current state of this project (`v0.1`) is a demo trained on a very small dataset (around 20m of IMU data with no labels or ground truth videos for manual labeling) adapted from the SubT-MRS dataset. The current architecture uses a reconstruction MSE scoring system which was able to capture some anomalies; however, I've run into serious issues in capturing low-frequency anomalies such as sensor drift. Furthermore, this version uses an `LSTM-AE`. Although this architecture performed fine on smoke tests at this scale, it is likely not the final architecture I would use at larger scale, especially for capturing cross-channel dependencies, an important skill for modeling physical systems.

## Data processing and training pipeline + architectural overview

The project pipeline has four major layers:

1. **Data processing**
   - restore clean overlap artifacts
   - build feature tables from IMU and derived features
   - fit fold-specific normalizers
   - generate train/validation window indices

2. **Modeling**
   - train LSTM and TCN autoencoder baselines on leave-one-run-out folds
   - score validation replay with reconstruction residuals
   - inject synthetic anomalies into replay windows for controlled evaluation
   - compute leaderboard metrics across folds and seeds

3. **Execution**
   - local Python path for development
   - CPU Docker path for local reproducibility
   - GPU Docker + ECR + EC2 launch path for larger search/training runs

4. **Demo / replay**
   - restore the selected final model
   - replay the held-out run `final_challenge_ugv2`
   - visualize anomaly score, threshold, grouped attribution, and injected anomalies

Architecturally, the final selected model is:

- `LSTM autoencoder`
- feature set: `raw_plus_derived`
- normalization: `zscore`
- batch size: `64`
- learning rate: `1e-3`

The model itself outputs a reconstruction of the normalized input window. The detector then computes squared residuals and collapses them into:

- a scalar anomaly score
- grouped residual scores for:
  - `quaternion`
  - `gyro`
  - `accel`

This was enough to support end-to-end training, evaluation, attribution, and replay. It was not enough to robustly capture all subtle anomaly types, which became clear during the held-out replay analysis.

Attribution in this version is group-level over `quaternion`, `gyro`, and `accel`, not per-channel root-cause attribution.

## Performance evaluation

Model selection was done in three stages:

- **Stage 1:** representative-fold search on `fold_2` and `fold_4`
- **Stage 2:** four-fold finalist evaluation
- **Stage 3:** repeated-seed stability check on the top two configs

The key selection outcome was:

- the TCN was the strongest single-seed finalist on the four-fold sweep
- the LSTM was the most stable across seeds
- because the dataset is small, I chose stability over the slightly stronger single-seed attribution result

Final selected model:

- `lstm / raw_plus_derived / zscore / bs64 / lr1e-3`

Selected quantitative results from the recorded experiment log:

- **Stage 2 TCN finalist**
  - detection rate: `1.0`
  - median time-to-detect: `0.18 s`
  - clean alerts per minute: `0.0`
  - group attribution top1: `1.0`

- **Stage 3 final LSTM selection**
  - detection rate: `1.0`
  - median time-to-detect: `0.18 s`
  - clean alerts per minute: `0.0`
  - group attribution top1: `0.9643`

For synthetic anomaly types that matter most to this demo:

- `gyro_bias_drift`
  - detection `1.0`
  - attribution `0.8333`
  - median TTD `0.58 s`
- `accel_freeze`
  - detection `1.0`
  - attribution `0.9167`
  - median TTD `0.58 s`
- `clipping`
  - detection `1.0`
  - attribution `1.0`
  - median TTD `0.18 s`

However, the held-out replay demo exposed an important weakness:

- on the real held-out run, the current score/thresholding path produces clustered alert-like regions in accel-dominant parts of the sequence
- the injected subtle demo anomalies do not reliably trigger alerts
- clipping increases grouped residuals clearly, but the scalar detection path still suppresses it
- drift/freeze often barely move the current scalar score at all

So the honest conclusion is:

- the full training/evaluation pipeline works
- the benchmarking loop is useful
- the current detector is not yet strong enough for subtle anomaly replay on unseen real data

These failures expose important information for future improvements.

## Next steps for `v0.2`

For the next iteration of this project, I'd like to tackle a much more ambitious scope, modeling full cross-channel dependencies and adjusting the architecture to capture a broader range of anomalies.

The first major change I'd like to make is the data collection pipeline. `v0.1` was trained on data adapted from the SubT-MRS dataset. For `v0.2` I'd like to augment this data collection process by generating synthetic data in IsaacSim to create a larger dataset where we can also have more insight into the ground truth of the runs and inject anomalies in a more grounded way.

Secondly, I’d like to change not only the model architecture, but also the scoring and detection formulation. The main lesson from `v0.1` is that a pooled reconstruction-residual score is too blunt for subtle faults like drift and freeze. For `v0.2`, I would explore:

- **forecasting-based detectors**
  - short-horizon prediction error instead of pure reconstruction error
  - better suited to faults that break temporal evolution rather than instantaneous shape
- **group-aware / state-aware scoring**
  - separate scoring for gyro, accel, and orientation-related features
  - thresholds conditioned on operating regime rather than one global cutoff
- **change-detection style methods**
  - cumulative or smoothed detectors that capture small persistent shifts
- **stronger sequence models**
  - transformer-style architectures or richer latent-variable models that can model cross-channel structure more explicitly than the current LSTM autoencoder

In short, `v0.2` should aim to solve three problems that `v0.1` surfaced clearly:

1. more realistic and scalable data
2. stronger sequence modeling of physical sensor dependencies
3. a better anomaly score than pooled reconstruction residuals

## Final Thoughts

I would love to continue working on this problem in the future and will continue to publish updates to this repo as well as potential future blog posts with more in-depth deep dives into the development process, likely on my [substack](https://substack.com/@benzhnng?). Feel free to reach out to bezh@seas.upenn.edu with any questions or comments.
