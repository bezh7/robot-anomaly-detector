from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.feature_contract import (
    DERIVED_IMU_FEATURES,
    FEATURE_TABLE_METADATA_COLUMNS,
    GT_CONTEXT_FEATURES,
    RAW_IMU_FEATURES,
)
from src.validate_feature_dataset import validate_feature_dataset

EPSILON = 1e-12
PLOT_WINDOW_SECONDS = 5.0
NS_PER_SECOND = 1_000_000_000.0


def _load_feature_tables(feature_root: Path) -> dict[str, pd.DataFrame]:
    feature_tables_dir = Path(feature_root) / 'feature_tables'
    return {
        path.stem: pd.read_parquet(path)
        for path in sorted(feature_tables_dir.glob('*.parquet'))
    }


def compute_sequence_feature_stats(feature_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for sequence_name, frame in feature_tables.items():
        duration_s = float(frame['timestamp_ns'].iloc[-1] - frame['timestamp_ns'].iloc[0]) / NS_PER_SECOND
        feature_columns = [
            column for column in frame.columns
            if column not in FEATURE_TABLE_METADATA_COLUMNS
        ]
        for column in feature_columns:
            values = frame[column].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                stats = {k: float('nan') for k in ['mean', 'std', 'min', 'p01', 'p50', 'p99', 'max']}
            else:
                stats = {
                    'mean': float(np.mean(finite_values)),
                    'std': float(np.std(finite_values, ddof=0)),
                    'min': float(np.min(finite_values)),
                    'p01': float(np.quantile(finite_values, 0.01)),
                    'p50': float(np.quantile(finite_values, 0.50)),
                    'p99': float(np.quantile(finite_values, 0.99)),
                    'max': float(np.max(finite_values)),
                }
            records.append(
                {
                    'sequence_name': sequence_name,
                    'feature_name': column,
                    'row_count': int(len(frame)),
                    'duration_s': duration_s,
                    'missing_rate': float(np.mean(~np.isfinite(values))),
                    **stats,
                }
            )
    return pd.DataFrame.from_records(records)


def compute_fold_window_counts(window_indices_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(Path(window_indices_dir).glob('*.parquet')):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        first = frame.iloc[0]
        records.append(
            {
                'path': str(path),
                'fold_name': str(first['fold_name']),
                'split_name': str(first['split_name']),
                'feature_set_name': str(first['feature_set_name']),
                'normalization_mode': str(first['normalization_mode']),
                'window_count': int(len(frame)),
                'window_size': int(first['window_size']),
                'stride': int(first['stride']),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_fold_feature_drift(
    feature_tables: dict[str, pd.DataFrame],
    split_manifest: dict[str, object],
) -> pd.DataFrame:
    feature_columns = RAW_IMU_FEATURES + DERIVED_IMU_FEATURES + GT_CONTEXT_FEATURES
    records: list[dict[str, object]] = []
    for fold_index, fold in enumerate(split_manifest['folds'], start=1):
        fold_name = f'fold_{fold_index}'
        training_sequences = list(fold['training_sequences'])
        validation_sequence = str(fold['validation_sequence'])
        train_frame = pd.concat([feature_tables[name][feature_columns] for name in training_sequences], ignore_index=True)
        validation_frame = feature_tables[validation_sequence][feature_columns]
        for column in feature_columns:
            train_values = train_frame[column].to_numpy(dtype=float)
            val_values = validation_frame[column].to_numpy(dtype=float)
            train_finite = train_values[np.isfinite(train_values)]
            val_finite = val_values[np.isfinite(val_values)]
            train_mean = float(np.mean(train_finite)) if train_finite.size else float('nan')
            train_std = float(np.std(train_finite, ddof=0)) if train_finite.size else float('nan')
            val_mean = float(np.mean(val_finite)) if val_finite.size else float('nan')
            smd = abs(train_mean - val_mean) / max(train_std, EPSILON) if np.isfinite(train_mean) and np.isfinite(val_mean) else float('nan')
            records.append(
                {
                    'fold_name': fold_name,
                    'validation_sequence': validation_sequence,
                    'feature_name': column,
                    'train_mean': train_mean,
                    'train_std': train_std,
                    'validation_mean': val_mean,
                    'validation_std': float(np.std(val_finite, ddof=0)) if val_finite.size else float('nan'),
                    'train_missing_rate': float(np.mean(~np.isfinite(train_values))),
                    'validation_missing_rate': float(np.mean(~np.isfinite(val_values))),
                    'standardized_mean_diff': float(smd),
                }
            )
    return pd.DataFrame.from_records(records)


def _relative_time_seconds(timestamps_ns: np.ndarray) -> np.ndarray:
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    return (timestamps_ns - int(timestamps_ns[0])) / NS_PER_SECOND


def _plot_overlay(sequence_name: str, feature_table: pd.DataFrame, raw_imu: pd.DataFrame, output_path: Path) -> None:
    end_ns = int(feature_table['timestamp_ns'].iloc[0] + PLOT_WINDOW_SECONDS * NS_PER_SECOND)
    feature_slice = feature_table[feature_table['timestamp_ns'] <= end_ns]
    raw_slice = raw_imu[raw_imu['timestamp_ns'] <= end_ns]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(_relative_time_seconds(raw_slice['timestamp_ns'].to_numpy()), raw_slice['ang_vel_x'].to_numpy(), label='raw', alpha=0.7)
    axes[0].plot(_relative_time_seconds(feature_slice['timestamp_ns'].to_numpy()), feature_slice['ang_vel_x'].to_numpy(), label='resampled', linewidth=2)
    axes[0].set_ylabel('ang_vel_x')
    axes[0].legend()
    axes[0].set_title(f'{sequence_name}: raw vs resampled overlay')

    axes[1].plot(_relative_time_seconds(raw_slice['timestamp_ns'].to_numpy()), raw_slice['lin_acc_x'].to_numpy(), label='raw', alpha=0.7)
    axes[1].plot(_relative_time_seconds(feature_slice['timestamp_ns'].to_numpy()), feature_slice['lin_acc_x'].to_numpy(), label='resampled', linewidth=2)
    axes[1].set_ylabel('lin_acc_x')
    axes[1].set_xlabel('seconds')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_derived(sequence_name: str, feature_table: pd.DataFrame, output_path: Path) -> None:
    feature_slice = feature_table.iloc[: min(len(feature_table), int(PLOT_WINDOW_SECONDS * 50))]
    t = _relative_time_seconds(feature_slice['timestamp_ns'].to_numpy())

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, column in zip(axes, ['gyro_norm', 'accel_norm', 'angular_accel_norm', 'jerk_norm'], strict=True):
        ax.plot(t, feature_slice[column].to_numpy(), linewidth=1.5)
        ax.set_ylabel(column)
    axes[0].set_title(f'{sequence_name}: derived feature traces')
    axes[-1].set_xlabel('seconds')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_gt_context(sequence_name: str, feature_table: pd.DataFrame, output_path: Path) -> None:
    feature_slice = feature_table.iloc[: min(len(feature_table), int(PLOT_WINDOW_SECONDS * 50))]
    t = _relative_time_seconds(feature_slice['timestamp_ns'].to_numpy())

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, column in zip(axes, ['gt_speed_mps', 'gt_yaw_rad', 'gt_yaw_rate_rps'], strict=True):
        ax.plot(t, feature_slice[column].to_numpy(), linewidth=1.5)
        ax.set_ylabel(column)
    axes[0].set_title(f'{sequence_name}: GT context traces')
    axes[-1].set_xlabel('seconds')
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _recommendation(max_model_missing: float, max_gt_missing: float, max_smd: float) -> str:
    if max_model_missing > 0.0:
        return 'fix pipeline before modeling'
    if max_gt_missing > 0.10 or max_smd > 5.0:
        return 'proceed with caveats'
    return 'proceed to modeling'


def _write_markdown_summary(
    output_path: Path,
    *,
    summary: dict[str, object],
    sequence_stats: pd.DataFrame,
    drift: pd.DataFrame,
    window_counts: pd.DataFrame,
) -> None:
    top_drift = drift.sort_values('standardized_mean_diff', ascending=False).head(5)
    lines = [
        '# Pre-Model QA Summary',
        '',
        f"Recommendation: **{summary['recommendation']}**",
        '',
        '## Overview',
        f"- Sequences: {summary['sequence_count']}",
        f"- Folds: {summary['fold_count']}",
        f"- Experiments: {summary['experiment_count']}",
        f"- Plots: {summary['plot_count']}",
        f"- Max model-feature missing rate: {summary['max_model_feature_missing_rate']:.4f}",
        f"- Max GT-context missing rate: {summary['max_gt_context_missing_rate']:.4f}",
        f"- Max standardized mean diff: {summary['max_standardized_mean_diff']:.4f}",
        '',
        '## Window counts',
    ]
    for _, row in window_counts.sort_values(['fold_name', 'split_name', 'feature_set_name']).iterrows():
        lines.append(
            f"- {row['fold_name']} / {row['split_name']} / {row['feature_set_name']} / {row['normalization_mode']}: {int(row['window_count'])} windows"
        )
    lines += ['', '## Top feature drift', '']
    for _, row in top_drift.iterrows():
        lines.append(
            f"- {row['fold_name']} / val={row['validation_sequence']} / {row['feature_name']}: standardized_mean_diff={row['standardized_mean_diff']:.4f}"
        )
    gt_missing = sequence_stats[sequence_stats['feature_name'].isin(GT_CONTEXT_FEATURES)]
    if not gt_missing.empty:
        lines += ['', '## GT missingness by sequence', '']
        for _, row in gt_missing.sort_values(['sequence_name', 'feature_name']).iterrows():
            if float(row['missing_rate']) > 0.0:
                lines.append(
                    f"- {row['sequence_name']} / {row['feature_name']}: missing_rate={row['missing_rate']:.4f}"
                )
    output_path.write_text('\n'.join(lines) + '\n')


def build_pre_model_qa_report(
    *,
    feature_root: Path,
    output_root: Path,
    clean_root: Path | None = None,
) -> tuple[Path, dict[str, object]]:
    feature_root = Path(feature_root)
    output_root = Path(output_root)
    clean_root = Path(clean_root) if clean_root is not None else None
    output_root.mkdir(parents=True, exist_ok=True)
    plots_dir = output_root / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    validate_feature_dataset(feature_root)
    feature_tables = _load_feature_tables(feature_root)
    split_manifest = json.loads((feature_root / 'split_manifest.json').read_text())

    sequence_stats = compute_sequence_feature_stats(feature_tables)
    window_counts = compute_fold_window_counts(feature_root / 'window_indices')
    drift = compute_fold_feature_drift(feature_tables, split_manifest)

    plot_count = 0
    for sequence_name, feature_table in feature_tables.items():
        _plot_derived(sequence_name, feature_table, plots_dir / f'{sequence_name}_derived.png')
        _plot_gt_context(sequence_name, feature_table, plots_dir / f'{sequence_name}_gt_context.png')
        plot_count += 2
        if clean_root is not None:
            raw_imu_path = clean_root / 'overlap' / f'{sequence_name}_imu.parquet'
            if raw_imu_path.exists():
                raw_imu = pd.read_parquet(raw_imu_path)
                _plot_overlay(sequence_name, feature_table, raw_imu, plots_dir / f'{sequence_name}_overlay.png')
                plot_count += 1

    max_model_missing = float(
        sequence_stats[sequence_stats['feature_name'].isin(RAW_IMU_FEATURES + DERIVED_IMU_FEATURES)]['missing_rate'].max()
    )
    gt_stats = sequence_stats[sequence_stats['feature_name'].isin(GT_CONTEXT_FEATURES)]
    max_gt_missing = float(gt_stats['missing_rate'].max()) if not gt_stats.empty else 0.0
    max_smd = float(drift['standardized_mean_diff'].max()) if not drift.empty else 0.0

    summary: dict[str, object] = {
        'sequence_count': len(feature_tables),
        'fold_count': len(split_manifest['folds']),
        'experiment_count': len(split_manifest['experiments']),
        'feature_count': len(RAW_IMU_FEATURES + DERIVED_IMU_FEATURES + GT_CONTEXT_FEATURES),
        'plot_count': plot_count,
        'max_model_feature_missing_rate': max_model_missing,
        'max_gt_context_missing_rate': max_gt_missing,
        'max_standardized_mean_diff': max_smd,
        'recommendation': _recommendation(max_model_missing, max_gt_missing, max_smd),
    }

    sequence_stats.to_csv(output_root / 'sequence_feature_stats.csv', index=False)
    window_counts.to_csv(output_root / 'fold_window_counts.csv', index=False)
    drift.to_csv(output_root / 'fold_feature_drift.csv', index=False)
    (output_root / 'pre_model_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_markdown_summary(output_root / 'qa_summary.md', summary=summary, sequence_stats=sequence_stats, drift=drift, window_counts=window_counts)

    return output_root, summary
