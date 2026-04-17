from __future__ import annotations

import io
import zipfile
from pathlib import Path

UGV_SEQUENCE_NAMES = [
    'final_challenge_ugv1',
    'final_challenge_ugv2',
    'final_challenge_ugv3',
    'urban_challenge_ugv1',
    'urban_challenge_ugv2',
]


def write_sample_imu_csv(
    path: Path,
    *,
    timestamps: list[int],
    gyro_scale: float = 1.0,
    accel_scale: float = 1.0,
    spaced_header: bool = True,
) -> Path:
    header = (
        'timestamp, q_x, q_y, q_z, q_w, ang_vel_x, ang_vel_y, ang_vel_z, '
        'lin_acc_x, lin_acc_y, lin_acc_z\n'
        if spaced_header
        else 'timestamp,q_x,q_y,q_z,q_w,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n'
    )
    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.append(
            ', '.join(
                [
                    str(timestamp),
                    f'{0.01 * (index + 1):.4f}',
                    '0.0000',
                    '0.0000',
                    '1.0000',
                    f'{gyro_scale * (index + 1):.4f}',
                    f'{gyro_scale * (index + 2):.4f}',
                    f'{gyro_scale * (index + 3):.4f}',
                    f'{accel_scale * (index + 4):.4f}',
                    f'{accel_scale * (index + 5):.4f}',
                    f'{accel_scale * (index + 6):.4f}',
                ]
            )
        )
    path.write_text(header + '\n'.join(rows) + '\n')
    return path


def write_sample_gt_csv(path: Path, *, timestamps: list[int], sign_flip_index: int | None = None) -> Path:
    lines = [
        'timestamp,p_w_b_x,p_w_b_y,p_w_b_z,q_w_b_x,q_w_b_y,q_w_b_z,q_w_b_w'
    ]
    for index, timestamp in enumerate(timestamps):
        quaternion = [0.0, 0.0, 0.1 * (index + 1), 0.99]
        if sign_flip_index is not None and index == sign_flip_index:
            quaternion = [-value for value in quaternion]
        lines.append(
            ','.join(
                [
                    str(timestamp),
                    f'{1.0 * index:.4f}',
                    f'{2.0 * index:.4f}',
                    f'{0.5 * index:.4f}',
                    f'{quaternion[0]:.4f}',
                    f'{quaternion[1]:.4f}',
                    f'{quaternion[2]:.4f}',
                    f'{quaternion[3]:.4f}',
                ]
            )
        )
    path.write_text('\n'.join(lines) + '\n')
    return path


def write_sample_gt_zip(path: Path, *, timestamps: list[int], sign_flip_index: int | None = None) -> Path:
    csv_buffer = io.StringIO()
    csv_lines = [
        'timestamp,p_w_b_x,p_w_b_y,p_w_b_z,q_w_b_x,q_w_b_y,q_w_b_z,q_w_b_w'
    ]
    for index, timestamp in enumerate(timestamps):
        quaternion = [0.0, 0.0, 0.1 * (index + 1), 0.99]
        if sign_flip_index is not None and index == sign_flip_index:
            quaternion = [-value for value in quaternion]
        csv_lines.append(
            ','.join(
                [
                    str(timestamp),
                    f'{1.0 * index:.4f}',
                    f'{2.0 * index:.4f}',
                    f'{0.5 * index:.4f}',
                    f'{quaternion[0]:.4f}',
                    f'{quaternion[1]:.4f}',
                    f'{quaternion[2]:.4f}',
                    f'{quaternion[3]:.4f}',
                ]
            )
        )
    csv_buffer.write('\n'.join(csv_lines) + '\n')

    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('example_sequence/ground_truth_path.csv', csv_buffer.getvalue())
        archive.writestr('example_sequence/map_analysis.png', b'PNG')
    return path


def build_local_ugv_raw_root(root_dir: Path) -> Path:
    for sequence_index, sequence_name in enumerate(UGV_SEQUENCE_NAMES):
        sequence_dir = root_dir / sequence_name
        sequence_dir.mkdir(parents=True, exist_ok=True)

        imu_timestamps = [
            100 + sequence_index * 10,
            200 + sequence_index * 10,
            300 + sequence_index * 10,
            400 + sequence_index * 10,
            500 + sequence_index * 10,
        ]
        gt_timestamps = [
            150 + sequence_index * 10,
            250 + sequence_index * 10,
            350 + sequence_index * 10,
            450 + sequence_index * 10,
        ]

        write_sample_imu_csv(
            sequence_dir / 'imu_data.csv',
            timestamps=list(reversed(imu_timestamps)),
            gyro_scale=1.0 + sequence_index,
            accel_scale=2.0 + sequence_index,
        )
        write_sample_gt_zip(
            sequence_dir / f'{sequence_name}_gt.zip',
            timestamps=list(reversed(gt_timestamps)),
            sign_flip_index=1,
        )
        (sequence_dir / 'calibration.yaml').write_text('laser_to_imu: [0, 0, 0, 0, 0, 0, 1]\n')
        (sequence_dir / f'{sequence_name}_folder.zip').write_bytes(b'placeholder')
    return root_dir
