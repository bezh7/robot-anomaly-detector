from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
import subprocess

from src.cleaning_contract import GROUND_TRUTH_CSV_FILENAME
from src.io_utils import Runner, default_runner


def extract_ground_truth_csv(gt_zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / GROUND_TRUTH_CSV_FILENAME

    with zipfile.ZipFile(gt_zip_path) as archive:
        member_name = next(name for name in archive.namelist() if name.endswith(GROUND_TRUTH_CSV_FILENAME))
        with archive.open(member_name) as source_handle, output_path.open('wb') as output_handle:
            output_handle.write(source_handle.read())

    return output_path


def extract_ground_truth_csv_from_s3(
    s3_zip_path: str,
    output_dir: Path,
    runner: Runner = default_runner,
) -> Path:
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_zip_path = Path(temporary_directory) / Path(s3_zip_path).name
        subprocess.run(
            ['aws', 's3', 'cp', s3_zip_path, str(local_zip_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return extract_ground_truth_csv(local_zip_path, output_dir)
