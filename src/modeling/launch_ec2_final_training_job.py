from __future__ import annotations

import argparse
import json
import subprocess


def _parse_image_uri(image_uri: str) -> tuple[str, str]:
    registry = image_uri.split("/", 1)[0]
    parts = registry.split(".")
    if len(parts) < 6 or parts[1] != "dkr" or parts[2] != "ecr":
        raise ValueError(f"unsupported ECR image URI: {image_uri}")
    return registry, parts[3]


def build_user_data_script(
    *,
    image_uri: str,
    feature_s3_prefix: str,
    modeling_s3_prefix: str,
    architecture: str = "lstm",
    feature_set: str = "raw",
    normalization: str = "zscore",
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    seed: int = 7,
) -> str:
    registry, region = _parse_image_uri(image_uri)
    return f"""#!/bin/bash
set -euxo pipefail
mkdir -p /opt/rad/features /opt/rad/modeling
aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {registry}
docker pull {image_uri}
aws s3 sync {feature_s3_prefix} /opt/rad/features
docker run --rm --gpus all \\
  -v /opt/rad/features:/app/artifacts/features \\
  -v /opt/rad/modeling:/app/artifacts/modeling \\
  {image_uri} \\
  -m src.modeling.train_anomaly_model --artifact-root artifacts/features --output-root artifacts/modeling --final-train --architecture {architecture} --feature-set {feature_set} --normalization {normalization} --batch-size {batch_size} --learning-rate {learning_rate} --max-epochs {max_epochs} --seed {seed}
aws s3 sync /opt/rad/modeling {modeling_s3_prefix}
"""


def build_run_instances_command(
    *,
    ami_id: str,
    instance_type: str,
    instance_profile_name: str,
    key_name: str,
    security_group_id: str,
    subnet_id: str,
    tag_name: str,
    user_data_script: str,
) -> list[str]:
    return [
        "aws",
        "ec2",
        "run-instances",
        "--image-id",
        ami_id,
        "--instance-type",
        instance_type,
        "--iam-instance-profile",
        f"Name={instance_profile_name}",
        "--key-name",
        key_name,
        "--security-group-ids",
        security_group_id,
        "--subnet-id",
        subnet_id,
        "--tag-specifications",
        f"ResourceType=instance,Tags=[{{Key=Name,Value={tag_name}}}]",
        "--user-data",
        user_data_script,
    ]


def launch_instance(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    payload = json.loads(completed.stdout)
    return str(payload["Instances"][0]["InstanceId"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch an EC2 final-training job.")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--feature-s3-prefix", required=True)
    parser.add_argument("--modeling-s3-prefix", required=True)
    parser.add_argument("--ami-id", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-profile-name", required=True)
    parser.add_argument("--key-name", required=True)
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--tag-name", default="rad-final-train")
    parser.add_argument("--architecture", default="lstm")
    parser.add_argument("--feature-set", default="raw")
    parser.add_argument("--normalization", default="zscore")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    script = build_user_data_script(
        image_uri=args.image_uri,
        feature_s3_prefix=args.feature_s3_prefix,
        modeling_s3_prefix=args.modeling_s3_prefix,
        architecture=args.architecture,
        feature_set=args.feature_set,
        normalization=args.normalization,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )
    command = build_run_instances_command(
        ami_id=args.ami_id,
        instance_type=args.instance_type,
        instance_profile_name=args.instance_profile_name,
        key_name=args.key_name,
        security_group_id=args.security_group_id,
        subnet_id=args.subnet_id,
        tag_name=args.tag_name,
        user_data_script=script,
    )
    print(launch_instance(command))


if __name__ == "__main__":
    main()
