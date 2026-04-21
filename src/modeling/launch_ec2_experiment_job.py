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
  -m src.modeling.run_model_search --artifact-root artifacts/features --output-root artifacts/modeling
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
    parser = argparse.ArgumentParser(description="Launch an EC2 experiment/search job.")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--feature-s3-prefix", required=True)
    parser.add_argument("--modeling-s3-prefix", required=True)
    parser.add_argument("--ami-id", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-profile-name", required=True)
    parser.add_argument("--key-name", required=True)
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--tag-name", default="rad-stage1-search")
    args = parser.parse_args()

    script = build_user_data_script(
        image_uri=args.image_uri,
        feature_s3_prefix=args.feature_s3_prefix,
        modeling_s3_prefix=args.modeling_s3_prefix,
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
