from src.modeling.launch_ec2_experiment_job import (
    build_run_instances_command,
    build_user_data_script,
)


def test_build_user_data_script_restores_features_runs_search_and_syncs_outputs():
    script = build_user_data_script(
        image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/robot-anomaly-detector:git-abc123",
        feature_s3_prefix="s3://example-bucket/example-prefix/features",
        modeling_s3_prefix="s3://example-bucket/example-prefix/modeling",
    )

    assert "aws ecr get-login-password --region us-east-1" in script
    assert "docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com" in script
    assert "docker pull 123456789012.dkr.ecr.us-east-1.amazonaws.com/robot-anomaly-detector:git-abc123" in script
    assert "mkdir -p /opt/rad/features /opt/rad/modeling" in script
    assert "aws s3 sync s3://example-bucket/example-prefix/features /opt/rad/features" in script
    assert "docker run --rm --gpus all" in script
    assert '-v /opt/rad/features:/app/artifacts/features' in script
    assert '-v /opt/rad/modeling:/app/artifacts/modeling' in script
    assert (
        "-m src.modeling.run_model_search --artifact-root artifacts/features --output-root artifacts/modeling"
        in script
    )
    assert "aws s3 sync /opt/rad/modeling s3://example-bucket/example-prefix/modeling" in script
    assert "git clone" not in script
    assert "python3 -m venv" not in script
    assert "pip install -r requirements.txt" not in script


def test_build_run_instances_command_tags_experiment_job():
    command = build_run_instances_command(
        ami_id="ami-123456",
        instance_type="g4dn.xlarge",
        instance_profile_name="robot-anomaly-detector",
        key_name="demo-key",
        security_group_id="sg-123456",
        subnet_id="subnet-123456",
        tag_name="rad-stage1-search",
        user_data_script="#!/bin/bash\necho hi\n",
    )

    joined = " ".join(command)
    assert command[:4] == ["aws", "ec2", "run-instances", "--image-id"]
    assert "rad-stage1-search" in joined
    assert "robot-anomaly-detector" in joined
