from pathlib import Path


def test_requirements_are_split_between_runtime_dev_and_local():
    runtime = Path("requirements-runtime.txt").read_text()
    dev = Path("requirements-dev.txt").read_text()
    local = Path("requirements.txt").read_text()

    assert "torch" not in runtime
    assert "pytest" not in runtime
    assert "matplotlib" not in runtime
    assert "pytest" in dev
    assert "matplotlib" in dev
    assert "-r requirements-runtime.txt" in local
    assert "-r requirements-dev.txt" in local
    assert "torch==2.6.0" in local


def test_cpu_dockerfile_installs_runtime_requirements_and_cpu_torch():
    dockerfile = Path("Dockerfile.cpu").read_text()
    assert "requirements-runtime.txt" in dockerfile
    assert "download.pytorch.org/whl/cpu" in dockerfile
    assert "torch==2.6.0" in dockerfile
    assert '"src.modeling.run_model_search"' in dockerfile


def test_gpu_dockerfile_uses_aws_dlc_base_and_runtime_requirements():
    dockerfile = Path("Dockerfile.gpu").read_text()
    assert (
        "FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.6.0-gpu-py312-cu126-ubuntu22.04-sagemaker"
        in dockerfile
    )
    assert "requirements-runtime.txt" in dockerfile
    assert 'ENTRYPOINT ["python"]' in dockerfile
    assert '"src.modeling.run_model_search"' in dockerfile
    assert 'CMD ["-m", "src.modeling.run_model_search"' in dockerfile
    assert "pip install -r requirements-runtime.txt" in dockerfile


def test_docker_runner_mounts_artifacts_and_executes_search_with_cpu_image():
    script = Path("scripts/run_model_search_in_docker.sh").read_text()
    assert "Dockerfile.cpu" in script
    assert '-v "$PWD/artifacts/features:/app/artifacts/features"' in script
    assert '-v "$PWD/artifacts/modeling:/app/artifacts/modeling"' in script
