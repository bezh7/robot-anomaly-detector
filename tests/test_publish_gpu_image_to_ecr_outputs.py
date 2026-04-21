from pathlib import Path


def test_publish_script_builds_linux_amd64_gpu_image_and_pushes_to_ecr():
    script = Path("scripts/publish_gpu_image_to_ecr.sh").read_text()

    assert "docker buildx build --platform linux/amd64 -f Dockerfile.gpu" in script
    assert "--push" in script
    assert "aws ecr get-login-password" in script
    assert "docker login --username AWS --password-stdin" in script
    assert "docker push" not in script
    assert "create-repository" in script or "describe-repositories" in script
