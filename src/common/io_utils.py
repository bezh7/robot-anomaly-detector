import subprocess
from collections.abc import Callable


Runner = Callable[[list[str]], str]


def default_runner(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout


def list_s3_prefixes(s3_prefix: str, runner: Runner = default_runner) -> list[str]:
    output = runner(["aws", "s3", "ls", s3_prefix])
    prefixes: list[str] = []

    for line in output.splitlines():
        stripped_line = line.strip()
        if not stripped_line.startswith("PRE "):
            continue
        prefixes.append(stripped_line.removeprefix("PRE ").rstrip("/"))

    return prefixes


def read_s3_text(s3_path: str, runner: Runner = default_runner) -> str:
    return runner(["aws", "s3", "cp", s3_path, "-"])
