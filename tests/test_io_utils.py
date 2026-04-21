from src.common.io_utils import list_s3_prefixes


def test_list_s3_prefixes_parses_aws_cli_output():
    seen_commands = []
    raw_prefix = "s3://example-bucket/raw/"

    def runner(command: list[str]) -> str:
        seen_commands.append(command)
        return "                           PRE corridor01/\n                           PRE final_challenge_ugv1/\n"

    prefixes = list_s3_prefixes(raw_prefix, runner=runner)

    assert prefixes == ["corridor01", "final_challenge_ugv1"]
    assert seen_commands == [["aws", "s3", "ls", raw_prefix]]
