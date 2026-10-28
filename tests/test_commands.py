from app import commands


def test_clear_is_the_only_command() -> None:
    assert [command.name for command in commands.COMMANDS] == ["clear"]
