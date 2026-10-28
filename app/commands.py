"""Slash commands offered by the chat box: typing `/` shows this list.

For now there is one, /clear. It is not sent to the agent: app.main handles it before the model
(see app.chat), so it is deterministic and costs no tokens.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str  # what the user types after the slash
    summary: str  # one line shown in the picker


COMMANDS: tuple[Command, ...] = (Command("clear", "Clear this chat (your tasks stay)"),)
