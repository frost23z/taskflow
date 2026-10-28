# Demo script

The exact messages to record, in order. They are the same scenes that `demo/test_demo.py` plays
automatically, so a manual recording and the scripted one match.

## Before you start

```bash
uv run python -m app.seed   # 24 demo tasks (only inserted if the table is empty)
uv run fastapi dev          # http://127.0.0.1:8000
```

Use a fresh database so the board starts with the seeded tasks and nothing else. Wait for the board
to load before you type the first message. To record it automatically instead, run
`uv run pytest demo --headed` (see the README).

## Prompts

Type each one into the chat, send it, and give the viewer a moment to read the reply and see the
board change.

| # | Type this | What to point out |
| --- | --- | --- |
| 1 | `Create a high-priority task to submit the expense report tomorrow` | A new task appears with priority High and due date Tomorrow. The task count goes up by one |
| 2 | `Show all unfinished tasks related to documentation` | Semantic search: it also finds tasks that never use the word "documentation", and leaves out done ones |
| 3 | `Move the demo task to In progress` | Two demo tasks exist, so nothing changes and the agent asks which one |
| 3b | `The video one` | "Record demo video" moves to In progress |
| 4 | `Mark the passport task as done and change the priority of the electricity bill to Low` | Two actions from one message: passport is Done, electricity bill is Low |
| 5 | `Delete the groceries task` | The agent asks for confirmation and nothing is removed yet |
| 5b | `yes` | The task is deleted and the count goes down by one |

Then refresh the page. The board keeps every change: the new task, the status and priority edits,
and the deleted task staying gone.

## Notes

- The agent is a real LLM, so the wording of replies differs between runs. What must hold is the
  state of the board, not the exact sentence.
- If the first model is rate limited you may see a longer pause on one message. The fallback chain
  moves to the next model on its own, and the server log shows a warning for each fallback.
- Message 3 is the clarification case. If the agent acts without asking, rerun on a fresh database,
  because the "Record demo video" and "Finish demo slides" tasks must both still exist.
