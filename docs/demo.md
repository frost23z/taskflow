# Demo script

The exact messages to record, in order. They are the same scenes that `demo/test_demo.py` plays
automatically, so a manual recording and the scripted one match. All of them except the last two
come straight from the assignment's example instructions.

## Before you start

```bash
uv run python -m app.seed --demo   # the six tasks of the UI reference (only inserted if empty)
uv run fastapi dev                 # http://127.0.0.1:8000
```

Use a fresh database so the board starts with exactly six tasks. Record the browser at about
1440x900 so the whole board stays visible while it changes: no scrolling during the video. Wait for
the board to load before you type the first message. To record it automatically instead, run
`uv run pytest demo --headed` (see the README).

## Prompts

Type each one into the chat, send it, and give the viewer a moment to read the reply and see the
board change. Rows that the chat just created or changed flash on the board.

| # | Type this | What to point out |
| --- | --- | --- |
| 1 | `Create a high-priority task to finish the demo tomorrow` | A new task appears with priority High and due date Tomorrow. The count goes from 6 to 7 |
| 2 | `Show all unfinished tasks related to documentation` | Semantic search: it finds "Update API documentation" and leaves out the done README task |
| 3 | `Move the demo task to In progress and mark the weekly report Done` | Two actions from one message. Both rows flash. If the agent asks which task you mean, answer `The one I just created` |
| 4 | `Change the priority of the API task to Low` | "The API task" can mean the documentation task or the pull request about API endpoints. If the agent asks, answer `The documentation one`; otherwise the priority just changes |
| 5 | Type `/`, then Esc | The command picker lists `/clear`. Close it again |
| 5b | `Delete the README task` | The agent asks for confirmation. Nothing is removed yet |
| 5c | `yes` | The task is deleted and the count goes down by one |

Then refresh the page. The board keeps every change, and the chat history is restored.

Two more things worth showing if you have the time: type a long instruction to see the chat box
grow to three lines and then scroll (Shift+Enter adds a line), and finish with `/clear` from the picker to empty the chat. The tasks stay.

## Notes

- The agent is a real LLM, so the wording of replies differs between runs. What must hold is the
  state of the board, not the exact sentence.
- Scenes 3 and 4 are the clarification cases. Whether the agent asks depends on how close the
  matching tasks score, so you may see either. The safeguard that never varies is scene 5: a
  delete always asks for confirmation first.
- If the first model is rate limited you may see a longer pause on one message. The fallback chain
  moves to the next model on its own, and the server log shows a warning for each fallback.
