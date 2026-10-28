"""Scripted walkthrough of TaskFlow for the demo video. Every instruction goes through the chat.

    uv run playwright install chromium
    uv run pytest demo --headed

Writes demo/output/taskflow-demo.webm. The scenes follow the assignment's example instructions:
create, semantic search, clarification, several actions in one message, delete with confirmation,
and persistence after a refresh.

The agent is a real LLM, so wording varies between runs. Checks are collected and reported at the
end instead of stopping at the first miss, so a run always finishes and always leaves a video.
"""

import re

from playwright.sync_api import Locator, Page

REPLY_TIMEOUT_MS = 90_000  # the LLM fallback chain can be slow on free tiers
TYPE_DELAY_MS = 35  # per key, so viewers can read what is being typed


class Demo:
    def __init__(self, page: Page) -> None:
        self.page = page
        self.failed: list[str] = []
        self.input = page.locator("#chat-input")
        self.send_button = page.locator("#send-btn")
        self.bubbles = page.locator("#chat-log > div")

    def beat(self, ms: int = 1500) -> None:
        """Hold the frame so the viewer can read it."""
        self.page.wait_for_timeout(ms)

    def check(self, label: str, passed: bool, detail: str = "") -> None:
        print(f"{'ok  ' if passed else 'MISS'} {label}")
        if not passed:
            if detail:
                print(f"     assistant said: {detail!r}")
            self.failed.append(label)

    def task_count(self) -> int:
        return int(self.page.locator("#task-count").inner_text())

    def row(self, title_part: str) -> Locator:
        return self.page.locator("#task-list li").filter(has_text=re.compile(title_part, re.I))

    def row_text(self, title_part: str) -> str:
        """Text of the first matching row, or "" if none (a miss fails a check, not the run)."""
        row = self.row(title_part)
        return row.first.inner_text() if row.count() else ""

    def show(self, title_part: str) -> None:
        """Scroll a task into view and hold, so the viewer sees the change on the board."""
        row = self.row(title_part)
        if row.count():
            row.first.scroll_into_view_if_needed()
            self.beat(2000)

    def ask(self, message: str) -> str:
        """Type an instruction, send it, wait for the assistant, and return its reply."""
        self.input.click()
        self.input.press_sequentially(message, delay=TYPE_DELAY_MS)
        self.beat(500)
        self.send_button.click()
        # The form disables itself while the agent works and re-enables when the reply arrives.
        self.page.wait_for_function(
            "() => !document.getElementById('send-btn').disabled", timeout=REPLY_TIMEOUT_MS
        )
        reply = self.bubbles.last.inner_text()
        print(f"you> {message}\nbot> {reply}")
        self.beat(2500)
        return reply


def test_taskflow_demo(page: Page) -> None:
    demo = Demo(page)
    page.goto("/")
    page.locator("#task-list li").first.wait_for()
    demo.beat(3000)
    start_count = demo.task_count()
    demo.check("board shows the seeded tasks", start_count >= 20)

    # 1. Create a task from a sentence: priority and due date are understood.
    demo.ask("Create a high-priority task to submit the expense report tomorrow")
    demo.check("create: new task on the board", demo.task_count() == start_count + 1)
    demo.check("create: priority is High", "High" in demo.row_text("expense report"))
    demo.check("create: due tomorrow", "Tomorrow" in demo.row_text("expense report"))
    page.get_by_role("link", name="View updated task board").last.click()
    demo.show("expense report")

    # 2. Semantic search: "documentation" also finds tasks that never use the word.
    reply = demo.ask("Show all unfinished tasks related to documentation")
    demo.check("search: reply mentions documentation tasks", "documentation" in reply.lower())

    # 3. Ambiguity: two demo tasks exist. If the agent asks, answer it; either way the video task
    #    must end up In progress.
    reply = demo.ask("Move the demo task to In progress")
    if "?" in reply:
        demo.ask("The video one")
    demo.check(
        "move: Record demo video is In progress",
        "In progress" in demo.row_text("demo video"),
    )
    demo.show("demo video")

    # 4. Several instructions in one message.
    reply = demo.ask(
        "Mark the passport task as done and change the priority of the electricity bill to Low"
    )
    demo.check("multi: passport is Done", "Done" in demo.row_text("passport"), reply)
    demo.check("multi: electricity bill is Low", "Low" in demo.row_text("electricity"), reply)
    demo.show("passport")
    demo.show("electricity")

    # 5. Delete asks for confirmation first, then removes the task.
    before_delete = demo.task_count()
    reply = demo.ask("Delete the groceries task")
    asked = "?" in reply or "confirm" in reply.lower() or "sure" in reply.lower()
    demo.check("delete: agent asks to confirm", asked, reply)
    demo.check("delete: nothing removed before confirming", demo.task_count() == before_delete)
    demo.ask("yes")
    demo.check("delete: task removed after confirming", demo.task_count() == before_delete - 1)
    demo.check("delete: groceries row gone", demo.row("groceries").count() == 0)

    # 6. Persistence: a refresh keeps every change.
    count_before_reload = demo.task_count()
    page.reload()
    page.locator("#task-list li").first.wait_for()
    demo.beat(3000)
    demo.check("reload: same number of tasks", demo.task_count() == count_before_reload)
    demo.check("reload: created task still there", demo.row("expense report").count() == 1)
    demo.check("reload: groceries still gone", demo.row("groceries").count() == 0)

    assert not demo.failed, f"checks that did not pass: {demo.failed}"
