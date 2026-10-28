"""Scripted walkthrough of TaskFlow for the demo video. Every instruction goes through the chat.

    uv run playwright install chromium
    uv run pytest demo --headed

Writes demo/output/taskflow-demo.webm. It starts from the six-task board of the assignment's UI
reference and uses the assignment's own example instructions: create, semantic search, several
actions in one message, a priority change, the / command picker, delete with its confirmation, and
persistence after a refresh. When an instruction is ambiguous the agent asks, the script answers.

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

    def type(self, text: str) -> None:
        self.input.click()
        self.input.press_sequentially(text, delay=TYPE_DELAY_MS)

    def send_and_wait(self, shown: str, *, press_enter: bool = False) -> str:
        """Send what is typed, wait for the assistant, and return its reply."""
        self.beat(500)
        if press_enter:
            self.input.press("Enter")
        else:
            self.send_button.click()
        # The form disables itself while the agent works and re-enables when the reply arrives.
        self.page.wait_for_function(
            "() => !document.getElementById('send-btn').disabled", timeout=REPLY_TIMEOUT_MS
        )
        reply = self.bubbles.last.inner_text()
        print(f"you> {shown}\nbot> {reply}")
        self.beat(2500)
        return reply

    def ask(self, message: str) -> str:
        """Type an instruction, send it, wait for the assistant, and return its reply."""
        self.type(message)
        return self.send_and_wait(message)

    def show_picker(self) -> None:
        """Type `/` so the viewer sees the command list, then leave it again."""
        self.type("/")
        self.page.locator("#slash-menu").wait_for()
        self.beat(2500)
        self.input.press("Escape")
        self.input.fill("")
        self.beat(500)

    def clear_chat(self) -> str:
        """Pick /clear from the picker and send it."""
        self.type("/cl")
        self.beat(1000)
        self.input.press("Enter")  # chooses /clear
        return self.send_and_wait("/clear", press_enter=True)


def test_taskflow_demo(page: Page) -> None:
    demo = Demo(page)
    page.goto("/")
    page.locator("#task-list li").first.wait_for()
    demo.beat(3000)
    start_count = demo.task_count()
    demo.check("board starts with the six reference tasks", start_count == 6)

    # 1. Create a task from a sentence: priority and due date are understood.
    demo.ask("Create a high-priority task to finish the demo tomorrow")
    demo.check("create: new task on the board", demo.task_count() == start_count + 1)
    demo.check("create: priority is High", "High" in demo.row_text("demo"))
    demo.check("create: due tomorrow", "Tomorrow" in demo.row_text("demo"))
    page.get_by_role("link", name="View updated task board").last.click()
    demo.show("demo")

    # 2. Semantic search: unfinished tasks only, found by meaning rather than by exact words.
    reply = demo.ask("Show all unfinished tasks related to documentation")
    demo.check("search: finds the API documentation task", "api documentation" in reply.lower())

    # 3. Two actions in one message. "The demo task" should be the one just created; if the agent
    #    is unsure it asks, and the script answers.
    reply = demo.ask("Move the demo task to In progress and mark the weekly report Done")
    if "?" in reply:
        demo.ask("The one I just created")
    demo.check("move: demo task is In progress", "In progress" in demo.row_text("demo"))
    demo.check("move: weekly report is Done", "Done" in demo.row_text("weekly report"))
    demo.show("demo")
    demo.show("weekly report")

    # 4. "The API task" can mean the documentation task or the pull request about API endpoints.
    #    If the agent asks which one, the script answers.
    reply = demo.ask("Change the priority of the API task to Low")
    if "?" in reply:
        demo.ask("The documentation one")
    demo.check("priority: API documentation is Low", "Low" in demo.row_text("api documentation"))
    demo.show("api documentation")

    # 5. The / picker, then delete: the agent asks for confirmation before removing anything.
    demo.show_picker()
    before_delete = demo.task_count()
    reply = demo.ask("Delete the README task")
    asked = "?" in reply or "confirm" in reply.lower() or "sure" in reply.lower()
    demo.check("delete: agent asks to confirm", asked, reply)
    demo.check("delete: nothing removed before confirming", demo.task_count() == before_delete)
    demo.ask("yes")
    demo.check("delete: task removed after confirming", demo.task_count() == before_delete - 1)
    demo.check("delete: README row gone", demo.row("readme").count() == 0)

    # 6. Persistence: a refresh keeps every change.
    count_before_reload = demo.task_count()
    page.reload()
    page.locator("#task-list li").first.wait_for()
    demo.beat(3000)
    demo.check("reload: same number of tasks", demo.task_count() == count_before_reload)
    demo.check("reload: created task still there", demo.row("demo").count() == 1)
    demo.check("reload: README still gone", demo.row("readme").count() == 0)

    # 7. /clear empties the chat (and only the chat): the board keeps its tasks.
    demo.clear_chat()
    demo.check("clear: only the confirmation is left in the chat", demo.bubbles.count() == 1)
    demo.check("clear: tasks untouched", demo.task_count() == count_before_reload)

    assert not demo.failed, f"checks that did not pass: {demo.failed}"
