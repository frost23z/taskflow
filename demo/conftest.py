"""Fixtures for the demo recording: a throwaway TaskFlow server and a 720p recorded browser.

The server gets its own temporary SQLite file and Chroma index, seeded with the demo tasks, so a
run never touches your real data. It reads LLM keys from your normal .env, so the chat is real.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.sync_api import Page

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "output"
VIDEO_SIZE = {"width": 1280, "height": 720}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def base_url() -> Iterator[str]:
    """Start the app on a free port with a temporary, freshly seeded database."""
    data_dir = Path(tempfile.mkdtemp(prefix="taskflow-demo-"))
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{data_dir / 'tasks.db'}",
        "CHROMA_PATH": str(data_dir / "chroma"),
        "APP_ENV": "prod",  # no SQL logging in the server output
    }
    subprocess.run([sys.executable, "-m", "app.seed"], cwd=ROOT, env=env, check=True)

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                urllib.request.urlopen(f"{url}/api/tasks", timeout=2)
                break
            except urllib.error.URLError, ConnectionError:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("TaskFlow server did not start") from None
                time.sleep(0.5)
        yield url
    finally:
        server.terminate()
        server.wait(timeout=10)
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args: dict) -> dict:
    return {
        **browser_context_args,
        "viewport": VIDEO_SIZE,
        "record_video_size": VIDEO_SIZE,
        "color_scheme": "light",
    }


@pytest.fixture(autouse=True)
def save_demo_video(page: Page) -> Iterator[None]:
    """Copy the recording to demo/output/taskflow-demo.webm once the test is over."""
    yield
    video = page.video
    page.close()  # the video file is only finished when the page is closed
    if video:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        video.save_as(OUTPUT / "taskflow-demo.webm")
