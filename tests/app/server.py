"""Serve an ASGI app with uvicorn on a free loopback port, for HTTP tests.

The socket is bound before the server starts (port 0, so the OS picks a free
port), which leaves no window for another process to take the port.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import uvicorn


class ServerDidNotStart(RuntimeError):
    pass


@contextmanager
def serve(app, *, timeout: float = 10.0) -> Iterator[str]:
    """Yield the base URL of `app` running in a background thread."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="critical", lifespan="on"))
    exit_codes: list[object] = []

    def run() -> None:
        # uvicorn calls sys.exit() when the app fails to start; keep the code
        # instead of letting it surface as an unhandled thread exception.
        try:
            server.run(sockets=[sock])
        except SystemExit as exc:
            exit_codes.append(exc.code)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + timeout
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise ServerDidNotStart(f"the app did not start (exit codes: {exit_codes})")
            time.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout)
        sock.close()
