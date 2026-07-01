"""Progress reporting for long-running harness jobs (SSE / UI)."""

from __future__ import annotations

import threading
from typing import Callable

Callback = Callable[[str, str], None]

_lock = threading.Lock()
_callbacks: list[Callback] = []


def register_callback(callback: Callback) -> None:
    with _lock:
        if callback not in _callbacks:
            _callbacks.append(callback)


def unregister_callback(callback: Callback) -> None:
    with _lock:
        if callback in _callbacks:
            _callbacks.remove(callback)


def report(step: str, detail: str = "") -> None:
    with _lock:
        listeners = list(_callbacks)
    for cb in listeners:
        try:
            cb(step, detail)
        except Exception:
            pass
