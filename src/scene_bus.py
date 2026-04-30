"""
scene_bus.py — Lightweight publish/subscribe event bus

All modules communicate through named topics.  The bus is the single
source of truth for inter-module messaging and keeps every component
decoupled from one another.

Topics used in this project:
  "scene_state"   payload: (state: dict, snapshot_path: str)
  "task_result"   payload: (success: bool, context: str)
  "plan_ready"    payload: (plan: dict)

Usage:
    bus = SceneBus()

    # Publisher
    bus.publish("scene_state", state_dict, "/path/to/snapshot.jpg")

    # Subscriber
    def on_scene(state, snapshot_path):
        ...
    bus.subscribe("scene_state", on_scene)
"""

import threading
from collections import defaultdict
from typing import Callable


class SceneBus:
    """
    Thread-safe in-process publish/subscribe event bus.

    Subscribers are called synchronously from the publisher's thread.
    If you need the subscriber to not block the publisher (e.g. for
    long-running LLM calls) the subscriber itself should spawn a thread.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, callback: Callable) -> None:
        """Register *callback* to be called whenever *topic* is published."""
        with self._lock:
            self._subscribers[topic].append(callback)
        print(f"[SceneBus] Subscribed to '{topic}': {callback.__qualname__}")

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Remove a previously registered callback."""
        with self._lock:
            try:
                self._subscribers[topic].remove(callback)
            except ValueError:
                pass

    def publish(self, topic: str, *args, **kwargs) -> int:
        """
        Call all subscribers registered for *topic*.

        Returns the number of subscribers that were called.
        Subscriber exceptions are caught and printed but do NOT propagate.
        """
        with self._lock:
            callbacks = list(self._subscribers.get(topic, []))

        for cb in callbacks:
            try:
                cb(*args, **kwargs)
            except Exception as exc:
                print(f"[SceneBus] ERROR — subscriber {cb.__qualname__} "
                      f"raised on topic '{topic}': {exc}")

        return len(callbacks)

    def topics(self) -> list[str]:
        """Return a list of topics that have at least one subscriber."""
        with self._lock:
            return [t for t, subs in self._subscribers.items() if subs]
