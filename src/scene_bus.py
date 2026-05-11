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
        self._topic_subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._thread_lock = threading.Lock()
        self._latest_payloads: dict[str, tuple] = {}


    def subscribe(self, topic: str, callback: Callable) -> None:
        """Register *callback* to be called whenever *topic* is published."""
        with self._thread_lock:
            self._topic_subscribers[topic].append(callback)
        print(f"[SceneBus] Subscribed to '{topic}': {callback.__qualname__}")

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        """Remove a previously registered callback."""
        with self._thread_lock:
            try:
                self._topic_subscribers[topic].remove(callback)
            except ValueError:
                pass

    def publish(self, topic: str, *args, **kwargs) -> int:
        """
        Call all subscribers registered for *topic*.

        Returns the number of subscribers that were called.
        Subscriber exceptions are caught and printed but do NOT propagate.
        """
        with self._thread_lock:
            callbacks = list(self._topic_subscribers.get(topic, []))

        for callback in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as exc:
                print(f"[SceneBus] ERROR — subscriber {callback.__qualname__} "
                      f"raised on topic '{topic}': {exc}")
        try:
            self._latest_payloads[topic] = (args, kwargs)
        except Exception:
            pass

        return len(callbacks)

    def get_latest(self, topic: str):
        """Return the most-recent payload published to *topic*.

        Returns the first positional argument if present (e.g. the `state` dict
        for "scene_state"), or None if nothing has been published yet.
        """
        with self._thread_lock:
            payload_tuple = self._latest_payloads.get(topic)
        if not payload_tuple:
            return None
        args, kwargs = payload_tuple
        return args[0] if args else None

    def topics(self) -> list[str]:
        """Return a list of topics that have at least one subscriber."""
        with self._thread_lock:
            return [topic for topic, subscribers in self._topic_subscribers.items() if subscribers]
