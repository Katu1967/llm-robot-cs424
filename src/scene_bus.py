
import threading
from collections import defaultdict
from typing import Callable


class SceneBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._latest: dict[str, tuple] = {}

    def subscribe(self, topic: str, callback: Callable) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

        print(f"[SceneBus] subscribed {topic} -> {callback.__qualname__}")

    def unsubscribe(self, topic: str, callback: Callable) -> None:
        with self._lock:
            try:
                self._subscribers[topic].remove(callback)
            except ValueError:
                pass

    def publish(self, topic: str, *args, **kwargs) -> int:
        with self._lock:
            callbacks_snapshot = list(self._subscribers.get(topic, []))

        for subscriber_callback in callbacks_snapshot:
            try:
                subscriber_callback(*args, **kwargs)
            except Exception as exc:
                print(f"[SceneBus] subscriber error on {topic}: {subscriber_callback.__qualname__}: {exc}")

        with self._lock:
            try:
                self._latest[topic] = (args, kwargs)
            except Exception:
                pass

        return len(callbacks_snapshot)

    def get_latest(self, topic: str):
        with self._lock:
            cached_entry = self._latest.get(topic)

        if not cached_entry:
            return None

        published_args, published_kwargs = cached_entry
        return published_args[0] if published_args else None

    def topics(self) -> list[str]:
        with self._lock:
            return [topic for topic, subs in self._subscribers.items() if subs]
