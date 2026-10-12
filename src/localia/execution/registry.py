from __future__ import annotations

from collections.abc import Iterable

from localia.execution.executors import Executor
from localia.jev import Route


class ExecutorRegistry:
    """Registry of concrete route executors."""

    def __init__(self, executors: Iterable[Executor] = ()) -> None:
        self._executors: dict[Route, Executor] = {}
        for executor in executors:
            self.register(executor)

    def register(self, executor: Executor) -> None:
        route = executor.route
        if route in self._executors:
            raise ValueError(f"duplicate executor for route: {route.value}")
        self._executors[route] = executor

    def get(self, route: Route) -> Executor:
        try:
            return self._executors[route]
        except KeyError as exc:
            raise KeyError(f"no executor registered for route: {route.value}") from exc

    def has(self, route: Route) -> bool:
        return route in self._executors

    def routes(self) -> tuple[Route, ...]:
        return tuple(sorted(self._executors, key=lambda item: item.value))
