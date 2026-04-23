"""Pluggable progress-bar backend (Rich or tqdm).

Rich progress bars render in-place and are invisible to non-interactive
contexts (e.g. AI agents running in the background).  Switching to tqdm
produces line-by-line output that is captured properly.

Usage::

    from src.core.io.progress import create_progress

    with create_progress() as progress:
        task = progress.add_task("Working...", total=100)
        for i in range(100):
            progress.advance(task)

The backend is selected via :func:`set_default_backend` (called once at
startup from the Hydra config ``logging.progress_backend``).
"""

from __future__ import annotations

import re
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ---------------------------------------------------------------------------
# Module-level default
# ---------------------------------------------------------------------------

_default_backend: str = "rich"


def set_default_backend(backend: str) -> None:
    """Set the global progress-bar backend (``"rich"`` or ``"tqdm"``)."""
    global _default_backend
    _default_backend = backend


# ---------------------------------------------------------------------------
# Rich backend
# ---------------------------------------------------------------------------

_DEFAULT_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TimeRemainingColumn(),
)

_TRAINING_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)


class RichProgressManager:
    """Thin wrapper around :class:`rich.progress.Progress`."""

    def __init__(
        self,
        *,
        transient: bool = False,
        console: Console | None = None,
        columns: str | None = None,
        expand: bool = False,
    ) -> None:
        kwargs: dict[str, Any] = {"transient": transient, "expand": expand}
        if console is not None:
            kwargs["console"] = console

        if columns == "training":
            cols = _TRAINING_COLUMNS
        else:
            cols = _DEFAULT_COLUMNS

        self._progress = Progress(*cols, **kwargs)

    # -- context manager --------------------------------------------------

    def __enter__(self) -> RichProgressManager:
        self._progress.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._progress.__exit__(*args)

    def start(self) -> None:
        self._progress.start()

    def stop(self) -> None:
        self._progress.stop()

    # -- task management ---------------------------------------------------

    def add_task(
        self,
        description: str,
        total: float | None = None,
        visible: bool = True,
    ) -> int:
        return self._progress.add_task(description, total=total, visible=visible)

    def update(
        self,
        task_id: int,
        *,
        advance: float = 0,
        completed: float | None = None,
        description: str | None = None,
        total: float | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if advance:
            kwargs["advance"] = advance
        if completed is not None:
            kwargs["completed"] = completed
        if description is not None:
            kwargs["description"] = description
        if total is not None:
            kwargs["total"] = total
        self._progress.update(task_id, **kwargs)

    def advance(self, task_id: int, amount: float = 1) -> None:
        self._progress.advance(task_id, amount)

    def remove_task(self, task_id: int) -> None:
        self._progress.remove_task(task_id)

    # -- output ------------------------------------------------------------

    def print(self, message: str) -> None:
        self._progress.console.print(message)


# ---------------------------------------------------------------------------
# tqdm backend
# ---------------------------------------------------------------------------

_RICH_MARKUP_RE = re.compile(r"\[/?[a-zA-Z_.# ]+\]")


class TqdmProgressManager:
    """Progress manager backed by :mod:`tqdm`.

    Each *task* maps to an independent ``tqdm`` progress bar that prints
    line-by-line output — suitable for non-interactive / background contexts.
    """

    def __init__(self, **_kwargs: Any) -> None:
        from tqdm import tqdm  # noqa: F401 — validate availability

        self._bars: dict[int, Any] = {}
        self._next_id: int = 0

    # -- context manager (no-op) ------------------------------------------

    def __enter__(self) -> TqdmProgressManager:
        return self

    def __exit__(self, *args: object) -> None:
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()

    # -- task management ---------------------------------------------------

    def add_task(
        self,
        description: str,
        total: float | None = None,
        visible: bool = True,
    ) -> int:
        from tqdm import tqdm

        task_id = self._next_id
        self._next_id += 1
        self._bars[task_id] = tqdm(
            total=total,
            desc=description,
            disable=not visible,
        )
        return task_id

    def update(
        self,
        task_id: int,
        *,
        advance: float = 0,
        completed: float | None = None,
        description: str | None = None,
        total: float | None = None,
    ) -> None:
        bar = self._bars.get(task_id)
        if bar is None:
            return
        if total is not None:
            bar.total = total
        if description is not None:
            bar.set_description(description)
        if completed is not None:
            bar.n = completed
            bar.refresh()
        elif advance:
            bar.update(advance)

    def advance(self, task_id: int, amount: float = 1) -> None:
        bar = self._bars.get(task_id)
        if bar is not None:
            bar.update(amount)

    def remove_task(self, task_id: int) -> None:
        bar = self._bars.pop(task_id, None)
        if bar is not None:
            bar.close()

    # -- output ------------------------------------------------------------

    def print(self, message: str) -> None:
        from tqdm import tqdm

        clean = _RICH_MARKUP_RE.sub("", message)
        tqdm.write(clean)


# ---------------------------------------------------------------------------
# Type alias (for function signatures that accept either backend)
# ---------------------------------------------------------------------------

ProgressManager = RichProgressManager | TqdmProgressManager


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_progress(
    *,
    backend: str | None = None,
    transient: bool = False,
    console: Console | None = None,
    columns: str | None = None,
    expand: bool = False,
) -> ProgressManager:
    """Create a progress manager using the configured backend.

    Args:
        backend: ``"rich"`` or ``"tqdm"``.  Defaults to the value set by
            :func:`set_default_backend` (which itself defaults to ``"rich"``).
        transient: If ``True``, the progress display is removed after
            completion (Rich only).
        console: Rich console instance (Rich only).
        columns: Column preset name — ``"training"`` for the training-loop
            layout, or ``None`` for the default layout (Rich only).
        expand: Expand the progress bar to fill the terminal (Rich only).
    """
    backend = backend or _default_backend
    if backend == "tqdm":
        return TqdmProgressManager()
    return RichProgressManager(
        transient=transient,
        console=console,
        columns=columns,
        expand=expand,
    )
