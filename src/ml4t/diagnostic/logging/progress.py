"""
Progress Indicators for Long-Running Operations

Provides progress bars and indicators for batch operations.
"""

import sys
import time
from collections.abc import Iterable, Sized
from contextlib import contextmanager
from typing import Any, cast


class ProgressBar:
    """
    Simple progress bar for terminal output.

    Example:
        >>> progress = ProgressBar(total=100, description="Processing")
        >>> for i in range(100):
        ...     progress.update(1)
        >>> progress.close()
    """

    def __init__(
        self,
        total: int,
        description: str = "",
        width: int = 50,
        show_percentage: bool = True,
        show_count: bool = True,
    ):
        """
        Initialize progress bar.

        Args:
            total: Total number of items
            description: Progress description
            width: Width of progress bar in characters
            show_percentage: Show percentage complete
            show_count: Show current/total count
        """
        self.total = total
        self.description = description
        self.width = width
        self.show_percentage = show_percentage
        self.show_count = show_count
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """
        Update progress by n items.

        Args:
            n: Number of items processed
        """
        self.current += n
        self._render()

    def _render(self):
        """Render progress bar to terminal."""
        # Calculate percentage
        percentage = self.current / self.total if self.total > 0 else 0

        # Build progress bar
        filled = int(self.width * percentage)
        bar = "█" * filled + "░" * (self.width - filled)

        # Build status text
        parts = []
        if self.description:
            parts.append(self.description)

        parts.append(f"[{bar}]")

        if self.show_percentage:
            parts.append(f"{percentage * 100:.1f}%")

        if self.show_count:
            parts.append(f"({self.current}/{self.total})")

        # Add elapsed time
        elapsed = time.time() - self.start_time
        if elapsed > 1:
            parts.append(f"{elapsed:.1f}s")

        # Write to stderr (doesn't interfere with stdout)
        sys.stderr.write("\r" + " ".join(parts))
        sys.stderr.flush()

    def close(self):
        """Close progress bar and move to next line."""
        sys.stderr.write("\n")
        sys.stderr.flush()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@contextmanager
def progress_indicator(iterable: Iterable[Any], total: int | None = None, description: str = ""):
    """
    Context manager for iterating with progress indicator.

    Args:
        iterable: Items to iterate over
        total: Total number of items (or len(iterable))
        description: Progress description

    Yields:
        Items from iterable with progress updates

    Example:
        >>> items = range(100)
        >>> with progress_indicator(items, description="Processing") as progress:
        ...     for item in progress:
        ...         process(item)
    """
    # Try to get length if not provided
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(cast(Sized, iterable))
        else:
            total = 0  # Unknown length

    progress = ProgressBar(total=total, description=description)

    try:
        for item in iterable:
            yield item
            progress.update(1)
    finally:
        progress.close()


def spinner(description: str = "Working"):
    """
    Simple spinner for indefinite operations.

    Args:
        description: Operation description

    Example:
        >>> spin = spinner("Computing")
        >>> next(spin)  # Show next frame
        >>> next(spin)
        >>> # When done, just stop calling next()
    """
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0

    while True:
        frame = frames[i % len(frames)]
        sys.stderr.write(f"\r{frame} {description}")
        sys.stderr.flush()
        i += 1
        yield


class ProgressTracker:
    """
    Track progress across multiple stages.

    Example:
        >>> tracker = ProgressTracker(["load", "process", "save"])
        >>> tracker.start("load")
        >>> # ... loading ...
        >>> tracker.complete("load")
        >>> tracker.start("process")
        >>> # ... processing ...
        >>> tracker.complete("process")
    """

    def __init__(self, stages: list[str]):
        """
        Initialize progress tracker.

        Args:
            stages: List of stage names
        """
        self.stages = stages
        self.current_stage: str | None = None
        self.completed_stages: set[str] = set()
        self.start_times: dict[str, float] = {}

    def start(self, stage: str):
        """
        Start a stage.

        Args:
            stage: Stage name
        """
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}")

        self.current_stage = stage
        self.start_times[stage] = time.time()

        # Display progress
        current_idx = self.stages.index(stage)
        total = len(self.stages)
        print(f"[{current_idx + 1}/{total}] Starting: {stage}", file=sys.stderr)

    def complete(self, stage: str):
        """
        Mark stage as complete.

        Args:
            stage: Stage name
        """
        self.completed_stages.add(stage)
        self.current_stage = None

        # Display completion
        if stage in self.start_times:
            elapsed = time.time() - self.start_times[stage]
            print(f"✓ Completed: {stage} ({elapsed:.2f}s)", file=sys.stderr)

    def progress(self) -> float:
        """
        Get overall progress (0.0 to 1.0).

        Returns:
            Progress fraction
        """
        if not self.stages:
            return 1.0
        return len(self.completed_stages) / len(self.stages)
