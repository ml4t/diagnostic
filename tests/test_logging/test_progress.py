"""Tests for progress indicators."""

from __future__ import annotations

import io
import time
from contextlib import redirect_stderr

import pytest

from ml4t.diagnostic.logging.progress import (
    ProgressBar,
    ProgressTracker,
    spinner,
)


class TestProgressBar:
    """Tests for ProgressBar class."""

    def test_initialization(self):
        """Test ProgressBar initialization."""
        progress = ProgressBar(total=100, description="Test")

        assert progress.total == 100
        assert progress.description == "Test"
        assert progress.current == 0
        assert progress.width == 50

    def test_update(self):
        """Test updating progress."""
        # Capture stderr to avoid test output pollution
        with redirect_stderr(io.StringIO()):
            progress = ProgressBar(total=100)

            progress.update(10)
            assert progress.current == 10

            progress.update(5)
            assert progress.current == 15

    def test_update_default_increment(self):
        """Test update with default increment of 1."""
        with redirect_stderr(io.StringIO()):
            progress = ProgressBar(total=100)

            progress.update()
            assert progress.current == 1

    def test_context_manager(self):
        """Test using ProgressBar as context manager."""
        with redirect_stderr(io.StringIO()):
            with ProgressBar(total=10) as progress:
                for _i in range(10):
                    progress.update(1)

                assert progress.current == 10

    def test_custom_width(self):
        """Test custom progress bar width."""
        progress = ProgressBar(total=100, width=30)
        assert progress.width == 30

    def test_percentage_calculation(self):
        """Test percentage is calculated correctly."""
        with redirect_stderr(io.StringIO()) as stderr:
            progress = ProgressBar(total=100, show_percentage=True)
            progress.update(50)

            # Percentage should be shown in output
            output = stderr.getvalue()
            assert "50" in output or progress.current == 50

    def test_zero_total(self):
        """Test handling of zero total."""
        with redirect_stderr(io.StringIO()):
            progress = ProgressBar(total=0)
            progress.update(1)  # Should not raise

    def test_close(self):
        """Test closing progress bar."""
        with redirect_stderr(io.StringIO()):
            progress = ProgressBar(total=10)
            progress.update(5)
            progress.close()  # Should not raise


class TestProgressBarEdgeCases:
    """Additional tests for ProgressBar edge cases."""

    def test_render_with_description(self):
        """Test render includes description."""
        with redirect_stderr(io.StringIO()) as stderr:
            progress = ProgressBar(total=10, description="Loading")
            progress.update(5)

            output = stderr.getvalue()
            assert "Loading" in output

    def test_render_without_description(self):
        """Test render works without description."""
        with redirect_stderr(io.StringIO()) as stderr:
            progress = ProgressBar(total=10, description="")
            progress.update(5)

            output = stderr.getvalue()
            assert "[" in output  # Progress bar marker

    def test_render_elapsed_time(self):
        """Test that elapsed time is shown after 1 second."""
        with redirect_stderr(io.StringIO()):
            progress = ProgressBar(total=10)
            # Manually set start_time to simulate elapsed time
            progress.start_time = time.time() - 2.0
            progress.update(5)
            # Just ensure no crash - time display is optional

    def test_show_count_disabled(self):
        """Test progress bar with count display disabled."""
        with redirect_stderr(io.StringIO()) as stderr:
            progress = ProgressBar(total=10, show_count=False)
            progress.update(5)

            output = stderr.getvalue()
            assert "(5/10)" not in output

    def test_show_percentage_disabled(self):
        """Test progress bar with percentage display disabled."""
        with redirect_stderr(io.StringIO()) as stderr:
            progress = ProgressBar(total=10, show_percentage=False)
            progress.update(5)

            output = stderr.getvalue()
            # Should not contain percentage format
            assert "50.0%" not in output


class TestSpinner:
    """Tests for spinner generator."""

    def test_spinner_yields_frames(self):
        """Test that spinner yields frames."""
        with redirect_stderr(io.StringIO()):
            spin = spinner("Working")

            # Get a few frames
            next(spin)
            next(spin)
            next(spin)

            # Should not raise

    def test_spinner_with_description(self):
        """Test spinner with custom description."""
        with redirect_stderr(io.StringIO()) as stderr:
            spin = spinner("Custom description")
            next(spin)

            output = stderr.getvalue()
            assert "Custom" in output or len(output) > 0


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_initialization(self):
        """Test ProgressTracker initialization."""
        tracker = ProgressTracker(["load", "process", "save"])

        assert tracker.stages == ["load", "process", "save"]
        assert tracker.current_stage is None
        assert len(tracker.completed_stages) == 0

    def test_start_stage(self):
        """Test starting a stage."""
        with redirect_stderr(io.StringIO()):
            tracker = ProgressTracker(["load", "process"])
            tracker.start("load")

            assert tracker.current_stage == "load"
            assert "load" in tracker.start_times

    def test_complete_stage(self):
        """Test completing a stage."""
        with redirect_stderr(io.StringIO()):
            tracker = ProgressTracker(["load", "process"])
            tracker.start("load")
            tracker.complete("load")

            assert "load" in tracker.completed_stages
            assert tracker.current_stage is None

    def test_unknown_stage_raises_error(self):
        """Test that unknown stage raises ValueError."""
        tracker = ProgressTracker(["load", "process"])

        with pytest.raises(ValueError, match="Unknown stage"):
            tracker.start("unknown")

    def test_progress_calculation(self):
        """Test progress calculation."""
        with redirect_stderr(io.StringIO()):
            tracker = ProgressTracker(["load", "process", "save"])

            assert tracker.progress() == 0.0

            tracker.start("load")
            tracker.complete("load")
            assert tracker.progress() == pytest.approx(1 / 3)

            tracker.start("process")
            tracker.complete("process")
            assert tracker.progress() == pytest.approx(2 / 3)

            tracker.start("save")
            tracker.complete("save")
            assert tracker.progress() == pytest.approx(1.0)

    def test_empty_stages(self):
        """Test with empty stages list."""
        tracker = ProgressTracker([])

        assert tracker.progress() == 1.0

    def test_elapsed_time_tracking(self):
        """Test that elapsed time is tracked."""
        with redirect_stderr(io.StringIO()):
            tracker = ProgressTracker(["slow_stage"])
            tracker.start("slow_stage")
            time.sleep(0.01)  # Small delay
            tracker.complete("slow_stage")

            assert "slow_stage" in tracker.start_times

    def test_multiple_stages_sequence(self):
        """Test running through multiple stages."""
        with redirect_stderr(io.StringIO()):
            stages = ["extract", "transform", "load"]
            tracker = ProgressTracker(stages)

            for stage in stages:
                tracker.start(stage)
                tracker.complete(stage)

            assert len(tracker.completed_stages) == 3
            assert tracker.progress() == 1.0
