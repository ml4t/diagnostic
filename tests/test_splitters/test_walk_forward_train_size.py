"""Regression tests: WalkForwardCV honours train_size in every forward split path (#293).

``train_size`` is the training window that precedes the first test fold. The
first test fold starts at ``train_size + gap``; with ``expanding=False`` every
fold trains on exactly ``train_size`` rows or sessions before purging, and with
``expanding=True`` the first fold trains on ``train_size`` and later folds grow.
Before the fix, the default ``consecutive=False`` layout started the first test
fold at ``test_size`` in the row, session, calendar and held-out forward paths,
so the first fold trained on ``test_size`` rows and expanding runs ignored
``train_size`` entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml4t.diagnostic.splitters import WalkForwardCV

N_SPLITS = 3
TRAIN = 252
TEST = 63


def _row_splits(n_samples: int = 1000, **kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
    params = {"n_splits": N_SPLITS, "train_size": TRAIN, "test_size": TEST, "calendar": None}
    params.update(kwargs)
    cv = WalkForwardCV(**params)
    return list(cv.split(np.zeros((n_samples, 1))))


def _assert_contiguous(indices: np.ndarray, start: int, stop: int) -> None:
    np.testing.assert_array_equal(indices, np.arange(start, stop))


class TestRowPath:
    def test_expanding_first_fold_trains_on_train_size(self) -> None:
        """Prevents: expanding mode ignoring train_size (first fold trained on test_size rows)."""
        splits = _row_splits()

        train0, test0 = splits[0]
        _assert_contiguous(train0, 0, TRAIN)
        assert test0[0] == TRAIN
        previous_end = 0
        for train, test in splits:
            assert train[0] == 0
            assert train[-1] + 1 == test[0]
            assert len(test) == TEST
            assert test[0] >= previous_end
            previous_end = test[-1] + 1

    def test_rolling_every_fold_trains_on_train_size(self) -> None:
        """Prevents: the first rolling fold training on test_size rows instead of train_size."""
        splits = _row_splits(expanding=False)

        previous_end = 0
        for train, test in splits:
            _assert_contiguous(train, test[0] - TRAIN, test[0])
            assert len(test) == TEST
            assert test[0] >= previous_end
            previous_end = test[-1] + 1
        assert splits[0][1][0] == TRAIN

    @pytest.mark.parametrize("consecutive", [False, True])
    def test_gap_does_not_shorten_the_first_rolling_window(self, consecutive: bool) -> None:
        """Prevents: the gap being cut out of the first training window (train_size - gap rows)."""
        gap = 5
        splits = _row_splits(expanding=False, gap=gap, consecutive=consecutive)

        assert splits[0][1][0] == TRAIN + gap
        for train, test in splits:
            _assert_contiguous(train, test[0] - gap - TRAIN, test[0] - gap)

    def test_purging_trims_the_train_size_window(self) -> None:
        """Prevents: purging applied to a first window that starts at test_size, not train_size."""
        horizon = 5
        splits = _row_splits(expanding=False, label_horizon=horizon)

        assert splits[0][1][0] == TRAIN
        for train, test in splits:
            _assert_contiguous(train, test[0] - TRAIN, test[0] - horizon)

    def test_default_test_size_divides_data_after_training_window(self) -> None:
        """Prevents: test_size=None placing the first test fold inside the training window."""
        splits = _row_splits(n_samples=100, train_size=30, test_size=None, expanding=False)

        assert [int(test[0]) for _, test in splits] == [30, 53, 76]
        assert splits[-1][1][-1] == 99
        for train, test in splits:
            _assert_contiguous(train, test[0] - 30, test[0])

    def test_insufficient_data_raises(self) -> None:
        """Prevents: silently shrinking the first fold when train_size and folds do not fit."""
        with pytest.raises(ValueError, match="Insufficient data"):
            _row_splits(n_samples=400)


def _session_panel(n_sessions: int = 300, n_assets: int = 3) -> pd.DataFrame:
    sessions = pd.date_range("2024-01-02", periods=n_sessions, freq="B")
    return pd.DataFrame(
        {
            "session_date": np.repeat(sessions, n_assets),
            "asset": np.tile([f"A{i}" for i in range(n_assets)], n_sessions),
            "feature": 0.0,
        }
    )


def _session_positions(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    codes = pd.factorize(df["session_date"], sort=True)[0]
    return np.unique(codes[indices])


class TestSessionPath:
    TRAIN_SESSIONS = 100
    TEST_SESSIONS = 20

    def _splits(self, df: pd.DataFrame, **kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
        cv = WalkForwardCV(
            n_splits=N_SPLITS,
            train_size=self.TRAIN_SESSIONS,
            test_size=self.TEST_SESSIONS,
            align_to_sessions=True,
            calendar=None,
            **kwargs,
        )
        return list(cv.split(df))

    def test_expanding_first_fold_trains_on_train_size(self) -> None:
        """Prevents: session-aligned expanding mode ignoring train_size."""
        df = _session_panel()
        splits = self._splits(df)

        train0, test0 = splits[0]
        _assert_contiguous(_session_positions(df, train0), 0, self.TRAIN_SESSIONS)
        assert _session_positions(df, test0)[0] == self.TRAIN_SESSIONS
        for train, test in splits:
            train_s, test_s = _session_positions(df, train), _session_positions(df, test)
            _assert_contiguous(train_s, 0, test_s[0])
            assert len(test_s) == self.TEST_SESSIONS

    def test_rolling_every_fold_trains_on_train_size(self) -> None:
        """Prevents: the first session-aligned rolling fold training on test_size sessions."""
        df = _session_panel()
        splits = self._splits(df, expanding=False)

        previous_end = 0
        for train, test in splits:
            train_s, test_s = _session_positions(df, train), _session_positions(df, test)
            _assert_contiguous(train_s, test_s[0] - self.TRAIN_SESSIONS, test_s[0])
            assert test_s[0] >= previous_end
            previous_end = test_s[-1] + 1
            assert len(set(train) & set(test)) == 0


def _nyse_daily_frame(n_days: int = 300) -> pd.DataFrame:
    # Business days include NYSE holidays, which the calendar path filters out.
    index = pd.date_range("2024-01-02", periods=n_days, freq="B", tz="America/New_York")
    return pd.DataFrame({"feature": np.zeros(n_days)}, index=index)


class TestCalendarPath:
    TRAIN_SESSIONS = 100
    TEST_SESSIONS = 20

    def _splits(self, df: pd.DataFrame, **kwargs) -> list[tuple[np.ndarray, np.ndarray]]:
        cv = WalkForwardCV(
            n_splits=N_SPLITS,
            train_size=self.TRAIN_SESSIONS,
            test_size=self.TEST_SESSIONS,
            calendar="NYSE",
            **kwargs,
        )
        return list(cv.split(df))

    @staticmethod
    def _trading_positions(df: pd.DataFrame) -> pd.Series:
        cv = WalkForwardCV(calendar="NYSE")
        assert cv.calendar is not None
        _, trading = cv.calendar.get_sessions_and_mask(df.index)
        positions = np.full(len(df), -1)
        positions[np.asarray(trading)] = np.arange(int(np.asarray(trading).sum()))
        return pd.Series(positions)

    def test_expanding_first_fold_trains_on_train_size(self) -> None:
        """Prevents: the calendar path ignoring train_size in expanding mode."""
        df = _nyse_daily_frame()
        positions = self._trading_positions(df)
        splits = self._splits(df)

        train0, test0 = splits[0]
        _assert_contiguous(positions[train0].to_numpy(), 0, self.TRAIN_SESSIONS)
        assert positions[test0[0]] == self.TRAIN_SESSIONS
        for train, test in splits:
            assert train.max() < test.min()
            assert len(test) == self.TEST_SESSIONS

    def test_rolling_every_fold_trains_on_train_size(self) -> None:
        """Prevents: the first calendar rolling fold training on test_size sessions."""
        df = _nyse_daily_frame()
        positions = self._trading_positions(df)
        splits = self._splits(df, expanding=False)

        previous_end = -1
        for train, test in splits:
            train_s, test_s = positions[train].to_numpy(), positions[test].to_numpy()
            _assert_contiguous(train_s, test_s[0] - self.TRAIN_SESSIONS, test_s[0])
            assert test_s[0] > previous_end
            previous_end = test_s[-1]


class TestHeldOutForwardPath:
    def test_rolling_every_fold_trains_on_train_size(self) -> None:
        """Prevents: forward folds before a held-out test starting at test_size, not train_size."""
        cv = WalkForwardCV(
            n_splits=N_SPLITS,
            train_size=200,
            test_size=50,
            test_period=100,
            expanding=False,
            calendar=None,
        )
        splits = list(cv.split(np.zeros((600, 1))))

        assert len(splits) == N_SPLITS
        assert splits[0][1][0] == 200
        for train, val in splits:
            _assert_contiguous(train, val[0] - 200, val[0])
            assert val[-1] < 500

    def test_default_val_size_divides_data_after_training_window(self) -> None:
        """Prevents: test_size=None validation folds overlapping each other after train_size."""
        cv = WalkForwardCV(
            n_splits=N_SPLITS,
            train_size=200,
            test_period=100,
            expanding=False,
            calendar=None,
        )
        splits = list(cv.split(np.zeros((600, 1))))

        assert [(int(val[0]), len(val)) for _, val in splits] == [
            (200, 100),
            (300, 100),
            (400, 100),
        ]
        for train, val in splits:
            _assert_contiguous(train, val[0] - 200, val[0])
