"""Price-produced columns retain per-metric counts, periodic rates and output contracts."""

import importlib
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def api():
    return importlib.import_module("ml4t.diagnostic.compat.empyrical")


@pytest.fixture
def returns():
    prices = pd.DataFrame(
        {"complete": [100, 102, 101, 103, 104], "missing": [50, 51, 52, 51, np.nan]},
        index=pd.bdate_range("2025-01-02", periods=5, tz="UTC", name="date"),
    )
    return prices.pct_change(fill_method=None).iloc[1:]


def test_parent_compat_namespace_exposes_scalar_workflow_in_a_fresh_interpreter():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import pandas as pd; import ml4t.diagnostic.compat as compat; prices=pd.Series([100.,102.,104.]); returns=prices.pct_change(fill_method=None).iloc[1:]; value=compat.empyrical.annual_return(returns,annualization=2); assert abs(value-.04)<1e-12",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _expected(name, column, scale, rate=0.001):
    values = (
        [2 / 100, -1 / 102, 2 / 101, 1 / 103] if column == "complete" else [1 / 50, 1 / 51, -1 / 52]
    )
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    annual = (1.04 if column == "complete" else 1.02) ** (scale / 4) - 1
    if name in {"annual_return", "cagr"}:
        return annual
    if name == "annual_volatility":
        return std * math.sqrt(scale)
    if name == "calmar_ratio":
        return annual * (102 if column == "complete" else 52)
    if name == "sharpe_ratio":
        return (mean - rate) / std * math.sqrt(scale)
    downside = math.sqrt(sum(min(value - rate, 0) ** 2 for value in values) / len(values))
    return (
        downside * math.sqrt(scale)
        if name == "downside_risk"
        else (mean - rate) / downside * math.sqrt(scale)
    )


@pytest.mark.parametrize("scale", [1, 12, 52, 52.5, 252, 365])
@pytest.mark.parametrize(
    "name",
    [
        "annual_return",
        "cagr",
        "annual_volatility",
        "calmar_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "downside_risk",
    ],
)
def test_price_columns_use_metric_specific_missingness_and_explicit_periodic_rates(
    api, returns, name, scale
):
    before = returns.copy(deep=True)
    kwargs = {"period": "ignored-explicit-scale", "annualization": scale}
    if name == "sharpe_ratio":
        kwargs["risk_free"] = 0.001
    if name in {"sortino_ratio", "downside_risk"}:
        kwargs["required_return"] = 0.001
    function = getattr(api, name)
    expected = [_expected(name, column, scale) for column in ["complete", "missing"]]
    if name == "calmar_ratio":
        result = returns.apply(lambda column: function(column, **kwargs))
    else:
        result = function(returns, **kwargs)
        if name in {"annual_volatility", "sharpe_ratio"}:
            assert isinstance(result, np.ndarray)
        else:
            assert isinstance(result, pd.Series)
            assert list(result.index) == ["complete", "missing"]
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)
    for column, wanted in zip(["complete", "missing"], expected, strict=True):
        for sample in [returns[column], returns[column].to_numpy()]:
            actual = function(sample, **kwargs)
            assert np.ndim(actual) == 0
            np.testing.assert_allclose(actual, wanted, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(returns, before)


@pytest.mark.parametrize(
    "name", ["annual_volatility", "sharpe_ratio", "sortino_ratio", "downside_risk"]
)
def test_price_column_output_buffer_is_written_without_changing_input(api, returns, name):
    function = getattr(api, name)
    before = returns.copy(deep=True)
    out = np.empty(2)
    expected = function(returns.to_numpy(), annualization=12)
    actual = function(returns.to_numpy(), annualization=12, out=out)
    assert actual is out
    np.testing.assert_allclose(actual, expected)
    labeled = function(returns, annualization=12, out=out)
    assert np.shares_memory(labeled, out)
    if name in {"sortino_ratio", "downside_risk"}:
        assert isinstance(labeled, pd.Series)
        assert list(labeled.index) == ["complete", "missing"]
    scalar_out = np.empty(())
    scalar = function(returns["missing"], annualization=12, out=scalar_out)
    assert np.ndim(scalar) == 0 and scalar == scalar_out.item()
    invalid = np.full(3, -999.0)
    with pytest.raises(ValueError, match="shape"):
        function(returns, out=invalid)
    assert invalid.tolist() == [-999.0] * 3
    pd.testing.assert_frame_equal(returns, before)


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("scale", [12, 52.5])
def test_volatility_levy_exponent_scales_each_actual_price_column(api, returns, alpha, scale):
    daily = [_expected("annual_volatility", column, 1) for column in ["complete", "missing"]]
    expected = np.array(daily) * scale ** (1 / alpha)
    np.testing.assert_allclose(
        api.annual_volatility(returns, alpha=alpha, annualization=scale), expected
    )


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf])
def test_invalid_levy_exponent_does_not_silently_produce_complex_or_zero_volatility(
    api, returns, alpha
):
    with pytest.raises(ValueError, match="positive.*finite"):
        api.annual_volatility(returns, alpha=alpha)


def test_one_defined_price_return_retains_elapsed_rows_without_inventing_volatility(api, returns):
    sample = returns["missing"].where(np.arange(4) == 0)
    assert sample.notna().sum() == 1
    assert api.annual_return(sample, annualization=12) == pytest.approx(1.02**3 - 1)
    assert np.isnan(api.annual_volatility(sample))
    assert np.isnan(api.sharpe_ratio(sample))
    assert api.downside_risk(sample) == 0
    assert api.sortino_ratio(sample) == np.inf


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, True])
def test_invalid_explicit_annualization_is_rejected_instead_of_reinterpreted(api, scale):
    with pytest.raises(ValueError, match="positive.*finite"):
        api.annualization_factor("daily", scale)


def test_period_defaults_are_used_only_without_explicit_scale(api):
    assert [
        api.annualization_factor(period, None)
        for period in ["daily", "weekly", "monthly", "quarterly", "yearly"]
    ] == [252, 52, 12, 4, 1]
    assert api.annualization_factor("not-a-period", 52.5) == 52.5
    with pytest.raises(ValueError, match="period"):
        api.annualization_factor("not-a-period", None)


def test_negative_ending_price_wealth_does_not_return_complex_cagr(api):
    from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import annual_return

    prices = pd.Series([2.0, 3.0, -1.0, -2.0, -3.0], index=pd.bdate_range("2020-04-16", periods=5))
    sample = prices.pct_change(fill_method=None).iloc[1:]
    native = annual_return(sample, periods_per_year=52.5)
    assert np.isrealobj(native) and np.isnan(native)
    assert np.isnan(api.annual_return(sample, annualization=52.5))


def test_total_price_loss_is_retained_when_subsequent_zero_price_return_is_missing(api):
    prices = pd.Series([100.0, 110.0, 0.0, 0.0], index=pd.bdate_range("2025-01-02", periods=4))
    sample = prices.pct_change(fill_method=None).iloc[1:]
    assert sample.tolist()[:2] == pytest.approx([0.1, -1.0])
    assert np.isnan(sample.iloc[2])
    assert api.annual_return(sample, annualization=52.5) == -1
    assert api.calmar_ratio(sample, annualization=52.5) == -1


def test_annualizing_large_finite_price_gain_returns_infinity_not_python_overflow(api):
    from ml4t.diagnostic.evaluation.portfolio_analysis.metrics import annual_return

    sample = pd.Series([1e-100, 1e100]).pct_change(fill_method=None).iloc[1:]
    assert np.isfinite(sample.iloc[0])
    assert 252 * math.log(1e200) > math.log(np.finfo(float).max)
    assert annual_return(sample) == np.inf
    assert api.annual_return(sample) == np.inf


@pytest.mark.parametrize(
    "name",
    [
        "annual_return",
        "cagr",
        "annual_volatility",
        "calmar_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "downside_risk",
    ],
)
def test_empty_and_all_missing_price_returns_keep_each_metrics_undefined_contract(
    api, returns, name
):
    function = getattr(api, name)
    assert np.isnan(function(returns["missing"].iloc[:0]))
    missing = (
        pd.Series(np.nan, index=pd.bdate_range("2025-01-02", periods=5))
        .pct_change(fill_method=None)
        .iloc[1:]
    )
    actual = function(missing)
    if name in {"annual_return", "cagr"}:
        assert actual == 0
    else:
        assert np.isnan(actual)
    if name != "calmar_ratio":
        empty = function(returns.iloc[:0])
        if name in {"annual_return", "cagr"}:
            assert np.ndim(empty) == 0 and np.isnan(empty)
        else:
            assert isinstance(empty, np.ndarray) and empty.shape == (2,)
            assert np.isnan(empty).all()


def test_exact_constant_and_zero_price_growth_do_not_invent_risk(api):
    constant = pd.Series([1, 2, 4, 8, 16]).pct_change(fill_method=None).iloc[1:]
    zero = pd.Series([100, 100, 100, 100, 100]).pct_change(fill_method=None).iloc[1:]
    for sample, mean in [(constant, 1), (zero, 0)]:
        assert api.annual_return(sample, annualization=12) == (1 + mean) ** 12 - 1
        assert api.annual_volatility(sample) == api.downside_risk(sample) == 0
        assert np.isnan(api.calmar_ratio(sample))
        for name in ["sharpe_ratio", "sortino_ratio"]:
            value = getattr(api, name)(sample)
            assert value == np.inf if mean else np.isnan(value)


def test_singleton_frame_sortino_is_undefined_array_not_a_fabricated_labeled_estimate(api, returns):
    sample = returns.iloc[:1]
    assert isinstance(api.annual_return(sample), pd.Series)
    downside = api.downside_risk(sample, required_return=0.03, annualization=12)
    assert isinstance(downside, pd.Series)
    assert list(downside.index) == ["complete", "missing"]
    np.testing.assert_allclose(downside, [0.01 * math.sqrt(12)] * 2)
    for name in ["sortino_ratio", "sharpe_ratio", "annual_volatility"]:
        result = getattr(api, name)(sample)
        assert isinstance(result, np.ndarray) and result.shape == (2,)
        assert np.isnan(result).all()


def test_zero_price_recovery_infinity_is_not_silently_removed_from_risk_sample(api):
    sample = pd.Series([1.0, 0.0, 1.0, 1.0]).pct_change(fill_method=None).iloc[1:]
    assert sample.tolist() == [-1.0, np.inf, 0.0]
    assert np.isnan(api.annual_return(sample, annualization=12))
    assert np.isnan(api.annual_volatility(sample, annualization=12))
    assert np.isnan(api.sharpe_ratio(sample, annualization=12))
    assert api.downside_risk(sample, annualization=12) == pytest.approx(2)
    assert api.sortino_ratio(sample, annualization=12) == np.inf


def test_one_observation_sortino_guard_is_distinct_from_single_defined_observation(api, returns):
    from ml4t.diagnostic.metrics import periodic_sortino_ratio

    sample = returns["missing"].where(np.arange(4) == 0)
    assert np.isnan(periodic_sortino_ratio(sample))
    assert api.sortino_ratio(sample) == np.inf
    assert np.isnan(api.sortino_ratio(sample.iloc[:1]))
    assert api.downside_risk(
        sample.iloc[:1], required_return=0.03, annualization=12
    ) == pytest.approx(0.01 * math.sqrt(12))


@pytest.mark.parametrize("name", ["sharpe_ratio", "sortino_ratio", "downside_risk"])
def test_reordered_dated_periodic_rates_pair_with_actual_return_dates(api, returns, name):
    rates = pd.Series([0.003, 0.001, 0.004, 0.002], index=returns.index[[2, 0, 3, 1]])
    excess = [1 / 50 - 0.001, 1 / 51 - 0.002, -1 / 52 - 0.003]
    mean = sum(excess) / 3
    downside = math.sqrt(sum(min(value, 0) ** 2 for value in excess) / 3)
    if name == "sharpe_ratio":
        denominator = math.sqrt(sum((value - mean) ** 2 for value in excess) / 2)
        expected = mean / denominator * math.sqrt(12)
    else:
        expected = (
            downside * math.sqrt(12) if name == "downside_risk" else mean / downside * math.sqrt(12)
        )
    key = "risk_free" if name == "sharpe_ratio" else "required_return"
    actual = getattr(api, name)(returns["missing"], annualization=12, **{key: rates})
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("name", ["sharpe_ratio", "sortino_ratio", "downside_risk"])
@pytest.mark.parametrize("reverse_columns", [False, True])
@pytest.mark.parametrize("rate_kind", ["series", "frame"])
def test_frame_periodic_rates_align_by_column_not_series_position(
    api, returns, name, reverse_columns, rate_kind
):
    if reverse_columns:
        returns = returns[["missing", "complete"]]
    rates = pd.Series({"complete": 0.001, "missing": 0.002}).reindex(returns.columns[::-1])
    if rate_kind == "frame":
        rates = pd.DataFrame(
            np.tile(rates.to_numpy(), (len(returns), 1)),
            index=returns.index[::-1],
            columns=rates.index,
        )
    key = "risk_free" if name == "sharpe_ratio" else "required_return"
    actual = getattr(api, name)(returns, annualization=12, **{key: rates})
    expected = [
        _expected(name, column, 12, 0.001 if column == "complete" else 0.002)
        for column in returns.columns
    ]
    if isinstance(actual, pd.Series):
        assert list(actual.index) == list(returns.columns)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("supplied", [0.2, np.array([0.2, 0.3])])
def test_supplied_downside_deviation_is_annualized_and_preserves_column_output(
    api, returns, supplied
):
    deviations = np.broadcast_to(supplied, (2,))
    means = [(2 / 100 - 1 / 102 + 2 / 101 + 1 / 103) / 4, (1 / 50 + 1 / 51 - 1 / 52) / 3]
    result = api.sortino_ratio(
        returns, required_return=0.001, annualization=52.5, _downside_risk=supplied
    )
    assert list(result.index) == ["complete", "missing"]
    np.testing.assert_allclose(
        result,
        [
            (mean - 0.001) * 52.5 / deviation
            for mean, deviation in zip(means, deviations, strict=True)
        ],
    )


@pytest.mark.parametrize(
    "name", ["annual_volatility", "sharpe_ratio", "sortino_ratio", "downside_risk"]
)
def test_invalid_or_aliased_output_buffer_cannot_mutate_price_returns(api, returns, name):
    sample = returns.to_numpy(copy=True)
    before = sample.copy()
    function = getattr(api, name)
    with pytest.raises(TypeError, match="floating"):
        function(sample, out=np.empty(2, dtype=int))
    readonly = np.empty(2)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        function(sample, out=readonly)
    with pytest.raises(ValueError, match="overlap"):
        function(sample, out=sample[0])
    np.testing.assert_array_equal(sample, before)


@pytest.mark.parametrize(
    "name", ["annual_volatility", "sharpe_ratio", "sortino_ratio", "downside_risk"]
)
def test_float32_output_keeps_buffer_storage_and_its_explicit_precision(api, returns, name):
    out = np.empty(2, dtype=np.float32)
    kwargs = {"annualization": 12, "out": out}
    if name == "sharpe_ratio":
        kwargs["risk_free"] = 0.001
    if name in {"sortino_ratio", "downside_risk"}:
        kwargs["required_return"] = 0.001
    actual = getattr(api, name)(returns, **kwargs)
    assert np.asarray(actual).dtype == np.dtype("float32")
    assert np.shares_memory(actual, out)
    expected = [_expected(name, column, 12) for column in ["complete", "missing"]]
    np.testing.assert_allclose(actual, expected, rtol=2e-6)


@pytest.mark.parametrize(
    "name",
    [
        "annual_return",
        "cagr",
        "annual_volatility",
        "calmar_ratio",
        "sharpe_ratio",
        "sortino_ratio",
        "downside_risk",
    ],
)
@pytest.mark.parametrize("shape", ["series", "array1", "frame", "array2"])
def test_captured_price_workflow_matches_pinned_reference_values_and_supported_shapes(
    api, returns, name, shape
):
    reference = pytest.importorskip("empyrical")
    sample = (
        returns["missing"]
        if shape == "series"
        else returns["missing"].to_numpy()
        if shape == "array1"
        else returns
        if shape == "frame"
        else returns.to_numpy()
    )
    kwargs = {"period": "ignored-explicit-scale", "annualization": 52.5}
    if name == "sharpe_ratio":
        kwargs["risk_free"] = 0.001
    if name in {"sortino_ratio", "downside_risk"}:
        kwargs["required_return"] = 0.001
    if name == "calmar_ratio" and sample.ndim == 2:
        expected = np.array(
            [
                reference.calmar_ratio(
                    sample[:, index] if shape == "array2" else sample.iloc[:, index], **kwargs
                )
                for index in range(2)
            ]
        )
        actual = np.array(
            [
                api.calmar_ratio(
                    sample[:, index] if shape == "array2" else sample.iloc[:, index], **kwargs
                )
                for index in range(2)
            ]
        )
    else:
        expected = getattr(reference, name)(sample, **kwargs)
        actual = getattr(api, name)(sample, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    assert np.shape(actual) == np.shape(expected)
    if isinstance(expected, pd.Series):
        assert isinstance(actual, pd.Series)
        if name == "sortino_ratio":
            assert list(actual.index) == ["complete", "missing"]
            assert list(expected.index) in [["complete", "missing"], [0, 1]]
        else:
            pd.testing.assert_index_equal(actual.index, expected.index)
