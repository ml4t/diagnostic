"""Shared transaction notional, denominator and extra-cost definitions."""

from __future__ import annotations

from typing import Literal

import polars as pl


def execution_value_expr(
    *, absolute_quantity: bool = False, absolute_price: bool = False, integral: bool = False
) -> pl.Expr:
    """Execution cash value with explicit quantity/price sign conventions."""
    dtype = pl.Int128 if integral else pl.Float64
    quantity = pl.col("quantity").cast(dtype)
    if absolute_quantity:
        quantity = quantity.abs()
    price = pl.col("price").cast(dtype)
    if absolute_price:
        price = price.abs()
    return quantity * price * pl.col("multiplier").cast(dtype)


def filled_notional_expr(*, absolute_price: bool = True, integral: bool = False) -> pl.Expr:
    """Cash notional, with an explicit signed-price legacy analytical variant."""
    return execution_value_expr(
        absolute_quantity=True, absolute_price=absolute_price, integral=integral
    )


def absolute_quantity_expr(dtype: pl.DataType) -> pl.Expr:
    """Absolute execution units without signed64-bit minimum overflow."""
    values = pl.col("quantity")
    return values.cast(pl.Int128).abs() if dtype.is_integer() else values.abs()


def cash_fraction_expr(
    numerator: pl.Expr, denominator: pl.Expr, *, positive_finite: bool = True
) -> pl.Expr:
    """Normalize cash with native positive-NAV or explicit legacy IEEE division."""
    if not positive_finite:
        return numerator / denominator
    return (
        pl.when(denominator.is_finite() & (denominator > 0))
        .then(numerator / denominator)
        .otherwise(None)
    )


def average_gross_book(frame: pl.DataFrame, *, initial_gross: float | None = None) -> pl.DataFrame:
    """Adjacent complete gross values, with an explicit optional initial seed."""
    gross = pl.col("gross").cast(pl.Float64)
    previous = gross.shift(1)
    if initial_gross is not None:
        previous = pl.when(pl.int_range(pl.len()) == 0).then(initial_gross).otherwise(previous)
    return frame.with_columns(((gross + previous) / 2).alias("denominator"))


def adjusted_return_expr(
    returns: pl.Expr,
    cash_cost: pl.Expr,
    denominator: pl.Expr,
    *,
    method: Literal["additive_drag", "legacy_pnl"] = "additive_drag",
) -> pl.Expr:
    """Extra costs only; observed fees are not removed from supplied returns.

    Native additive drag keeps a known zero extra cost defined without NAV.
    Legacy PnL preserves missing-addend and IEEE division semantics, including
    undefined zero capital rather than silently replacing it.
    """
    if method == "additive_drag":
        drag = (
            pl.when(cash_cost == 0).then(0.0).otherwise(cash_fraction_expr(cash_cost, denominator))
        )
        return returns - drag
    if method != "legacy_pnl":
        raise ValueError("unknown return-cost formulation")
    returns, cash_cost, denominator = [
        value.fill_nan(None) for value in (returns, cash_cost, denominator)
    ]
    pnl = returns * denominator
    numerator = (
        pl.when(pnl.is_null() & cash_cost.is_null())
        .then(None)
        .otherwise(pnl.fill_null(0.0) - cash_cost.fill_null(0.0))
    )
    return cash_fraction_expr(numerator, denominator, positive_finite=False)
