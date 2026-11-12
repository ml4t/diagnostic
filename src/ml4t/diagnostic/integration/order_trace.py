"""Compare the recorded outcomes of one canonical order across two runs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

from ml4t.specs import CanonicalChildOrderIntent, CanonicalTargetIntent


def _decimal(value: float | int | str) -> Decimal:
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError("Order trace economics require finite values")
    return result


def _one(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    matches = [row for row in rows if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key}={value!r}, found {len(matches)}")
    return matches[0]


def _child_orders(rows: list[dict[str, Any]], child_id: str) -> dict[str, dict[str, Any]]:
    by_id = {row["order_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate native order ID in trace")
    selected = {}
    for order_id, row in by_id.items():
        current = row
        owner = current.get("child_intent_id")
        seen = {order_id}
        while (parent_id := current.get("replaces_order_id")) is not None:
            if parent_id in seen or parent_id not in by_id:
                raise ValueError(f"Broken replacement lineage for {order_id!r}")
            seen.add(parent_id)
            current = by_id[parent_id]
            parent_owner = current.get("child_intent_id")
            if owner is not None and parent_owner is not None and owner != parent_owner:
                raise ValueError(f"Replacement crosses canonical child IDs for {order_id!r}")
            owner = owner or parent_owner
        if owner == child_id:
            selected[order_id] = row
    return selected


def _summarize(
    trace: Mapping[str, Any],
    child_id: str,
    currency: str,
    decision_price: Decimal | None,
    terminal_price: Decimal | None,
) -> dict[str, Any]:
    from ml4t.specs import ExecutionFill

    if type(trace.get("schema_version")) is not int or trace["schema_version"] != 1:
        raise ValueError("Unsupported order trace schema version")
    child = CanonicalChildOrderIntent.from_mapping(
        _one(trace["children"], "child_intent_id", child_id)
    )
    target = CanonicalTargetIntent.from_mapping(
        _one(trace["targets"], "intent_id", child.target_intent_id)
    )
    orders = _child_orders(trace["orders"], child_id)
    all_order_ids = {row["order_id"] for row in trace["orders"]}
    risk_rejections = [
        row for row in trace.get("risk_rejections", ()) if row["child_intent_id"] == child_id
    ]
    states = [row for row in trace["reconciliations"] if row["child_intent_id"] == child_id]
    unresolved = []
    executions: dict[tuple[str | None, str], ExecutionFill] = {}
    for row in trace["fills"]:
        fill = ExecutionFill.from_mapping(row)
        if fill.order_id not in orders:
            if fill.asset == child.asset and fill.order_id not in all_order_ids:
                unresolved.append("unlinked_fill")
            continue
        if fill.asset != child.asset or fill.side != child.side:
            raise ValueError(f"Execution asset or side contradicts child {child_id!r}")
        if fill.execution_id is None:
            unresolved.append("missing_execution_id")
            continue
        identity = (fill.venue, fill.execution_id)
        previous = executions.get(identity)
        if previous is not None:
            previous_facts = previous.to_dict()
            current_facts = fill.to_dict()
            del previous_facts["received_at"], current_facts["received_at"]
            if previous_facts != current_facts:
                raise ValueError(f"Conflicting execution evidence for {identity!r}")
            continue
        executions[identity] = fill

    fills = list(executions.values())
    if not orders and not states and not risk_rejections:
        unresolved.append("no_order_outcome")
    quantities: dict[str, Decimal] = defaultdict(Decimal)
    by_price: dict[Decimal, Decimal] = defaultdict(Decimal)
    for fill in fills:
        quantities[fill.order_id] += _decimal(fill.quantity)
        by_price[_decimal(fill.price)] += _decimal(fill.quantity)
    latest_by_order = {row["order_id"]: row for row in states if row["order_id"] in orders}
    for order_id, row in latest_by_order.items():
        if quantities[order_id] != _decimal(row["filled_quantity"]):
            unresolved.append("incomplete_fill_history")
    for order_id, order in orders.items():
        if order["status"] == "filled" and not quantities[order_id]:
            unresolved.append("incomplete_fill_history")

    quantity = sum(quantities.values(), Decimal(0))
    requested = _decimal(child.quantity)
    remaining = requested - quantity
    fee_known = all(fill.fee is not None and fill.fee_currency == currency for fill in fills)
    units_known = all(fill.currency == currency and fill.multiplier is not None for fill in fills)
    multipliers = {_decimal(fill.multiplier) for fill in fills if fill.multiplier is not None}
    multiplier = next(iter(multipliers)) if len(multipliers) == 1 else None
    fees = sum((_decimal(fill.fee) for fill in fills if fill.fee is not None), Decimal(0))
    side = 1 if child.side.value == "buy" else -1
    filled_cost = None
    opportunity_cost = None
    if not unresolved and fee_known and units_known and multiplier is not None:
        if decision_price is not None:
            filled_cost = fees + sum(
                side * (price - decision_price) * size * multiplier
                for price, size in by_price.items()
            )
            if terminal_price is not None and remaining >= 0:
                opportunity_cost = side * (terminal_price - decision_price) * remaining * multiplier
    return {
        "target": target.to_dict(),
        "child": child.to_dict(),
        "orders": list(orders.values()),
        "reconciliations": states,
        "risk_rejections": risk_rejections,
        "fills": [fill.to_dict() for fill in fills],
        "account": trace.get("account"),
        "observed_filled_quantity": str(quantity),
        "remaining_quantity": None if unresolved else str(max(remaining, Decimal(0))),
        "overfilled_quantity": None if unresolved else str(max(-remaining, Decimal(0))),
        "filled_by_price": {str(price): str(size) for price, size in sorted(by_price.items())},
        "fees": str(fees) if fee_known and not unresolved else None,
        "execution_cost": str(filled_cost) if filled_cost is not None else None,
        "residual_opportunity_cost": str(opportunity_cost)
        if opportunity_cost is not None
        else None,
        "total_shortfall": (
            str(filled_cost + opportunity_cost)
            if filled_cost is not None and opportunity_cost is not None
            else None
        ),
        "unresolved": sorted(set(unresolved)),
    }


def compare_order_traces(
    backtest: Mapping[str, Any],
    live: Mapping[str, Any],
    *,
    child_intent_id: str,
    currency: str,
    decision_price: float | str | None = None,
    terminal_price: float | str | None = None,
) -> dict[str, Any]:
    """Compare two traces using shared intent IDs and explicit price references.

    Unknown linkage, costs or quote-based components stay unknown. The caller
    supplies a common benchmark and horizon mark in the comparison currency.
    Account checkpoints are retained independently of order-level attribution.
    """
    decision = _decimal(decision_price) if decision_price is not None else None
    terminal = _decimal(terminal_price) if terminal_price is not None else None
    left = _summarize(backtest, child_intent_id, currency, decision, terminal)
    right = _summarize(live, child_intent_id, currency, decision, terminal)

    outcome_keys = ("filled_by_price", "fees", "remaining_quantity")
    if left["target"] != right["target"]:
        stage = "decision"
    elif left["child"] != right["child"]:
        stage = "child_order"
    elif bool(left["risk_rejections"]) != bool(right["risk_rejections"]):
        stage = "risk_rejection"
    elif any(row["outcome"] == "rejected" for row in right["reconciliations"]) != any(
        row["outcome"] == "rejected" for row in left["reconciliations"]
    ):
        stage = "rejection"
    elif left["unresolved"] or right["unresolved"]:
        stage = "unresolved"
    elif sum(row.get("replaces_order_id") is not None for row in left["orders"]) != sum(
        row.get("replaces_order_id") is not None for row in right["orders"]
    ):
        stage = "replacement"
    elif any(left[key] != right[key] for key in outcome_keys):
        stage = "execution_outcome"
    else:
        stage = None
    left_cost, right_cost = left["total_shortfall"], right["total_shortfall"]
    comparable = stage not in {"decision", "child_order", "risk_rejection", "rejection"}
    return {
        "child_intent_id": child_intent_id,
        "currency": currency,
        "decision_price": str(decision) if decision is not None else None,
        "terminal_price": str(terminal) if terminal is not None else None,
        "first_differing_stage_compared": stage,
        "live_minus_backtest_shortfall": (
            str(_decimal(right_cost) - _decimal(left_cost))
            if comparable and left_cost is not None and right_cost is not None
            else None
        ),
        "backtest": left,
        "live": right,
    }
