"""Internal observation-key checks shared by producers and analysis boundaries."""

import polars as pl


def _validate_observation_keys(
    frame: pl.DataFrame, keys: list[str], name: str, *, unique: bool = True
) -> None:
    if missing := set(keys).difference(frame.columns):
        raise ValueError(f"{name} is missing key columns: {sorted(missing)}")
    for key in keys:
        values = frame[key]
        if values.null_count():
            raise ValueError(f"{name} {key} keys must be nonmissing")
        if values.dtype.is_numeric() and not values.is_finite().all():
            raise ValueError(f"{name} {key} keys must be finite")
    if unique and keys and frame.select(keys).is_duplicated().any():
        raise ValueError(f"duplicate {name} {'/'.join(keys)} keys are not allowed")
