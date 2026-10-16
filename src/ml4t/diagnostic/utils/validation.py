"""Internal observation-key checks shared by producers and analysis boundaries."""

import polars as pl


def _align_datetime_key_resolution(
    left: pl.DataFrame, right: pl.DataFrame, key: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    left_type, right_type = left[key].dtype, right[key].dtype
    if (
        not isinstance(left_type, pl.Datetime)
        or not isinstance(right_type, pl.Datetime)
        or left_type.time_zone != right_type.time_zone
        or left_type == right_type
    ):
        return left, right
    units = ["ms", "us", "ns"]
    unit = max([left_type.time_unit, right_type.time_unit], key=units.index)
    target = pl.Datetime(unit, left_type.time_zone)
    aligned = []
    for frame in (left, right):
        values = frame[key]
        try:
            refined = values.cast(target)
            lossless = refined.cast(values.dtype).equals(values)
        except pl.exceptions.PolarsError as exc:
            raise ValueError(f"{key} cannot be represented losslessly in {target}") from exc
        if not lossless:
            raise ValueError(f"{key} cannot be represented losslessly in {target}")
        aligned.append(frame.with_columns(refined))
    return aligned[0], aligned[1]


def _validate_observation_keys(
    frame: pl.DataFrame, keys: list[str], name: str, *, unique: bool = True
) -> None:
    keys = list(dict.fromkeys(keys))
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
