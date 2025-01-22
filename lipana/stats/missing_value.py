import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import numpy as np
import polars as pl
from numba import njit, prange

from ..base import cm

__all__ = [
    "count_df_selected_cols_nonnan",
    "AbstractMissingValueHandler",
    "NullMissingValueHandler",
    "fill_full_empty",
    "FullEmptyFillingMissingValueHandler",
    "sample_normal_dist",
    "NormalDistSamplingMissingValueHandler",
    "do_missing_value_handling",
    "do_missing_value_handling_on_df",
]

logger = logging.getLogger("lipana")


def count_df_selected_cols_nonnan(
    df: pl.DataFrame,
    cols: Sequence[str],
    count_col: Optional[str] = None,
):
    """Count non-NaN values in selected columns return a numpy array or attach it to the input dataframe as a new column."""
    counts = np.sum(~np.isnan(df.select(cols).to_numpy()), axis=1)
    if count_col is None:
        return counts
    return df.with_columns(pl.lit(counts, dtype=pl.Int8).alias(count_col))


@dataclass
class AbstractMissingValueHandler:
    """Base class for missing value handlers."""


@dataclass
class NullMissingValueHandler(AbstractMissingValueHandler):
    """Handler that does nothing with missing values."""


def _initialize_paired_values(
    paired_1: Optional[Any],
    paired_2: Optional[Any],
    overall: Optional[Any],
) -> tuple[Any, Any]:
    """Initialize paired values with an overall default if not provided."""
    if paired_1 is None:
        paired_1 = overall
    if paired_2 is None:
        paired_2 = overall
    if any((paired_1 is None, paired_2 is None)):
        raise ValueError(
            f"When at least one of the parameters of a pair is not provided, "
            f"a overall default value should be provided. Currently {paired_1=} and {paired_2=}, with {overall=}."
        )
    return paired_1, paired_2


@njit
def _check_and_fill_full_empty(
    a1: np.ndarray,
    a2: np.ndarray,
    a1_enough_rep: int = 3,
    a2_enough_rep: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Check and fill fully-empty rows based on non-NaN counts."""
    a1_nonnan_count = (~np.isnan(a1)).sum()
    a2_nonnan_count = (~np.isnan(a2)).sum()
    if (a1_nonnan_count >= a1_enough_rep) and (a2_nonnan_count == 0):
        a2[:a1_nonnan_count] = 0.0
    elif (a1_nonnan_count == 0) and (a2_nonnan_count >= a2_enough_rep):
        a1[:a2_nonnan_count] = 0.0
    else:
        pass
    return a1, a2


@njit
def fill_full_empty(
    mat1: np.ndarray,
    mat2: np.ndarray,
    a1_enough_rep: int = 3,
    a2_enough_rep: int = 3,
    do_copy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill fully-empty row in one array if that row in the other array has enough non-NA values.
    This is a pair-handling method, and can not be used on a group with more than two conditions.

    For each row index, `mat1[i]` has `n1` detections, and `mat2[i]` has `n2` detections.
    If `n1 >= a1_enough_rep` and `n2 == 0`, then `mat2[i][:n1] = 0.0`.
    If `n1 == 0` and `n2 >= a2_enough_rep`, then `mat1[i][:n2] = 0.0`.
    """
    if do_copy:
        mat1 = mat1.copy()
        mat2 = mat2.copy()
    for i in prange(mat1.shape[0]):
        mat1[i], mat2[i] = _check_and_fill_full_empty(
            mat1[i],
            mat2[i],
            a1_enough_rep,
            a2_enough_rep,
        )
    return mat1, mat2


@dataclass
class FullEmptyFillingMissingValueHandler(AbstractMissingValueHandler):
    """
    This method will fill a fully-empty row in one array if that row in the other array has enough non-NA values.
    See `fill_full_empty` for details
    """

    min_rep_count: Optional[int] = 3

    min_exp_rep_count: Optional[int] = None
    min_ctrl_rep_count: Optional[int] = None

    def __post_init__(self):
        self.min_exp_rep_count, self.min_ctrl_rep_count = _initialize_paired_values(
            self.min_exp_rep_count, self.min_ctrl_rep_count, self.min_rep_count
        )


@njit
def _check_and_sample_normal_dist(
    a1: np.ndarray,
    a2: np.ndarray,
    a1_enough_rep: int = 3,
    a2_enough_rep: int = 3,
    a1_impute_rep_range: tuple[int, int] = (1, 2),
    a2_impute_rep_range: tuple[int, int] = (1, 2),
    log_scale_minmax_diff: int = 1,
):
    a1_nonnan_count = (~np.isnan(a1)).sum()
    a2_nonnan_count = (~np.isnan(a2)).sum()
    if (
        (a1_nonnan_count >= a1_enough_rep)
        and (a2_nonnan_count <= a2_impute_rep_range[1])
        and (a2_nonnan_count >= a2_impute_rep_range[0])
        and ((np.nanmin(a1) - np.nanmax(a2)) > log_scale_minmax_diff)
    ):
        _n = a1_nonnan_count - a2_nonnan_count
        a2[np.where(np.isnan(a2))[0][:_n]] = np.clip(np.random.normal(np.nanmean(a2), np.nanstd(a1), _n), 1e-4, None)
    elif (
        (a2_nonnan_count >= a2_enough_rep)
        and (a1_nonnan_count <= a1_impute_rep_range[1])
        and (a1_nonnan_count >= a1_impute_rep_range[0])
        and ((np.nanmin(a2) - np.nanmax(a1)) > log_scale_minmax_diff)
    ):
        _n = a2_nonnan_count - a1_nonnan_count
        a1[np.where(np.isnan(a1))[0][:_n]] = np.clip(np.random.normal(np.nanmean(a1), np.nanstd(a2), _n), 1e-4, None)
    return a1, a2


@njit
def sample_normal_dist(
    mat1: np.ndarray,
    mat2: np.ndarray,
    a1_enough_rep: int = 3,
    a2_enough_rep: int = 3,
    a1_impute_rep_range: tuple[int, int] = (1, 2),
    a2_impute_rep_range: tuple[int, int] = (1, 2),
    log_scale_minmax_diff: int = 1,
    do_copy: bool = False,
):
    """
    Sample from a normal distribution to fill missing values in one array based on the other array.
    This method only handle paired two inputs, and can not be used on a group with more than two conditions.

    The imputation will be conducted if all the following rules are met:
    1. A row in one input array has enough detections >= defined `a[12]_enough_rep`.
    2. The other array has number of detections within the defined range `a[12]_impute_rep_range`.
    3. The difference between the minimum value in the more-detected array and the maximum value in the other array is larger than `log_scale_minmax_diff`.

    The mean of distribution is the mean of the less-detected array, and the standard deviation is from the more-detected array.
    """
    if do_copy:
        mat1 = mat1.copy()
        mat2 = mat2.copy()
    for i in prange(mat1.shape[0]):
        mat1[i], mat2[i] = _check_and_sample_normal_dist(
            mat1[i],
            mat2[i],
            a1_enough_rep,
            a2_enough_rep,
            a1_impute_rep_range,
            a2_impute_rep_range,
            log_scale_minmax_diff,
        )
    return mat1, mat2


@dataclass
class NormalDistSamplingMissingValueHandler(AbstractMissingValueHandler):
    """
    This method will sample from a normal distribution to fill missing values in one array based on the other array.
    See `sample_normal_dist` for details
    """

    min_rep_count: Optional[int] = 3
    do_imputation_for_rep_range: Optional[tuple[int, int]] = (1, 2)
    log_scale_minmax_diff: int = 1

    min_exp_rep_count: Optional[int] = None
    min_ctrl_rep_count: Optional[int] = None
    do_imputation_for_exp_rep_range: Optional[tuple[int, int]] = None
    do_imputation_for_ctrl_rep_range: Optional[tuple[int, int]] = None

    def __post_init__(self):
        self.min_exp_rep_count, self.min_ctrl_rep_count = _initialize_paired_values(
            self.min_exp_rep_count, self.min_ctrl_rep_count, self.min_rep_count
        )
        self.do_imputation_for_exp_rep_range, self.do_imputation_for_ctrl_rep_range = _initialize_paired_values(
            self.do_imputation_for_exp_rep_range,
            self.do_imputation_for_ctrl_rep_range,
            self.do_imputation_for_rep_range,
        )


def do_missing_value_handling(
    exp_quant_arr: np.ndarray,
    ctrl_quant_arr: np.ndarray,
    config: Optional[Union[AbstractMissingValueHandler, Sequence[AbstractMissingValueHandler]]] = None,
    do_copy: bool = False,
):
    """
    Handle missing values in two arrays based on the provided configurations.
    Configurations can be a single instance or a list of instances of objects that inherit from `AbstractMissingValueHandler`.
    """
    if config is None:
        config = NullMissingValueHandler()
    if isinstance(config, AbstractMissingValueHandler):
        config = (config,)
    for conf in config:
        if isinstance(conf, NullMissingValueHandler):
            pass
        elif isinstance(conf, FullEmptyFillingMissingValueHandler):
            exp_quant_arr, ctrl_quant_arr = fill_full_empty(
                exp_quant_arr,
                ctrl_quant_arr,
                conf.min_exp_rep_count,
                conf.min_ctrl_rep_count,
                do_copy,
            )
        elif isinstance(conf, NormalDistSamplingMissingValueHandler):
            exp_quant_arr, ctrl_quant_arr = sample_normal_dist(
                exp_quant_arr,
                ctrl_quant_arr,
                conf.min_exp_rep_count,
                conf.min_ctrl_rep_count,
                conf.do_imputation_for_exp_rep_range,
                conf.do_imputation_for_ctrl_rep_range,
                conf.log_scale_minmax_diff,
                do_copy,
            )
    return exp_quant_arr, ctrl_quant_arr


def do_missing_value_handling_on_df(
    df: pl.DataFrame,
    exp_runs: Sequence[str],
    ctrl_runs: Sequence[str],
    config: Optional[Union[AbstractMissingValueHandler, Sequence[AbstractMissingValueHandler]]] = None,
    filter_less_than_rep: Optional[Union[int, tuple[int, int]]] = None,
    attach_back: bool = False,
    annotation_col: Optional[Union[str, Sequence[str]]] = cm.precursor,
):
    """
    If `attach_back` is True, will return the input dataframe with new columns of imputed values attached, and the new columns have name as "{run}_imputed".
    Else, will return a new dataframe with only imputed values and the annotation column(s) (in this case, run column names are as the original ones, and `annotation_col` should be correctly provided).
    """
    result = pl.from_numpy(
        np.hstack(
            (
                do_missing_value_handling(
                    df.select(exp_runs).to_numpy(),
                    df.select(ctrl_runs).to_numpy(),
                    config=config,
                    do_copy=True,
                )
            )
        ),
        [*exp_runs, *ctrl_runs],
    )

    if attach_back:
        df = df.rename({r: f"{r}_raw" for r in [*exp_runs, *ctrl_runs]}).with_columns(result)
    else:
        df = result.with_columns(df.select(annotation_col))
    if filter_less_than_rep is None:
        return df
    if isinstance(filter_less_than_rep, int):
        filter_less_than_rep = (filter_less_than_rep, filter_less_than_rep)
    return df.filter(
        np.logical_and(
            *(
                count_df_selected_cols_nonnan(df, cols, count_col=None) >= filter_less_than_rep[idx]
                for idx, cols in enumerate((exp_runs, ctrl_runs))
            )
        )
    )
