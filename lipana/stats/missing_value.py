import copy
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl
from numba import njit, prange

from ..base import cm
from .stats_base import (
    AbstractGroupMissingValueHandler,
    AbstractMissingValueHandler,
    AbstractPairwiseMissingValueHandler,
    _T_CompareScope,
)

__all__ = [
    "count_df_selected_cols_nonnan",
    "NullMissingValueHandler",
    "fill_full_empty",
    "FullEmptyFillingMissingValueHandler",
    "sample_normal_dist",
    "NormalDistSamplingMissingValueHandler",
    "sequential_impute",
    "SequentialImputeMissingValueHandler",
    "do_pairwise_missing_value_handling",
    "do_group_missing_value_handling",
    "check_paired_mv",
    "CheckRemovalMissingValueHandler",
    "check_mv_on_df",
    "do_pairwise_missing_value_handling_on_df",
    "do_group_missing_value_handling_on_df",
]

logger = logging.getLogger("lipana")


def count_df_selected_cols_nonnan(
    df: pl.DataFrame,
    cols: Sequence[str],
    count_col: Optional[str] = None,
) -> np.ndarray | pl.DataFrame:
    """
    Count non-NaN values in selected columns and return a numpy array or attach it to the input dataframe as a new column.
    """
    counts = np.sum(~np.isnan(df.select(cols).to_numpy()), axis=1)
    if count_col is None:
        return counts
    return df.with_columns(pl.lit(counts, dtype=pl.Int8).alias(count_col))


@dataclass
class NullMissingValueHandler(AbstractMissingValueHandler):
    """Handler that does nothing with missing values."""

    _compare_scope: _T_CompareScope = "all"


def _initialize_paired_values(
    paired_1: Optional[Any],
    paired_2: Optional[Any],
    default: Optional[Any],
) -> tuple[Any, Any]:
    """Initialize paired values with a default value if not provided."""
    if paired_1 is None:
        paired_1 = default
    if paired_2 is None:
        paired_2 = default
    if any((paired_1 is None, paired_2 is None)):
        raise ValueError(
            f"When at least one of the parameters of a pair is not provided, "
            f"a default value should be provided. Currently {paired_1=} and {paired_2=}, with {default=}."
        )
    return paired_1, paired_2


@njit
def _check_and_fill_full_empty(
    a1: np.ndarray,
    a2: np.ndarray,
    a1_enough_rep: int = 3,
    a2_enough_rep: int = 3,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Check and fill fully-empty rows based on non-NaN counts."""
    a1_nonnan_count = (~np.isnan(a1)).sum()
    a2_nonnan_count = (~np.isnan(a2)).sum()
    is_filled = False
    if (a1_nonnan_count >= a1_enough_rep) and (a2_nonnan_count == 0):
        a2[:a1_nonnan_count] = 0.0
        is_filled = True
    elif (a1_nonnan_count == 0) and (a2_nonnan_count >= a2_enough_rep):
        a1[:a2_nonnan_count] = 0.0
        is_filled = True
    return a1, a2, is_filled


@njit
def fill_full_empty(
    mat1: np.ndarray,
    mat2: np.ndarray,
    a1_enough_rep: int = 3,
    a2_enough_rep: int = 3,
    do_copy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fill fully-empty array in an array pair if its paired array has enough non-NA values.
    `mat1` and `mat2` should have shape as (n, _), where `_` means the number of samples can be different in two matrices.
    This is a pair-wise handling method, and can not be used on a group with more than two conditions.

    For each row index, `mat1[i]` has `n1` detections, and `mat2[i]` has `n2` detections.
    If `n1 >= a1_enough_rep` and `n2 == 0`, then `mat2[i][:n1] = 0.0`.
    If `n1 == 0` and `n2 >= a2_enough_rep`, then `mat1[i][:n2] = 0.0`.
    """
    if do_copy:
        mat1 = mat1.copy()
        mat2 = mat2.copy()
    is_filled = np.zeros(mat1.shape[0], dtype=np.bool_)
    for i in prange(mat1.shape[0]):
        mat1[i], mat2[i], is_filled[i] = _check_and_fill_full_empty(
            mat1[i],
            mat2[i],
            a1_enough_rep,
            a2_enough_rep,
        )
    return mat1, mat2, is_filled


@dataclass
class FullEmptyFillingMissingValueHandler(AbstractPairwiseMissingValueHandler):
    """
    Configuration that will call `fill_full_empty` in `do_pairwise_missing_value_handling`.
    """

    min_rep_count: Optional[int] = 3

    min_exp_rep_count: Optional[int] = None
    min_ctrl_rep_count: Optional[int] = None

    _compare_scope: _T_CompareScope = "pairwise"

    def __post_init__(self):
        self.min_exp_rep_count, self.min_ctrl_rep_count = _initialize_paired_values(
            self.min_exp_rep_count,
            self.min_ctrl_rep_count,
            self.min_rep_count,
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
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Check and sample from a normal distribution to fill missing values in one array based on the other array."""
    a1_nonnan_count = (~np.isnan(a1)).sum()
    a2_nonnan_count = (~np.isnan(a2)).sum()
    is_filled = False
    if (
        (a1_nonnan_count >= a1_enough_rep)
        and (a2_nonnan_count <= a2_impute_rep_range[1])
        and (a2_nonnan_count >= a2_impute_rep_range[0])
        and ((np.nanmin(a1) - np.nanmax(a2)) > log_scale_minmax_diff)
    ):
        _n = a1_nonnan_count - a2_nonnan_count
        a2[np.where(np.isnan(a2))[0][:_n]] = np.clip(np.random.normal(np.nanmean(a2), np.nanstd(a1), _n), 1e-4, None)
        is_filled = True
    elif (
        (a2_nonnan_count >= a2_enough_rep)
        and (a1_nonnan_count <= a1_impute_rep_range[1])
        and (a1_nonnan_count >= a1_impute_rep_range[0])
        and ((np.nanmin(a2) - np.nanmax(a1)) > log_scale_minmax_diff)
    ):
        _n = a2_nonnan_count - a1_nonnan_count
        a1[np.where(np.isnan(a1))[0][:_n]] = np.clip(np.random.normal(np.nanmean(a1), np.nanstd(a2), _n), 1e-4, None)
        is_filled = True
    return a1, a2, is_filled


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample from a normal distribution to fill missing values in one array based on the other array.
    `mat1` and `mat2` should have shape as (n, _), where `_` means the number of samples can be different in two matrices.
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
    is_filled = np.zeros(mat1.shape[0], dtype=np.bool_)
    for i in prange(mat1.shape[0]):
        mat1[i], mat2[i], is_filled[i] = _check_and_sample_normal_dist(
            mat1[i],
            mat2[i],
            a1_enough_rep,
            a2_enough_rep,
            a1_impute_rep_range,
            a2_impute_rep_range,
            log_scale_minmax_diff,
        )
    return mat1, mat2, is_filled


@dataclass
class NormalDistSamplingMissingValueHandler(AbstractPairwiseMissingValueHandler):
    """
    Configuration that will call `sample_normal_dist` in `do_pairwise_missing_value_handling`.
    """

    min_rep_count: Optional[int] = 3
    do_imputation_for_rep_range: Optional[tuple[int, int]] = (1, 2)
    log_scale_minmax_diff: int = 1

    min_exp_rep_count: Optional[int] = None
    min_ctrl_rep_count: Optional[int] = None
    do_imputation_for_exp_rep_range: Optional[tuple[int, int]] = None
    do_imputation_for_ctrl_rep_range: Optional[tuple[int, int]] = None

    _compare_scope: _T_CompareScope = "pairwise"

    def __post_init__(self):
        self.min_exp_rep_count, self.min_ctrl_rep_count = _initialize_paired_values(
            self.min_exp_rep_count,
            self.min_ctrl_rep_count,
            self.min_rep_count,
        )
        self.do_imputation_for_exp_rep_range, self.do_imputation_for_ctrl_rep_range = _initialize_paired_values(
            self.do_imputation_for_exp_rep_range,
            self.do_imputation_for_ctrl_rep_range,
            self.do_imputation_for_rep_range,
        )


def sequential_impute(
    mat: np.ndarray,
    min_required_detections: int = 3,
    copy_input: bool = True,
) -> tuple[np.ndarray, bool]:
    """
    Sequentially impute missing values based on the assumption of co-linearity of the values in each column in input.
    This method will iteratively impute missing values row-by-row, from rows with lowerest number of NANs to rows with the highest number of NANs.

    Reference: Sequential imputation for missing values. doi: 10.1016/j.compbiolchem.2007.07.001.
    """
    _mat_raw_order = "C" if mat.flags.c_contiguous else "F"

    if copy_input:
        mat = copy.deepcopy(mat)

    nrows, ncols = mat.shape
    max_nan_for_impute = ncols - min_required_detections

    nan_mat = np.isnan(mat)
    row_nan_count = np.sum(nan_mat, axis=1)

    if row_nan_count.sum() == 0:
        return mat, np.zeros(mat.shape[0], dtype=np.bool_)

    fullfill_row_idx = np.where(row_nan_count == 0)[0]
    n_fullfill = len(fullfill_row_idx)

    row_to_impute_indices = np.where((row_nan_count > 0) & (row_nan_count <= max_nan_for_impute))[0]
    is_filled = np.zeros(mat.shape[0], dtype=np.bool_)
    is_filled[row_to_impute_indices] = True

    # Sort the index of rows to impute by increased nan number
    row_to_impute_indices = row_to_impute_indices[np.argsort(row_nan_count[row_to_impute_indices], kind="stable")]

    fullfill_subx = mat[fullfill_row_idx]
    for i, impute_row_idx in enumerate(row_to_impute_indices):
        fullfill_cov = np.cov(fullfill_subx, rowvar=False)
        fullfill_colmean = np.mean(fullfill_subx, axis=0)

        inv_fullfill_cov = np.linalg.pinv(fullfill_cov)

        nan_idx = np.where(nan_mat[impute_row_idx])[0]
        nonnan_idx = np.where(~nan_mat[impute_row_idx])[0]

        mat[impute_row_idx, nan_idx] = fullfill_colmean[nan_idx] - (
            np.linalg.inv(inv_fullfill_cov[np.ix_(nan_idx, nan_idx)])
            @ inv_fullfill_cov[np.ix_(nan_idx, nonnan_idx)]
            @ (mat[impute_row_idx][nonnan_idx] - fullfill_colmean[nonnan_idx])
        )

        n_fullfill += 1
        fullfill_row_idx = np.hstack((fullfill_row_idx, impute_row_idx))
        fullfill_subx = mat[fullfill_row_idx]

    return np.asarray(mat, order=_mat_raw_order), is_filled


@dataclass
class SequentialImputeMissingValueHandler(AbstractGroupMissingValueHandler):
    """
    Configuration that will call `sequential_impute` in `do_group_missing_value_handling`.
    """

    min_required_detections: int = 3
    copy_input: bool = True

    _compare_scope: _T_CompareScope = "all"


def do_pairwise_missing_value_handling(
    exp_quant_arr: np.ndarray,
    ctrl_quant_arr: np.ndarray,
    config: Union[AbstractPairwiseMissingValueHandler, Sequence[AbstractPairwiseMissingValueHandler]],
    do_copy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Handle missing values in two arrays based on the provided configurations.
    Configurations can be a single instance or a list of instances of objects that inherit from `AbstractPairwiseMissingValueHandler`.
    """
    if config is None:
        config = NullMissingValueHandler()
    if not isinstance(config, Iterable):
        config = (config,)
    fill_types = np.full(exp_quant_arr.shape[0], "", dtype="<U20")
    for conf in config:
        if isinstance(conf, NullMissingValueHandler):
            pass
        elif isinstance(conf, FullEmptyFillingMissingValueHandler):
            exp_quant_arr, ctrl_quant_arr, is_filled = fill_full_empty(
                exp_quant_arr,
                ctrl_quant_arr,
                conf.min_exp_rep_count,
                conf.min_ctrl_rep_count,
                do_copy,
            )
            _idxs = np.where(is_filled)[0]
            fill_types[_idxs] = np.char.add(fill_types[_idxs], "full_empty")
        elif isinstance(conf, NormalDistSamplingMissingValueHandler):
            exp_quant_arr, ctrl_quant_arr, is_filled = sample_normal_dist(
                exp_quant_arr,
                ctrl_quant_arr,
                conf.min_exp_rep_count,
                conf.min_ctrl_rep_count,
                conf.do_imputation_for_exp_rep_range,
                conf.do_imputation_for_ctrl_rep_range,
                conf.log_scale_minmax_diff,
                do_copy,
            )
            _idxs = np.where(is_filled)[0]
            fill_types[_idxs] = np.char.add(fill_types[_idxs], "sample_norm")
        else:
            raise ValueError(f"Unsupported configuration type: {type(conf)}")
    return exp_quant_arr, ctrl_quant_arr, fill_types


def do_group_missing_value_handling(
    mat: np.ndarray,
    config: Optional[Union[AbstractGroupMissingValueHandler, Sequence[AbstractGroupMissingValueHandler]]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Handle missing values in a group of arrays based on the provided configurations.
    """
    if config is None:
        config = NullMissingValueHandler()
    if isinstance(config, AbstractGroupMissingValueHandler):
        config = (config,)
    fill_types = np.full(mat.shape[0], "", dtype="<U20")
    for conf in config:
        if isinstance(conf, NullMissingValueHandler):
            pass
        elif isinstance(conf, SequentialImputeMissingValueHandler):
            mat, is_filled = sequential_impute(mat, conf.min_required_detections, conf.copy_input)
            _idxs = np.where(is_filled)[0]
            fill_types[_idxs] = np.char.add(fill_types[_idxs], "seq_imp")
        else:
            raise ValueError(f"Unsupported configuration type: {type(conf)}")
    return mat, fill_types


def check_paired_mv(
    mat1: np.ndarray,
    mat2: np.ndarray,
    a1_enough_rep: int = 3,
    a2_enough_rep: int = 3,
) -> np.ndarray:
    return np.logical_and(
        (~np.isnan(mat1)).sum(axis=1) >= a1_enough_rep,
        (~np.isnan(mat2)).sum(axis=1) >= a2_enough_rep,
    )


@dataclass
class CheckRemovalMissingValueHandler(AbstractPairwiseMissingValueHandler):
    """
    Configuration that will call `fill_full_empty` in `do_pairwise_missing_value_handling`.
    """

    min_rep_count: Optional[int] = 3
    annotation_col: str = "mv_check_passed"
    remove_not_passed: bool = False

    min_exp_rep_count: Optional[int] = None
    min_ctrl_rep_count: Optional[int] = None

    _compare_scope: _T_CompareScope = "pairwise"

    def __post_init__(self):
        self.min_exp_rep_count, self.min_ctrl_rep_count = _initialize_paired_values(
            self.min_exp_rep_count,
            self.min_ctrl_rep_count,
            self.min_rep_count,
        )


def check_mv_on_df(
    df: pl.DataFrame,
    exp_runs: Sequence[str],
    ctrl_runs: Sequence[str],
    mv_filter_config: CheckRemovalMissingValueHandler,
) -> pl.DataFrame:
    checks = check_paired_mv(
        df.select(exp_runs).to_numpy(),
        df.select(ctrl_runs).to_numpy(),
        mv_filter_config.min_exp_rep_count,
        mv_filter_config.min_ctrl_rep_count,
    )
    if mv_filter_config.annotation_col is not None:
        df = df.with_columns(pl.lit(checks, dtype=pl.Boolean).alias(mv_filter_config.annotation_col))
    if mv_filter_config.remove_not_passed:
        df = df.filter(~checks)
    return df


def do_pairwise_missing_value_handling_on_df(
    df: pl.DataFrame,
    exp_runs: Sequence[str],
    ctrl_runs: Sequence[str],
    config: Optional[Union[AbstractPairwiseMissingValueHandler, Sequence[AbstractPairwiseMissingValueHandler]]] = None,
    mv_filter_config: Optional[CheckRemovalMissingValueHandler] = None,
    attach_back: Union[Literal["none", "drop_raw"], str] = "drop_raw",
    annotation_col: Optional[Union[str, Sequence[str]]] = cm.precursor,
):
    """
    Handle missing values of two conditions in a dataframe.
    This function can generally be used for pairwise missing value handling methods, because once two conditions are modified at the same time, they can not be used in next iteration.

    The output is a dataframe and the columns in it depend on the `attach_back` parameter.
    - When set to "none", the output will have columns [*exp_runs, *ctrl_runs, *annotation_col, "missing_fill_type"], which means the new dataframe only has new quantity columns, "missing_fill_type", and those defined in `annotation_col`.
    - By default, `attach_back` is "drop_raw", and the output will have columns [*all_columns_in_input_df, "missing_fill_type"], where the quantity columns are new ones and the original ones are dropped.
    - Set to a string is similar as "drop_raw", but the original columns are not dropped and renamed with `attach_back` added as a suffix.
    """
    exp_result, ctrl_result, fill_types = do_pairwise_missing_value_handling(
        df.select(exp_runs).to_numpy(),
        df.select(ctrl_runs).to_numpy(),
        config=config,
        do_copy=True,
    )

    result = pl.from_numpy(
        np.hstack((exp_result, ctrl_result)),
        [*exp_runs, *ctrl_runs],
    ).with_columns(pl.lit(fill_types, dtype=pl.Utf8).alias("missing_fill_type"))

    match attach_back:
        case "none":
            df = result.with_columns(df.select(annotation_col))
        case "drop_raw":
            df = df.drop([*exp_runs, *ctrl_runs]).with_columns(result)
        case _:
            df = df.rename({r: f"{r}{attach_back}" for r in [*exp_runs, *ctrl_runs]}).with_columns(result)

    if mv_filter_config is None:
        return df
    return check_mv_on_df(df, exp_runs, ctrl_runs, mv_filter_config)


def do_group_missing_value_handling_on_df(
    df: pl.DataFrame,
    runs: Sequence[str],
    config: Optional[Union[AbstractGroupMissingValueHandler, Sequence[AbstractGroupMissingValueHandler]]] = None,
    raw_values_suffix: Optional[str] = "_raw",
):
    """
    Handle missing values of a group of runs in a dataframe.
    This function receives configs for group-wise missing value handling, and will return a dataframe with the missing values handled.

    Set `raw_values_suffix` to a string to have the original values in the input dataframe attached as a new column with the string added to the original column names.
    """
    mat, fill_types = do_group_missing_value_handling(df.select(runs).to_numpy(order="c"), config)
    result = pl.from_numpy(mat, runs).with_columns(pl.lit(fill_types, dtype=pl.Utf8).alias("missing_fill_type"))

    match raw_values_suffix:
        case None:
            df = df.drop(runs)
        case _:
            df = df.rename({r: f"{r}{raw_values_suffix}" for r in runs})
    return pl.concat([df, result], how="horizontal")
