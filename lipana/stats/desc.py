import itertools
import logging
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from ..utils import (
    AbstractDFManiConfig,
    _T_InputOrAll,
    do_df_mani,
    gather_value_or_all,
)

__all__ = [
    "agg_vec",
    "AbstractDescConfig",
    "calc_ratio",
    "calc_ratio_batch",
    "calc_ratio_on_df",
    "RatioCalcConfig",
    "iqr",
    "cv",
    "calc_cv_on_df",
    "CVCalcConfig",
    "do_desc_summary_on_df",
]

logger = logging.getLogger("lipana")


def agg_vec(
    vec: np.array,
    method: Literal["mean", "median", "absmax", "absmin", "interquartile_mean"],
):
    match method:
        case "mean":
            return np.mean(vec)
        case "median":
            return np.median(vec)
        case "absmax":
            return vec[np.argmax(np.abs(vec))]
        case "absmin":
            return vec[np.argmin(np.abs(vec))]
        case "interquartile_mean":
            return np.mean(np.sort(vec)[int(np.floor(vec.size / 4)) : int(np.ceil(vec.size / 4 * 3))])


@dataclass
class AbstractDescConfig:
    pass


def calc_ratio(
    arr1: np.ndarray,
    arr2: np.ndarray,
    is_log: bool = True,
    temp_reverse_log_scale: Optional[int] = None,
    div_method: Literal["agg_and_divide", "divide_and_agg"] = "divide_and_agg",
    agg_method: Literal["mean", "median", "absmax", "absmin", "interquartile_mean"] = "interquartile_mean",
):
    """
    Do aggregation on log-scaled values might result in results with larger variance.
    Use `temp_reverse_log_scale` to temporarily reverse the log scale for ratio calculation (the result will be reverted to original log scale again).
    """
    if all(np.isnan(arr1)) or all(np.isnan(arr2)):
        return np.nan
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]

    if temp_reverse_log_scale is True:
        temp_reverse_log_scale = 2
    elif temp_reverse_log_scale is False:
        temp_reverse_log_scale = None
    if is_log and (temp_reverse_log_scale is not None):
        arr1 = np.power(temp_reverse_log_scale, arr1)
        arr2 = np.power(temp_reverse_log_scale, arr2)
        is_log = False

    if div_method == "agg_and_divide":
        v1 = agg_vec(arr1, agg_method)
        v2 = agg_vec(arr2, agg_method)
        if is_log:
            result = v1 - v2
        else:
            result = v1 / v2
    elif div_method == "divide_and_agg":
        if is_log:
            result = agg_vec((arr1.reshape(1, -1) - arr2.reshape(-1, 1)).ravel(), agg_method)
        else:
            result = agg_vec((arr1.reshape(1, -1) / arr2.reshape(-1, 1)).ravel(), agg_method)
    else:
        raise ValueError(f"Unsupported div_method: {div_method}")
    if temp_reverse_log_scale is not None:
        return np.log(result) / np.log(temp_reverse_log_scale)
    return result


def calc_ratio_batch(
    arr: np.ndarray,
    arr2: np.ndarray = None,
    is_log: bool = True,
    temp_reverse_log_scale: Optional[int] = None,
    div_method: Literal["agg_and_divide", "divide_and_agg"] = "divide_and_agg",
    agg_method: Literal["mean", "median", "absmax", "absmin", "interquartile_mean"] = "interquartile_mean",
):
    if arr2 is None:
        if (len(arr.shape) != 3) or (arr.shape[1] != 2):
            raise ValueError(f"Expect a three-dim array with shape (n, 2, m) to calc ratios. Got {arr.shape}")
        return calc_ratio_batch(arr[:, 0], arr[:, 1], is_log, temp_reverse_log_scale, div_method, agg_method)
    else:
        if arr.shape[0] != arr2.shape[0]:
            raise ValueError(
                f"Two arrays should have the same shape to calculate ratios. Got {arr.shape} and {arr2.shape}"
            )
        return np.array(
            [
                calc_ratio(arr[i], arr2[i], is_log, temp_reverse_log_scale, div_method, agg_method)
                for i in range(arr.shape[0])
            ]
        )


def calc_ratio_on_df(
    df: pl.DataFrame,
    cond_to_cols_map: dict[str, Sequence[str]],
    base_cond: _T_InputOrAll = None,
    cond_pairs: Union[tuple[str, str], Sequence[tuple[str, str]]] = None,
    is_log: bool = True,
    temp_reverse_log_scale: Optional[int] = None,
    div_method: Literal["agg_and_divide", "divide_and_agg"] = "divide_and_agg",
    agg_method: Literal["mean", "median", "absmax", "absmin", "interquartile_mean"] = "interquartile_mean",
    new_colname_pattern: str = "ratio_{cond1}_to_{cond2}",
) -> pl.DataFrame:
    """
    Calculate the ratio of quantity values between two conditions and attach the results to the dataframe.
    Can set either base condition(s) or pair(s) of conditions, or a mixture of them.
    If both base_cond and cond_pairs are None, will calculate ratios for all available condition pairs.

    Parameters
    ----------
    df : pl.DataFrame
        The quantification dataframe.
    cond_to_cols_map: dict[str, Sequence[str]]
        A dictionary that maps condition names to the column names of the quantity values.
    base_cond : _T_InputOrAll, optional
        The base condition(s) for calculating the ratio, by default None.
        If "all", will calculate ratios for all available conditions pairs.
    cond_pairs : Union[tuple[str, str], Sequence[tuple[str, str]]], optional
        Explicitly define the expected condition pairs to calculate the ratio, by default None.
    is_log : bool, optional
        Whether the quantity values are log-transformed, by default False.
    temp_reverse_log_scale : Optional[int], optional
        The base of the log scale, by default None.
        When set, will temporarily reverse the log scale for ratio calculation (the result will be reverted to original log scale again).
    div_method : Literal[&quot;agg_and_divide&quot;, &quot;divide_and_agg&quot;], optional
        see `calc_ratio_batch` for details, by default "divide_and_agg"
    agg_method : Literal[&quot;mean&quot;, &quot;median&quot;, &quot;absmax&quot;, &quot;absmin&quot;, &quot;interquartile_mean&quot;], optional
        see `calc_ratio_batch` for details, by default "interquartile_mean"
    new_colname_pattern : str, optional
        The pattern of the column names that will be attached to quantification dataframe, by default "ratio_{cond1}_to_{cond2}".

    Returns
    -------
    pl.DataFrame
        The quantification dataframe.
    """
    if (base_cond is None) and (cond_pairs is None):
        base_cond = "all"

    if cond_pairs is None:
        cond_pairs = []
    else:
        if (len(cond_pairs) == 2) and all(isinstance(c, str) for c in cond_pairs):
            cond_pairs = [cond_pairs]
        elif isinstance(cond_pairs, Sequence):
            cond_pairs = list(cond_pairs)
        else:
            raise ValueError(f"Invalid cond_pairs: {cond_pairs}")

    if base_cond is not None:
        base_cond = gather_value_or_all(base_cond, list(cond_to_cols_map.keys()))
        for base in base_cond:
            for comp in cond_to_cols_map.keys():
                if (base != comp) and ((comp, base) not in cond_pairs):
                    cond_pairs.append((comp, base))
    cond_pairs = [
        (c1, c2)
        for c1, c2 in cond_pairs
        if all(col in df.columns for col in itertools.chain(cond_to_cols_map[c1], cond_to_cols_map[c2]))
    ]
    df = df.with_columns(
        pl.lit(
            calc_ratio_batch(
                df.select(cond_to_cols_map[c1]).to_numpy(),
                df.select(cond_to_cols_map[c2]).to_numpy(),
                is_log=is_log,
                temp_reverse_log_scale=temp_reverse_log_scale,
                div_method=div_method,
                agg_method=agg_method,
            ),
            dtype=pl.Float32,
        ).alias(new_colname_pattern.format(cond1=c1, cond2=c2))
        for c1, c2 in cond_pairs
    )
    return df


@dataclass
class RatioCalcConfig(AbstractDescConfig):
    cond_to_cols_map: dict[str, Sequence[str]]
    base_cond: _T_InputOrAll = None
    cond_pairs: Union[tuple[str, str], Sequence[tuple[str, str]]] = None
    is_log: bool = True
    temp_reverse_log_scale: Optional[int] = None
    div_method: Literal["agg_and_divide", "divide_and_agg"] = "divide_and_agg"
    agg_method: Literal["mean", "median", "absmax", "absmin", "interquartile_mean"] = "interquartile_mean"
    new_colname_pattern: str = "ratio_{cond1}_to_{cond2}"


def iqr(value: np.ndarray):
    return np.percentile(value, 75) - np.percentile(value, 25)


def cv(
    value_array: np.ndarray,
    min_reps: int = 3,
    std_ddof: int = 1,
    temp_reverse_log_scale: Optional[int] = None,
    make_percentage: bool = True,
    keep_na: bool = False,
    # round: Optional[int] = None,
    # return_iqr: bool = False,
):
    """
    Parameters
    ----------
    value_array : np.ndarray
        A two-dimensional array with rows as sample and cols as replicates. CV will be performed to each row (dim 0)
    min_reps : int, optional
        Required minimum number of non-NA values, by default 3
    std_ddof : int, optional
        ddof for std, by default 1
    temp_reverse_log_scale : Optional[int], optional
        The base of the log scale, by default None.
        When set, will temporarily reverse the log scale for ratio calculation (the result will be reverted to original log scale again).
    make_percentage : bool, optional
        If true, cv will multi 100, else do nothing, by default True
    keep_na : bool, optional
        Whether to return NAs for those CV-unavailable samples, by default False
    round : Optional[int], optional
        Round the output values, by default None
    return_iqr : bool, optional
        Whether to return IQR of calulated CVs, by default False.
        If True, a tuple like (cvs, iqr) will be returned, otherwise cvs only.
    """

    if len(value_array.shape) != 2:
        raise ValueError(
            f"Expect a two-dim array to calc CV with sample as rows and replicates as cols. "
            f"Current input array has shape {value_array.shape} with {len(value_array.shape)} dim"
        )
    value_array = np.asarray(value_array)
    if temp_reverse_log_scale is not None:
        value_array = np.power(temp_reverse_log_scale, value_array)

    sample_num, rep_num = value_array.shape
    if min_reps > rep_num:
        min_reps = rep_num

    cv_avail_value_idx = np.where((rep_num - np.isnan(value_array).sum(axis=1)) >= min_reps)[0]
    cv_avail_values = value_array[cv_avail_value_idx]
    cvs = np.nanstd(cv_avail_values, axis=1, ddof=std_ddof) / np.nanmean(cv_avail_values, axis=1)
    if make_percentage:
        cvs = cvs * 100

    if keep_na:
        temp = np.zeros(sample_num)
        temp.fill(np.nan)
        temp[cv_avail_value_idx] = cvs
        cvs = temp.copy()
    return cvs


def calc_cv_on_df(
    df: pl.DataFrame,
    cond_to_cols_map: dict[str, Sequence[str]],
    cond: _T_InputOrAll = "all",
    min_reps: int = 3,
    temp_reverse_log_scale: Optional[int] = 2,
    new_colname_pattern: str = "{cond}_cv_{min_reps}reps",
) -> pl.DataFrame:
    """
    Calculate the coefficient of variation (cv) for each condition and attach the result to the dataframe.
    Note: all cvs are made as percentage.

    Parameters
    ----------
    cond_to_cols_map: dict[str, Sequence[str]]
        A dictionary that maps condition names to the column names of the quantity values.
    cond : _T_InputOrAll, optional
        Selected conditions, by default "all"
    min_reps : int, optional
        Minimum quantity values required for calculating CV, by default 3.
        One entry with quantities less than this number in a certain condition will have a cv of NaN.
    temp_reverse_log_scale: Optional[int], optional
        To temporarily reverse the log scale of the quantity values for cv calculation, by default 2.
        If None, will use quantity values as they in the current dataframe.
    new_colname_pattern : str, optional
        The pattern of the column names that will be attached to quantification dataframe, by default "{cond}_cv_{min_reps}reps".
        Can set to something like "{cond}-CV" to omit the annotation of minimum replicates.

    Returns
    -------
    pl.DataFrame
        The dataframe with cv columns attached.
    """
    cond = gather_value_or_all(cond, list(cond_to_cols_map.keys()))
    df = df.with_columns(
        pl.lit(
            cv(
                df.select(cond_to_cols_map[c]).to_numpy(),
                min_reps=min_reps,
                std_ddof=1,
                temp_reverse_log_scale=temp_reverse_log_scale,
                make_percentage=True,
                keep_na=True,
            ),
            dtype=pl.Float32,
        ).alias(
            new_colname_pattern.format(cond=c, min_reps=min_reps)
            if ("min_reps" in new_colname_pattern)
            else new_colname_pattern.format(cond=c)
        )
        for c in cond
    )
    return df


@dataclass
class CVCalcConfig(AbstractDescConfig):
    cond_to_cols_map: dict[str, Sequence[str]]
    cond: _T_InputOrAll = "all"
    min_reps: int = 3
    temp_reverse_log_scale: Optional[int] = 2
    new_colname_pattern: str = "{cond}_cv_{min_reps}reps"


def do_desc_summary_on_df(
    df: pl.DataFrame,
    config: Union[AbstractDescConfig, Sequence[Union[AbstractDescConfig, AbstractDFManiConfig]]],
):
    if isinstance(config, AbstractDescConfig):
        config = (config,)
    for conf in config:
        if isinstance(conf, RatioCalcConfig):
            df = calc_ratio_on_df(df, **conf.__dict__)
        elif isinstance(conf, CVCalcConfig):
            df = calc_cv_on_df(df, **conf.__dict__)
        elif isinstance(conf, AbstractDFManiConfig):
            df = do_df_mani(df, conf)
        else:
            raise ValueError(f"An unexpected config is passed, with type: {type(conf)}")
    return df
