import logging
import random
import string
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union
import itertools

import numpy as np
import polars as pl
import scipy.stats

from ..annotations import attach_annotation_from_other_df
from ..base import AbstractQuantificationReport, ExperimentLayout
from .stats_base import AbstractHypoTestConfig, AbstractTestAdjustConfig, _T_CompareScope
from ..utils import AbstractDFManiConfig, do_df_mani, exec_r_script, read_df_from_parquet_or_tsv

__all__ = [
    "ttest",
    "ttest_batch",
    "ttest_on_df",
    "TTestConfig",
    "exec_limma",
    "LimmaPairwiseConfig",
    "scipy_comb_p_with_nan",
    "combine_pvalues_in_group",
    "PvalueCombineConfig",
    "agg_values_in_group",
    "ValueAggregationConfig",
    "assign_sign",
    "filter_sign_in_group",
    "filter_sign_select_min_one",
    "SignVotingConfig",
    "sign_voting_in_group",
    "scipy_fdr_with_nan",
    "fdr_on_df",
    "FDRConfig",
    "do_hypo_test_on_df",
]

logger = logging.getLogger("lipana")


def ttest(
    arr1: np.ndarray,
    arr2: np.ndarray,
    equal_var: bool = False,
    one_side_alt_when_full_zero: bool = False,
    min_nonnan_count: int = 3,
) -> tuple[float, float]:
    """
    A wrapper of `scipy.stats.ttest_ind` with four changes:
    1. Only support 1-d arrays.
    2. Will return (np.nan, np.nan) if the number of non-NA values in either array is less than `min_nonnan_count`.
    3. Will return (0.0, 1.0) if all values in both arrays are zero.
    4. By default, the alternative hypothesis will be two-sided.
    When `one_side_alt_when_full_zero` is True, the alternative will be "less" if all values in `arr1` are zero,
    and "greater" if all values in `arr2` are zero.

    Parameters
    ----------
    arr1 : np.ndarray
        The first 1-dimensional array of values for the t-test.
    arr2 : np.ndarray
        The second 1-dimensional array of values for the t-test.
    equal_var : bool, optional
        If True, perform a standard independent 2-sample test assuming equal population variances.
        If False, perform Welch's t-test, which does not assume equal population variance.
        Default is False.
    one_side_alt_when_full_zero : bool, optional
        If True, the alternative hypothesis will be one-sided ("less" or "greater") when all values in one of the arrays are zero.
        If False, the alternative hypothesis will always be two-sided.
        Default is False.
    min_nonnan_count : int, optional
        The minimum number of non-NA values required in each array to perform the t-test.
        If either array has fewer than this number of non-NA values, the function will return (np.nan, np.nan).
        Default is 3.
    """
    arr1 = arr1[~np.isnan(arr1)]
    arr2 = arr2[~np.isnan(arr2)]
    if (len(arr1) < min_nonnan_count) or (len(arr2) < min_nonnan_count):
        return np.nan, np.nan

    arr1_nonzero_count = (arr1 != 0).sum()
    arr2_nonzero_count = (arr2 != 0).sum()
    if (arr1_nonzero_count == 0) and (arr2_nonzero_count == 0):
        return 0.0, 1.0

    if one_side_alt_when_full_zero:
        if arr1_nonzero_count == 0:
            alt = "less"
        elif arr2_nonzero_count == 0:
            alt = "greater"
        else:
            alt = "two-sided"
    else:
        alt = "two-sided"
    r = scipy.stats.ttest_ind(arr1, arr2, equal_var=equal_var, alternative=alt, nan_policy="omit")
    return r.statistic, r.pvalue


def ttest_batch(
    mat1: np.ndarray,
    mat2: np.ndarray,
    equal_var: bool = False,
    one_side_alt_when_full_zero: bool = False,
    min_nonnan_count: int = 3,
) -> np.ndarray:
    """
    A batch version of `ttest` to calculate t-test statistics and p-values for each row in two arrays.
    """
    if mat1.shape != mat2.shape:
        raise ValueError(f"Two input arrays should have the same shape. Got {mat1.shape} and {mat2.shape}")
    return np.array(
        [
            ttest(mat1[i], mat2[i], equal_var, one_side_alt_when_full_zero, min_nonnan_count)
            for i in range(mat1.shape[0])
        ]
    )


def _flatten_condition_pairs(
    condition_pairs: Optional[Union[str, Sequence[Union[str, tuple[str, str]]]]],
    all_conditions: Optional[Sequence[str]],
    to_str: bool = False,
):
    """
    condition_pairs : Optional[Union[str, Sequence[Union[str, tuple[str, str]]]]], optional
        A sequence of condition pairs for the hypothesis test.
        When `None`, will omit this argument for `do_limma_pair.R` script, which will do hypothesis tests on all possible condition pairs.
        When str, will be passed to `do_limma_pair.R` script as `condition_pairs`, which will set the provided condition as the control condition.
        When Sequence, will do join. For example, [A, (A, B), (C, B)] will be "A;;A//B;;C//B", which will be flattened as "B//A;C//A;A//B;C//B".

    """
    if condition_pairs is None:
        condition_pairs = list(itertools.combinations(all_conditions, 2))
    elif isinstance(condition_pairs, str):
        condition_pairs = []
        for pair in condition_pairs.split(";;"):
            if "//" in pair:
                condition_pairs.append(tuple(pair.split("//")))
            else:
                condition_pairs.extend([(pair, cond) for cond in all_conditions if pair != cond])
    elif isinstance(condition_pairs, Sequence):
        _condition_pairs = []
        for pair in condition_pairs:
            if isinstance(pair, Sequence):
                if len(pair) != 2:
                    raise ValueError(f"Invalid condition pair: {pair}")
                _condition_pairs.append(pair)
            else:
                _condition_pairs.extend([(pair, cond) for cond in all_conditions if pair != cond])
        condition_pairs = _condition_pairs
    else:
        raise ValueError(f"Invalid condition pairs: {condition_pairs}")

    condition_pairs = sorted(set(condition_pairs))
    if to_str:
        return ";;".join([f"{pair[0]}//{pair[1]}" for pair in condition_pairs])
    return condition_pairs


def ttest_on_df(
    df: pl.DataFrame,
    condition_pairs: Optional[Union[str, Sequence[Union[str, tuple[str, str]]]]] = None,
    exp_layout: Optional[ExperimentLayout] = None,
    entry_col: Optional[str] = "precursor",
    recollected_columns: Optional[Union[str, Sequence[str]]] = None,
    equal_var: bool = False,
    one_side_alt_when_full_zero: bool = False,
    min_nonnan_count: int = 3,
    t_col: str = "t",
    p_col: str = "pvalue",
) -> pl.DataFrame:
    """
    Perform t-tests on defined groups of runs in a dataframe.
    """
    condition_pairs = _flatten_condition_pairs(condition_pairs, exp_layout.all_conditions, to_str=False)

    result = pl.concat(
        (
            pl.from_numpy(
                ttest_batch(
                    df.select(exp_layout.condition_to_runs_map[condition_pair[0]]).to_numpy(),
                    df.select(exp_layout.condition_to_runs_map[condition_pair[1]]).to_numpy(),
                    equal_var,
                    one_side_alt_when_full_zero,
                    min_nonnan_count,
                ),
                [t_col, p_col],
            ).with_columns(
                pl.Series(
                    name="pair",
                    values=[f"{condition_pair[0]}_vs_{condition_pair[1]}" for condition_pair in condition_pairs],
                ),
                df[entry_col],
            ),
        )
        for condition_pair in condition_pairs
    )

    if recollected_columns is not None:
        result = attach_annotation_from_other_df(
            result,
            other_df=df,
            annotation_cols=recollected_columns,
            on=entry_col,
        )

    return result


@dataclass
class TTestConfig(AbstractHypoTestConfig):
    """
    Configuration for calling `ttest_on_df` in `do_stats_on_df`.
    """

    exp_layout: Optional[ExperimentLayout] = None
    entry_col: Optional[str] = None
    recollected_columns: Optional[Union[str, Sequence[str]]] = None

    equal_var: bool = False
    one_side_alt_when_full_zero: bool = False
    min_nonnan_count: int = 3
    t_col: str = "t"
    p_col: str = "pvalue"

    _compare_scope: _T_CompareScope = "pairwise"


default_limma_output_column_map = {
    "logFC": "log2_fc",
    "t": "t",
    "P.Value": "pvalue",
    "adj.P.Val": "adj_pvalue",
}


def exec_limma(
    in_data: Union[AbstractQuantificationReport, pl.DataFrame, str, Path],
    condition_pairs: Optional[Union[str, Sequence[Union[str, tuple[str, str]]]]] = None,
    exp_layout: Optional[ExperimentLayout] = None,
    entry_name: Optional[str] = None,
    output_column_map: Optional[Union[Literal["default"], dict[str, str]]] = "default",
    output_column_map_override: Optional[dict[str, str]] = None,
    recollected_columns: Optional[Union[str, Sequence[str]]] = None,
    dump_in_df_to: Optional[Union[str, Path]] = None,
    del_files: bool = False,
    rscript_exec: Union[str, Path] = "Rscript",
) -> tuple[pl.DataFrame, Path, Path]:
    """
    Perform hypothesis tests by running external R script `do_limma_pair.R`.
    Receives input quantification data, annotation data, and optional condition pairs.
    Returns the output dataframe, input path, and output path.

    Input data should have quantity columns requried in annotation data, and an entry column to annotate each row.

    The original output file contains 8 columns: "ID", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B", "pair".
    Where the first 7 columns are from Limma, and "pair" has values as "{cond1}_vs_{cond2}".
    If `output_column_map` is None, the dataframe will be returned as these columns as.
    By default, `output_column_map` is set to "default" and the returned dataframe will have column names as:
    `entry_name` (default "Entry"), "log2_fc", "AveExpr", "t", "pvalue", "adj_pvalue", "B", "pair".

    Note: In the script for running Limma, each condition requires at least 2 runs with quantification values. This is a fixed number and can not be defined here.
    Note: Limma receives quantity values in log2 scale.

    Parameters
    ----------
    in_data : Union[AbstractQuantificationReport, pl.DataFrame, str, Path]
        Input data as "AbstractQuantificationReport", polars DataFrame, or path.
    exp_layout : Optional[ExperimentLayout], optional
        `ExperimentLayout` to specify the experiment layout for condition to runs mapping.
        Only when `in_data` is `AbstractQuantificationReport`, this parameter can be None.
    condition_pairs : Optional[Union[str, Sequence[Union[str, tuple[str, str]]]]], optional
        A sequence of condition pairs for the hypothesis test.
        When `None`, will omit this argument for `do_limma_pair.R` script, which will do hypothesis tests on all possible condition pairs.
        When str, will be passed to `do_limma_pair.R` script as `condition_pairs`, which will set the provided condition as the control condition.
        When Sequence, will do join. For example, [A, (A, B), (C, B)] will be "A;;A//B;;C//B", which will be flattened as "B//A;C//A;A//B;C//B".
    entry_name : Optional[str], optional
        An optional string to specify the entry name in the input data, by default None.
        When `in_data` is "AbstractQuantificationReport", this parameter will be `in_data.entry_level`.
        Else, the default value is "Entry", which should be in the dataframe.
    output_column_map : Optional[dict[str, str]], optional
        An optional dict to map the output column names to the desired names, like {"ID": "cut_site", ...}, by default None.
    dump_in_df_to : Optional[Union[str, Path]], optional
        Optional path to dump the dataframe of quantification data, by default None.
        If in_data is not a path and dump_to is None, a temporary input file will be created, and the output file will also be in the same directory.
        For annotation file, the path will be the same as the input file with a suffix ".annotation.txt".
    del_files : bool, optional
        Whether to delete the input and output files, by default False.
    rscript_exec : Union[str, Path], optional
        Path to Rscript executable, by default "Rscript" to use the system default.

    Returns
    -------
    tuple[pl.DataFrame, Path, Path]
        returns a three-element tuple of output dataframe, the limma input path, and the limma output path
    """

    rscript_exec = str(rscript_exec)
    script_path = Path(__file__).resolve().parent.parent.joinpath("ext", "do_limma_pair.R")
    logger.info(f'Run hypothesis test via Limma, using Rscript: "{rscript_exec}", and script: "{str(script_path)}"')

    if isinstance(in_data, (str, Path)):
        input_path = Path(in_data).resolve()
        logger.info(f'limma input file path: "{str(input_path)}"')
        in_data = read_df_from_parquet_or_tsv(input_path)
    elif isinstance(in_data, (pl.DataFrame, AbstractQuantificationReport)):
        input_path = (
            Path(dump_in_df_to).resolve()
            if (dump_in_df_to is not None)
            else Path(tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name)
        )
        logger.info(f'write temporary limma input file to: "{str(input_path)}"')
        if isinstance(in_data, AbstractQuantificationReport):
            in_data.dump(path=input_path)
            exp_layout = in_data.exp_layout
            entry_name = in_data.entry_level
            in_data = in_data.df
        else:
            in_data.write_csv(input_path, separator="\t")
    else:
        raise ValueError(f"Unsupported input data type: `{type(in_data)}`")
    args = [str(input_path)]

    if exp_layout is None:
        raise ValueError("Experiment layout information is required for limma test")

    if condition_pairs is not None:
        if not isinstance(condition_pairs, str):
            _condition_pairs = []
            for cond in condition_pairs:
                if isinstance(cond, tuple):
                    _condition_pairs.append(f"{cond[0]}//{cond[1]}")
                else:
                    _condition_pairs.append(cond)
            condition_pairs = ";;".join(_condition_pairs)

    anno_path = (
        Path(dump_in_df_to).resolve().with_suffix(".annotation.txt")
        if (dump_in_df_to is not None)
        else Path(tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name)
    )
    logger.info(f'write temporary annotation file to: "{str(anno_path)}"')
    exp_layout.dump(anno_path)
    args.append(str(anno_path))

    if entry_name is None:
        entry_name = "Entry"
    args.append(entry_name)

    if condition_pairs is not None:
        args.append(condition_pairs)

    exec_r_script(rscript_exec, script_path, *args)

    output_path = input_path.parent.joinpath(f"{input_path.name}-limma_output.txt")
    logger.info(f'Load limma output from: "{str(output_path)}"')
    stats_data = pl.read_csv(output_path, separator="\t", null_values="NA")

    if del_files:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()

    if output_column_map is not None:
        if isinstance(output_column_map, str) and (output_column_map == "default"):
            output_column_map = default_limma_output_column_map
            output_column_map["ID"] = entry_name
        if output_column_map_override is not None:
            output_column_map.update(output_column_map_override)
    if (output_column_map is None) and (output_column_map_override is not None):
        output_column_map = output_column_map_override
    if output_column_map is not None:
        stats_data = stats_data.rename(output_column_map)
    if recollected_columns is not None:
        stats_data = attach_annotation_from_other_df(
            stats_data,
            other_df=in_data,
            annotation_cols=recollected_columns,
            on=entry_name,
        )
    return stats_data, input_path, output_path


@dataclass
class LimmaPairwiseConfig(AbstractHypoTestConfig):
    exp_layout: Optional[ExperimentLayout] = None
    entry_name: Optional[str] = None
    output_column_map: Optional[Union[Literal["default"], dict[str, str]]] = "default"
    output_column_map_override: Optional[dict[str, str]] = None
    recollected_columns: Optional[Union[str, Sequence[str]]] = None
    dump_in_df_to: Optional[Union[str, Path]] = None
    del_files: bool = True
    rscript_exec: Union[str, Path] = "Rscript"

    _compare_scope: _T_CompareScope = "pairwise"


def scipy_comb_p_with_nan(
    x,
    method="fisher",
    weights=None,
    ignore_nan: bool = True,
    return_p_only: bool = True,
) -> Union[float, tuple]:
    """
    Wrapper for scipy.stats.combine_pvalues that handles NaN values.

    Parameters
    ----------
    x : array-like
        Array of p-values to combine
    method : str, optional
        Method for combining p-values
    weights : array-like, optional
        Weights for combining p-values
    ignore_nan : bool, optional
        If True, skip NaN values. If False, return NaN if any value is NaN
    return_p_only : bool, optional
        If True, return only combined p-value without test statistic

    Returns
    -------
    Union[float, tuple]
        Combined p-value if return_p_only=True, else (statistic, p-value) tuple
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    else:
        x = np.asarray(x)
    if ignore_nan:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    r = scipy.stats.combine_pvalues(x, method=method, weights=weights)
    if return_p_only:
        return r[1]
    return r


def combine_pvalues_in_group(
    df: pl.DataFrame,
    group_col: Union[str, Sequence[str]],
    filter_condition_for_combining: Optional[pl.Expr] = None,
    method: str = "fisher",
    p_col: str = "pvalue",
    new_p_col: Optional[str] = None,
    ignore_nan: bool = True,
    return_p_only: bool = True,
) -> pl.DataFrame:
    """
    Combine p-values in a group of rows based on the provided method.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing p-values to be combined
    group_col : Union[str, Sequence[str]]
        Column name(s) to group by before combining p-values
    filter_condition_for_combining : Optional[pl.Expr], optional
        Polars expression to indicate the rows to do p-value combining.
        If None, all rows within groups are used.
    method : str, optional
        Method to combine p-values. See `scipy.stats.combine_pvalues`
    p_col : str, optional
        Name of column containing p-values to combine, defaults to "pvalue"
    new_p_col : Optional[str], optional
        Name for the output column with combined p-values.
        If None, overwrites the input p-value column.
    ignore_nan : bool, optional
        If True, skip NaN values when combining p-values.
        If False, return NaN if any input p-value is NaN.
        Defaults to True.
    return_p_only : bool, optional
        If True, return only combined p-values.
        If False, return additional statistics depending on method.
        Defaults to True.

    Returns
    -------
    pl.DataFrame
        DataFrame with added column containing combined p-values per group.
        Original DataFrame structure is preserved, with combined p-values
        repeated for all rows within each group.
    """
    if new_p_col is None:
        new_p_col = p_col
    comb_expr = (
        pl.col(p_col)
        .map_batches(
            partial(
                scipy_comb_p_with_nan,
                method=method,
                ignore_nan=ignore_nan,
                return_p_only=return_p_only,
            )
        )
        .over(pl.col(group_col))
    )
    if filter_condition_for_combining is not None:
        comb_expr = pl.when(filter_condition_for_combining).then(comb_expr).otherwise(pl.lit(np.nan))
    return df.with_columns(comb_expr)


@dataclass
class PvalueCombineConfig(AbstractTestAdjustConfig):
    """
    Configuration for combining p-values within groups.

    Attributes
    ----------
    group_col : Union[str, Sequence[str]]
        Column(s) to group by before combining p-values
    filter_condition_for_combining : Optional[pl.Expr]
        Filter condition for rows to include in combination
    method : str
        Method for combining p-values
    p_col : str
        Column containing p-values
    new_p_col : Optional[str]
        Name for output column with combined p-values
    ignore_nan : bool
        Whether to ignore NaN values
    return_p_only : bool
        Whether to return only p-values without test statistics
    """

    group_col: Union[str, Sequence[str]] = "cut_site"
    filter_condition_for_combining: Optional[pl.Expr] = None
    method: str = "fisher"
    p_col: str = "pvalue"
    new_p_col: Optional[str] = None
    ignore_nan: bool = True
    return_p_only: bool = True

    _compare_scope: _T_CompareScope = "all"


def agg_values_in_group(
    df: pl.DataFrame,
    group_col: Union[str, Sequence[str]] = ("pair", "cut_site"),
    agg_col: Union[str, Sequence[str]] = ("fc", "pvalue"),
    agg_func: Union[Callable, Sequence[Callable]] = (
        np.nanmedian,
        scipy_comb_p_with_nan,
    ),
    recollected_columns: Optional[Union[str, Sequence[str]]] = None,
    recollection_attach_method: Literal["leftjoin", "agg_leftjoin"] = "leftjoin",
):
    """
    Aggregate values within groups using specified functions.
    Generally, this method is used to combine p-values and aggregate FCs within a group at the same time.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    group_col : Union[str, Sequence[str]]
        Column(s) to group by
    agg_col : Union[str, Sequence[str]]
        Column(s) to aggregate
    agg_func : Union[Callable, Sequence[Callable]]
        Function(s) for aggregation (e.g. np.nanmedian, scipy_comb_p_with_nan)
    recollected_columns : Optional[Union[str, Sequence[str]]]
        Additional columns to include in output
    recollection_attach_method : Literal["leftjoin", "agg_leftjoin"]
        Method for attaching recollected columns

    Returns
    -------
    pl.DataFrame
        DataFrame with aggregated values per group
    """
    if isinstance(agg_col, str):
        agg_col = [agg_col]
    if isinstance(agg_func, (Callable, PvalueCombineConfig)):
        agg_func = [agg_func]
    if len(agg_col) != len(agg_func):
        raise ValueError(
            f"Length of `agg_col` and `agg_method` should be the same. Got {len(agg_col)} and {len(agg_func)}"
        )
    agg_col = list(agg_col)
    agg_func = list(agg_func)

    agg_df = (
        df.group_by(group_col)
        .agg(pl.col(c).map_batches(f) for c, f in zip(agg_col, agg_func))
        .with_columns(pl.col(c).list.get(0) for c in agg_col)
    )
    if recollected_columns is not None:
        agg_df = attach_annotation_from_other_df(
            agg_df,
            other_df=df,
            annotation_cols=recollected_columns,
            on=group_col,
            method=recollection_attach_method,
        )
    return agg_df


@dataclass
class ValueAggregationConfig(AbstractTestAdjustConfig):
    """
    Configuration for aggregating values within groups.

    Attributes
    ----------
    group_col : Union[str, Sequence[str]]
        Column(s) to group by for aggregation
    agg_col : Union[str, Sequence[str]]
        Column(s) to aggregate
    agg_func : Union[Callable, Sequence[Callable]]
        Function(s) to use for aggregation
    recollected_columns : Optional[Union[str, Sequence[str]]]
        Additional columns to recollect after aggregation
    """

    group_col: Union[str, Sequence[str]] = ("pair", "cut_site")
    agg_col: Union[str, Sequence[str]] = ("fc", "pvalue")
    agg_func: Union[Callable, Sequence[Callable]] = (
        np.nanmedian,
        scipy_comb_p_with_nan,
    )
    recollected_columns: Optional[Union[str, Sequence[str]]] = None

    _compare_scope: _T_CompareScope = "all"


def assign_sign(
    df: pl.DataFrame,
    value_col: Optional[Union[str, float]] = None,
    pos_sign_gt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    pos_sign_lt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    neg_sign_gt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    neg_sign_lt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    group_col: Optional[Union[str, Sequence[str]]] = ("pair", "cut_site"),
    row_sign_col: str = "row_sign",
    group_sign_col: str = "group_sign",
):
    """
    Assign a column `row_sign_col` to the dataframe based on the sign of the values in the input columns.
    If `group_col` is provided, will also assign a column `group_sign_col` to the dataframe based on the sign of the values in the group.

    Singly define `value_col` will assign the sign (+1, -1, or 0) of the values in that column.
    When `value_col` is None, will assign the sign based on the provided conditions from four parameters `[pos|neg]_sign_[gt|lt]_col_value`.
    `pos_sign_gt_col_value` and related parameters should have the format like `("col_name", value)`, or a list of such tuples.

    For example of inputs, a dataframe has two entry columns "cut_site" and "precursor".
    Here `value_col` can be `"t"`, and `group_col` can be `("pair", "cut_site")`.
    Else, `value_col` is `None`, `pos_sign_gt_col_value` is `("log2_fc", 1.0)`, `pos_sign_lt_col_value` is `("pvalue", 0.05)`, and `neg_sign_lt_col_value` is `[("log2_fc", -1.0), ("pvalue", 0.05)]`.

    Note: The group sign is determined by the sum of row signs in each group, which means row sign of 0 contributes to nothing.
    This can leads to a group sign equal to +1 or -1 even if there are 10 rows with sign equals 0 and only one row with sign +1 or -1.
    This can happen if the row sign is got by strict rules, such as some hard thresholds for fold change and p-value.
    If hard rules are used, should be caution in further analysis. Can refer to `filter_sign_manual_rule` to see the additional processes to handle this.
    """

    def cond_expr(
        col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]],
        direction: Literal["gt", "lt"],
    ):
        if col_value is None:
            return None
        if (len(col_value) == 2) and (isinstance(col_value[0], str) and isinstance(col_value[1], (int, float))):
            col_value = [col_value]
        expr = pl.lit(True)
        for c, v in col_value:
            expr = expr & (getattr(pl.col(c), direction)(v))
        return expr

    def comb_expr(exprs: Sequence[pl.Expr]):
        exprs = [e for e in exprs if e is not None]
        if len(exprs) == 0:
            return pl.lit(False)
        expr = exprs[0]
        for e in exprs[1:]:
            expr = expr & e
        return expr

    if value_col is None:
        pos_expr = comb_expr((cond_expr(pos_sign_gt_col_value, "gt"), cond_expr(pos_sign_lt_col_value, "lt")))
        neg_expr = comb_expr((cond_expr(neg_sign_gt_col_value, "gt"), cond_expr(neg_sign_lt_col_value, "lt")))
        df = df.with_columns(
            pl.when(pos_expr).then(pl.lit(1)).when(neg_expr).then(pl.lit(-1)).otherwise(pl.lit(0)).alias(row_sign_col)
        )
    else:
        df = df.with_columns(pl.col(value_col).sign().alias(row_sign_col))

    if group_col is None:
        return df
    return df.with_columns(
        pl.col(row_sign_col)
        .drop_nans()
        .sum()
        .over(pl.col(group_col))
        .sign()
        # .replace_strict({1: "pos", 0: "zero", -1: "neg"})
        .alias(group_sign_col)
    )


def filter_sign_in_group(
    df: pl.DataFrame,
    row_sign_col: str = "row_sign",
    group_sign_col: str = "group_sign",
    drop_opposite_sign: bool = True,
    drop_bidirectional_balanced_group: bool = True,
):
    """
    Filter input dataframe via the sign in defined columns within each group.
    When `drop_opposite_sign` is True, will drop rows with `row_sign_col != group_sign_col` in each group.
    When `drop_bidirectional_balanced_group` is True, will drop rows with `group_sign_col == 0` in each group.
    """
    match drop_opposite_sign, drop_bidirectional_balanced_group:
        case True, True:
            return df.filter(pl.col(row_sign_col).eq(pl.col(group_sign_col)))
        case True, False:
            return df.filter(
                pl.when(pl.col(group_sign_col).ne(0))
                .then(pl.col(row_sign_col).eq(pl.col(group_sign_col)))
                .otherwise(pl.lit(True))
            )
        case False, True:
            return df.filter(pl.col(group_sign_col).ne(0))
    return df


def filter_sign_select_min_one(
    df: pl.DataFrame,
    value_col: str = "pvalue",
    group_col: Union[str, Sequence[str]] = ("pair", "cut_site"),
    row_sign_col: str = "row_sign",
    group_sign_col: str = "group_sign",
):
    n_per_group_col = "".join(random.sample(string.ascii_lowercase, 10))
    sel_min_expr = pl.col(value_col).eq(pl.col(value_col).min().over(group_col))

    return (
        df.with_columns(pl.col(row_sign_col).len().over(group_col).alias(n_per_group_col))
        .filter(
            pl
            # First filter out the groups have sum of direction equals to 0
            .when(pl.col(group_sign_col).eq(0))
            .then(pl.lit(False))
            # Then filter out the groups have abs of sum of direction <= number of rows / 2
            .when((pl.col(row_sign_col).sum().abs().le(pl.col(row_sign_col).len().truediv(2))).over(group_col))
            .then(pl.lit(False))
            # Groups have only one row with sign +/- 1
            .when(pl.col(n_per_group_col).eq(1))
            .then(pl.lit(True))
            # Groups have two rows with same sign +1 or -1, select one based on min of a column
            .when(pl.col(n_per_group_col).eq(2))
            .then(sel_min_expr)
            # Because groups have already been filtered to have at least half of rows with +1 or -1 signs
            # When rows >=3 in each group, select one with min of a column from those rows with same sign of the group
            # .when(pl.col(n_per_group_col).mod(2).eq(1))
            # .when(pl.col(n_per_group_col).mod(2).eq(0))
            .otherwise(
                pl.when(pl.col(row_sign_col).eq(pl.col(group_sign_col))).then(sel_min_expr).otherwise(pl.lit(False))
            )
        )
        .drop(n_per_group_col)
    )


@dataclass
class SignVotingConfig(AbstractTestAdjustConfig):
    """
    By default, `drop_opposite_sign`, `drop_bidirectional_balanced_group`, and `filter_value_col` are all None.
    And no filtering will be conducted, just assign the sign columns.

    If `drop_opposite_sign` or `drop_bidirectional_balanced_group` is True, will filter the dataframe based on `filter_sign_in_group`.
    Else, and `filter_value_col` is provided, will filter the dataframe based on `filter_sign_manual_rule`.
    """

    sign_value_col: Optional[Union[str, float]] = None
    pos_sign_gt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None
    pos_sign_lt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None
    neg_sign_gt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None
    neg_sign_lt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None

    group_col: Optional[Union[str, Sequence[str]]] = ("pair", "cut_site")
    row_sign_col: str = "row_sign"
    group_sign_col: str = "group_sign"

    drop_opposite_sign: Optional[bool] = None
    drop_bidirectional_balanced_group: Optional[bool] = None
    filter_value_col: Optional[str] = None

    _compare_scope: _T_CompareScope = "all"


def sign_voting_in_group(
    df: pl.DataFrame,
    config: SignVotingConfig,
):
    df = assign_sign(
        df,
        value_col=config.sign_value_col,
        pos_sign_gt_col_value=config.pos_sign_gt_col_value,
        pos_sign_lt_col_value=config.pos_sign_lt_col_value,
        neg_sign_gt_col_value=config.neg_sign_gt_col_value,
        neg_sign_lt_col_value=config.neg_sign_lt_col_value,
        group_col=config.group_col,
        row_sign_col=config.row_sign_col,
        group_sign_col=config.group_sign_col,
    )
    if config.drop_opposite_sign is True or config.drop_bidirectional_balanced_group is True:
        return filter_sign_in_group(
            df,
            row_sign_col=config.row_sign_col,
            group_sign_col=config.group_sign_col,
            drop_opposite_sign=config.drop_opposite_sign,
            drop_bidirectional_balanced_group=config.drop_bidirectional_balanced_group,
        )
    if config.filter_value_col is not None:
        return filter_sign_select_min_one(
            df,
            value_col=config.filter_value_col,
            group_col=config.group_col,
            row_sign_col=config.row_sign_col,
            group_sign_col=config.group_sign_col,
        )
    return df


def scipy_fdr_with_nan(x, method=Literal["BH", "BY"]) -> np.ndarray:
    """
    A wrapper of `scipy.stats.false_discovery_control` to handle NaN values in the input array.
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy().copy()
    else:
        x = np.asarray(x)
    idx = np.where(~np.isnan(x))[0]
    x[idx] = scipy.stats.false_discovery_control(x[idx], method=method)
    return x


def fdr_on_df(
    df: pl.DataFrame,
    group: Optional[Union[str, Sequence[str]]] = None,
    pre_filter: Optional[pl.Expr] = None,
    p_col: str = "pvalue",
    new_col_name: str = "adjp",
    method: Literal["BH", "BY"] = "BH",
):
    """
    FDR for whole column or within a group.
    This function will perform FDR correction on the input DataFrame (optionally by group).

    Generally, a group can be a condition pair so that FDR is calculated within the context of paired two conditions.
    When group contains the protein group, the context can be further narrowed down to the protein level,
    which can be helpful to find significantly different quantities (or surface accessibility) of entries from one protein.

    Do pre-filtering will only calculate FDR for those expected entries, for example to filter out fully-trp peptides or restricted cut sites.
    In this case, only those kept entries have FDR values, and the rest will be NaN.
    """
    if pre_filter is None:
        pre_filter = True

    fdr_expr = pl.col(p_col).map_batches(partial(scipy_fdr_with_nan, method=method))
    if group is not None:
        fdr_expr = fdr_expr.over(group)

    return df.with_columns(pl.when(pre_filter).then(fdr_expr).otherwise(pl.lit(np.nan)).alias(new_col_name))


@dataclass
class FDRConfig(AbstractTestAdjustConfig):
    group: Optional[Union[str, Sequence[str]]] = None
    pre_filter: Optional[pl.Expr] = None
    p_col: str = "pvalue"
    new_col_name: str = "adj_pvalue"
    method: Literal["BH", "BY"] = "BH"

    _compare_scope: _T_CompareScope = "all"


def do_hypo_test_on_df(
    df: pl.DataFrame,
    config: Optional[Union[AbstractHypoTestConfig, Sequence[AbstractHypoTestConfig]]] = None,
    condition_pairs: Optional[Union[str, Sequence[Union[str, tuple[str, str]]]]] = None,
    exp_layout: Optional[ExperimentLayout] = None,
) -> pl.DataFrame:
    """
    Perform statistical analysis on a DataFrame based on the provided configuration(s).

    This function applies a series of statistical operations to the DataFrame, such as t-tests, Limma analysis,
    p-value combination, value aggregation, sign voting, and FDR correction. The operations are determined by the
    configuration(s) provided in the `config` parameter.

    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing the data to be analyzed.
    config : Union[AbstractHypoTestConfig, Sequence[AbstractHypoTestConfig]]
        A single configuration object or a sequence of configuration objects that define the statistical operations
        to be performed. The configuration objects can be instances of `TTestConfig` or `LimmaConfig`.

    """
    if isinstance(config, AbstractHypoTestConfig):
        config = (config,)
    if sum(isinstance(conf, (TTestConfig, LimmaPairwiseConfig)) for conf in config) > 1:
        raise ValueError("Only one of `TTestConfig` and `LimmaConfig` can be provided in the config list")
    for conf in config:
        if isinstance(conf, TTestConfig):
            df = ttest_on_df(
                df,
                condition_pairs=condition_pairs,
                exp_layout=exp_layout,
                entry_col=conf.entry_col,
                recollected_columns=conf.recollected_columns,
                equal_var=conf.equal_var,
                one_side_alt_when_full_zero=conf.one_side_alt_when_full_zero,
                min_nonnan_count=conf.min_nonnan_count,
                t_col=conf.t_col,
                p_col=conf.p_col,
            )
        elif isinstance(conf, LimmaPairwiseConfig):
            df, _, _ = exec_limma(
                df,
                condition_pairs=condition_pairs,
                exp_layout=exp_layout,
                entry_name=conf.entry_name,
                output_column_map=conf.output_column_map,
                output_column_map_override=conf.output_column_map_override,
                recollected_columns=conf.recollected_columns,
                dump_in_df_to=conf.dump_in_df_to,
                del_files=conf.del_files,
                rscript_exec=conf.rscript_exec,
            )
        else:
            raise ValueError(f"An unexpected config is passed, with type: {type(conf)}")
    return df


_T_TestAdjustConfig = Union[
    AbstractTestAdjustConfig,
    AbstractDFManiConfig,
    Callable,
]


def do_test_adjust_on_df(
    df: pl.DataFrame,
    config: Optional[Union[_T_TestAdjustConfig, Sequence[_T_TestAdjustConfig]]] = None,
) -> pl.DataFrame:
    """
    Perform adjustments on hypothesis test results based on the provided configuration(s).

    This function applies a series of statistical operations to the DataFrame, such as p-value combination,
    value aggregation, sign voting, and FDR correction. The operations are determined by the
    configuration(s) provided in the `config` parameter.
    """
    if isinstance(config, _T_TestAdjustConfig):
        config = (config,)
    for conf in config:
        if isinstance(conf, PvalueCombineConfig):
            df = combine_pvalues_in_group(
                df,
                group_col=conf.group_col,
                filter_condition_for_combining=conf.filter_condition_for_combining,
                method=conf.method,
                p_col=conf.p_col,
                new_p_col=conf.new_p_col,
                ignore_nan=conf.ignore_nan,
                return_p_only=conf.return_p_only,
            )
        elif isinstance(conf, ValueAggregationConfig):
            df = agg_values_in_group(
                df,
                group_col=conf.group_col,
                agg_col=conf.agg_col,
                agg_func=conf.agg_func,
                recollected_columns=conf.recollected_columns,
            )
        elif isinstance(conf, SignVotingConfig):
            df = sign_voting_in_group(df, conf)
        elif isinstance(conf, FDRConfig):
            df = fdr_on_df(
                df,
                group=conf.group,
                pre_filter=conf.pre_filter,
                p_col=conf.p_col,
                new_col_name=conf.new_col_name,
                method=conf.method,
            )
        elif isinstance(conf, AbstractDFManiConfig):
            df = do_df_mani(df, conf)
        elif isinstance(conf, Callable):
            df = conf(df)
        else:
            raise ValueError(f"An unexpected config is passed, with type: {type(conf)}")
    return df
