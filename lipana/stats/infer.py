import itertools
import logging
import random
import string
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl
import scipy.stats

from ..annotations import attach_annotation_from_other_df
from ..base import AbstractQuantificationReport, ExperimentLayout
from ..utils import (
    AbstractDFManiConfig,
    do_df_mani,
    exec_r_script,
    filter_top_n_by_group,
    flatten_list,
    read_df_from_parquet_or_tsv,
)
from .stats_base import AbstractHypoTestConfig, AbstractTestAdjustConfig, _T_CompareScope

__all__ = [
    "_prepare_filter_condition",
    "_generate_nan_null_check_expr",
    "_flatten_condition_pairs",
    "ttest",
    "ttest_batch",
    "ttest_on_df",
    "TTestConfig",
    "exec_limma",
    "LimmaPairwiseConfig",
    "assign_sign",
    "check_sign_direction_in_group",
    "SignCheckingConfig",
    "check_sign_in_group",
    "TopKSelectionConfig",
    "select_top_k_in_group",
    "scipy_comb_p_with_nan",
    "combine_pvalues_in_group",
    "PvalueCombineConfig",
    "agg_values_in_group",
    "ValueAggregationConfig",
    "scipy_fdr_with_nan",
    "fdr_on_df",
    "FDRConfig",
    "do_hypo_test_on_df",
    "do_test_adjust_on_df",
]

logger = logging.getLogger("lipana")


def _prepare_filter_condition(
    condition: Optional[Union[str, pl.Expr]] = None,
    true_expr_if_none: bool = True,
) -> Optional[pl.Expr]:
    if condition is None:
        if true_expr_if_none:
            return pl.lit(True)
        else:
            return None
    elif isinstance(condition, str):
        return pl.col(condition)
    elif isinstance(condition, pl.Expr):
        return condition
    else:
        raise ValueError(f"`condition` must be a string or a pl.Expr, got {type(condition)}")


def _generate_nan_null_check_expr(col: Union[str, Sequence[str]]) -> pl.Expr:
    if isinstance(col, str):
        return (
            pl.when(pl.col(col).is_nan())
            .then(pl.lit(True))
            .when(pl.col(col).is_null())
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
        )
    else:
        c1 = col[0]
        expr = pl.when(pl.col(c1).is_nan()).then(pl.lit(True)).when(pl.col(c1).is_null()).then(pl.lit(True))
        for c in col[1:]:
            expr = expr.when(pl.col(c).is_nan()).then(pl.lit(True)).when(pl.col(c).is_null()).then(pl.lit(True))
        return expr.otherwise(pl.lit(False))


def _broadcast_func_out_len(func: Callable, retain_nan: bool = True) -> Union[pl.Series, np.ndarray]:
    """
    Broadcast the output length of a function to the length of the input.
    """

    def _inner(x):
        x = np.asarray(x)
        in_len = len(x)
        func_out = func(x)
        if hasattr(func_out, "__len__"):
            if in_len != len(func_out):
                raise ValueError(f"Length of input and output are different. Got {in_len} and {len(func_out)}")
            return pl.Series(func_out)
        else:
            out = np.full(in_len, func_out)
            if retain_nan:
                idx = np.where(np.isnan(x))[0]
                out[idx] = np.nan
            return pl.Series(out)

    return _inner


def _output_same_len_series(x, value: Any = None) -> pl.Series:
    if value is None:
        return pl.Series([None] * len(x))
    else:
        return pl.Series(np.full(len(x), value))


def _mark_first_nonnan_as_true(x):
    x = np.asarray(x)
    if np.isnan(x).all():
        return np.full(len(x), False, dtype=np.bool_)
    out = np.full(len(x), False, dtype=np.bool_)
    out[np.where(~np.isnan(x))[0][0]] = True
    return out


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
        _condition_pairs = []
        for pair in condition_pairs.split(";;"):
            if "//" in pair:
                _condition_pairs.append(tuple(pair.split("//")))
            else:
                _condition_pairs.extend([(pair, cond) for cond in all_conditions if pair != cond])
        condition_pairs = _condition_pairs
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
    filter_by_col: Optional[str] = None,
) -> pl.DataFrame:
    """
    Perform t-tests on defined groups of runs in a dataframe.
    """
    condition_pairs = _flatten_condition_pairs(condition_pairs, exp_layout.all_conditions, to_str=False)

    if filter_by_col is not None:
        ignored = df.filter(~pl.col(filter_by_col))
        df = df.filter(pl.col(filter_by_col))

    result = pl.concat(
        (
            pl.concat(
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
                    ).with_columns(df[entry_col]),
                    pl.DataFrame if (filter_by_col is None) else ignored.select(entry_col),
                ),
                how="diagonal",
            ).with_columns(
                pl.Series(
                    name="pair",
                    values=[f"{condition_pair[0]}_vs_{condition_pair[1]}" for condition_pair in condition_pairs],
                )
            )
            for condition_pair in condition_pairs
        ),
        how="vertical",
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

    keep_by_col: Optional[str] = None

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
    filter_by_col: Optional[str] = None,
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
    filter_by_col: Optional[str], optional
        A column that contains boolean values to keep rows.
        If provided, will only do the hypothesis test on the rows with True values in this column, and other entries will be added to the final result table with values as nan.

    Returns
    -------
    tuple[pl.DataFrame, Path, Path]
        returns a three-element tuple of output dataframe, the limma input path, and the limma output path
    """

    rscript_exec = str(rscript_exec)
    script_path = Path(__file__).resolve().parent.parent.joinpath("ext", "do_limma_pair.R")
    logger.info(f'Run hypothesis test via Limma, using Rscript: "{rscript_exec}", and script: "{str(script_path)}"')

    ignored = None
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
            exp_layout = in_data.exp_layout
            entry_name = in_data.entry_level
            in_data = in_data.df
        if filter_by_col is not None:
            ignored = in_data.filter(~pl.col(filter_by_col))
            in_data = in_data.filter(pl.col(filter_by_col))
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
    keep_by_col: Optional[str] = None

    _compare_scope: _T_CompareScope = "pairwise"


def assign_sign(
    df: pl.DataFrame,
    value_col: Optional[Union[str, float]] = None,
    pos_sign_gt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    pos_sign_lt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    neg_sign_gt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    neg_sign_lt_col_value: Optional[Union[tuple[str, float], Sequence[tuple[str, float]]]] = None,
    row_sign_col: str = "row_sign",
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
            return None, None
        if (len(col_value) == 2) and (isinstance(col_value[0], str) and isinstance(col_value[1], (int, float))):
            col_value = [col_value]
        cols = []
        expr = pl.lit(True)
        for c, v in col_value:
            cols.append(c)
            expr = expr & (getattr(pl.col(c), direction)(v))
        return expr, cols

    def comb_expr(exprs: Sequence[tuple[pl.Expr, Union[str, Sequence[str]]]]):
        exprs, cols = list(zip(*exprs, strict=True))
        idxs = [i for i, e in enumerate(exprs) if e is not None]
        exprs = [exprs[i] for i in idxs]
        cols = [cols[i] for i in idxs]
        if len(exprs) == 0:
            return pl.lit(False), []
        expr = exprs[0]
        for e in exprs[1:]:
            expr = expr & e
        return expr, flatten_list(cols)

    if value_col is not None:
        df = df.with_columns(
            pl.when(_generate_nan_null_check_expr(value_col))
            .then(pl.lit(None))
            .otherwise(pl.col(value_col).sign())
            .alias(row_sign_col)
        )
    elif (
        (pos_sign_gt_col_value is not None)
        or (pos_sign_lt_col_value is not None)
        or (neg_sign_gt_col_value is not None)
        or (neg_sign_lt_col_value is not None)
    ):
        pos_expr, pos_expr_cols = comb_expr(
            (cond_expr(pos_sign_gt_col_value, "gt"), cond_expr(pos_sign_lt_col_value, "lt"))
        )
        neg_expr, neg_expr_cols = comb_expr(
            (cond_expr(neg_sign_gt_col_value, "gt"), cond_expr(neg_sign_lt_col_value, "lt"))
        )
        used_cols = pos_expr_cols + neg_expr_cols
        df = df.with_columns(
            pl.when(_generate_nan_null_check_expr(used_cols))
            .then(pl.lit(None))
            .when(pos_expr)
            .then(pl.lit(1))
            .when(neg_expr)
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .alias(row_sign_col)
        )
    else:
        raise ValueError("No value column or sign condition provided")

    return df


_sign_map = {
    1: "pos",
    0: "zero",
    -1: "neg",
}


def _map_group_sign(batch: pl.Series):
    batch = batch.clone()
    nonnull_idx = np.where(~batch.is_null())[0]
    if len(nonnull_idx) == 0:
        return batch.cast(pl.String)
    count = Counter(batch[np.where(~batch.is_null())[0]].to_list())
    max_count = max(count.values())
    major_sign_name = ";".join([_sign_map[k] for k, v in count.items() if v == max_count])
    batch = batch.cast(pl.String)
    batch[nonnull_idx] = major_sign_name
    return batch


def check_sign_direction_in_group(
    df: pl.DataFrame,
    row_sign_col: str = "row_sign",
    group_col: Optional[Union[str, Sequence[str]]] = ("pair", "protein_group"),
    drop_unpassed: bool = False,
):
    """
    Mark (and drop) groups that have amibiguous sign directions.

    This function will check the input dataframe and add three columns to it:
    - "is_group_sign_balanced": mark as `True` when +1 and -1 are balanced in a group, else `False`.
    - "group_major_sign": find the sign(s) that is the majority in a group. This can be "pos", "neg", "zero", or any combination of them like "pos;zero" means +1 and 0 have the same number of rows.
    - "sign_filter_passed": mark as `True` when
        1. the group has a major sign as +1 or -1;
        2. the row has sign as the same as the major sign of that group.

    If `drop_unpassed` is `True`, will drop the rows with `sign_filter_passed` equals to `False`.
    """
    df = df.with_columns(
        pl.col(row_sign_col).len().over(group_col).alias("n_detection_in_group"),
        pl.when(pl.col(row_sign_col).sum().over(group_col).eq(0))
        .then(pl.when(pl.col(row_sign_col).eq(1).any().over(group_col)).then(pl.lit(True)).otherwise(pl.lit(False)))
        .otherwise(pl.lit(False))
        .alias("is_group_sign_balanced"),
        pl.col(row_sign_col).map_batches(_map_group_sign).over(group_col).alias("group_major_sign"),
    ).with_columns(
        pl.when(pl.col("is_group_sign_balanced").eq(True))
        .then(pl.lit(False))
        .otherwise(
            pl.when(pl.col("group_major_sign").is_in(["pos", "neg"]))
            .then(
                pl.when(
                    pl.col(row_sign_col)
                    .replace_strict(_sign_map, return_dtype=pl.String)
                    .eq(pl.col("group_major_sign"))
                )
                .then(pl.lit(True))
                .otherwise(pl.lit(False))
            )
            .otherwise(pl.lit(False))
        )
        .alias("sign_filter_passed")
    )

    if drop_unpassed:
        df = df.filter(pl.col("sign_filter_passed").eq(True))
    return df


def __filter_sign_select_min_one(
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
class SignCheckingConfig(AbstractTestAdjustConfig):
    """
    Config for function `check_sign_in_group`.

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
    drop_unpassed: bool = False

    _compare_scope: _T_CompareScope = "all"


def check_sign_in_group(
    df: pl.DataFrame,
    config: SignCheckingConfig,
):
    """
    Check the sign of each row in each group.
    Receives a dataframe and a config :py:class:`SignCheckingConfig`.
    Will do :py:func:`assign_sign` and :py:func:`check_sign_direction_in_group` on the dataframe.
    """
    df = assign_sign(
        df,
        value_col=config.sign_value_col,
        pos_sign_gt_col_value=config.pos_sign_gt_col_value,
        pos_sign_lt_col_value=config.pos_sign_lt_col_value,
        neg_sign_gt_col_value=config.neg_sign_gt_col_value,
        neg_sign_lt_col_value=config.neg_sign_lt_col_value,
        row_sign_col=config.row_sign_col,
    )
    df = check_sign_direction_in_group(
        df,
        row_sign_col=config.row_sign_col,
        group_col=config.group_col,
        drop_unpassed=config.drop_unpassed,
    )
    return df


@dataclass
class TopKSelectionConfig(AbstractTestAdjustConfig):
    """
    Configuration for selecting top K rows in each group.
    """

    group_col: Union[str, Sequence[str]] = ("pair", "cut_site")
    value_col: str = "pvalue"
    top_k: int = 1
    min_or_max: Literal["min", "max"] = "min"
    mark_col: str = "top_k_selected"
    drop_unselected: bool = False
    filter_condition: Optional[Union[str, pl.Expr]] = None


def select_top_k_in_group(
    df: pl.DataFrame,
    group_col: Union[str, Sequence[str]] = ("pair", "cut_site"),
    value_col: str = "pvalue",
    top_k: int = 1,
    min_or_max: Literal["min", "max"] = "min",
    mark_col: str = "top_k_selected",
    drop_unselected: bool = False,
    filter_condition: Optional[Union[str, pl.Expr]] = None,
):
    """
    Select top K rows in each group.
    """
    filter_condition = _prepare_filter_condition(filter_condition)

    df = df.with_columns(
        pl.when(filter_condition)
        .then(
            pl.when(
                filter_top_n_by_group(
                    group_by=group_col,
                    value_col=value_col,
                    n=top_k,
                    use_min=(min_or_max == "min"),
                    as_pl_expr=True,
                )
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
        )
        .otherwise(pl.lit(False))
        .alias(mark_col)
    )
    if drop_unselected:
        df = df.filter(pl.col(mark_col).eq(True))
    return df


def scipy_comb_p_with_nan(
    x,
    method="fisher",
    weights=None,
    ignore_nan: bool = True,
    return_p_only: bool = True,
    broadcast_to_in_len: bool = True,
) -> Union[float, tuple]:
    """
    Wrapper for scipy.stats.combine_pvalues that handles NaN values, and can broadcast the length of the output to the length of the input.

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
    broadcast_to_in_len : bool, optional
        If True (and `return_p_only` is True), broadcast the length of the output to the length of the input (NaN will be filled with NaN).
        If False, the output will be a scalar or a named tuple (depends on `return_p_only`)

    Returns
    -------
    Union[float, tuple]
        Combined p-value if return_p_only=True, else (statistic, p-value) tuple
    """
    if isinstance(x, pl.Series):
        x = x.to_numpy()
    else:
        x = np.asarray(x)
    raw_len = len(x)
    if ignore_nan:
        nonnan_idx = ~np.isnan(x)
        x = x[nonnan_idx]
    if len(x) == 0:
        if broadcast_to_in_len:
            return pl.Series(np.full(raw_len, np.nan))
        else:
            return np.nan
    r = scipy.stats.combine_pvalues(x, method=method, weights=weights)
    if return_p_only:
        if broadcast_to_in_len:
            arr = np.full(raw_len, np.nan)
            arr[nonnan_idx] = r[1]
            return pl.Series(arr, dtype=pl.Float64)
        else:
            return r[1]
    return r


def combine_pvalues_in_group(
    df: pl.DataFrame,
    group_col: Union[str, Sequence[str]],
    filter_condition: Optional[pl.Expr] = None,
    method: str = "fisher",
    p_col: str = "pvalue",
    new_p_col: Optional[str] = "pvalue_combined",
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
                ignore_nan=True,
                return_p_only=True,
                broadcast_to_in_len=True,
            )
        )
        .over(group_col)
    )
    if filter_condition is not None:
        comb_expr = pl.when(_prepare_filter_condition(filter_condition)).then(comb_expr).otherwise(pl.lit(np.nan))
    return df.with_columns(comb_expr.alias(new_p_col))


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
    method: str = "fisher"
    p_col: str = "pvalue"
    new_p_col: Optional[str] = "pvalue_combined"
    ignore_nan: bool = True
    return_p_only: bool = True
    filter_condition: Optional[Union[str, pl.Expr]] = None

    _compare_scope: _T_CompareScope = "all"


def agg_values_in_group(
    df: pl.DataFrame,
    group_col: Union[str, Sequence[str]] = ("pair", "cut_site"),
    agg_col: Union[str, Sequence[str]] = ("log2_fc", "pvalue"),
    agg_func: Union[Callable, Sequence[Callable]] = (
        np.nanmedian,
        scipy_comb_p_with_nan,
    ),
    new_col_name: Optional[Union[str, Sequence[str]]] = ("log2_fc_combined", "pvalue_combined"),
    mark_col: Optional[str] = "first_nonnan_in_combined",
    filter_condition: Optional[Union[str, pl.Expr]] = None,
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
        The input function will be called by `map_batches`, and the input will be a pl.Series of the values to aggregate.
        Besides customized functions, any function from numpy should be compatible with polars, e.g. np.nanmean and np.nanmedian.
    new_col_name : Optional[Union[str, Sequence[str]]]
        Name for the output column(s) with aggregated values.
        If None, overwrites the input column(s).
    filter_condition: Optional[Union[str, pl.Expr]]
        Filter condition for rows to include in aggregation

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
    if new_col_name is None:
        new_col_name = agg_col
    if isinstance(new_col_name, str):
        new_col_name = [new_col_name]
    if len(new_col_name) != len(agg_col):
        raise ValueError(
            f"Length of `new_col_name` and `agg_col` should be the same. Got {len(new_col_name)} and {len(agg_col)}"
        )
    agg_col = list(agg_col)
    agg_func = list(agg_func)
    new_col_name = list(new_col_name)

    filter_condition = _prepare_filter_condition(filter_condition)

    df = df.with_columns(
        pl.when(filter_condition)
        .then(pl.col(_col).map_batches(_broadcast_func_out_len(_func)).over(group_col))
        .otherwise(pl.col(_col).map_batches(_output_same_len_series))
        .alias(new_col_name[idx])
        for idx, (_col, _func) in enumerate(zip(agg_col, agg_func, strict=True))
    )
    df = df.with_columns(
        pl.col(new_col_name[0]).map_batches(_mark_first_nonnan_as_true).over(group_col).alias(mark_col)
    )

    return df


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
    new_col_name : Optional[Union[str, Sequence[str]]]
        Name for the output column(s) with aggregated values.
        If None, overwrites the input column(s).
    filter_condition: Optional[Union[str, pl.Expr]]
        Filter condition for rows to include in aggregation
    """

    group_col: Union[str, Sequence[str]] = ("pair", "cut_site")
    agg_col: Union[str, Sequence[str]] = ("log2_fc", "pvalue")
    agg_func: Union[Callable, Sequence[Callable]] = (
        np.nanmedian,
        scipy_comb_p_with_nan,
    )
    new_col_name: Optional[Union[str, Sequence[str]]] = ("log2_fc_combined", "pvalue_combined")
    mark_col: Optional[str] = "first_nonnan_in_combined"
    filter_condition: Optional[Union[str, pl.Expr]] = None

    _compare_scope: _T_CompareScope = "all"


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
    p_col: str = "pvalue",
    new_col_name: str = "adjp",
    method: Literal["BH", "BY"] = "BH",
    filter_condition: Optional[Union[str, pl.Expr]] = None,
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
    filter_condition = _prepare_filter_condition(filter_condition)

    fdr_expr = pl.col(p_col).map_batches(partial(scipy_fdr_with_nan, method=method))
    if group is not None:
        fdr_expr = fdr_expr.over(group)

    return df.with_columns(pl.when(filter_condition).then(fdr_expr).otherwise(pl.lit(np.nan)).alias(new_col_name))


@dataclass
class FDRConfig(AbstractTestAdjustConfig):
    group: Optional[Union[str, Sequence[str]]] = None
    p_col: str = "pvalue"
    new_col_name: str = "adj_pvalue"
    method: Literal["BH", "BY"] = "BH"
    filter_condition: Optional[Union[str, pl.Expr]] = None

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
                filter_by_col=conf.keep_by_col,
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
                filter_by_col=conf.keep_by_col,
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
        if isinstance(conf, SignCheckingConfig):
            df = check_sign_in_group(df, conf)
        elif isinstance(conf, TopKSelectionConfig):
            df = select_top_k_in_group(
                df,
                group_col=conf.group_col,
                value_col=conf.value_col,
                top_k=conf.top_k,
                min_or_max=conf.min_or_max,
                mark_col=conf.mark_col,
                drop_unselected=conf.drop_unselected,
                filter_condition=conf.filter_condition,
            )
        elif isinstance(conf, PvalueCombineConfig):
            df = combine_pvalues_in_group(
                df,
                group_col=conf.group_col,
                method=conf.method,
                p_col=conf.p_col,
                new_p_col=conf.new_p_col,
                filter_condition=conf.filter_condition,
            )
        elif isinstance(conf, ValueAggregationConfig):
            df = agg_values_in_group(
                df,
                group_col=conf.group_col,
                agg_col=conf.agg_col,
                agg_func=conf.agg_func,
                new_col_name=conf.new_col_name,
                mark_col=conf.mark_col,
                filter_condition=conf.filter_condition,
            )
        elif isinstance(conf, FDRConfig):
            df = fdr_on_df(
                df,
                group=conf.group,
                p_col=conf.p_col,
                new_col_name=conf.new_col_name,
                method=conf.method,
                filter_condition=conf.filter_condition,
            )
        elif isinstance(conf, AbstractDFManiConfig):
            df = do_df_mani(df, conf)
        elif isinstance(conf, Callable):
            df = conf(df)
        else:
            raise ValueError(f"An unexpected config is passed, with type: {type(conf)}")
    return df
