import functools
import logging
import re
from functools import partial
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from ..annotations import attach_annotation_from_other_df
from ..base import ComparisonDesign
from ..utils import (
    AbstractDFManiConfig,
    DFAddLitColConfig,
    DFDropColConfig,
    do_df_mani,
    flatten_list,
    subtract_list,
    unique_list_ordered,
)
from .desc import RatioCalcConfig, do_desc_summary_on_df
from .infer import (
    FDRConfig,
    LimmaPairwiseConfig,
    SignCheckingConfig,
    TopKSelectionConfig,
    ValueAggregationConfig,
    do_hypo_test_on_df,
    do_test_adjust_on_df,
    scipy_comb_p_with_nan,
)
from .missing_value import (
    CheckRemovalMissingValueHandler,
    FullEmptyFillingMissingValueHandler,
    SequentialImputeMissingValueHandler,
    count_df_selected_cols_nonnan,
    do_group_missing_value_handling_on_df,
    do_pairwise_missing_value_handling_on_df,
)
from .stats_base import (
    AbstractDescConfig,
    AbstractHypoTestConfig,
    AbstractMissingValueHandler,
    AbstractPairwiseMissingValueHandler,
    AbstractTestAdjustConfig,
)

__all__ = [
    "do_stats_pipe_direct",
    "do_stats_pipe_agg",
]

logger = logging.getLogger("lipana")


_T_stats_chain = Union[
    AbstractDescConfig,
    AbstractMissingValueHandler,
    AbstractHypoTestConfig,
    AbstractTestAdjustConfig,
    AbstractDFManiConfig,
    Callable,
]


def _exec_chain(df: pl.DataFrame, chain: Sequence[Callable]) -> pl.DataFrame:
    for c in chain:
        df = c(df)
    return df


def do_stats_pipe_direct(
    qdf: pl.DataFrame,
    design: ComparisonDesign,
    entry_level: str,
    mv_config: Optional[AbstractPairwiseMissingValueHandler] = None,
    min_rep_count: int = 3,
    annotation_df: Optional[pl.DataFrame] = None,
    group_entry_level: Optional[Union[str, Sequence[str]]] = None,
    annotation_cols: Optional[Union[str, Sequence[str]]] = None,
    return_chains: bool = False,
):
    """
    The retrieval of `group_entry_level` and `annotation_cols` is from `annotation_df`, with `entry_level` as the key.
    """

    do_group_wise_fdr = False
    attach_group_entry_cols = []
    if (group_entry_level is not None) and (group_entry_level != entry_level):
        do_group_wise_fdr = True
        attach_group_entry_cols = subtract_list(
            [group_entry_level] if isinstance(group_entry_level, str) else group_entry_level,
            qdf.columns,
        )

    attach_annotation_cols = []
    if annotation_cols is not None:
        attach_annotation_cols = subtract_list(
            [annotation_cols] if isinstance(annotation_cols, str) else annotation_cols,
            qdf.columns,
            attach_group_entry_cols,
        )

    if len(attach_group_entry_cols) > 0 or len(attach_annotation_cols) > 0:
        if annotation_df is None:
            raise ValueError(
                "`annotation_df` is required when `group_entry_level` and/or `annotation_cols` are provided."
            )
        if cols := subtract_list([*attach_group_entry_cols, *attach_annotation_cols], annotation_df.columns):
            raise ValueError(
                f"The following columns are required for annotation, but are not found in `annotation_df`: {cols}"
            )

    used_conditions = design.used_conditions
    used_runs = flatten_list([design.exp_layout.condition_to_runs_map[c] for c in used_conditions])
    if cols := subtract_list(used_runs, qdf.columns):
        raise ValueError(f"The following runs are not found in `qdf`: {cols}")
    original_qdf_anno_cols = subtract_list(qdf.columns, used_runs, [entry_level])

    chains = []
    for treatment, control in design.pairwise_comparisons:
        treatment_runs = design.exp_layout.condition_to_runs_map[treatment]
        control_runs = design.exp_layout.condition_to_runs_map[control]
        one_chain = [
            partial(
                count_df_selected_cols_nonnan,
                cols=treatment_runs,
                count_col="detected_runs-numerator",
            ),
            partial(
                count_df_selected_cols_nonnan,
                cols=control_runs,
                count_col="detected_runs-denominator",
            ),
            partial(
                do_df_mani,
                config=DFDropColConfig(cols=subtract_list(design.exp_layout.all_runs, treatment_runs, control_runs)),
            ),
            partial(
                do_pairwise_missing_value_handling_on_df,
                exp_runs=treatment_runs,
                ctrl_runs=control_runs,
                config=mv_config,
                mv_filter_config=CheckRemovalMissingValueHandler(
                    min_rep_count=min_rep_count,
                    annotation_col="mv_check_passed",
                    remove_not_passed=False,
                ),
                attach_back="drop_raw",
            ),
            partial(
                do_desc_summary_on_df,
                config=RatioCalcConfig(
                    cond_to_cols_map=design.exp_layout.condition_to_runs_map,
                    base_cond=control,
                    cond_pairs=None,
                    is_log=True,
                    temp_reverse_log_scale=2,
                    div_method="agg_and_divide",
                    agg_method="mean",
                    new_colname_pattern="log2_fc",
                ),
            ),
            partial(
                do_hypo_test_on_df,
                config=LimmaPairwiseConfig(
                    exp_layout=design.exp_layout,
                    entry_name=entry_level,
                    output_column_map="default",
                    output_column_map_override={
                        "logFC": "log2_fc_limma",
                        "adj.P.Val": "adj_pvalue_limma",
                    },
                    recollected_columns=unique_list_ordered(
                        [
                            "log2_fc",
                            "detected_runs-numerator",
                            "detected_runs-denominator",
                            "missing_fill_type",
                            "mv_check_passed",
                            *original_qdf_anno_cols,
                        ]
                    ),
                    dump_in_df_to=None,
                    del_files=True,
                ),
                condition_pairs=[(treatment, control)],
                exp_layout=design.exp_layout,
            ),
            partial(
                do_test_adjust_on_df,
                config=FDRConfig(
                    group=None,
                    filter_condition=None,  # here no need to filter via mv_check_passed, because nan will be ignored in fdr calculation
                    p_col="pvalue",
                    new_col_name="adj_pvalue_exp_wise",
                    method="BH",
                ),
            ),
        ]

        if do_group_wise_fdr:
            if len(attach_group_entry_cols) > 0:
                one_chain.append(
                    partial(
                        attach_annotation_from_other_df,
                        other_df=annotation_df.select([*attach_group_entry_cols, entry_level]).unique(),
                        annotation_cols=attach_group_entry_cols,
                        on=entry_level,
                        method="leftjoin",
                    )
                )
            one_chain.append(
                partial(
                    do_test_adjust_on_df,
                    config=FDRConfig(
                        group=group_entry_level,
                        p_col="pvalue",
                        new_col_name="adj_pvalue_group_wise",
                        method="BH",
                        filter_condition=None,
                    ),
                )
            )

        one_chain.append(
            partial(
                do_df_mani,
                config=DFAddLitColConfig(col_name="pair", value=f"{treatment}_vs_{control}"),
            )
        )
        if len(attach_annotation_cols) > 0:
            one_chain.append(
                partial(
                    attach_annotation_from_other_df,
                    other_df=annotation_df.select([*attach_annotation_cols, entry_level]).unique(),
                    annotation_cols=attach_annotation_cols,
                    on=entry_level,
                    method="leftjoin",
                )
            )
        chains.append(one_chain)
    if return_chains:
        return qdf, chains
    return pl.concat([_exec_chain(qdf, chain) for chain in chains])


def do_stats_pipe_agg(
    qdf: pl.DataFrame,
    base_entry: str,
    target_entry: str,
    group_entry: Optional[str] = None,
    annotation_df: Optional[pl.DataFrame] = None,
    annotation_cols: Optional[Union[str, Sequence[str]]] = None,
    pipeline: Union[
        str,
        Literal["sel_min_p", "sel_min_p_direction_check", "combine_p", "combine_p_direction_check"],
    ] = "sel_min_p",
):
    sel_min_k = None
    if pipeline == "combine_p":
        direction_check = False
    elif pipeline == "combine_p_direction_check":
        pipeline = "combine_p"
        direction_check = True
    elif r := re.search(r"sel_min(\d)?_p", pipeline):
        if r.group(1) is None:
            sel_min_k = 1
        else:
            sel_min_k = int(r.group(1))
        direction_check = pipeline.endswith("_direction_check")
    else:
        raise ValueError(f"Invalid pipeline: {pipeline}")

    if annotation_cols is not None:
        qdf = attach_annotation_from_other_df(
            qdf,
            annotation_df,
            annotation_cols=annotation_cols,
            on=base_entry,
            pre_filter=None,
            unique_on_key_only=False,
            method="leftjoin",
        )

    configs = []
    filter_expr = pl.lit(True)
    used_p_col = "pvalue"

    if direction_check:
        configs.append(
            SignCheckingConfig(
                sign_value_col="t",
                group_col=("pair", target_entry),
                row_sign_col="row_sign",
                drop_unpassed=False,
            )
        )
        filter_expr = filter_expr & pl.col("sign_filter_passed")
    if sel_min_k is not None:
        configs.append(
            TopKSelectionConfig(
                group_col=("pair", target_entry),
                value_col="pvalue",
                top_k=sel_min_k,
                min_or_max="min",
                mark_col="top_k_selected",
                drop_unselected=False,
                filter_condition=filter_expr,
            )
        )
        filter_expr = filter_expr & pl.col("top_k_selected")
    if pipeline == "combine_p":
        configs.append(
            ValueAggregationConfig(
                group_col=("pair", target_entry),
                agg_col=(
                    "log2_fc",
                    "log2_fc_limma",
                    "pvalue",
                ),
                agg_func=(np.nanmedian, np.nanmedian, scipy_comb_p_with_nan),
                new_col_name=("log2_fc_combined", "log2_fc_limma_combined", "pvalue_combined"),
                mark_col="first_nonnan_in_combined",
                filter_condition=filter_expr,
            )
        )
        filter_expr = filter_expr & pl.col("first_nonnan_in_combined")
        used_p_col = "pvalue_combined"

    configs.append(
        FDRConfig(
            group="pair",
            p_col=used_p_col,
            new_col_name="adj_pvalue_combined_exp_wise",
            method="BH",
            filter_condition=filter_expr,
        )
    )
    if group_entry is not None:
        configs.append(
            FDRConfig(
                group=("pair", group_entry),
                p_col=used_p_col,
                new_col_name="adj_pvalue_combined_group_wise",
                method="BH",
                filter_condition=filter_expr,
            )
        )
    return do_test_adjust_on_df(
        qdf,
        config=configs,
    )


def __do_stats_pipeline_pairwise(
    qdf: pl.DataFrame,
    main_df: pl.DataFrame,
    design: ComparisonDesign,
    target_entry_col: str,
    base_entry_col: Optional[str] = None,
    group_entry_col: Optional[str] = None,
    missing_value_config: Optional[AbstractMissingValueHandler] = None,
    pipeline: Literal["sel_min_p", "sel_min_p_from_all", "combine_p", "direct_test"] = "sel_min_p",
    return_chains: bool = False,
):
    """

    This is a general entry point for pairwise comparison pipelines.
    Three pre-defined pipelines are supported:
    - "sel_min_p": select the one with the lowest p-value for each target entry. (e.g. select the precursor with min p-value for each protein)
    - "sel_min_p_from_all": select the one with the lowest p-value for each target entry without pre-filtering.
    - "combine_p": combine p-values for each target entry. (e.g. combine p-values of precursors for each cut site)
    - "direct_test": directly test on target entry level. (e.g. do test on quantification matrix of peptide directly)

    Parameters
    ----------
    qdf: pl.DataFrame
        The quantification matrix dataframe.
        When `pipeline` is "sel_min_p" or "combine_p", this dataframe should be the quantification matrix of the base entry.
        When `pipeline` is "direct_test", this dataframe should be the quantification matrix of the target entry.
    main_df: pl.DataFrame
        The main report dataframe.
        This dataframe is used to provide the information for `base_entry_col` and `group_entry_col`.
    design: ComparisonDesign
        The design of the tests.
    target_entry_col: str
        The column name of the target entry.
    base_entry_col: Optional[str]
        The column name of the base entry.
        When this parameter is given, tests will be done on the base entry level and the results will be aggregate to the target entry level.
        "sel_min_p" and "combine_p" pipelines require this parameter.
    group_entry_col: Optional[str]
        The column name of the group entry.
        When this parameter is given, except the experiment-wise FDR control, an additional FDR control will be done within groups.
    missing_value_config: Optional[AbstractMissingValueHandler]
        The configuration for missing value handling.
        If None, by default a `FullEmptyFillingMissingValueHandler` will be used.
    pipeline: Literal["sel_min_p", "sel_min_p_from_all", "combine_p", "direct_test"]
        1. "sel_min_p": This pipeline will do tests on the base entry and select the one with the lowest p-value for each target entry.
            Before selection, there will be a pre-filtering step to remove the rows with opposite signs.
            For example, set `target_entry` to "cut_site" and `base_entry` to "precursor", then the pipeline will select the precursor with the lowest p-value for each cut site.
            If group_entry is provided, do FDR control within groups, else do FDR control on all data.
        2. "sel_min_p_from_all": Similar to "sel_min_p", but without the pre-filtering step.
        3. "combine_p": This pipeline will do tests on the base entry andcombine p-values for each target entry.
            For example, set `target_entry` to "cut_site" and `base_entry` to "precursor", then the pipeline will combine p-values of precursors for each cut site.
        4. "direct_test": This pipeline will do tests on the target entry level.
            For example, when the target is stripped peptide, `qdf` should be the quantification matrix of peptide, and `target_entry` is "stripped_peptide" and `base_entry` should be None.
    """
    used_conditions = design.used_conditions
    used_runs = flatten_list([design.exp_layout.condition_to_runs_map[c] for c in used_conditions])

    _do_pairwise_mv_handling = True
    if missing_value_config is None:
        missing_value_config = FullEmptyFillingMissingValueHandler(min_rep_count=2)
    if isinstance(missing_value_config, SequentialImputeMissingValueHandler):
        qdf = do_group_missing_value_handling_on_df(
            qdf,
            runs=used_runs,
            config=missing_value_config,
            raw_values_suffix=None,
        )

        missing_value_config = None
        _do_pairwise_mv_handling = False

    if pipeline in ["sel_min_p", "sel_min_p_from_all", "combine_p"]:
        if base_entry_col is None:
            raise ValueError(
                "`base_entry_col` is required when `pipeline` is `sel_min_p`, `sel_min_p_from_all` or `combine_p`."
            )
        primary_annotation_col = base_entry_col
    elif pipeline == "direct_test":
        if base_entry_col is not None:
            if base_entry_col != target_entry_col:
                raise ValueError(
                    "`base_entry_col` should be None or the same as `target_entry_col` when `pipeline` is `direct_test`."
                )
        primary_annotation_col = target_entry_col
    else:
        raise ValueError(f"Invalid pipeline: {pipeline}")

    chains = []
    for treatment, control in design.pairwise_comparisons:
        treatment_runs = design.exp_layout.condition_to_runs_map[treatment]
        control_runs = design.exp_layout.condition_to_runs_map[control]
        one_chain = [
            functools.partial(
                do_pairwise_missing_value_handling_on_df,
                config=missing_value_config,
                filter_less_than_rep=2,
                exp_runs=treatment_runs,
                ctrl_runs=control_runs,
                annotation_col=primary_annotation_col,
            ),
            functools.partial(
                do_desc_summary_on_df,
                config=RatioCalcConfig(
                    cond_to_cols_map=design.exp_layout.condition_to_runs_map,
                    base_cond=control,
                    cond_pairs=None,
                    is_log=True,
                    temp_reverse_log_scale=2,
                    div_method="agg_and_divide",
                    agg_method="mean",
                    new_colname_pattern="log2_fc",
                ),
            ),
            functools.partial(
                do_hypo_test_on_df,
                config=LimmaPairwiseConfig(
                    exp_layout=design.exp_layout,
                    entry_name=primary_annotation_col,
                    output_column_map="default",
                    output_column_map_override={
                        "logFC": "log2_fc_limma",
                        "adj.P.Val": "adj_pvalue_limma",
                    },
                    recollected_columns="log2_fc",
                    dump_in_df_to=None,
                    del_files=True,
                ),
                condition_pairs=[(treatment, control)],
                exp_layout=design.exp_layout,
            ),
        ]

        if pipeline in ["sel_min_p", "sel_min_p_from_all", "combine_p"]:
            one_chain.append(
                functools.partial(
                    attach_annotation_from_other_df,
                    other_df=main_df.select([base_entry_col, target_entry_col]).unique(),
                    annotation_cols=target_entry_col,
                    on=base_entry_col,
                    pre_filter=None,
                    method="leftjoin",
                )
            )
            if pipeline == "sel_min_p":
                one_chain.append(
                    functools.partial(
                        do_test_adjust_on_df,
                        config=SignCheckingConfig(
                            sign_value_col="t",
                            group_col=target_entry_col,
                            row_sign_col="row_sign",
                            group_sign_col="group_sign",
                            filter_value_col="pvalue",
                        ),
                    )
                )
            elif pipeline == "sel_min_p_from_all":
                one_chain.append(
                    lambda df: df.filter(pl.col("pvalue").eq(pl.col("pvalue").min().over(target_entry_col)))
                )
            else:
                one_chain.append(
                    functools.partial(
                        do_test_adjust_on_df,
                        config=[
                            SignCheckingConfig(
                                sign_value_col="t",
                                group_col=target_entry_col,
                                row_sign_col="row_sign",
                                group_sign_col="group_sign",
                                drop_opposite_sign=True,
                                drop_bidirectional_balanced_group=True,
                            ),
                            ValueAggregationConfig(
                                group_col=target_entry_col,
                                agg_col=(
                                    "log2_fc",
                                    "log2_fc_limma",
                                    "pvalue",
                                    "adj_pvalue_limma",
                                ),
                                agg_func=(
                                    np.nanmedian,
                                    np.nanmedian,
                                    scipy_comb_p_with_nan,
                                    scipy_comb_p_with_nan,
                                ),
                                recollected_columns=None,
                            ),
                        ],
                    )
                )

        one_chain.append(
            functools.partial(
                do_test_adjust_on_df,
                config=FDRConfig(
                    group=None,
                    filter_condition=None,
                    p_col="pvalue",
                    new_col_name="adj_pvalue_exp_wise",
                    method="BH",
                ),
            )
        )
        if (group_entry_col is not None) and (group_entry_col != target_entry_col):
            one_chain.append(
                functools.partial(
                    attach_annotation_from_other_df,
                    other_df=main_df.select([group_entry_col, target_entry_col]).unique(),
                    annotation_cols=group_entry_col,
                    on=target_entry_col,
                    method="leftjoin",
                )
            )
            one_chain.append(
                functools.partial(
                    do_test_adjust_on_df,
                    config=FDRConfig(
                        group=group_entry_col,
                        filter_condition=None,
                        p_col="pvalue",
                        new_col_name="adj_pvalue_group_wise",
                        method="BH",
                    ),
                )
            )
        one_chain.append(
            functools.partial(
                do_df_mani,
                config=DFAddLitColConfig(col_name="pair", value=f"{treatment}_vs_{control}"),
            )
        )
        chains.append(one_chain)
    if return_chains:
        return qdf, chains
    return pl.concat([_exec_chain(qdf, chain) for chain in chains])
