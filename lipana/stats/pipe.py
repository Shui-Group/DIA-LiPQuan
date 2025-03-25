import functools
import logging
from typing import Callable, Literal, Optional, Union

import numpy as np
import polars as pl

from ..annotations import attach_annotation_from_other_df
from ..base import ComparisonDesign
from ..utils import (
    AbstractDFManiConfig,
    DFAddLitColConfig,
    do_df_mani,
    flatten_nested_list,
)
from .desc import RatioCalcConfig, do_desc_summary_on_df
from .infer import (
    FDRConfig,
    LimmaPairwiseConfig,
    SignVotingConfig,
    ValueAggregationConfig,
    do_hypo_test_on_df,
    do_test_adjust_on_df,
    scipy_comb_p_with_nan,
)
from .missing_value import (
    FullEmptyFillingMissingValueHandler,
    SequentialImputeMissingValueHandler,
    do_group_missing_value_handling_on_df,
    do_pairwise_missing_value_handling_on_df,
)
from .stats_base import (
    AbstractDescConfig,
    AbstractHypoTestConfig,
    AbstractMissingValueHandler,
    AbstractTestAdjustConfig,
)

__all__ = [
    "do_stats_pipeline_pairwise",
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


def _exec_chain(df, chain):
    for c in chain:
        df = c(df)
    return df


def do_stats_pipeline_pairwise(
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
    used_runs = flatten_nested_list([design.exp_layout.condition_to_runs_map[c] for c in used_conditions])

    _do_pairwise_mv_handling = True
    if missing_value_config is None:
        missing_value_config = FullEmptyFillingMissingValueHandler(min_rep_count=2)
    if isinstance(missing_value_config, SequentialImputeMissingValueHandler):
        qdf = do_group_missing_value_handling_on_df(
            qdf, runs=used_runs, config=missing_value_config, raw_values_suffix=None
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
                raise ValueError("`base_entry_col` should be None or the same as `target_entry_col` when `pipeline` is `direct_test`.")
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
                    output_column_map_override={"logFC": "log2_fc_limma", "adj.P.Val": "adj_pvalue_limma"},
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
                        config=SignVotingConfig(
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
                            SignVotingConfig(
                                sign_value_col="t",
                                group_col=target_entry_col,
                                row_sign_col="row_sign",
                                group_sign_col="group_sign",
                                drop_opposite_sign=True,
                                drop_bidirectional_balanced_group=True,
                            ),
                            ValueAggregationConfig(
                                group_col=target_entry_col,
                                agg_col=("log2_fc", "log2_fc_limma", "pvalue", "adj_pvalue_limma"),
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
                    pre_filter=None,
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
                        pre_filter=None,
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


def pipe_sel_min_p(): ...


def pipe_combine_p(): ...


def pipe_direct_target_test(): ...
