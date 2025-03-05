import logging
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import polars as pl

from ..base import AbstractQuantificationReport, ExperimentLayout
from ..utils import AbstractDFManiConfig, do_df_mani, read_df_from_parquet_or_tsv
from .desc import do_desc_summary_on_df
from .infer import do_hypo_test_on_df, do_test_adjust_on_df
from .missing_value import do_pairwise_missing_value_handling_on_df
from .stats_base import (
    AbstractDescConfig,
    AbstractHypoTestConfig,
    AbstractMissingValueHandler,
    AbstractTestAdjustConfig,
)

logger = logging.getLogger("lipana")


_T_stats_chain = Union[
    AbstractDescConfig,
    AbstractMissingValueHandler,
    AbstractHypoTestConfig,
    AbstractTestAdjustConfig,
    AbstractDFManiConfig,
    Callable,
]


def exec_chain(
    df: Union[AbstractQuantificationReport, pl.DataFrame],
    chain: Sequence[Union[_T_stats_chain, Sequence[_T_stats_chain]]],
    condition_pair: tuple[str, str] = None,
    exp_layout: ExperimentLayout = None,
):
    """
    Execute a chain of stats operations.
    """
    if isinstance(df, AbstractQuantificationReport):
        df = df.df
    if not isinstance(df, pl.DataFrame):
        raise ValueError(f"Invalid input data type: {type(df)}")

    for step in chain:
        if isinstance(step, Sequence):
            df = exec_chain(df, step)
        elif isinstance(step, AbstractDescConfig):
            df = do_desc_summary_on_df(df, step)
        elif isinstance(step, AbstractMissingValueHandler):
            df = do_pairwise_missing_value_handling_on_df(
                df,
                exp_layout.condition_to_runs_map[condition_pair[0]],
                exp_layout.condition_to_runs_map[condition_pair[1]],
                step,
            )
        elif isinstance(step, AbstractHypoTestConfig):
            df = do_hypo_test_on_df(df, step, condition_pair, exp_layout)
        elif isinstance(step, AbstractTestAdjustConfig):
            df = do_test_adjust_on_df(df, step)
        elif isinstance(step, AbstractDFManiConfig):
            df = do_df_mani(df, step)
        elif callable(step):
            df = step(df)
        else:
            raise ValueError(f"Invalid chain step type: {type(step)}")

    return df


def do_stats_pipeline_pairwise(
    in_data: Union[AbstractQuantificationReport, pl.DataFrame, str, Path],
    configs: Sequence[_T_stats_chain],
    target_entry: str = None,
    base_entry: Optional[str] = None,
    group_entry: Optional[str] = None,
):
    """
    base_entry is provided, do combine
    group entry is provided, do FDR control within groups
    """
    if isinstance(in_data, (str, Path)):
        df = read_df_from_parquet_or_tsv(in_data)
    elif isinstance(in_data, pl.DataFrame):
        df = in_data
    elif isinstance(in_data, AbstractQuantificationReport):
        df = in_data.df
    else:
        raise ValueError(f"Invalid input data type: {type(in_data)}")

    _pairwise = False
    for config in configs:
        if hasattr(config, "_compare_scope"):
            if config._compare_scope == "pairwise":
                _pairwise = True
                break

    for config in configs:
        ...

    return exec_chain(in_data, configs, condition_pair, exp_layout)
