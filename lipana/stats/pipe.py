import logging
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import polars as pl

from ..base import (
    AbstractQuantificationReport,
    AbstractSearchReport,
    AbstractStatsReport,
    ExperimentSetting,
)
from .desc import (
    calc_ratio_batch,
)
from .infer import (
    scipy_comb_p_with_nan,
    scipy_fdr_with_nan,
    ttest_batch,
)
from .missing_value import (
    AbstractMissingValueHandler,
    FullEmptyFillingMissingValueHandler,
    NormalDistSamplingMissingValueHandler,
    do_missing_value_handling_on_df,
)

logger = logging.getLogger("lipana")


def do_stats_pipeline(
    in_data: Union[AbstractQuantificationReport, pl.DataFrame, str, Path],
    target_entry: str,
    base_entry: Optional[str] = None,
    group_entry: Optional[str] = None,
    # missing_value_config: Optional[]
):
    """
    If base_entry is provided, will do combine
    If group entry is provided, will do FDR control within groups
    """
