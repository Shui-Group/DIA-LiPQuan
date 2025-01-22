import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import polars as pl

from ..annotations import annotate_common_info
from ..base import ExperimentSetting, cm
from ..fasta import ParsedFasta
from ..utils import resume_file, write_df_to_parquet_or_tsv
from .report import SearchReport

__all__ = [
    "diann_report_loading_filter",
    "DIANNReport",
    "load_diann_search_report",
]

logger = logging.getLogger("lipana")

diann_report_loading_filter = {
    "Basic": (
        (pl.col("Q.Value") < 0.01)
        & (pl.col("Lib.PG.Q.Value") < 0.01)
        & (pl.col("Protein.Group").is_not_null())
        & (pl.col("Precursor.Quantity").is_not_null())
        & (pl.col("Precursor.Quantity") > 1.1)
    )
}


class DIANNReport(SearchReport):
    @classmethod
    def load_search_report(
        cls,
        path: Union[str, Path],
        exp_setting: ExperimentSetting,
        parsed_fasta: ParsedFasta,
        do_species_annotation: bool = False,
        pre_annotation_filter: Optional[pl.Expr] = diann_report_loading_filter["Basic"],
        post_annotation_filter: Optional[pl.Expr] = None,
        restricted_cut_sites: Sequence[str] = ("K", "R"),
        expand_to_cut_site_level: bool = True,
        resume: Union[bool, str, Path] = True,
        write_processed_report: bool = True,
        processed_report_filename_suffix: str = "-processed.parquet",
        batch_size: int = 10_000,
        n_jobs: int = -1,
    ) -> "DIANNReport":
        df = load_diann_search_report(
            path=path,
            exp_setting=exp_setting,
            parsed_fasta=parsed_fasta,
            do_species_annotation=do_species_annotation,
            pre_annotation_filter=pre_annotation_filter,
            post_annotation_filter=post_annotation_filter,
            restricted_cut_sites=restricted_cut_sites,
            expand_to_cut_site_level=expand_to_cut_site_level,
            resume=resume,
            write_processed_report=write_processed_report,
            processed_report_filename_suffix=processed_report_filename_suffix,
            batch_size=batch_size,
            n_jobs=n_jobs,
        )
        return cls(df=df, exp_setting=exp_setting, workspace=Path(path).parent)


def load_diann_search_report(
    path: Union[str, Path],
    exp_setting: ExperimentSetting,
    parsed_fasta: ParsedFasta,
    do_species_annotation: bool = False,
    pre_annotation_filter: Optional[pl.Expr] = diann_report_loading_filter["Basic"],
    post_annotation_filter: Optional[pl.Expr] = None,
    restricted_cut_sites: Sequence[str] = ("K", "R"),
    expand_to_cut_site_level: bool = True,
    resume: Union[bool, str, Path] = True,
    write_processed_report: bool = True,
    processed_report_filename_suffix: str = "-processed.parquet",
    batch_size: int = 10_000,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """
    use resume to directly define the path to load processed report
    set resume to True to use input file path with defined suffix, which has default value as "-processed.parquet"
    set resume to False to process from scratch

    """
    df, processed_report_path = resume_file(
        path=path, resume=resume, processed_filename_suffix=processed_report_filename_suffix
    )
    if df is not None:
        return df

    logger.info(f"Load and process search report from {str(path)}")
    df = pl.read_csv(path, separator="\t")
    if pre_annotation_filter is not None:
        df = df.filter(pre_annotation_filter)

    df = df.rename(
        {
            "Run": cm.run,
            "Protein.Group": cm.protein_group,
            "Stripped.Sequence": cm.stripped_peptide,
            "Modified.Sequence": cm.modified_peptide,
            "Precursor.Charge": cm.precursor_charge,
            "Precursor.Quantity": cm.precursor_quantity,
            "Precursor.Normalised": cm.precursor_quantity_normalised,
            "Ms1.Area": cm.precursor_quantity_ms1,
            "Ms1.Normalised": cm.precursor_quantity_ms1_normalised,
        }
    ).with_columns(
        pl.col(cm.precursor_quantity).alias(cm.precursor_quantity_ms2),
        pl.col(cm.precursor_quantity_normalised).alias(cm.precursor_quantity_ms2_normalised),
    )

    df = df.join(exp_setting.exp_df, on=cm.run, how="left", coalesce=True)

    df = annotate_common_info(
        df,
        parsed_fasta=parsed_fasta,
        do_species_annotation=do_species_annotation,
        post_annotation_filter=post_annotation_filter,
        restricted_cut_sites=restricted_cut_sites,
        expand_to_cut_site_level=expand_to_cut_site_level,
        cut_site_report_unique_on=(cm.run, cm.cut_site, cm.precursor),
        batch_size=batch_size,
        n_jobs=n_jobs,
    )

    if write_processed_report:
        write_df_to_parquet_or_tsv(df, processed_report_path)
    return df
