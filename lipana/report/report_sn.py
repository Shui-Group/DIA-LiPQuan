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
    "spectronaut_report_loading_filter",
    "SpectronautReport",
    "load_spectronaut_search_report",
]

logger = logging.getLogger("lipana")

spectronaut_report_loading_filter = {
    "Basic": (
        (pl.col("PG.ProteinGroups").is_not_null())
        & (pl.col("FG.Quantity").is_not_null())
        & (pl.col("FG.Quantity") > 1.1)
    ),
}


class SpectronautReport(SearchReport):
    @classmethod
    def load_search_report(
        cls,
        path: Union[str, Path],
        exp_setting: ExperimentSetting,
        parsed_fasta: ParsedFasta,
        do_species_annotation: bool = False,
        pre_annotation_filter: Optional[pl.Expr] = spectronaut_report_loading_filter["Basic"],
        post_annotation_filter: Optional[pl.Expr] = None,
        restricted_cut_sites: Sequence[str] = ("K", "R"),
        expand_to_cut_site_level: bool = True,
        resume: Union[bool, str, Path] = True,
        write_processed_report: bool = True,
        processed_report_filename_suffix: str = "-processed.parquet",
        batch_size: int = 1e4,
        n_jobs: int = -1,
    ) -> "SpectronautReport":
        df = load_spectronaut_search_report(
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


mod_to_unimod_map = {
    "[Acetyl (Protein N-term)]": "(UniMod:1)",
    "[Carbamidomethyl (C)]": "(UniMod:4)",
    "[Oxidation (M)]": "(UniMod:35)",
}


def load_spectronaut_search_report(
    path: Union[str, Path],
    exp_setting: ExperimentSetting,
    parsed_fasta: ParsedFasta,
    do_species_annotation: bool = False,
    pre_annotation_filter: Optional[pl.Expr] = spectronaut_report_loading_filter["Basic"],
    post_annotation_filter: Optional[pl.Expr] = None,
    restricted_cut_sites: Sequence[str] = ("K", "R"),
    expand_to_cut_site_level: bool = True,
    resume: Union[bool, str, Path] = True,
    write_processed_report: bool = True,
    processed_report_filename_suffix: str = "-processed.parquet",
    batch_size: int = 1e4,
    n_jobs: int = -1,
) -> pl.DataFrame:
    df, processed_report_path = resume_file(
        path=path, resume=resume, processed_filename_suffix=processed_report_filename_suffix
    )
    if df is not None:
        return df

    logger.info(f"Load and process search report from {str(path)}")
    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides={
            "PEP.IsProteinGroupSpecific": pl.String,
            "PEP.IsProteotypic": pl.String,
            "PEP.NrOfMissedCleavages": pl.String,
        },
    )
    if pre_annotation_filter is not None:
        df = df.filter(pre_annotation_filter)

    df = (
        df.rename(
            {
                "R.FileName": cm.run,
                "PG.ProteinGroups": cm.protein_group,
                "PEP.StrippedSequence": cm.stripped_peptide,
                "EG.ModifiedPeptide": cm.modified_peptide,
                "FG.Charge": cm.precursor_charge,
                # "PG.Quantity": cm.protein_group_quantity,
                # "PEP.Quantity": cm.stripped_peptide_quantity,
                # "EG.TotalQuantity (Settings)": cm.modified_peptide_quantity,
                "FG.Quantity": cm.precursor_quantity_normalised,
                "FG.MS1RawQuantity": cm.precursor_quantity_ms1,
                "FG.MS1Quantity": cm.precursor_quantity_ms1_normalised,
                "FG.MS2RawQuantity": cm.precursor_quantity_ms2,
                "FG.MS2Quantity": cm.precursor_quantity_ms2_normalised,
            }
        )
        .with_columns(
            pl.col(cm.precursor_quantity_ms2).alias(cm.precursor_quantity),
        )
        .with_columns(
            pl.col(cm.modified_peptide).str.replace_many(mod_to_unimod_map).str.strip_prefix("_").str.strip_suffix("_")
        )
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
