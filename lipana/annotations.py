"""
nterm_enzymatic_type: "restricted", "non_restricted", "terminal", "after_m_terminal"
cterm_enzymatic_type: "restricted", "non_restricted", "terminal"
"""

import functools
import logging
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Sequence, Union

import polars as pl

from .base import AbstractSearchReport, cm
from .fasta import ParsedFasta
from .utils import exec_r_script

__all__ = [
    "annotate_nterm_enzymatic_specificity",
    "annotate_cterm_enzymatic_specificity",
    "annotate_nterm_enzymatic_specificity_plexpr",
    "annotate_cterm_enzymatic_specificity_plexpr",
    "annotate_peptide_two_side_aa",
    "annotate_peptide_enzymatic_specificity",
    "annotate_cut_sites",
    "annotate_species",
    "annotate_common_info",
    "get_enzymatic_specificity",
    "construct_common_identification_report",
    "construct_cut_site_identification_report",
    "attach_annotation_from_other_df",
    "construct_input_for_quant_aggregation",
    "do_quant_aggregation_via_iq",
    "do_quant_aggregation_via_topk",
    "attach_wide_quant_to_long_report",
    "default_recollected_cols",
    "convert_long_report_to_wide",
]

logger = logging.getLogger("lipana")


def annotate_nterm_enzymatic_specificity(
    cut_aa: str,
    restricted_cut_sites: Iterable[str] = ("K", "R"),
    pep_pos: Optional[int] = None,
    mark_after_m_terminal: bool = False,
) -> Literal["restricted", "non_restricted", "terminal", "after_m_terminal"]:
    if mark_after_m_terminal:
        if pep_pos is None:
            raise ValueError("pep_pos should be provided when mark_after_m_terminal is True")
        if (pep_pos == 2) and (cut_aa == "M"):
            return "after_m_terminal"
    if cut_aa == "_":
        return "terminal"
    if cut_aa in restricted_cut_sites:
        return "restricted"
    return "non_restricted"


def annotate_cterm_enzymatic_specificity(
    n_cut_aa: str,
    c_cut_aa: Optional[str] = None,
    restricted_cut_sites: Iterable[str] = ("K", "R"),
    mark_prot_terminal: bool = True,
) -> Literal["restricted", "non_restricted", "terminal"]:
    if mark_prot_terminal:
        if c_cut_aa is None:
            raise ValueError("c_cut_aa should be provided when mark_prot_terminal is True")
        if c_cut_aa == "_":
            return "terminal"
    if n_cut_aa in restricted_cut_sites:
        return "restricted"
    return "non_restricted"


def annotate_nterm_enzymatic_specificity_plexpr(
    cut_aa_col: str,
    restricted_cut_sites: Iterable[str] = ("K", "R"),
    pep_pos_col: Optional[str] = None,
    mark_after_m_terminal: bool = False,
) -> pl.Expr:
    """
    polars expression for annotating n-term enzymatic specificity
    """
    if mark_after_m_terminal:
        if pep_pos_col is None:
            raise ValueError("pep_pos should be provided when mark_after_m_terminal is True")

    return (
        pl.when((pl.col(pep_pos_col).eq(2) & pl.col(cut_aa_col).eq("M")) if mark_after_m_terminal else pl.lit(False))
        .then(pl.lit("after_m_terminal"))
        .when(pl.col(cut_aa_col).eq("_"))
        .then(pl.lit("terminal"))
        .when(pl.col(cut_aa_col).is_in(restricted_cut_sites))
        .then(pl.lit("restricted"))
        .otherwise(pl.lit("non_restricted"))
        .alias(cm.nterm_enzymatic_specificity)
    )


def annotate_cterm_enzymatic_specificity_plexpr(
    n_cut_aa_col: str,
    c_cut_aa_col: Optional[str] = None,
    restricted_cut_sites: Iterable[str] = ("K", "R"),
    mark_prot_terminal: bool = True,
) -> pl.Expr:
    """
    polars expression for annotating c-term enzymatic specificity
    """
    if mark_prot_terminal:
        if c_cut_aa_col is None:
            raise ValueError("c_cut_aa_col should be provided when mark_prot_terminal is True")
    return (
        pl.when(mark_prot_terminal & pl.col(c_cut_aa_col).eq("_"))
        .then(pl.lit("terminal"))
        .when(pl.col(n_cut_aa_col).is_in(restricted_cut_sites))
        .then(pl.lit("restricted"))
        .otherwise(pl.lit("non_restricted"))
        .alias(cm.cterm_enzymatic_specificity)
    )


def annotate_peptide_two_side_aa(
    df: pl.DataFrame,
    protein_id_to_seq: dict[str, str],
    peptide_position_col: Optional[str] = None,
    protein_col: str = cm.first_protein,
    peptide_col: str = cm.stripped_peptide,
) -> pl.DataFrame:
    if peptide_position_col is None:
        df = df.with_columns(
            pl.col(protein_col)
            .replace(protein_id_to_seq)
            .str.find(pl.col(peptide_col), literal=True)
            .add(1)
            .alias(cm.peptide_start_position)
        )
        peptide_position_col = cm.peptide_start_position
    return (
        df.with_columns(pl.col(protein_col).replace(protein_id_to_seq).alias("temp_prot_seq"))
        .with_columns(
            pl.col(peptide_position_col).add(pl.col(peptide_col).str.len_chars()).add(-1).alias(cm.peptide_end_position)
        )
        .with_columns(
            pl.when(pl.col(peptide_position_col).eq(1))
            .then(pl.lit("_"))
            .otherwise(pl.col("temp_prot_seq").str.slice(pl.col(peptide_position_col).add(-2), 1))
            .alias(cm.prev_aa),
            pl.when(pl.col(cm.peptide_end_position).eq(pl.col("temp_prot_seq").str.len_chars()))
            .then(pl.lit("_"))
            .otherwise(pl.col("temp_prot_seq").str.slice(pl.col(cm.peptide_end_position), 1))
            .alias(cm.next_aa),
            pl.col(peptide_col).str.slice(0, 1).alias(cm.peptide_n_term_aa),
            pl.col(peptide_col).str.slice(-1, 1).alias(cm.peptide_c_term_aa),
        )
        .drop("temp_prot_seq")
    )


def annotate_peptide_enzymatic_specificity(
    df: pl.DataFrame,
    do_term_enzymatic_specificity_annotation: bool = False,
    restricted_cut_sites: Iterable[str] = ("K", "R"),
) -> pl.DataFrame:
    """
    When do_term_enzymatic_specificity_annotation is True, the following columns are required:
    - prev_aa
    - next_aa
    # - peptide_n_term_aa
    - peptide_c_term_aa
    - peptide_start_position

    When do_term_enzymatic_specificity_annotation is False, the following columns are required:
    - nterm_enzymatic_specificity
    - cterm_enzymatic_specificity

    Will return the input dataframe with one or three additional columns:
    - nterm_enzymatic_specificity (if do_term_enzymatic_specificity_annotation is True)
    - cterm_enzymatic_specificity (if do_term_enzymatic_specificity_annotation is True)
    - peptide_enzymatic_specificity

    Note: might need to control what role the protein terminal plays in the enzymatic specificity
    Currently the protein terminal is always regarded as the enzyme restriction sites,
    which makes the peptide a fully-specific one if the cut site is restricted, and a semi-specific if the cut site is not restricted.
    """
    if do_term_enzymatic_specificity_annotation:
        df = df.with_columns(
            annotate_nterm_enzymatic_specificity_plexpr(
                cut_aa_col=cm.prev_aa,
                restricted_cut_sites=restricted_cut_sites,
                pep_pos_col=cm.peptide_start_position,
                mark_after_m_terminal=True,
            ),
            annotate_cterm_enzymatic_specificity_plexpr(
                n_cut_aa_col=cm.peptide_c_term_aa,
                c_cut_aa_col=cm.next_aa,
                restricted_cut_sites=restricted_cut_sites,
                mark_prot_terminal=True,
            ),
        )
    return df.with_columns(
        pl.when(
            pl.col(cm.nterm_enzymatic_specificity).eq("restricted")
            & pl.col(cm.cterm_enzymatic_specificity).eq("restricted")
        )
        .then(pl.lit("fully_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).is_in(("terminal", "after_m_terminal"))
            & pl.col(cm.cterm_enzymatic_specificity).eq("restricted")
        )
        .then(pl.lit("fully_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).eq("restricted")
            & pl.col(cm.cterm_enzymatic_specificity).eq("terminal")
        )
        .then(pl.lit("fully_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).eq("restricted")
            & pl.col(cm.cterm_enzymatic_specificity).eq("non_restricted")
        )
        .then(pl.lit("semi_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).eq("non_restricted")
            & pl.col(cm.cterm_enzymatic_specificity).eq("restricted")
        )
        .then(pl.lit("semi_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).is_in(("terminal", "after_m_terminal"))
            & pl.col(cm.cterm_enzymatic_specificity).eq("non_restricted")
        )
        .then(pl.lit("semi_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).eq("non_restricted")
            & pl.col(cm.cterm_enzymatic_specificity).eq("terminal")
        )
        .then(pl.lit("semi_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).eq("non_restricted")
            & pl.col(cm.cterm_enzymatic_specificity).eq("non_restricted")
        )
        .then(pl.lit("non_specific"))
        .when(
            pl.col(cm.nterm_enzymatic_specificity).is_in(("terminal", "after_m_terminal"))
            & pl.col(cm.cterm_enzymatic_specificity).eq("terminal")
        )
        .then(pl.lit("undigested"))
        .otherwise(pl.lit("undefined"))
        .alias(cm.peptide_enzymatic_specificity)
    )


def annotate_cut_sites(
    df: pl.DataFrame,
    protein_col: str = cm.first_protein,
    peptide_col: str = cm.stripped_peptide,
    peptide_position_col: str = cm.peptide_start_position,
) -> pl.DataFrame:
    if cm.peptide_end_position not in df.columns:
        df = df.with_columns(
            pl.col(peptide_position_col).add(pl.col(peptide_col).str.len_chars()).add(-1).alias(cm.peptide_end_position)
        )
    return df.with_columns(
        pl.col(protein_col)
        .add("-")
        .add(pl.col(peptide_position_col).add(-1).cast(pl.String))
        .add("_")
        .add(pl.col(peptide_position_col).cast(pl.String))
        .alias(cm.n_cut_site),
        pl.col(protein_col)
        .add("-")
        .add(pl.col(cm.peptide_end_position).cast(pl.String))
        .add("_")
        .add(pl.col(cm.peptide_end_position).add(1).cast(pl.String))
        .alias(cm.c_cut_site),
    )


def annotate_species(
    df: pl.DataFrame,
    parsed_fasta: ParsedFasta,
    protein_col: str = cm.first_protein,
    peptide_col: str = cm.stripped_peptide,
    annotate_via_protein: bool = True,
    annotate_via_peptide: bool = True,
    batch_size: int = 10_000,
    n_jobs: int = -1,
) -> pl.DataFrame:
    if annotate_via_protein:
        df = df.with_columns(
            pl.col(protein_col)
            .replace_strict(parsed_fasta.prot_acc_to_species, default="NotMapped")
            .alias(cm.mapped_species_from_protein)
        )

    if annotate_via_peptide:
        from functools import partial

        from joblib import Parallel, delayed

        def _process_batch(pep_batch, species_to_cat_seqs):
            return {
                pep: ";".join([species for species, concat_seq in species_to_cat_seqs.items() if pep in concat_seq])
                for pep in pep_batch
            }

        peps = list(set(df[peptide_col].unique()) - set(parsed_fasta.peptide_to_species_cache))
        pep_batches = [peps[i : i + batch_size] for i in range(0, len(peps), batch_size)]
        partial_process = partial(_process_batch, species_to_cat_seqs=parsed_fasta.species_to_concat_seqs)
        results = Parallel(n_jobs=n_jobs)(delayed(partial_process)(batch) for batch in pep_batches)
        for result in results:
            parsed_fasta.peptide_to_species_cache.update(result)

        df = df.with_columns(
            pl.col(cm.stripped_peptide)
            .replace_strict(parsed_fasta.peptide_to_species_cache)
            .alias(cm.mapped_species_from_peptide)
        )
    return df


def annotate_common_info(
    df: pl.DataFrame,
    parsed_fasta: ParsedFasta,
    do_species_annotation: bool = False,
    post_annotation_filter: Optional[pl.Expr] = None,
    restricted_cut_sites: Sequence[str] = ("K", "R"),
    expand_to_cut_site_level: bool = True,
    cut_site_report_unique_on: Sequence[str] = (cm.run, cm.cut_site, cm.precursor),
    batch_size: int = 10_000,
    n_jobs: int = -1,
) -> pl.DataFrame:
    """
    Annotate common information for a input dataframe.
    The following columns are required:
    - protein_group
    - stripped_peptide
    - modified_peptide
    - precursor_charge
    """

    df = df.with_columns(
        (pl.col(cm.modified_peptide) + "/" + pl.col(cm.precursor_charge).cast(pl.String)).alias(cm.precursor),
        pl.col(cm.protein_group).str.split(";").list.first().alias(cm.first_protein),
    )

    df = annotate_peptide_two_side_aa(
        df,
        protein_id_to_seq=parsed_fasta.prot_acc_to_seq,
        peptide_position_col=None,
        protein_col=cm.first_protein,
        peptide_col=cm.stripped_peptide,
    )
    df = df.filter(pl.col(cm.peptide_start_position).is_not_null())
    df = annotate_peptide_enzymatic_specificity(
        df,
        do_term_enzymatic_specificity_annotation=True,
        restricted_cut_sites=restricted_cut_sites,
    )
    df = annotate_cut_sites(
        df,
        protein_col=cm.first_protein,
        peptide_col=cm.stripped_peptide,
        peptide_position_col=cm.peptide_start_position,
    )

    if do_species_annotation:
        df = annotate_species(
            df,
            parsed_fasta=parsed_fasta,
            protein_col=cm.first_protein,
            peptide_col=cm.stripped_peptide,
            annotate_via_protein=True,
            annotate_via_peptide=True,
            batch_size=int(batch_size),
            n_jobs=n_jobs,
        )

    if post_annotation_filter is not None:
        df = df.filter(post_annotation_filter)

    if expand_to_cut_site_level:
        df = construct_cut_site_identification_report(
            df,
            restricted_cut_sites=restricted_cut_sites,
            protein_terminal_role="drop",
            do_unique_on=cut_site_report_unique_on,
        )
    return df


def get_enzymatic_specificity(
    pep_seq: str,
    prot_seq: str,
    restricted_enzyme_cuts: str = "KR",
    # nterm_m_as_terminal: bool = True,
    # allow_prot_terminal: bool = True,
) -> tuple[str, int, str, str]:
    """
    This is a pure function that receives one peptide sequence and its corresponding protein sequence, and returns the enzymatic specificity of the peptide.
    Returns a tuple of enzymatic_specificity, peptide_start_position, prev_aa, next_aa

    Note: protein n-term M will be always regarded as the protein terminal, and will only report the first position on protein sequence that matches peptide sequence
    """
    pos = prot_seq.find(pep_seq) + 1
    prot_seq_len = len(prot_seq)
    prev_aa = "_" if (pos == 1) else prot_seq[pos - 2]
    next_aa = "_" if ((pos + len(pep_seq) - 1) == prot_seq_len) else prot_seq[pos + len(pep_seq) - 1]
    is_n_end = (prev_aa == "_") or ((prev_aa == "M") and (pos == 2))
    is_c_end = next_aa == "_"

    match ((prev_aa in restricted_enzyme_cuts), (pep_seq[-1] in restricted_enzyme_cuts), is_n_end, is_c_end):
        case _, _, True, True:
            t = "Nonspecific"
        # Pep on prot n-term
        case _, True, True, False:
            t = "Fully"
        case _, False, True, False:
            t = "Semi"  # can be "Nonspecific"
        # Pep on prot c-term
        case True, _, False, True:
            t = "Fully"
        case False, _, False, True:
            t = "Semi"  # can be "Nonspecific"
        # Pep on prot internal
        case True, True, False, False:
            t = "Fully"
        case True, False, False, False:
            t = "Semi"
        case False, True, False, False:
            t = "Semi"
        case False, False, False, False:
            t = "Nonspecific"
        case _:
            raise ValueError(f"Peptide {pep_seq} can not be processed")
    return t, pos, prev_aa, next_aa


def construct_common_identification_report(
    df: pl.DataFrame,
    run_colname: str = cm.run,
    entry_colname: str = cm.protein_group,
):
    """
    For most entry types, a report focusing on the identification is to make the dataframe unique on (run, entry)
    Should be cautious before quantity aggregation from low-level entry before use this function
    """
    return df.unique([run_colname, entry_colname])


def construct_cut_site_identification_report(
    df: pl.DataFrame,
    restricted_cut_sites: Sequence[str] = ("K", "R"),
    protein_terminal_role: Literal["drop", "restricted"] = "drop",
    do_unique_on: Sequence[str] = (cm.run, cm.cut_site, cm.precursor),
):
    """
    This function will construct a cut site identification report from the input dataframe
    The input dataframe will be repeated twice, one for n-terminal cut site and one for c-terminal cut site
    New columns will be added
    - cut_site: the cut site
    - cut_site_n_aa: the amino acid before the cut site
    - cut_site_c_aa: the amino acid after the cut site
    - cut_site_on_term: whether the cut site is on n-terminal or c-terminal
    - cut_site_is_restricted: whether the cut site is restricted (c-terminal aa is in `restricted_cut_sites`) or not

    Set protein_terminal_role to "drop" to drop the sites that are on protein terminal
    Set protein_terminal_role to "restricted" to keep those sites on protein terminal, and mark them as restricted

    By default, the unique will be done on (run, cut_site, precursor), to ensure cut site quantification based on precursors
    """
    return (
        pl.concat(
            (
                df.with_columns(
                    pl.col(cm.n_cut_site).alias(cm.cut_site),
                    pl.col(cm.prev_aa).alias(cm.cut_site_n_aa),
                    pl.col(cm.peptide_n_term_aa).alias(cm.cut_site_c_aa),
                    pl.lit("n").alias(cm.cut_site_on_term),
                ),
                df.with_columns(
                    pl.col(cm.c_cut_site).alias(cm.cut_site),
                    pl.col(cm.peptide_c_term_aa).alias(cm.cut_site_n_aa),
                    pl.col(cm.next_aa).alias(cm.cut_site_c_aa),
                    pl.lit("c").alias(cm.cut_site_on_term),
                ),
            )
        )
        .filter(
            (~((pl.col(cm.cut_site_c_aa) == "_") | (pl.col(cm.cut_site_n_aa) == "_")))
            if (protein_terminal_role == "drop")
            else True
        )
        .with_columns(
            (
                pl.col(cm.cut_site_n_aa).is_in(restricted_cut_sites)
                | ((pl.col(cm.cut_site_c_aa) == "_") if (protein_terminal_role == "restricted") else False)
            ).alias(cm.cut_site_is_restricted)
        )
        .unique(do_unique_on)
    )


def attach_annotation_from_other_df(
    df: pl.DataFrame,
    other_df: pl.DataFrame,
    annotation_cols: Union[str, Sequence[str]],
    on: Union[str, Sequence[str]],
    pre_filter: Optional[pl.Expr] = None,
    method: Literal["leftjoin", "agg_leftjoin"] = "leftjoin",
    check_unique: bool = True,
    concat_char: str = ";",
) -> pl.DataFrame:
    """
    Attach annotation columns from another DataFrame to the target DataFrame using specified join keys.
    Will return a new dataframe with the annotation columns attached.
    This equals to left-join the quantification dataframe and the selected columns in main report on the entry column.

    - For "leftjoin" method:
       - Do unique on join keys and collected columns before joining
    - For "agg_leftjoin" method:
       - If `check_unique` is True and values are unique within groups, keeps the first value
       - If values are not unique, concatenates them using `concat_char`
       - Non-string columns will be cast to string before concatenation

    Parameters:
    -----------
    df : pl.DataFrame
        The target DataFrame to which annotations will be attached
    other_df : pl.DataFrame
        The source DataFrame containing the annotation columns
    annotation_cols : Union[str, Sequence[str]]
        Column name(s) to attach from the other_df. Can be a single string or a sequence of strings.
    on : Union[str, Sequence[str]]
        Column name(s) to use as join keys. Can be a single string or a sequence of strings.
    pre_filter : Optional[pl.Expr], optional
        A filter expression to apply to the other_df before joining. Default is None.
        This can be used to avoid the potential error of having unexpected additional annotations.
        For example, if we attach the `cut_site` column from the `other_df` to `df` with `precursor` as key,
        and we are actually handling the non-restricted cut sites. In this case, we can use `pre_filter` to
        filter out the restricted cut sites, as `(~pl.col(cut_site_is_restricted))`, else we will make each row
        duplicated because actually each precursor can map to two cut sites, in which only one or zero site is
        non-restricted.
    method : Literal["leftjoin", "agg_leftjoin"], optional
        Join method to use:
        - "leftjoin" (default): Direct left join with unique values,
        this might let input `df` have repeated rows if the value is not unique for join key(s).
        - "agg_leftjoin": Aggregate values before joining
    check_unique : bool, optional
        When using "agg_leftjoin", check if values are unique within groups before concatenation.
        If True and values are unique, keeps the first value instead of concatenating.
        Default is True.
    concat_char : str, optional
        Character used to concatenate values in "agg_leftjoin" mode when values are not unique.
        Default is ";".

    Returns
    -------
    pl.DataFrame
        A new DataFrame with the annotation columns attached
    """
    if isinstance(annotation_cols, str):
        annotation_cols = [annotation_cols]
    if isinstance(on, str):
        on = [on]

    if missing_cols := [col for col in annotation_cols if col not in other_df.columns]:
        raise ValueError(f"Columns not found in other_df: {missing_cols}")

    join_cols = [col for col in annotation_cols if col not in on]

    if pre_filter is not None:
        other_df = other_df.filter(pre_filter)
    other_df = other_df.select([*on, *join_cols]).unique()

    if method == "leftjoin":
        # Simple left join with unique values
        return df.join(
            other_df,
            on=on,
            how="left",
        )
    if method == "agg_leftjoin":
        # Prepare aggregation expressions
        agg_exprs = []
        for col in join_cols:
            if (
                check_unique
                and other_df.group_by(on).agg(pl.col(col).n_unique().eq(1)).select(pl.col(col).all()).item()
            ):
                agg_exprs.append(pl.col(col).first())
            else:
                agg_exprs.append(pl.col(col).cast(pl.String).str.concat(concat_char))

        # Perform the aggregation and join
        return df.join(
            other_df.group_by(on).agg(agg_exprs),
            on=on,
            how="left",
        )
    raise ValueError(f"Unknown method: {method}. Use 'leftjoin' or 'agg_leftjoin'")


def construct_input_for_quant_aggregation(
    df: pl.DataFrame,
    filter_condition: Optional[pl.Expr] = None,
    run_col: str = cm.run,
    primary_entry_col: str = cm.cut_site,
    low_level_entry_col: Union[str, Sequence[str]] = cm.precursor,
    base_quant_col: Union[str, Sequence[str]] = cm.precursor_quantity,
    require_expansion: Union[bool, Sequence[bool], str, Sequence[str]] = False,
    concat_entry_after_expansion: Optional[Union[str, Sequence[str]]] = None,
    remove_below_threshold: Optional[Union[float, Sequence[float]]] = 1.1,
) -> pl.DataFrame:
    """
    Construct a dataframe for quantity aggregation from low-level entry to high-level entry, with the following columns:
    - SampleIds: run_col
    - PrimaryIds: primary_entry_col
    - AggregationIds: low_level_entry_col
    - BaseQuant: base_quant_col
    AggregationIds will have a final format as "{concat_entry_after_expansion}_{low_level_entry_col}_{base_quant_col}" to ensure unique id, and "{concat_entry_after_expansion}_" can be omitted if not needed

    Use require_expansion to do explode on low level entry if needed
    - False to not expand
    - True indicates the column is already explodable (i.e. any polars type that supports explode method, like List)
    - str to define a delimiter for spliting the column
    Set concat_entry_after_expansion to concatenate the exploded entries after expansion to make unique ids

    Note: will do unique on (run_col, primary_entry_col, low_level_entry_col)
    add unique id for low level entry if same entry id might have varied quant from different source
    """
    if isinstance(base_quant_col, str):
        base_quant_col = [base_quant_col]
    if isinstance(low_level_entry_col, str):
        low_level_entry_col = [low_level_entry_col] * len(base_quant_col)
    if len(low_level_entry_col) != len(base_quant_col):
        raise ValueError(
            f"Length of low_level_entry_col and base_quant_col should be the same, now {len(low_level_entry_col)} and {len(base_quant_col)}"
        )

    if isinstance(require_expansion, (bool, str)):
        require_expansion = [require_expansion] * len(low_level_entry_col)
    if len(require_expansion) != len(low_level_entry_col):
        raise ValueError(
            f"Length of require_expansion should be the same as low_level_entry_col, now {len(require_expansion)} and {len(low_level_entry_col)}"
        )
    if isinstance(concat_entry_after_expansion, str):
        concat_entry_after_expansion = [concat_entry_after_expansion] * len(low_level_entry_col)
    if concat_entry_after_expansion is None:
        concat_entry_after_expansion = [None] * len(low_level_entry_col)
    if len(concat_entry_after_expansion) != len(low_level_entry_col):
        raise ValueError(
            f"Length of concat_entry_after_expansion should be the same as low_level_entry_col, now {len(concat_entry_after_expansion)} and {len(low_level_entry_col)}"
        )

    if remove_below_threshold is not None:
        if isinstance(remove_below_threshold, (float, int)):
            remove_below_threshold = [remove_below_threshold] * len(low_level_entry_col)
        if len(remove_below_threshold) != len(low_level_entry_col):
            raise ValueError(
                f"Length of remove_below_threshold should be the same as low_level_entry_col, now {len(remove_below_threshold)} and {len(low_level_entry_col)}"
            )
        thres_filter = [(pl.col(base_quant_col[i]) > t) for i, t in enumerate(remove_below_threshold)]
    else:
        thres_filter = [True] * len(low_level_entry_col)

    if filter_condition is not None:
        df = df.filter(filter_condition)

    dfs = []
    for i, (entry_col, quant_col, thres) in enumerate(
        zip(low_level_entry_col, base_quant_col, thres_filter, strict=True)
    ):
        if require_expansion[i]:
            if isinstance(require_expansion[i], str):
                char = require_expansion[i]
                new_df = df.with_columns(
                    pl.col(entry_col).str.strip_chars_end(char).str.split(char),
                    pl.col(quant_col).str.strip_chars_end(char).str.split(char),
                )
            else:
                new_df = df
            new_df = new_df.explode([entry_col, quant_col])
            if concat_entry_after_expansion[i] is not None:
                new_df = new_df.with_columns(
                    (pl.col(concat_entry_after_expansion[i]).add("_").add(pl.col(entry_col))).alias(entry_col),
                )
        else:
            new_df = df
        dfs.append(
            new_df.select([run_col, primary_entry_col, entry_col, quant_col])
            .with_columns(
                pl.col(entry_col).add("_").add(quant_col).alias(entry_col),  # ensure unique id
            )
            .filter(thres)
            .rename(
                {
                    run_col: "SampleIds",
                    primary_entry_col: "PrimaryIds",
                    entry_col: "AggregationIds",
                    quant_col: "BaseQuant",
                }
            )
        )

    return pl.concat(dfs).unique(["SampleIds", "PrimaryIds", "AggregationIds"])


def do_quant_aggregation_via_iq(
    in_data: Union[pl.DataFrame, str, Path],
    reverse_log2: bool = False,
    output_entry_name: Optional[str] = None,
    output_column_map: Optional[dict[str, str]] = None,
    dump_in_df_to: Optional[Union[str, Path]] = None,
    del_files: bool = False,
    rscript_exec: Union[str, Path] = "Rscript",
) -> tuple[pl.DataFrame, Path, Path]:
    """
    Perform quantification aggregation by running external R script `do_iq.R`.
    Receives input dataframe or path, and returns the output dataframe, input path, and output path.

    Input data should have four columns, see ``construct_input_for_quant_aggregation`` for details:
    "SampleIds, PrimaryIds, AggregationIds, and BaseQuant"

    Note: iq receives quantity values in raw scale, and the output is in log2 scale.

    Parameters
    ----------
    in_data : Union[pl.DataFrame, str, Path]
        Input data as polars DataFrame or path
    reverse_log2: bool, optional
        Whether to reverse the log2 scale to origin in the iq output, by default False
    output_entry_name : Optional[str], optional
        Optional name for the output entry, by default None
        The defined name will replace the name "PrimaryIds" in output.
    output_column_map : Optional[dict[str, str]], optional
        An optional dict to map the output column names to the desired names, by default None.
        Useful when the input data has short names for runs in column "SampleIds". For example, {"run1": "original_long_run_name", ...}.
    dump_in_df_to : Optional[Union[str, Path]], optional
        Optional path to dump the dataframe of quantification input, by default None
        If in_data is a dataframe and dump_to is None, a temporary file will be created, and the output file will also be in the same directory
    del_files : bool, optional
        Whether to delete the input and output files, by default False
    rscript_exec : Union[str, Path], optional
        Path to Rscript executable, by default "Rscript" to use the system default

    Returns
    -------
    tuple[pl.DataFrame, Path, Path]
        returns a three-element tuple of output quant dataframe, the iq input path, and the iq output path
    """
    rscript_exec = str(rscript_exec)
    script_path = Path(__file__).resolve().parent.joinpath("ext", "do_iq.R")
    logger.info(f'Run MaxLFQ estimation via iq, using Rscript: "{rscript_exec}", and script: "{str(script_path)}"')

    if isinstance(in_data, (str, Path)):
        input_path = Path(in_data).resolve()
        logger.info(f'iq input file path: "{str(input_path)}"')
    elif isinstance(in_data, pl.DataFrame):
        input_path = (
            Path(dump_in_df_to).resolve()
            if (dump_in_df_to is not None)
            else Path(tempfile.NamedTemporaryFile(suffix=".txt", delete=False).name)
        )
        logger.info(f'write temporary iq input file to: "{str(input_path)}"')
        in_data.write_csv(input_path, separator="\t")
    else:
        raise ValueError(f"Unsupported input data type: `{type(in_data)}`")

    exec_r_script(rscript_exec, script_path, str(input_path))

    output_path = input_path.parent.joinpath(f"{input_path.name}-iq_output.txt")
    logger.info(f'Load iq output from: "{str(output_path)}"')
    quant_data = pl.read_csv(output_path, separator="\t", null_values="NA")

    if reverse_log2:
        quant_data = quant_data.with_columns(
            (2 ** pl.col(c).cast(pl.Float32)).alias(c) for c in quant_data.columns if c != "PrimaryIds"
        )

    if del_files:
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()

    if output_entry_name is not None:
        quant_data = quant_data.rename({"PrimaryIds": output_entry_name})
    if output_column_map is not None:
        quant_data = quant_data.rename(output_column_map)
    return quant_data, input_path, output_path


def do_quant_aggregation_via_topk(
    in_data: Union[pl.DataFrame, str, Path],
    topk: int = 3,
    do_log_scale: Optional[Union[int, float]] = 2,
    reverse_log_scale: Optional[Union[int, float]] = None,
    output_entry_name: Optional[str] = None,
    output_column_map: Optional[dict[str, str]] = None,
    agg_method: Literal["mean", "median", "max", "min"] = "mean",
    save_output_to_file: Optional[Union[str, Path]] = None,
) -> tuple[pl.DataFrame, None, Path]:
    """
    Perform quantification aggregation by selecting top k values for each entry in each run.
    Receives input dataframe or path, and returns the output dataframe, None, and the output path.

    Input data should have four columns, see ``construct_input_for_quant_aggregation`` for details:
    "SampleIds, PrimaryIds, AggregationIds, and BaseQuant"

    Parameters
    ----------
    in_data : Union[pl.DataFrame, str, Path]
        Input data as polars DataFrame or path
    topk : int, optional
        Number of top values to select for each entry in each run, by default 3
    do_log_scale : Optional[Union[int, float]], optional
        Do log scale transformation on the input data, by default 2
        Set this to None to skip log scale transformation.
    reverse_log_scale : bool, optional
        Whether to reverse the log scale to origin in the output, by default None.
        This will be ignored if `do_log_scale` is not None.
    output_entry_name : Optional[str], optional
        Optional name for the output entry, by default None
        The defined name will replace the name "PrimaryIds" in output.
    output_column_map : Optional[dict[str, str]], optional
        Optional column map for the output dataframe, by default None
        A dict to map the output column names to the desired names.
        Useful when the input data has short names for runs in column "SampleIds".
    agg_method : Literal["mean", "median", "max", "min"], optional
        How to aggregate the top k values, by default "mean"
    save_output_to_file : Optional[Union[str, Path]], optional
        Optional path to save the output dataframe, by default None
    """
    if isinstance(in_data, (str, Path)):
        input_path = Path(in_data).resolve()
        in_data = pl.read_csv(input_path, separator="\t")

    run_col = "SampleIds"
    entry_col = "PrimaryIds"
    quant_col = "BaseQuant"
    group_cols = (run_col, entry_col)

    if do_log_scale is not None:
        in_data = in_data.with_columns(pl.col(quant_col).log(do_log_scale).alias(quant_col))
    elif reverse_log_scale is not None:
        in_data = in_data.with_columns((reverse_log_scale ** pl.col(quant_col).cast(pl.Float32)).alias(quant_col))

    quant_data = (
        in_data.group_by(group_cols, maintain_order=False)
        .agg(pl.col(quant_col).top_k(topk))
        .with_columns(getattr(pl.col(quant_col).list, agg_method)())
        .pivot(run_col, index=entry_col, values=quant_col, aggregate_function=None)
    )

    if output_entry_name is not None:
        quant_data = quant_data.rename({"PrimaryIds": output_entry_name})
    if output_column_map is not None:
        quant_data = quant_data.rename(output_column_map)
    if save_output_to_file is not None:
        quant_data.write_csv(save_output_to_file, separator="\t")
    return quant_data, None, save_output_to_file


def attach_wide_quant_to_long_report(
    report: pl.DataFrame,
    quant_data: pl.DataFrame,
    new_quant_col: str,
    report_run_col: str = cm.run,
    report_idx_col: str = cm.cut_site,
    quant_data_idx_col: str = "PrimaryIds",
    reverse_log: Optional[int] = None,
):
    """
    Receives a report dataframe in long format and a quantification dataframe in wide format.
    This function will attach a new column of quantity values from the quantification dataframe to the report dataframe,
    and return the updated report.
    If the quantity values in the quantification dataframe is log-transformed and they are expected to be reversed to original scale,
    set reverse_log to the base of log, else set to None (default).
    """
    if reverse_log is not None:
        if not isinstance(reverse_log, int):
            raise ValueError(f"When `reverse_log` is not None, it should be int, now {type(reverse_log)}")
    return report.join(
        quant_data.unpivot(
            index=quant_data_idx_col,
            variable_name=report_run_col,
            value_name=new_quant_col,
        )
        .with_columns(
            pl.col(new_quant_col).cast(pl.Float64).alias(new_quant_col)
            if reverse_log is None
            else (reverse_log ** pl.col(new_quant_col).cast(pl.Float64)).alias(new_quant_col)
        )
        .rename({quant_data_idx_col: report_idx_col}),
        on=[report_run_col, report_idx_col],
        how="left",
        coalesce=True,
    )


default_recollected_cols: dict[str, Sequence[str]] = {
    "pep": (
        cm.n_cut_site,
        cm.c_cut_site,
        cm.protein_group,
        cm.peptide_start_position,
        cm.peptide_end_position,
        cm.prev_aa,
        cm.next_aa,
        cm.peptide_n_term_aa,
        cm.peptide_c_term_aa,
        cm.nterm_enzymatic_specificity,
        cm.cterm_enzymatic_specificity,
        cm.peptide_enzymatic_specificity,
        cm.mapped_species_from_peptide,
    ),
    cm.cut_site: (
        cm.cut_site,
        cm.protein_group,
        cm.cut_site_n_aa,
        cm.cut_site_c_aa,
        cm.cut_site_is_restricted,
    ),
}
for _ in (cm.stripped_peptide, cm.modified_peptide, cm.precursor):
    default_recollected_cols[_] = default_recollected_cols["pep"]


def convert_long_report_to_wide(
    df: Union[pl.DataFrame, AbstractSearchReport],
    index_col: str = cm.precursor,
    column_col: str = cm.run,
    value_col: str = cm.precursor_quantity,
    agg_method: Optional[str] = "mean",
    do_log_scale: Optional[Union[int, float]] = 2,
    reverse_log_scale: Optional[Union[int, float]] = None,
    pl_filter: Optional[Union[Literal[True], pl.Expr]] = None,
    do_unique: Union[bool, str, Sequence[str]] = True,
    recollected_annotation_cols: Optional[Sequence[str]] = None,
    strict_recollection: bool = False,
):
    """
    Wrap the polars.pivot function to convert long format report to wide, with
    - optional unique for the (run, entry) group or specified columns
    - optional re-annotation for each entry after conversion

    Parameters
    ----------
    df : Union[pl.DataFrame, AbstractSearchReport]
        Input dataframe in long format
    index_col : str
        Row index, by default cm.precursor
    column_col : str
        Column index, by default cm.run
    value_col : str
        The quantification column used for this entry, by default cm.precursor_quantity
    agg_method : str, optional
        How to aggregate the quantification values if they have more than one for each entry in one run, by default "mean"
    do_log_scale : Optional[Union[int, float]], optional
        Do log scale transformation on the input data, by default 2
        Set this to None to skip log scale transformation.
    reverse_log_scale : bool, optional
        Whether to reverse the log scale to origin in the output, by default None.
        This will be ignored if `do_log_scale` is not None.
    pl_filter : Optional[Union[Literal[True], pl.Expr]], optional
        Polars expression to filter the input dataframe before conversion, by default None
        Note this will be conducted before unique, pivot, and re-annotation
    do_unique : Union[bool, str, Sequence[str]], optional
        Whether to do unique for the (run, entry) group, by default True
        - If True, do unique on columns (column_col, index_col)
        - If False, do nothing
        - If str or Sequence[str], do unique on the specified columns (column_col and index_col should be manurally included in this case)
    recollected_annotation_cols : Optional[Sequence[str]], optional
        Which columns should be re-annotated for each entry after conversion, by default None
        This should only include those columns that are consistent for each entry in every run, else only the first one will be kept
    strict_recollection : bool
        Requires the recollected columns present in dataframe, by default False
        If True, and columns defined in recollected_annotation_cols are not in df, will raise an error
    """
    if recollected_annotation_cols is not None:
        if isinstance(recollected_annotation_cols, str):
            recollected_annotation_cols = [recollected_annotation_cols]
        if not strict_recollection:
            recollected_annotation_cols = [c for c in recollected_annotation_cols if c in df.columns]
        if index_col not in recollected_annotation_cols:
            recollected_annotation_cols = [index_col, *recollected_annotation_cols]

    if isinstance(df, AbstractSearchReport):
        df = df.df

    if pl_filter is not None:
        if isinstance(pl_filter, (bool, pl.Expr)):
            df = df.filter(pl_filter)
        else:
            raise ValueError(f"Unsupported report_filter type: {type(pl_filter)}")

    if isinstance(do_unique, bool):
        if do_unique:
            df = df.unique([column_col, index_col])
    else:
        df = df.unique(do_unique)

    if do_log_scale is not None:
        df = df.with_columns(pl.col(value_col).log(do_log_scale).alias(value_col))
    elif reverse_log_scale is not None:
        df = df.with_columns((reverse_log_scale ** pl.col(value_col).cast(pl.Float32)).alias(value_col))

    df = df.pivot(column_col, index=index_col, values=value_col, aggregate_function=agg_method)
    if recollected_annotation_cols is None:
        return df
    return df.join(
        df.select(recollected_annotation_cols).unique(index_col),
        on=index_col,
        how="left",
        coalesce=True,
    )


construct_quant_mat_funcs: dict[str, Callable] = {
    "protein_group": functools.partial(convert_long_report_to_wide, index_col=cm.protein_group),
    "stripped_peptide": functools.partial(
        convert_long_report_to_wide,
        index_col=cm.stripped_peptide,
        recollected_annotation_cols=default_recollected_cols[cm.stripped_peptide],
    ),
    "modified_peptide": functools.partial(
        convert_long_report_to_wide,
        index_col=cm.modified_peptide,
        recollected_annotation_cols=default_recollected_cols[cm.modified_peptide],
    ),
    "precursor": functools.partial(
        convert_long_report_to_wide,
        index_col=cm.precursor,
        recollected_annotation_cols=default_recollected_cols[cm.precursor],
    ),
    "cut_site": functools.partial(
        convert_long_report_to_wide,
        index_col=cm.cut_site,
        recollected_annotation_cols=default_recollected_cols[cm.cut_site],
    ),
}
