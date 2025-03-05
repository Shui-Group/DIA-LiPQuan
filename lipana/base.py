import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Union

import polars as pl

from .utils import write_df_to_parquet_or_tsv

__all__ = [
    "_T_EntryLevels",
    "ColumnMap",
    "column_map",
    "cm",
    "AbstractSearchReport",
    "AbstractQuantificationReport",
    "AbstractStatsReport",
    "ExperimentLayout",
    "ComparisonDesign",
]

logger = logging.getLogger("lipana")

_T_EntryLevels = Literal[
    "protein",
    "protein_group",
    "gene",
    "gene_group",
    "cut_site",
    "ptm_site",
    "stripped_peptide",
    "modified_peptide",
    "precursor",
]


@dataclass
class ColumnMap:
    run: str = "run"
    condition: str = "condition"
    replicate: str = "replicate"
    cond_rep: str = "cond_rep"

    protein_group: str = "protein_group"
    first_protein: str = "first_protein"
    precursor: str = "precursor"
    modified_peptide: str = "modified_peptide"
    precursor_charge: str = "precursor_charge"
    stripped_peptide: str = "stripped_peptide"
    n_cut_site: str = "n_cut_site"
    c_cut_site: str = "c_cut_site"
    cut_site: str = "cut_site"
    cut_site_on_term: str = "cut_site_on_term"
    cut_site_n_aa: str = "cut_site_n_aa"
    cut_site_c_aa: str = "cut_site_c_aa"
    cut_site_is_restricted: str = "cut_site_is_restricted"

    protein_group_quantity: str = "protein_group_quantity"
    stripped_peptide_quantity: str = "stripped_peptide_quantity"
    modified_peptide_quantity: str = "modified_peptide_quantity"
    precursor_quantity: str = "precursor_quantity"
    precursor_quantity_normalised: str = "precursor_quantity_normalised"
    precursor_quantity_ms1: str = "precursor_quantity_ms1"
    precursor_quantity_ms1_normalised: str = "precursor_quantity_ms1_normalised"
    precursor_quantity_ms2: str = "precursor_quantity_ms2"
    precursor_quantity_ms2_normalised: str = "precursor_quantity_ms2_normalised"

    peptide_start_position: str = "peptide_start_position"
    peptide_end_position: str = "peptide_end_position"
    prev_aa: str = "prev_aa"
    next_aa: str = "next_aa"
    peptide_n_term_aa: str = "peptide_n_term_aa"
    peptide_c_term_aa: str = "peptide_c_term_aa"

    nterm_enzymatic_specificity: str = "nterm_enzymatic_specificity"
    cterm_enzymatic_specificity: str = "cterm_enzymatic_specificity"
    peptide_enzymatic_specificity: str = "peptide_enzymatic_specificity"

    mapped_species_from_protein: str = "mapped_species_from_protein"
    mapped_species_from_peptide: str = "mapped_species_from_peptide"


column_map = ColumnMap()
cm = column_map


class AbstractSearchReport:
    pass


class AbstractQuantificationReport:
    pass


class AbstractStatsReport:
    pass


@dataclass
class ExperimentLayout:
    """
    exp_df should have 3 columns named "run", "condition", and "replicate".
    See `from_df`, `from_file`, and `from_run_to_condition_map` for the initialization of this class.
    """

    exp_df: pl.DataFrame

    @classmethod
    def from_df(
        cls,
        exp_df: pl.DataFrame,
        run_col: str = "run",
        condition_col: str = "condition",
        replicate_col: Optional[str] = "replicate",
    ):
        """
        Input a experiment setting dataframe with at least two columns indicating run name and condition name
        If replicate_col is None, a new column will be created with the name "replicate" with values starting from 1 for each condition
        """
        if replicate_col is None:
            replicate_col = cm.replicate
            exp_df = exp_df.with_columns(pl.int_range(1, pl.len() + 1).over(condition_col).alias(replicate_col))
        exp_df = exp_df.rename(
            {
                run_col: cm.run,
                condition_col: cm.condition,
                replicate_col: cm.replicate,
            }
        )
        return cls(exp_df)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        run_col: str = "run",
        condition_col: str = "condition",
        replicate_col: Optional[str] = "replicate",
    ):
        path = Path(path).resolve().absolute()
        logger.info(f"Loading experiment setting from {str(path)}")
        return cls.from_df(pl.read_csv(path, separator="\t"), run_col, condition_col, replicate_col)

    @classmethod
    def from_run_to_condition_map(
        cls,
        run_to_condition_map: dict[str, str],
    ):
        return cls.from_df(
            pl.DataFrame(list(run_to_condition_map.items()), schema=[cm.run, cm.condition], orient="row"),
            run_col=cm.run,
            condition_col=cm.condition,
            replicate_col=None,
        )

    def __post_init__(self):
        self.run_to_condition_map = dict(self.exp_df.select([cm.run, cm.condition]).rows())
        self.run_to_condrep_map = dict(
            zip(self.exp_df[cm.run].to_list(), self.exp_df.select([cm.condition, cm.replicate]).rows(), strict=True)
        )

        self.condition_to_runs_map = dict(self.exp_df.group_by(cm.condition).agg(pl.col(cm.run)).rows())
        self.condition_to_condreps_map = {
            cond: [self.run_to_condrep_map[r] for r in runs] for cond, runs in self.condition_to_runs_map.items()
        }

        self.condrep_to_run_map = {condrep: run for run, condrep in self.run_to_condrep_map.items()}
        self.condrep_to_condition_map = {
            condrep: cond for cond, condreps in self.condition_to_condreps_map.items() for condrep in condreps
        }

        self.all_conditions = list(self.condition_to_runs_map.keys())
        self.all_runs = list(self.run_to_condition_map.keys())
        self.all_condreps = list(self.run_to_condrep_map.values())

        self.n_replicates_map = dict(self.exp_df.group_by(cm.condition).len().rows())

    def dump(self, path: Union[str, Path]) -> Path:
        path = Path(path).resolve().absolute()
        write_df_to_parquet_or_tsv(self.exp_df, path)
        return path


@dataclass
class ComparisonDesign:
    exp_layout: ExperimentLayout
    pairwise_comparisons: list[tuple[str, str]] = field(default_factory=list)
    # contrast_matrix: Optional[list[list[int]]] = None

    used_conditions: list[str] = field(default_factory=list, init=False)

    def add_pairwise_comparison(
        self,
        treatment: Optional[str] = None,
        control: Optional[str] = None,
    ) -> "ComparisonDesign":
        """
        Add one or more comparison pairs to the comparison design.
        Set both `treatment` and `control` to add a single comparison pair "treatment" to "control".
        Set only `treatment` to add all comparisons with `treatment` as the treatment condition.
        Set only `control` to add all comparisons with `control` as the control condition.
        Set both `treatment` and `control` to `None` to add all available comparisons.

        Parameters
        ----------
        treatment : Optional[str], optional
            The treatment condition to compare, by default None.
        control : Optional[str], optional
            The control condition to compare, by default None.
        """
        if treatment is None:
            treatment = self.exp_layout.all_conditions
        if control is None:
            control = self.exp_layout.all_conditions
        if isinstance(treatment, str):
            treatment = [treatment]
        if isinstance(control, str):
            control = [control]
        self.pairwise_comparisons.extend((t, c) for t in treatment for c in control)
        self._update_inner_states()
        return self

    def delete_pairwise_comparison(
        self,
        treatment: Optional[str] = None,
        control: Optional[str] = None,
    ) -> "ComparisonDesign":
        if treatment is None:
            treatment = self.exp_layout.all_conditions
        if control is None:
            control = self.exp_layout.all_conditions
        if isinstance(treatment, str):
            treatment = [treatment]
        if isinstance(control, str):
            control = [control]
        self.pairwise_comparisons = [
            p for p in self.pairwise_comparisons if (p[0] not in treatment) and (p[1] not in control)
        ]
        self._update_inner_states()
        return self

    def _update_inner_states(self):
        self.used_conditions = list(
            set(p[0] for p in self.pairwise_comparisons) | set(p[1] for p in self.pairwise_comparisons)
        )

    def __post_init__(self):
        self._update_inner_states()
