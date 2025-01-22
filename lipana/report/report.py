import functools
import logging
import pickle
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Union

import numpy as np
import polars as pl

from ..annotations import (
    attach_annotation_from_other_df,
    construct_cut_site_identification_report,
    construct_input_for_quant_aggregation,
    convert_long_report_to_wide,
    do_quant_aggregation_via_iq,
    do_quant_aggregation_via_topk,
)
from ..base import (
    AbstractQuantificationReport,
    AbstractSearchReport,
    AbstractStatsReport,
    ExperimentSetting,
    _T_EntryLevels,
    cm,
)
from ..stats import (
    calc_cv_on_df,
    calc_ratio_on_df,
    count_df_selected_cols_nonnan,
)
from ..utils import (
    _T_InputOrAll,
    flatten_nested_list,
    gather_value_or_all,
    lookup_dict_with_tuple_key,
    read_df_from_parquet_or_tsv,
    write_df_to_parquet_or_tsv,
)

__all__ = [
    "SearchReport",
    "EntryQuantificationReport",
    "EntryStatsReport",
]

logger = logging.getLogger("lipana")


class SearchReport(AbstractSearchReport):
    def __init__(
        self,
        df: pl.DataFrame,
        exp_setting: ExperimentSetting,
        workspace: Optional[Union[str, Path]] = None,
    ):
        """
        The main class for the search report, which receives an annotated search report dataframe,
        an experiment setting object, and an optional workspace folder.

        The workspace folder is required for dumping all objects stored in this class to disk,
        for restoring the state of this class by loading from disk.

        Generally this class should be initialized by the class method `load_search_report`,
        which is defined in classes inheriting this class for specific report sources.
        There is usually a concomitant function for loading and annotating report defined in each report source module,
        like `load_diann_search_report` in `report_diann.py`, and the returned dataframe can be used as the input for this class.
        """
        self.df = df
        self.exp_setting = exp_setting
        self.workspace = workspace

        self._quant_input: dict[Union[_T_EntryLevels, tuple[_T_EntryLevels, str]], pl.DataFrame] = {}
        self._quant_data: dict[Union[_T_EntryLevels, tuple[_T_EntryLevels, str]], EntryQuantificationReport] = {}
        self._stats_result: dict[tuple[Union[_T_EntryLevels, tuple[_T_EntryLevels, str]], str], EntryStatsReport] = {}

    @property
    def workspace(self):
        return self._workspace

    @workspace.setter
    def workspace(self, path: Union[str, Path]):
        self._workspace = Path(path).resolve().absolute()

    @property
    def id_report(self):
        return self.df

    @property
    def quant_input(self):
        return self._quant_input

    @property
    def quant_data(self):
        return self._quant_data

    @property
    def stats_result(self):
        return self._stats_result

    def __getitem__(self, key):
        return self.df[key]

    def get_id_report(
        self,
        entry_name: Optional[Union[_T_EntryLevels, str]] = None,
        run_colname: str = cm.run,
        pl_filter: Optional[pl.Expr] = None,
    ):
        """
        Get the identification report, w/ or w/o unique on (run, entry).
        Basically the returned dataframe of this function will be based on the long-format report with cut site annotated.
        That is, this dataframe has each cut site on each row, and each precursor will be repeated once.
        When entry_name is defeind, the returned dataframe will be unique on (run, entry_name).
        The defined filter is performed after the unique action.
        """
        if pl_filter is None:
            pl_filter = True
        if entry_name is None:
            return self.df.filter(pl_filter)
        return self.df.unique([run_colname, entry_name]).filter(pl_filter)

    def get_quant_input(
        self,
        entry_name: Union[_T_EntryLevels, str],
        input_type_name: str,
    ):
        """
        Retrieve the quantification input dataframe from the internal dict.
        """
        return lookup_dict_with_tuple_key(
            self._quant_input,
            (entry_name, input_type_name),
            fallback_second_key=(None, "NotDefined"),
            retrieve_if_first_key_is_unique=False,  # don't retrieve because quant input might usually have second key as None
            raise_on_missing=True,
        )

    def get_quant_data(
        self,
        entry_name: Optional[Union[_T_EntryLevels, str]] = None,
        quant_name: Optional[str] = None,
        main_df_entry_filter: Optional[pl.Expr] = None,
        quant_df_filter: Optional[pl.Expr] = None,
        annotation_cols: Optional[Union[str, Sequence[str]]] = None,
        annotation_attach_method: Literal["leftjoin", "agg_leftjoin"] = "leftjoin",
    ) -> Union["EntryQuantificationReport", pl.DataFrame]:
        """
        Get the quantification data for the specified entry and quantification method.

        When both `main_df_entry_filter` and `quant_df_filter` is None, this method will return raw `EntryQuantificationReport` object.
        Else, this method will return the filtered dataframe of the quantification data.
        """
        if entry_name is None:
            quant_data = list(self.quant_data.values())[0]
        else:
            quant_data = lookup_dict_with_tuple_key(
                self.quant_data,
                (entry_name, quant_name),
                fallback_second_key=(None, "NotDefined"),
                retrieve_if_first_key_is_unique=True,
                raise_on_missing=True,
            )
        if (main_df_entry_filter is None) and (quant_df_filter is None):
            return quant_data

        qdf = quant_data.filter_entry_by_main_report(main_df_entry_filter)
        if quant_df_filter is not None:
            qdf = qdf.filter(quant_df_filter)
        if annotation_cols is not None:
            qdf = attach_annotation_from_other_df(
                qdf,
                self.df,
                annotation_cols=annotation_cols,
                on=entry_name,
                method=annotation_attach_method,
            )
        return qdf

    def get_stats_result(
        self,
        entry_name: Optional[Union[_T_EntryLevels, str]] = None,
        quant_name: Optional[str] = None,
        stats_method: Optional[str] = None,
        extra_annotation: Optional[str] = None,
        pl_filter: Optional[pl.Expr] = None,
    ) -> pl.DataFrame:
        key = (entry_name, quant_name, stats_method)
        if (data := self.stats_result.get(key)) is None:
            key = (*key, extra_annotation)
            data = self.stats_result.get(key)
        if data is None:
            raise ValueError(f"Stats result not found for key: {key}")
        if pl_filter is None:
            return data.df
        return data.df.filter(pl_filter)

    def list_quant_data_names(
        self,
        entry_name: Union[_T_EntryLevels, str] = None,
        sort: bool = True,
    ):
        if entry_name is None:
            keys = list(self.quant_data.keys())
        else:
            keys = [k for k in self.quant_data.keys() if k[0] == entry_name]
        return sorted(keys) if sort else keys

    def list_stats_result_names(
        self,
        entry_name: Union[_T_EntryLevels, str] = None,
        sort: bool = True,
    ):
        if entry_name is None:
            keys = list(self.stats_result.keys())
        else:
            keys = [k for k in self.stats_result.keys() if k[0] == entry_name]
        return sorted(keys) if sort else keys

    def attach_quant_input(
        self,
        entry_name: Union[_T_EntryLevels, str],
        input_type_name: str,
        quant_input: pl.DataFrame,
    ):
        """
        Attach a quantification input dataframe to this report object.
        This function helps to save the additional steps to construct the same quant input dataframe for the same entry but with different quant method.
        """
        self._quant_input[(entry_name, input_type_name)] = quant_input

    def clear_quant_input(
        self,
        entry_name: Union[_T_EntryLevels, str] = None,
        input_type_name: str = None,
        omit_unexist: bool = False,
    ):
        """
        Remove the quantification input dataframe from the internal dict.

        There are three scenarios:
        1. entry_name is None: remove all quant input dataframes.
        2. entry_name not None & input_type_name is None: remove all quant input dataframes for the specified entry.
        3. entry_name not None & input_type_name not None: remove the specified quant input dataframe.
        """
        if entry_name is None:
            self._quant_input.clear()
        elif input_type_name is None:
            for k in list(self._quant_input.keys()):
                if k[0] == entry_name:
                    del self._quant_input[k]
        else:
            if omit_unexist:
                if (entry_name, input_type_name) in self._quant_input:
                    del self._quant_input[(entry_name, input_type_name)]
            else:
                del self._quant_input[(entry_name, input_type_name)]

    def attach_quant_data(
        self,
        quant_data: Union[pl.DataFrame, "EntryQuantificationReport"],
        entry_level: Optional[Union[_T_EntryLevels, str]] = None,
        quant_method: Optional[str] = None,
    ) -> "EntryQuantificationReport":
        """
        Attach a quantification data to this report object.
        The quantification data can be a dataframe or an EntryQuantificationReport object.

        Parameters
        ----------
        entry_name : Union[_T_EntryLevels, str]
            The name of entry for this quantification data.
        quant_name : str
            The name of this quantification data. Will be used together with entry_name as the key to store the quantification data.
        quant_data : Union[pl.DataFrame, &quot;EntryQuantificationReport&quot;]
            A quantification dataframe or an EntryQuantificationReport object.
            If a dataframe is given, an EntryQuantificationReport object will be constructed with the given dataframe.
        """
        if isinstance(quant_data, pl.DataFrame):
            if entry_level is None:
                raise ValueError("When quant_data is a dataframe, entry_name should be defined")
            quant_data = EntryQuantificationReport(
                quant_data,
                exp_setting=self.exp_setting,
                entry_level=entry_level,
                quant_method=quant_method,
            )
        elif isinstance(quant_data, EntryQuantificationReport):
            pass
        else:
            raise ValueError(f"Unsupported quant_data type: {type(quant_data)}")
        self.quant_data[(quant_data.entry_level, quant_data.quant_method)] = quant_data
        return quant_data

    def construct_and_attach_quant_data(
        self,
        quant_name: Optional[str] = None,
        method: Optional[Union[Literal["maxlfq", "topk"], str]] = None,
        filter_condition: Union[Literal[True], pl.Expr] = True,
        run_col: str = cm.run,
        primary_entry_col: str = cm.cut_site,
        low_level_entry_col: Union[str, Sequence[str]] = cm.precursor,
        base_quant_col: Union[str, Sequence[str]] = cm.precursor_quantity,
        require_expansion: Union[bool, Sequence[bool], str, Sequence[str]] = False,
        concat_entry_after_expansion: Optional[Union[str, Sequence[str]]] = None,
        remove_below_threshold: Optional[Union[float, Sequence[float]]] = 1.1,
        quant_input_name: Optional[str] = None,
        attach_quant_input: bool = False,
    ):
        _shared_log_text = f'"{primary_entry_col}" from "{low_level_entry_col}" and "{base_quant_col}"'
        if method is None:
            method = "maxlfq"
            agg_func = functools.partial(do_quant_aggregation_via_iq, del_files=True)
            logger.info(f'Quantification aggregation method for {_shared_log_text} is None, set to default "maxlfq"')
        elif method.startswith("top"):
            if method.endswith("k"):
                k = 3
            elif method[3:].isdigit():
                k = int(method[3:])
            else:
                raise ValueError(f'Quantification aggregation method "{method}" is not supported')
            method = f"top{k}"
            agg_func = functools.partial(do_quant_aggregation_via_topk, topk=k)
            logger.info(f'Quantification aggregation method for {_shared_log_text} is set to "top{k}"')
        elif method.lower() == "maxlfq":
            method = "maxlfq"
            agg_func = functools.partial(do_quant_aggregation_via_iq, del_files=True)
            logger.info(f'Quantification aggregation method for {_shared_log_text} is set to "maxlfq"')
        else:
            raise ValueError(f'Quantification aggregation method "{method}" is not supported')

        quant_in = None
        if quant_input_name is not None:
            quant_in = self.get_quant_input(primary_entry_col, quant_input_name)
            if quant_in is None:
                logger.info(
                    f'Reuse quantification input "{quant_input_name}" for "{primary_entry_col}" is defined but not found, construct new one according to the given parameters'
                )
        if quant_in is None:
            quant_in = construct_input_for_quant_aggregation(
                df=self.df,
                filter_condition=filter_condition,
                run_col=run_col,
                primary_entry_col=primary_entry_col,
                low_level_entry_col=low_level_entry_col,
                base_quant_col=base_quant_col,
                require_expansion=require_expansion,
                concat_entry_after_expansion=concat_entry_after_expansion,
                remove_below_threshold=remove_below_threshold,
            )

        if attach_quant_input:
            self.attach_quant_input(primary_entry_col, quant_input_name, quant_in)

        quant_data, inp, outp = agg_func(quant_in, output_entry_name=primary_entry_col)

        if quant_name is None:
            quant_name = f"{method}-{base_quant_col if isinstance(base_quant_col, str) else '-'.join(base_quant_col)}"
            logger.info(
                f'For quantification estimation for {_shared_log_text}, with method "{method}", `quant_name` is not explicitly defined, set to "{quant_name}"'
            )
        quant_data = self.attach_quant_data(quant_data, primary_entry_col, quant_name)
        return quant_data

    def attach_stats_result(
        self,
        report: Union[pl.DataFrame, "EntryStatsReport"],
        entry_level: Optional[Union[_T_EntryLevels, str]] = None,
        quant_method: Optional[str] = None,
        stats_method: Optional[str] = None,
        extra_annotation: Optional[str] = None,
    ) -> "EntryStatsReport":
        if isinstance(report, pl.DataFrame):
            if entry_level is None:
                raise ValueError("When report is a dataframe, entry_name should be defined")
            report = EntryStatsReport(
                report,
                entry_level=entry_level,
                quant_method=quant_method,
                stats_method=stats_method,
                extra_annotation=extra_annotation,
            )
        if extra_annotation is None:
            key = (report.entry_level, report.quant_method, report.stats_method)
        else:
            key = (report.entry_level, report.quant_method, report.stats_method, extra_annotation)
        self.stats_result[key] = report
        return report

    def expand_to_cut_site_level(
        self,
        restricted_cut_sites: Sequence[str] = ("K", "R"),
        protein_terminal_role: Literal["drop", "restricted"] = "drop",
        do_unique_on: Sequence[str] = (cm.run, cm.cut_site, cm.precursor),
    ):
        """
        Expand the report to cut site level, with cut site annotation.
        Generally, the functions for loading report defined in this package, like `load_diann_search_report`,
        will have a parameter `expand_to_cut_site_level` and this default to True,
        which means no need to call this method manurally after obtaining the SearchReport object.

        This method is a wrap of function `construct_cut_site_identification_report`.
        """
        self.df = construct_cut_site_identification_report(
            self.df,
            restricted_cut_sites=restricted_cut_sites,
            protein_terminal_role=protein_terminal_role,
            do_unique_on=do_unique_on,
        )

    def count_detections(
        self,
        entry_name: Union[_T_EntryLevels, str],
        cond: _T_InputOrAll = None,
        run: _T_InputOrAll = None,
        min_reps: int = 1,
        pre_filter: Optional[pl.Expr] = None,
        run_col: str = cm.run,
        return_scaler: bool = True,
    ) -> int | dict[str, int]:
        """
        Count the number of detections.

        When both `cond` and `run` are None, will return the total number of entries detected in the whole experiment.
        Parameters `cond` and `run` are not mutually exclusive, and all defined conditions and runs will be the keys in the returned dict.

        Parameters
        ----------
        entry_name : Union[_T_EntryLevels, str]
            The column name of the entry.
        cond : _T_InputOrAll, optional
            Selected conditions, by default None
            - If None, count detections among all conditions
            - If str or Sequence[str] or "all", count detections in the specified conditions
        run : _T_InputOrAll, optional
            Selected runs, by default None
            - If None, count detections among all runs
            - If str or Sequence[str] or "all", count detections in the specified runs
        min_reps : int, optional
            Minimum number of replicates for a detection, by default 1
        """
        if pre_filter is not None:
            df = self.df.filter(pre_filter)
        else:
            df = self.df

        if (cond is None) and (run is None):
            return df.n_unique(entry_name)

        cond = gather_value_or_all(cond, list(self.exp_setting.condition_to_runs_map.keys()))
        run = gather_value_or_all(run, list(self.exp_setting.run_to_condition_map.keys()))
        if (len(cond) == 0) and (len(run) == 0):
            return {}

        min_reps = 1 if (min_reps is None) else min_reps

        df = df.unique([run_col, entry_name])

        result = {}
        for c in cond:
            result[c] = (
                df.filter(pl.col(run_col).is_in(self.exp_setting.condition_to_runs_map[c]))
                .filter(pl.col(run_col).len().over(entry_name).ge(min_reps))
                .n_unique(entry_name)
            )
        for r in run:
            result[r] = df.filter(pl.col(run_col).eq(r)).n_unique(entry_name)

        if return_scaler and (len(result) == 1):
            return result[list(result.keys())[0]]
        return result

    def show_status(self, name: str = None, compact: bool = True) -> None:
        msgs = [
            f"Search report{f' "{name}"' if (name is not None) else ''} object with main report shape: {self.df.shape}",
            f"Number of quantification data: {len(self.quant_data)}",
            f"Number of statistics results: {len(self.stats_result)}",
            f"Workspace: {self.workspace}",
        ]
        if compact:
            logger.info(" | ".join(msgs))
        else:
            for msg in msgs:
                logger.info(msg)

    def dump(
        self,
        folder_or_path: Union[str, Path] = None,
        save_type: Literal["individual", "pkl"] = "individual",
    ):
        """
        Dump this report object to disk.

        When `save_type` is "individual", the `folder_or_path` should be a folder,
        and long-format report, quantification data, stats results,
        and other meta data will be saved as individual files in the defined folder.

        When `save_type` is "pkl", the `folder_or_path` should be a file path,
        and the whole report object will be saved as a pickle file.

        By default, can leave `folder_or_path` as None, and a default folder "lipana_analysis" will be created in the folder of `self.workspace`.
        """
        if folder_or_path is None:
            if self.workspace is None:
                raise ValueError("When `folder_or_path` is None, `self.workspace` can not be None")
            folder_or_path = self.workspace.joinpath("lipana_analysis")
            logger.info(f'`folder_or_path` is None, will dump to "{folder_or_path}"')
            save_type = "individual"

        if save_type == "individual":
            folder_or_path.mkdir(exist_ok=True)
            self.exp_setting.dump(folder_or_path.joinpath("experiment_setting.tsv"))
            self.df.write_parquet(folder_or_path.joinpath("search_report.parquet"))
            for k, q in self._quant_input.items():
                pass
            for q in self.quant_data.values():
                q.dump(folder_or_path)
            for s in self.stats_result.values():
                s.dump(folder_or_path)
        elif save_type == "pkl":
            folder_or_path = Path(folder_or_path).resolve()
            with open(folder_or_path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(f"Unsupported save_type: {save_type}")
        return folder_or_path

    @classmethod
    def load(
        cls,
        folder_or_path: Union[str, Path],
        load_type: Literal["individual", "pkl"] = "individual",
        show_status_after_load: bool = False,
        name_in_shown_status: Optional[str] = None,
    ):
        """
        Load a report object from disk.

        When `load_type` is "individual", the `folder_or_path` should be a folder,
        and the report object will be constructed by loading individual files in the defined folder.

        When `load_type` is "pkl", the `folder_or_path` should be a file path,
        and the report object will be loaded from the pickle file.
        """
        if load_type == "individual":
            folder_or_path = Path(folder_or_path).resolve()
            exp_setting = ExperimentSetting.from_file(folder_or_path.joinpath("experiment_setting.tsv"))
            df = pl.read_parquet(folder_or_path.joinpath("search_report.parquet"))

            obj = cls(df, exp_setting, workspace=folder_or_path.parent)

            for f in folder_or_path.iterdir():
                if f.name.startswith("quant_input!!"):
                    pass
                elif f.name.startswith("quant!!"):
                    obj.attach_quant_data(EntryQuantificationReport.load(f, exp_setting, main_report=obj))
                elif f.name.startswith("stats!!"):
                    pass
                else:
                    pass
            if show_status_after_load:
                obj.show_status(name=name_in_shown_status)
            return obj
        elif load_type == "pkl":
            with open(folder_or_path, "rb") as f:
                obj: SearchReport = pickle.load(f)
            if show_status_after_load:
                obj.show_status(name=name_in_shown_status)
            return obj
        else:
            raise ValueError(f"Unsupported load_type: {load_type}")


class EntryQuantificationReport(AbstractQuantificationReport):
    def __init__(
        self,
        df: pl.DataFrame,
        exp_setting: ExperimentSetting,
        entry_level: _T_EntryLevels,
        quant_method: Optional[str] = None,
        main_report: Optional[SearchReport] = None,
    ):
        """
        A quantification report and its meta-data at a specific entry level.

        Parameters
        ----------
        df : pl.DataFrame
            A quantification dataframe which is in wide-format.
            Use `from_long_report` to init this class by giving a long-format report as the input.
        exp_setting : ExperimentSetting
            The experiment setting object that defines the mapping between runs and conditions.
        entry_level : _T_EntryLevels
            The entry level of this quantification report, and should be the column name of the entry if the entry column exists in the dataframe.
        quant_method : Optional[str], optional
            The quantification method used for this report, by default None.
            This will be set to "NotDefined" if not given.
        """
        self.df = df
        if exp_setting is None:
            raise ValueError("ExperimentSetting object should be provided.")
        self.exp_setting = exp_setting
        self.cond_to_run_map = exp_setting.condition_to_runs_map
        self.entry_level = entry_level
        self.quant_method = quant_method if (quant_method is not None) else "NotDefined"
        self.__input_columns = list(self.df.columns)
        self._main_report = main_report

    @classmethod
    def from_long_report(
        cls,
        long_report: Union[pl.DataFrame, SearchReport],
        exp_setting: ExperimentSetting = None,
        entry_quant_col: str = "cut_site_quantity",
        entry_col: str = cm.cut_site,
        run_col: str = cm.run,
        quant_agg_method: Optional[str] = "mean",
        pl_filter: Optional[Union[Literal[True], pl.Expr]] = True,
        do_unique: Union[bool, str, Sequence[str]] = True,
        recollected_annotation_cols: Sequence[str] = None,
    ) -> "EntryQuantificationReport":
        """
        Construct a quantification report object from a long-format report.
        When the input is a SearchReport object, the dataframe in the object will be used as the input, and exp_setting can be None.
        """
        if isinstance(long_report, SearchReport):
            exp_setting = long_report.exp_setting
            long_report = long_report.df
        return cls(
            convert_long_report_to_wide(
                df=long_report,
                index_col=entry_col,
                column_col=run_col,
                value_col=entry_quant_col,
                agg_method=quant_agg_method,
                pl_filter=pl_filter,
                do_unique=do_unique,
                recollected_annotation_cols=recollected_annotation_cols,
                strict_recollection=False,
            ),
            exp_setting=exp_setting,
            entry_level=entry_col,
            quant_method=entry_quant_col,
        )

    def keys(self) -> tuple[_T_EntryLevels, str]:
        return (self.entry_level, self.quant_method)

    def remove_additional_columns(self):
        self.df = self.df.select(self.__input_columns)

    def __str__(self) -> str:
        return f"Quantification report at {self.entry_level} level, quantified by {self.quant_method}"

    def __getitem__(self, key):
        return self.df[key]

    def filter_entry_by_main_report(
        self,
        main_report_filter: Optional[pl.Expr] = None,
        # unique_on: Optional[Union[Union[str, Sequence[str]]]] = None,
    ) -> pl.DataFrame:
        if self._main_report is None:
            raise ValueError("To filter entry by main report, the main report should be provided.")
        if main_report_filter is None:
            return self.df
        return self.df.join(self._main_report.df.filter(main_report_filter), on=self.entry_level, how="semi")

    def attach_annotation_via_entry(
        self,
        annotation_cols: Union[str, Sequence[str]],
        persistent: bool = False,
        annotation_attach_method: Literal["leftjoin", "agg_leftjoin"] = "leftjoin",
    ) -> pl.DataFrame:
        """
        Attach annotation columns to the quantification dataframe from the main report by using entry as key.
        Will return a new dataframe with the annotation columns attached.
        This equals to left-join the quantification dataframe and the selected columns in main report on the entry column.
        If `persistent` is True, the dataframe stored in the object will also be updated, else only return a new dataframe with expected columns.
        """
        if self._main_report is None:
            raise ValueError("To attach annotation via entry, the main report should be provided.")
        if isinstance(annotation_cols, str):
            annotation_cols = [annotation_cols]
        df = attach_annotation_from_other_df(
            self.df,
            self._main_report.df,
            annotation_cols=annotation_cols,
            on=self.entry_level,
            method=annotation_attach_method,
        )
        if persistent:
            self.df = df
        return df

    def select_conditions(
        self,
        cond: _T_InputOrAll = None,
        agg_method: Optional[Literal["mean", "median", "max"]] = None,
        keep_entry_col: bool = True,
        keep_other_cols: bool = False,
    ) -> pl.DataFrame:
        """
        Select conditions from the quantification matrix w/ or w/o quantity aggregation.

        Parameters
        ----------
        cond : _T_InputOrAll, optional
            Selected conditions, by default None
            - If None or "all" (case-insensitive), select all conditions
            - If str or Sequence[str], select the specified conditions
        agg_method : Optional[Literal[&quot;mean&quot;, &quot;median&quot;, &quot;max&quot;]], optional
            Aggregation method for values from multiple runs, by default "mean"
            - If None, return the selected runs without aggregation (this will return runs as columns instead of conditions)
            - If "mean", "median", or "max", aggregate the values using the specified method, and the returned df will have defined conditions as columns instead of runs
        keep_entry_col : bool, optional
            Whether to include the entry column in result, by default True
        keep_other_cols : bool, optional
            Whether to include all other columns (exclude all runs and entry col) in result, by default False

        """
        conds = gather_value_or_all(cond, list(self.exp_setting.all_conditions))
        runs = flatten_nested_list([self.exp_setting.condition_to_runs_map[c] for c in conds])

        kept_cols = [self.entry_level] if keep_entry_col else []
        if keep_other_cols:
            kept_cols.extend([col for col in self.df.columns if (col not in runs) and (col != self.entry_level)])

        if agg_method is None:
            return self.df.select([*kept_cols, *runs])

        np_agg_func: Callable = getattr(np, f"nan{agg_method}")
        return pl.concat(
            (
                self.df.select(kept_cols),
                pl.DataFrame(
                    {c: np_agg_func(self.df.select(self.exp_setting.condition_to_runs_map[c]), axis=1) for c in conds}
                ),
            ),
            how="horizontal",
        )

    def select_runs(
        self,
        run: _T_InputOrAll = None,
        keep_entry_col: bool = True,
        keep_other_cols: bool = False,
    ) -> pl.DataFrame:
        """
        Select runs from the quantification matrix.

        Parameters
        ----------
        run : _T_InputOrAll, optional
            Selected runs, by default None
            - If None or "all" (case-insensitive), select all runs
            - If str or Sequence[str], select the defined runs
        keep_entry_col : bool, optional
            Whether to include the entry column in result, by default True
        keep_annotation_cols : bool, optional
            Whether to include all other columns (exclude selected runs/conditions and entry col) in result, by default False
        """

        runs = gather_value_or_all(run, list(self.exp_setting.all_runs))

        kept_cols = [self.entry_level] if keep_entry_col else []
        if keep_other_cols:
            kept_cols.extend([col for col in self.df.columns if (col not in runs) and (col != self.entry_level)])

        return self.df.select([*kept_cols, *runs])

    def calc_cv(
        self,
        cond: _T_InputOrAll = "all",
        min_reps: int = 3,
        temp_reverse_log_scale: Optional[int] = 2,
        new_colname_pattern: str = "{cond}_cv_{min_reps}reps",
    ) -> pl.DataFrame:
        """
        Calculate the coefficient of variation (cv) for each condition and attach the result to the dataframe.
        Note: all cvs are made as percentage.

        Parameters
        ----------
        cond : _T_InputOrAll, optional
            Selected conditions, by default "all"
        min_reps : int, optional
            Minimum quantity values required for calculating CV, by default 3.
            One entry with quantities less than this number in a certain condition will have a cv of NaN.
        temp_reverse_log_scale: Optional[int], optional
            To temporarily reverse the log scale of the quantity values for cv calculation, by default 2.
            If None, will use quantity values as they in the current dataframe.
        new_colname_pattern : str, optional
            The pattern of the column names that will be attached to quantification dataframe, by default "{cond}_cv_{min_reps}reps".
            Can set to something like "{cond}-CV" to omit the annotation of minimum replicates.

        Returns
        -------
        pl.DataFrame
            The dataframe with cv columns attached.
        """
        self.df = calc_cv_on_df(
            self.df,
            cond_to_cols_map=self.exp_setting.condition_to_runs_map,
            cond=cond,
            min_reps=min_reps,
            temp_reverse_log_scale=temp_reverse_log_scale,
            new_colname_pattern=new_colname_pattern,
        )
        return self.df

    def count_detected_replicates(
        self,
        cond="all",
        new_colname_pattern: str = "{cond}_detected_reps",
    ) -> pl.DataFrame:
        """
        Count the number of detected replicates for defined conditions and attach the results to the dataframe.

        Parameters
        ----------
        cond : Optional[_T_InputOrAll], optional
            Selected conditions, by default "all"
        new_colname_pattern : str, optional
            The pattern of the column names that will be attached to quantification dataframe, by default "{cond}_detected_reps".

        Returns
        -------
        pl.DataFrame
            The quantification dataframe.
        """
        cond = gather_value_or_all(cond, list(self.cond_to_run_map.keys()))
        self.df = self.df.with_columns(
            pl.lit(
                count_df_selected_cols_nonnan(self.df, cols=self.cond_to_run_map[c], count_col=False),
                dtype=pl.Int8,
            ).alias(new_colname_pattern.format(cond=c))
            for c in cond
        )
        return self.df

    def count_detections_below_cv(
        self,
        cond: _T_InputOrAll = "all",
        min_reps: Optional[int] = None,
        detected_rep_colname_pattern: str = "{cond}_detected_reps",
        cv_threshold: Optional[float] = 20.0,
        cv_colname_pattern: str = "{cond}_cv_3reps",
    ) -> dict[str, int]:
        """
        Count the number of detected entries for defined conditions and return a dict with pairs {condition: number}.
        Can set min_reps and/or cv_threshold to filter the entries to be counted.

        Parameters
        ----------
        cond : _T_InputOrAll, optional
            Selected conditions, by default "all"
        min_reps : Optional[int], optional
            The minimum number of replicates required for counting, by default None
            If defined, only those entries with quantity values more than this number will be counted.
            Generally this parameter has no effect if cv_threshold is not None, since the calculation of cv has a requirement of least replicates.
        cv_threshold : Optional[float], optional
            The minimum cv value required for counting, by default 20.0
            If defined, only those entries with cv values below this threshold will be counted, else no contribution to the counting.
        cv_colname_pattern : str, optional
            Which column to use for cv values, by default "{cond}_cv_3reps"

        Returns
        -------
        dict[str, int]
            A dict with pairs {condition: number of entries}
        """
        cond = gather_value_or_all(cond, list(self.cond_to_run_map.keys()))

        return {
            c: self.df.filter(
                True if (min_reps is None) else (pl.col(detected_rep_colname_pattern.format(cond=c)) >= min_reps)
            )
            .filter(True if (cv_threshold is None) else (pl.col(cv_colname_pattern.format(cond=c)) < cv_threshold))
            .n_unique(self.entry_level)
            for c in cond
        }

    def calc_ratio(
        self,
        base_cond: _T_InputOrAll = None,
        cond_pairs: Union[tuple[str, str], Sequence[tuple[str, str]]] = None,
        is_log: bool = False,
        temp_reverse_log_scale: Optional[int] = 2,
        div_method: Literal["agg_and_divide", "divide_and_agg"] = "agg_and_divide",
        agg_method: Literal["mean", "median", "absmax", "absmin", "interquartile_mean"] = "mean",
        new_colname_pattern: str = "ratio_{cond1}_to_{cond2}",
    ) -> pl.DataFrame:
        """
        Calculate the ratio of quantity values between two conditions and attach the results to the dataframe.
        Can set either base condition(s) or pair(s) of conditions, or a mixture of them.
        If both base_cond and cond_pairs are None, will calculate ratios for all available condition pairs.

        Parameters
        ----------
        base_cond : _T_InputOrAll, optional
            The base condition(s) for calculating the ratio, by default None.
            If "all", will calculate ratios for all available conditions pairs.
        cond_pairs : Union[tuple[str, str], Sequence[tuple[str, str]]], optional
            Explicitly define the expected condition pairs to calculate the ratio, by default None.
        is_log : bool, optional
            Whether the quantity values are log-transformed, by default False.
        temp_reverse_log_scale : Optional[int], optional
            The base of the log scale, by default None.
            When set, will temporarily reverse the log scale for ratio calculation (the result will be reverted to original log scale again).
        div_method : Literal[&quot;agg_and_divide&quot;, &quot;divide_and_agg&quot;], optional
            see `calc_ratio_batch` for details, by default "divide_and_agg"
        agg_method : Literal[&quot;mean&quot;, &quot;median&quot;, &quot;absmax&quot;, &quot;absmin&quot;, &quot;interquartile_mean&quot;], optional
            see `calc_ratio_batch` for details, by default "interquartile_mean"
        new_colname_pattern : str, optional
            The pattern of the column names that will be attached to quantification dataframe, by default "ratio_{cond1}_to_{cond2}".

        Returns
        -------
        pl.DataFrame
            The quantification dataframe.
        """
        self.df = calc_ratio_on_df(
            self.df,
            cond_to_cols_map=self.exp_setting.condition_to_runs_map,
            base_cond=base_cond,
            cond_pairs=cond_pairs,
            is_log=is_log,
            temp_reverse_log_scale=temp_reverse_log_scale,
            div_method=div_method,
            agg_method=agg_method,
            new_colname_pattern=new_colname_pattern,
        )
        return self.df

    def get_cond_quants(
        self,
        cond: _T_InputOrAll = "all",
        main_report_filter: Optional[pl.Expr] = None,
        temp_reverse_log_scale: Optional[int] = 2,
        agg_method: Literal["mean", "median", "max"] = "mean",
        min_reps: Optional[int] = None,
        detected_rep_colname_pattern: str = "{cond}_detected_reps",
        cv_threshold: Optional[float] = None,
        cv_colname_pattern: str = "{cond}_cv_3reps",
    ) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """
        Get the aggregated quantity values for defined conditions.
        Will return a numpy array if only one condition is selected, else a dict with pairs {condition: np.ndarray}.
        """
        cond = gather_value_or_all(cond, list(self.cond_to_run_map.keys()))
        if main_report_filter is not None:
            df = self.filter_entry_by_main_report(main_report_filter)
        else:
            df = self.df
        result = {}
        for c in cond:
            sub_df = df
            if min_reps is not None:
                sub_df = sub_df.filter(pl.col(detected_rep_colname_pattern.format(cond=c)).ge(min_reps))
            if cv_threshold is not None:
                sub_df = sub_df.filter(pl.col(cv_colname_pattern.format(cond=c)).lt(cv_threshold))

            agg_func = getattr(np, f"nan{agg_method}")
            arr = sub_df.select(self.cond_to_run_map[c]).to_numpy()
            if temp_reverse_log_scale is not None:
                arr = np.log(agg_func(np.power(temp_reverse_log_scale, arr), axis=1)) / np.log(temp_reverse_log_scale)
            else:
                arr = agg_func(arr, axis=1)
            result[c] = arr
        if len(result) == 1:
            return list(result.values())[0]
        return result

    def dump(self, folder: Union[str, Path] = None, path: Union[str, Path] = None):
        """
        Dump the quantification report to disk.
        Will first use `folder` if defined, else use `path`.

        When `folder` is defined, the file name will be constructed as "quant!!{entry_level}!!{quant_method}.parquet".
        """
        if folder is not None:
            path = Path(folder).joinpath(f"quant!!{self.entry_level}!!{self.quant_method}.parquet")
        elif path is not None:
            path = Path(path)
        else:
            raise ValueError("Either `folder` or `path` should be provided.")
        write_df_to_parquet_or_tsv(self.df, path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        exp_setting: ExperimentSetting,
        entry_level: Optional[_T_EntryLevels] = None,
        quant_method: Optional[str] = None,
        main_report: Optional[SearchReport] = None,
    ):
        """
        Load a quantification report from disk.
        """

        file_basename = Path(path).stem
        if file_basename.count("!!") == 2:
            entry_level, quant_method = file_basename.split("!!")[1:]
        else:
            if entry_level is None:
                raise ValueError(
                    "When `path` does not contain entry level info, `entry_level` should be explicitly defined."
                )
        return cls(read_df_from_parquet_or_tsv(path), exp_setting, entry_level, quant_method, main_report)

    def copy(self):
        return EntryQuantificationReport(
            self.df.clone(), self.exp_setting, self.entry_level, self.quant_method, self._main_report
        )


class EntryStatsReport(AbstractStatsReport):
    """
    #>    ProteinName       PeptideSequence                 FULL_PEPTIDE        Label
    #> 1:      P14164               ILQNDLK               P14164_ILQNDLK Ctrl vs Osmo
    #> 2:      P16622 SHLQSNQLYSNQLPLDFALGK P16622_SHLQSNQLYSNQLPLDFALGK Ctrl vs Osmo
    #>        log2FC          SE      Tvalue DF      pvalue adj.pvalue issue
    #> 1:  1.0601391 0.219397767   4.8320413  4 0.008448744 0.03942747    NA
    #> 2:  0.1655540 0.277961791   0.5955998  3 0.593383886 0.71358286    NA
    #>    MissingPercentage ImputationPercentage fully_TRI NSEMI_TRI CSEMI_TRI
    #> 1:         0.0000000                    0      TRUE     FALSE     FALSE
    #> 2:         0.1666667                    0      TRUE     FALSE     FALSE
    #>    CTERMINUS NTERMINUS StartPos EndPos
    #> 1:     FALSE     FALSE      358    365
    #> 2:     FALSE     FALSE      354    375
    """

    def __init__(
        self,
        df: pl.DataFrame,
        entry_level: _T_EntryLevels,
        quant_method: Optional[str] = None,
        stats_method: Optional[str] = None,
        extra_annotation: Optional[str] = None,
    ):
        self.df = df
        self.entry_level = entry_level
        self.quant_method = quant_method
        self.stats_method = stats_method
        self.extra_annotation = extra_annotation

    @property
    def report(self):
        return self.df

    @classmethod
    def from_quant_data(
        cls,
        quant_data: EntryQuantificationReport,
        stats_method: str,
        extra_annotation: str = None,
    ):
        pass

    def dump(self, folder: Union[str, Path] = None, path: Union[str, Path] = None):
        """
        Dump the statistical result to disk.
        Will first use `folder` if defined, else use `path`.

        When `folder` is defined, the file name will be constructed as "stats!!{entry_level}!!{quant_method}!!{stats_method}.parquet".
        """
        if folder is not None:
            if self.extra_annotation is None:
                path = Path(folder).joinpath(
                    f"stats!!{self.entry_level}!!{self.quant_method}!!{self.stats_method}.parquet"
                )
            else:
                path = Path(folder).joinpath(
                    f"stats!!{self.entry_level}!!{self.quant_method}!!{self.stats_method}!!{self.extra_annotation}.parquet"
                )
        elif path is not None:
            path = Path(path)
        else:
            raise ValueError("Either `folder` or `path` should be provided.")
        write_df_to_parquet_or_tsv(self.df, path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        entry_level: Optional[_T_EntryLevels] = None,
        quant_method: Optional[str] = None,
        stats_method: Optional[str] = None,
        extra_annotation: Optional[str] = None,
    ):
        """
        Load a statistical report from disk.
        """
        file_basename = Path(path).stem
        if file_basename.count("!!") == 3:
            entry_level, quant_method, stats_method = file_basename.split("!!")[1:]
        elif file_basename.count("!!") == 4:
            entry_level, quant_method, stats_method, extra_annotation = file_basename.split("!!")[1:]
        else:
            if entry_level is None:
                raise ValueError(
                    "When `path` does not contain entry level info, `entry_level` should be explicitly defined."
                )
        return cls(read_df_from_parquet_or_tsv(path), entry_level, quant_method, stats_method, extra_annotation)


class StatsTrack:
    pass
