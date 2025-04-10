import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

__all__ = [
    "flatten_list",
    "subtract_list",
    "unique_list_ordered",
    "exec_r_script",
    "normalize_tuple",
    "lookup_dict_with_tuple_key",
    "_T_InputOrAll",
    "gather_value_or_all",
    "check_query_in_vec",
    "filter_top_n_by_group",
    "add_bool_mark_by_expr",
    "write_df_to_parquet_or_tsv",
    "read_df_from_parquet_or_tsv",
    "resume_file",
    "AbstractDFManiConfig",
    "DFUniqueConfig",
    "DFFilterConfig",
    "DFDropColConfig",
    "DFConcatConfig",
    "DFColRenameConfig",
    "DFAddLitColConfig",
    "do_df_mani",
]

logger = logging.getLogger("lipana")


def flatten_list(nested_list: list[Union[list, Any]]) -> list[Any]:
    flattened = []
    for elem in nested_list:
        if isinstance(elem, list):
            flattened.extend(flatten_list(elem))
        else:
            flattened.append(elem)
    return flattened


def subtract_list(inlist: list[Any], *sublists: list[Any]) -> list[Any]:
    """
    Remove elements from the input list if they appear in any of the provided sublists.

    Parameters
    ----------
    inlist : list[Any]
        The input list to filter elements from
    *sublists : list[Any]
        One or more lists containing elements to be removed from inlist

    Returns
    -------
    list[Any]
        A new list with elements from inlist that are not present in any of the sublists
    """
    if not sublists:
        return inlist.copy()

    # Combine all sublists into a single set for efficient lookups
    elements_to_remove = set()
    for sublist in sublists:
        elements_to_remove.update(sublist)

    # Return elements from inlist that are not in elements_to_remove
    return [item for item in inlist if item not in elements_to_remove]


def unique_list_ordered(items: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def exec_r_script(
    rscript_exec: str,
    script_path: Union[str, Path],
    *script_args,
) -> None:
    """
    Execute R script and handle errors.
    """
    logger.info(f"Executing R script: {Path(script_path).name}, args: {script_args}")
    try:
        result = subprocess.run(
            [rscript_exec, str(script_path), *script_args],
            capture_output=False,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            error_msg = result.stderr or f"Error code {result.returncode}"
            raise ValueError(f"R script execution failed: {error_msg}")
    except FileNotFoundError as e:
        raise ValueError(f"Failed to find Rscript executable: {str(e)}\nRequired process needs R installed") from e
    except Exception as e:
        raise ValueError(f"Failed to execute R script: {str(e)}") from e


def normalize_tuple(data: Optional[Union[Any, Iterable[Any]]]) -> Tuple[Any, ...]:
    """
    Normalize input data to a tuple.
    1. None -> (None,)
    2. Iterable -> tuple
    3. Other -> (data,)
    """
    if data is None:
        return (None,)
    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return tuple(data)
    return (data,)


def lookup_dict_with_tuple_key(
    d: Mapping[Union[Any, Tuple[Any, Any]], Any],
    key: Union[Any, Tuple[Any, Any]],
    default: Any = None,
    fallback_second_key: Optional[Union[Any, Iterable[Any]]] = None,
    retrieve_if_first_key_is_unique: bool = True,
    raise_on_missing: bool = False,
) -> Any:
    """
    Retrieve a value from a dictionary with both tuple and single keys,
    supporting fallback values for the second element of tuple keys.

    The dictionary may contain:
    - Single keys, such as `{"key1": "value1"}`.
    - Tuple keys, such as `{("key1_part1", "key1_part2"): "value1"}` or `{("k2_p1", "k2_p2", "k2_p3", ...): "value2"}`.

    Search Order:
    1. Try to find `key` directly in the dictionary, whether it's a tuple or single key.
    2. If `key` is a tuple and its second element is in `fallback_second_key`,
       retry with only the first element `key[0]` as the key.
    3. If no match is found, form new keys `(key[0] or key, fallback_second_key[i])`
       for each element in `fallback_second_key` and search iteratively.
    4. If `retrieve_if_first_key_is_unique` is True, search for all matching keys
       with the same first element and return the value if there is only one match.
    5. If no key matches, return `default` or raise a `KeyError`.

    Parameters
    ----------
    d : Mapping[Union[Any, Tuple[Any, Any]], Any]
        The dictionary to search.
    key : Union[Any, Tuple[Any, Any]]
        The lookup key. Can be a single value or a tuple.
    default : Any, optional
        The value to return if no match is found and `raise_on_missing` is False.
        Defaults to None.
    fallback_second_key : Union[Any, Iterable[Any]], optional
        The fallback second component(s) to use if the exact key is not found.
        If iterable, searches iteratively through the values.
        Anything defined here will be tried, which means the default None is also a valid value, and `(key[0], None)` or `(key, None)` will be tried.
        Defaults to None.
    retrieve_if_first_key_is_unique : bool, optional
        If True, searches for all keys sharing the same first element as `key[0]` or `key`
        and returns the value if there's exactly one match. Defaults to True.
    raise_on_missing : bool, optional
        If True, raises a `KeyError` when no value is found. Defaults to False.

    Returns
    -------
    Any
        The value associated with the key in the dictionary.

    Raises
    ------
    KeyError
        If no value is found and `raise_on_missing` is True.
    """

    if key in d:
        return d[key]

    fallback_second_key = normalize_tuple(fallback_second_key)

    if isinstance(key, tuple) and (key[1] in fallback_second_key):
        if key[0] in d:
            return d[key[0]]

    base_key = key[0] if isinstance(key, tuple) else key
    for second_key in fallback_second_key:
        new_key = (base_key, second_key)
        if new_key in d:
            return d[new_key]

    if retrieve_if_first_key_is_unique:
        matching_keys = [k for k in d.keys() if isinstance(k, tuple) and k[0] == base_key]
        if len(matching_keys) == 1:
            return d[matching_keys[0]]

    if raise_on_missing:
        raise KeyError(f"Key `{key}` not found. Tried fallback second keys: {fallback_second_key}.")
    return default


def lookup_dict_with_tuple_keys():
    """
    To iterate over the dictionary and find the value for a tuple key.

    First find the max number of elements of each tuple key in the dictionary.
    Then use fallback_key to iterately form new keys from "key[0]" to (key[0], ...) where ... until the max number of elements.

    - Tuple keys, such as `{("key1_part1", "key1_part2"): "value1"}` or `{("k2_p1", "k2_p2", "k2_p3", ...): "value2"}`.
    """
    pass


_T_InputOrAll = Union[None, str, Sequence[Union[None, str]], Literal["all"]]


def gather_value_or_all(
    value: _T_InputOrAll = None,
    all_values: Optional[Sequence[str]] = None,
    keep_none: bool = False,
) -> list[str]:
    """
    Receives an input and/or a sequence of values, and return a list of values.
    When the inpute `value` is "all", will return `all_values` and `all_values` can not be None.
    When the input `value` is None, will return None if `keep_none` is True, else will return empty list `[]` (default will be `[]`).
    Otherwise, will return the input value as a list.

    TODO: use "each" and "all"? "all" will has behavior as current None
    """
    if value is None:
        if keep_none:
            return None
        return []
    elif isinstance(value, str):
        if value.lower() == "all":
            if all_values is None:
                raise ValueError("Either value or all_values should be provided")
            return all_values
        else:
            return [value]
    else:
        return list(value)


def check_query_in_vec(
    query: Union[str, int, float, Iterable],
    vec: Union[tuple, list, np.ndarray],
) -> Union[bool, list[bool]]:
    if isinstance(query, (str, int, float)):
        return query in vec
    return [q in vec for q in query]


def filter_top_n_by_group(
    df: Optional[pl.DataFrame] = None,
    group_by: Union[str, Sequence[str]] = None,
    value_col: str = None,
    n: int = 1,
    use_min: bool = True,
    as_pl_expr: bool = False,
) -> Union[pl.Expr, pl.DataFrame]:
    """
    Create expression or filter DataFrame to keep top N rows within each group.

    Generates a filter expression that selects N rows with smallest values
    (if use_min=True) or largest values (if use_min=False) of value_col
    within each group defined by group_by.

    Parameters
    ----------
    df : Optional[pl.DataFrame], default=None
        Input DataFrame to filter. Required unless as_pl_expr=True.
    group_by : Union[str, Sequence[str]]
        Column(s) to group by
    value_col : str
        Column to rank values within each group
    n : int, default=1
        Number of rows to keep in each group
    use_min : bool, default=True
        If True, keep n smallest values; if False, keep n largest values
    as_pl_expr : bool, default=False
        If True, return a boolean expression; if False, return filtered DataFrame

    Returns
    -------
    Union[pl.Expr, pl.DataFrame]
        A boolean expression for filtering if as_pl_expr=True,
        otherwise the filtered DataFrame
    """
    if not as_pl_expr and df is None:
        raise ValueError("DataFrame must be provided when as_pl_expr=False")
    if group_by is None:
        raise ValueError("group_by parameter is required")
    if value_col is None:
        raise ValueError("value_col parameter is required")

    groups = [group_by] if isinstance(group_by, str) else list(group_by)
    descending = not use_min
    expr = pl.col(value_col).rank(descending=descending, method="min").le(n).over(groups)

    if as_pl_expr:
        return expr
    return df.filter(expr)


def add_bool_mark_by_expr(
    df: pl.DataFrame,
    expr: Union[pl.Expr, str, bool],
    mark_col: str,
) -> pl.DataFrame:
    """
    Will add a column `mark_col` to the input dataframe and return it.
    The value of the column will be True if the expression is True, otherwise False.

    Parameters
    ----------
    df : pl.DataFrame
        The input dataframe.
    expr : Union[pl.Expr, str, bool]
        The expression to evaluate.
        When `expr` is a string, it will be evaluated as a column name.
        When `expr` is a boolean, it will be used directly as the value.
    mark_col : str
        The name of the column to add.

    Returns
    -------
    pl.DataFrame
        The input dataframe with the new column.
    """
    if isinstance(expr, pl.Expr):
        return df.with_columns(pl.when(expr).then(pl.lit(True)).otherwise(pl.lit(False)).alias(mark_col))
    elif isinstance(expr, str):
        return df.with_columns(pl.when(pl.col(expr)).then(pl.lit(True)).otherwise(pl.lit(False)).alias(mark_col))
    elif isinstance(expr, bool):
        return df.with_columns(pl.lit(expr).alias(mark_col))
    else:
        raise ValueError(f"Invalid expression type: {type(expr)}")


def write_df_to_parquet_or_tsv(
    df: pl.DataFrame,
    path: Union[str, Path],
    **kwargs,
) -> Path:
    path = Path(path).resolve()
    if path.suffix.lower() == ".parquet":
        df.write_parquet(path, **kwargs)
    else:
        df.write_csv(path, separator="\t", **kwargs)
    return path


def read_df_from_parquet_or_tsv(
    path: Union[str, Path],
    **kwargs,
) -> pl.DataFrame:
    path = Path(path).resolve()
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path, **kwargs)
    else:
        return pl.read_csv(path, separator="\t", **kwargs)


def resume_file(
    path: Union[str, Path],
    resume: Union[bool, str, Path] = True,
    processed_filename_suffix: str = "-processed.parquet",
) -> tuple[Optional[pl.DataFrame], Path]:
    """
    Resume processed file (in parquet or tsv format) from disk if exists.
    Return a two-element tuple of the processed file or None, and the path of the processed file.
    When processed file has suffix as ".parquet", will load a parquet file, else consider the file in ".tsv" format.
    """
    path = Path(path).resolve()
    if isinstance(resume, (str, Path)):
        processed_file_path = Path(resume).resolve()
    else:
        processed_file_path = path.parent.joinpath(f"{path.name}{processed_filename_suffix}")

    if (resume is True) or isinstance(resume, (str, Path)):
        if processed_file_path.exists():
            logger.info(f"Load processed file from {processed_file_path}")
            return read_df_from_parquet_or_tsv(processed_file_path), processed_file_path
    return None, processed_file_path


@dataclass
class AbstractDFManiConfig:
    pass


@dataclass
class DFUniqueConfig(AbstractDFManiConfig):
    """
    Set `on` to `None` will pass this unique action, but not on all columns.
    To do unique on all columns, set `on` to `pl.all()`.
    """

    on: Union[str, Sequence[str]] = None


@dataclass
class DFFilterConfig(AbstractDFManiConfig):
    condition: Optional[pl.Expr] = None


@dataclass
class DFDropColConfig(AbstractDFManiConfig):
    cols: Union[str, Sequence[str]]


@dataclass
class DFConcatConfig(AbstractDFManiConfig):
    how: Literal["vertical", "horizontal"] = "vertical"


@dataclass
class DFColRenameConfig(AbstractDFManiConfig):
    rename_dict: Mapping[str, str]


@dataclass
class DFAddLitColConfig(AbstractDFManiConfig):
    col_name: str
    value: Any


def do_df_mani(
    df: pl.DataFrame,
    config: Optional[Union[AbstractDFManiConfig, Sequence[AbstractDFManiConfig]]] = None,
):
    if config is None:
        return df
    if isinstance(config, AbstractDFManiConfig):
        config = (config,)

    for conf in config:
        if isinstance(conf, DFUniqueConfig):
            if conf.on is not None:
                df = df.unique(conf.on)
        elif isinstance(conf, DFFilterConfig):
            if conf.condition is not None:
                df = df.filter(conf.condition)
        elif isinstance(conf, DFDropColConfig):
            df = df.drop(conf.cols)
        elif isinstance(conf, DFConcatConfig):
            df = pl.concat(df, how=conf.how)
        elif isinstance(conf, DFColRenameConfig):
            df = df.rename(conf.rename_dict)
        elif isinstance(conf, DFAddLitColConfig):
            df = df.with_columns(pl.lit(conf.value).alias(conf.col_name))
        else:
            raise ValueError(f"Unexpected DataFrame manipulation config: {conf}")
    return df
