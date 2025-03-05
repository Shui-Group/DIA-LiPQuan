import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl

__all__ = [
    "flatten_nested_list",
    "exec_r_script",
    "normalize_tuple",
    "lookup_dict_with_tuple_key",
    "_T_InputOrAll",
    "gather_value_or_all",
    "check_query_in_vec",
    "write_df_to_parquet_or_tsv",
    "read_df_from_parquet_or_tsv",
    "resume_file",
    "AbstractDFManiConfig",
    "DFUniqueConfig",
    "DFFilterConfig",
    "do_df_mani",
]

logger = logging.getLogger("lipana")


def flatten_nested_list(nested_list: list[list]) -> list:
    temp = []
    for _ in nested_list:
        temp.extend(_)
    return temp


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
class DFConcatConfig(AbstractDFManiConfig):
    how: Literal["vertical", "horizontal"] = "vertical"


@dataclass
class DFColRenameConfig(AbstractDFManiConfig):
    rename_dict: Mapping[str, str]


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
        elif isinstance(conf, DFColRenameConfig):
            df = df.rename(conf.rename_dict)
        elif isinstance(conf, DFConcatConfig):
            df = pl.concat(df, how=conf.how)
        else:
            raise ValueError(f"Unexpected DataFrame manipulation config: {conf}")
    return df
