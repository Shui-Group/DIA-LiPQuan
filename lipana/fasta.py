import itertools
import logging
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

__all__ = [
    "ParsedFasta",
    "parse_fasta",
]

logger = logging.getLogger("lipana")


@dataclass
class ParsedFasta:
    """
    A parsed FASTA object containing:
    - prot_acc_to_seq: Map of protein accession to protein sequence
    - prot_acc_to_species: Map of protein accession to species
    - species_to_concat_seqs: Map of species to concatenated protein sequences

    Additional attribute:
    - peptide_to_species_cache: Cache of peptide to species mapping

    Generally, this object should be initialized by `parse_fasta` function,
    and can be loaded from the dumped file by class method `load`.
    """

    prot_acc_to_seq: dict[str, str]
    prot_acc_to_species: dict[str, str]
    species_to_concat_seqs: dict[str, str]

    peptide_to_species_cache: dict[str, str] = None

    _path: Optional[Union[str, Path]] = None
    _field_names = ("prot_acc_to_seq", "prot_acc_to_species", "species_to_concat_seqs", "peptide_to_species_cache")

    def __post_init__(self):
        if self.peptide_to_species_cache is None:
            self.peptide_to_species_cache = {}

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = Path(path).resolve().absolute()
        logger.info(f"Loading parsed FASTA from {str(path)}")
        with open(path, "rb") as h:
            return cls(**pickle.load(h), _path=path)

    def dump(self, path: Optional[Union[str, Path]] = None):
        if path is not None:
            self.path = path
        if self.path is None:
            raise ValueError("Set `path` before dumping the parsed FASTA data")
        logger.info(f"Dumping parsed FASTA to {str(self.path)}")
        with open(self.path, "wb") as h:
            pickle.dump({k: getattr(self, k) for k in self._field_names}, h)

    @property
    def path(self):
        if self._path is None:
            raise ValueError("ParsedFasta object is not loaded from a file, or has not been dumped to a file")
        return self._path

    @path.setter
    def path(self, path: Union[str, Path]):
        self._path = Path(path).resolve().absolute()

    def __getitem__(self, key):
        return getattr(self, key)


def read_fasta(
    fasta_file: Union[str, Path],
    sep: Optional[str] = "|",
    ident_idx: int = 1,
    ident_process_func: Optional[Callable] = None,
    open_mode: str = "r",
    encoding: str = "utf-8",
    skip_row: Optional[int] = None,
    ignore_blank: bool = False,
) -> dict:
    """
    Use `sep=None` to skip parsing title
    """
    fasta_dict = dict()
    seq_list = []
    with open(fasta_file, open_mode) as f:
        if isinstance(skip_row, int):
            [f.readline() for _ in range(skip_row)]
        for row in f:
            if open_mode == "rb":
                row = row.decode(encoding)
            if row.startswith(">"):
                if seq_list:
                    fasta_dict[acc] = "".join(seq_list)

                if ident_process_func is not None:
                    acc = ident_process_func(row)
                else:
                    if sep is None:
                        acc = row.strip("\n")
                    else:
                        acc = row.strip("\n").split(sep)[ident_idx]

                seq_list = []
            elif (not row) or (row == "\n"):
                if ignore_blank:
                    continue
                else:
                    raise ValueError(
                        "Blank line in target FASTA file. "
                        "Check completeness of FASTA file, or use `ignore_blank=True` to ignore this error."
                    )
            else:
                seq_list.append(row.strip("\n"))
        if seq_list:
            fasta_dict[acc] = "".join(seq_list)
    return fasta_dict


def _parse_uniprot_fasta_title(title: str) -> dict[str, str]:
    """
    A simple approach to parse UniProt title into a dictionary containing:
    - protein: Protein ID
    - species: Species name
    """
    acc, entry_name = title.split(" ")[0].split("|")[1:]
    species = entry_name.split("_")[1]
    return {"protein": acc, "species": species}


def _parse_fasta_title_regex(
    title: str,
    regex: Union[str, Sequence[str]] = r">[^|\s]+?\|(?P<protein>[^|\s]+?)\|[^\s]+?_(?P<species>[^\s]+)[$\s].*",
) -> dict[str, str]:
    """
    Parse a FASTA title with one or more regexes, and return a dictionary containing:
    - protein: Protein ID
    - species: Species name

    In the case with multiple regexes, they will be tried in order, and stop until two keys are found.
    If iteration is finished but species is not found, return the first result that contains "protein" key.
    If iteration is finished but "protein" is not found, raise an error.
    """
    if isinstance(regex, str):
        regex = [regex]
    records = []
    for r in regex:
        r = re.match(r, title)
        if r:
            if "protein" in r.groupdict() and "species" in r.groupdict():
                return r.groupdict()
            else:
                records.append(r.groupdict())
    if len(records) == 0:
        raise ValueError(f"Can not find protein and species in title: {title}, by regex: {regex}")
    for r in records:
        if "protein" in r:
            return r
    raise ValueError(f"Can not find protein in title: {title}, by regex: {regex}")


def parse_fasta(
    fasta_path: Union[str, Path, Sequence[Union[str, Path]]] = None,
    contam_fasta_path: Union[str, Path, Sequence[Union[str, Path]]] = None,
    contaminations: Optional[Union[str, Sequence[str]]] = None,
    fasta_title_regex: Optional[str] = None,
    gen_species_to_concat_seqs: bool = True,
    workspace: Optional[Union[str, Path]] = None,
    resume: Union[bool, str, Path] = True,
    write_parsed_fasta: bool = True,
) -> ParsedFasta:
    """
    Parse one or more FASTA files into a ParsedFasta object, which can be regarded as a dictionary containing:
    - prot_acc_to_seq: Map of protein accession to protein sequence
    - prot_acc_to_species: Map of protein accession to species
    - species_to_concat_seqs: Map of species to concatenated protein sequences

    Currently only UniProt format is supported, and species is extracted from the title
    (e.g. will be "BOVIN" if ">sp|P02662|CASA1_BOVIN ....." is given)

    Parameters
    ----------
    fasta_path : Union[str, Path, Sequence[Union[str, Path]]], optional
        Paths of target fasta files, by default None.
        Set to None only when 1. resume is a string or Path and the file exists; 2. resume is True and "parsed_fasta.pkl" exists in workspace.
    contam_fasta_path : Union[str, Path, Sequence[Union[str, Path]]], optional
        Paths of contamination fasta files, by default None
        These files will be loaded before the target files, and all proteins in these files will be marked as contaminations
    contaminations : Optional[Union[str, Sequence[str]]], optional
        Directly define which entry should be contaminations, by default None
        This can be a single string or a sequence of strings, and each string can be a protein accession or species name
    fasta_title_regex : Optional[Union[str, Sequence[str]]], optional
        A regex or a sequence of regexes to parse the title of the fasta file, by default None
        If not given or None, the title will be parsed as UniProt format in a simple way
        If given, the value should be a regex or a sequence of regexes, and the title will be parsed by the regexes in order
        Example:
        - If the title is ">sp|P02662|CASA1_BOVIN .....", the default method will extract "P02662" as protein and "BOVIN" as species
        - If the title is ">sp|P02662|CASA1_BOVIN .....", and fasta_title_regex is r">[^|\s]+?\|(?P<protein>[^|\s]+?)\|[^\s]+?_(?P<species>[^\s]+)[$\s].*", the protein will be "P02662" and the species will be "BOVIN"
        - If there are titles with mixed formats, like ">sp|P02662|CASA1_BOVIN ....." and ">Q32MB2 TREMBL:Q32MB2 Tax_Id=9606 Gene_Symbol=KRT73", set regex to [r">[^|\s]+?\|(?P<protein>[^|\s]+?)\|[^\s]+?_(?P<species>[^\s]+)[$\s].*", r">(?P<protein>[^\ ]+).+?Tax_Id=(?P<species>[^\ ]+).*?"] can handle this case
        The protein identifier must be found in all titles, otherwise an error will be raised
        When the protein can be found but species can not be found, the species will be set to "unknown"
    gen_species_to_concat_seqs : bool, optional
        Whether to generate the species to concatenated protein sequences, by default True
    workspace : Optional[Union[str, Path]], optional
        Where to dump or resume the existing file, by default None
        This only works when resume is True or write_parsed_fasta is True
        If not given, will be set to the parent folder of the first fasta file
    resume : Union[bool, str, Path], optional
        Resume existing file, by default True
        - if resume itself is a string or Path, will try to directly load the file from the given path, and in this case all other arguments will be ignored
        - if resume is True, and workspace is given, will try to load the file from workspace with the default file name "parsed_fasta.pkl"
        - if resume is True, but workspace is None or the file is not found, when write_parsed_fasta is True, will parse the FASTA file(s) and dump the result in the workspace with the default file name "parsed_fasta.pkl"
    write_parsed_fasta : bool, optional
        Whether to write parsed fasta to disk, by default True

    Returns
    -------
    ParsedFasta
    """

    if isinstance(resume, (str, Path)):
        dump_path = Path(resume).resolve().absolute()
        if dump_path.exists():
            return ParsedFasta.load(dump_path)
    elif resume is True and workspace is not None:
        dump_path = Path(workspace).joinpath("parsed_fasta.pkl")
        if dump_path.exists():
            return ParsedFasta.load(dump_path)
    else:
        dump_path = None

    if fasta_path is None:
        raise ValueError("At least one FASTA file must be given")

    def _check_path(path: Optional[Union[str, Path]]):
        if path is None:
            return []
        if isinstance(path, (str, Path)):
            path = [Path(path).resolve().absolute()]
        elif isinstance(path, Sequence):
            path = [Path(p).resolve().absolute() for p in path]
        else:
            raise ValueError("`fasta_path` must be a string, Path, or a sequence of strings or Paths")
        for p in path:
            if not p.exists():
                raise FileNotFoundError(f"FASTA file not found: {str(p)}")
        return path

    contam_fasta_path = _check_path(contam_fasta_path)
    fasta_path = [p for p in _check_path(fasta_path) if p not in contam_fasta_path]

    f0 = fasta_path[0]
    workspace = Path(workspace) if workspace is not None else Path(f0).parent
    if dump_path is None:
        dump_path = workspace.joinpath("parsed_fasta.pkl")

    if contaminations is None:
        contaminations = set()
    elif isinstance(contaminations, str):
        contaminations = {
            contaminations,
        }
    else:
        contaminations = set(contaminations)

    prot_acc_to_seq = {}
    prot_acc_to_species = {}
    if gen_species_to_concat_seqs:
        species_to_concat_seqs = defaultdict(str)
    else:
        species_to_concat_seqs = None

    is_whole_contam = [True] * len(contam_fasta_path) + [False] * len(fasta_path)
    n_files = len(is_whole_contam)

    for file_idx, path in enumerate(itertools.chain(contam_fasta_path, fasta_path)):
        logger.info(f"Loading FASTA file {file_idx + 1}/{n_files}: {path.name}")
        fasta_data = read_fasta(path, sep=None, ignore_blank=True)
        for title, seq in fasta_data.items():
            if fasta_title_regex is None:
                title_re = _parse_uniprot_fasta_title(title)
            else:
                title_re = _parse_fasta_title_regex(title, fasta_title_regex)
            acc = title_re["protein"]
            species = title_re.get("species", "unknown")
            if acc in prot_acc_to_seq:
                continue

            if is_whole_contam[file_idx] or (acc in contaminations) or (species in contaminations):
                species = "Contam"
                contaminations.add(acc)

            prot_acc_to_seq[acc] = seq
            prot_acc_to_species[acc] = species
            if gen_species_to_concat_seqs:
                species_to_concat_seqs[species] += f"{seq}-"

    parsed_fasta = ParsedFasta(
        prot_acc_to_seq=prot_acc_to_seq,
        prot_acc_to_species=prot_acc_to_species,
        species_to_concat_seqs=species_to_concat_seqs,
    )
    parsed_fasta.path = dump_path

    if write_parsed_fasta:
        parsed_fasta.dump(dump_path)

    return parsed_fasta
