from typing import Iterable, Literal


def generate_input_for_diann_lib_pred():
    pass


def generate_input_for_peptdeep_pred():
    pass


def generate_input_for_library_pred(
    entry_list,
    entry_level: Literal["peptide", "precursor"],
    input_format: Literal["diann", "peptdeep"],
):
    pass


class LibraryPeptides:
    def __init__(
        self,
        in_entries: Iterable[str] | Iterable[tuple[str]],
    ):
        pass
