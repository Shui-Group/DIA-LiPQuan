import functools
import re
from typing import Iterable, Literal, Union

import numpy as np
from numba import njit

from .utils import flatten_list

__all__ = [
    "TED",
]

Enzymes = {
    "Trypsin": {
        "CleavageSite": [("K", 1, "P"), ("R", 1, "P")],
        "RE": "[KR](?!P)",
        "RE_AllEnd": ".*?[KR](?!P)|.+",
    },
    "Trypsin/P": {
        "CleavageSite": [("K", 1, None), ("R", 1, None)],
        "RE": "[KR]",
        "RE_AllEnd": ".*?[KR]|.+",
    },
    "LysC": {
        "CleavageSite": [("K", 1, "P")],
        "RE": "K(?!P)",
        "RE_AllEnd": ".*?K(?!P)|.+",
    },
    "LysC/P": {
        "CleavageSite": [("K", 1, "P")],
        "RE": "K",
        "RE_AllEnd": ".*?K|.+",
    },
    "LysN": {
        "CleavageSite": [("K", -1, None)],
    },
    "AspC": {
        "CleavageSite": [("D", 1, None)],
    },
    "AspN": {
        "CleavageSite": [("D", -1, None)],
    },
    "Chymotrypsin": {
        "CleavageSite": [("F", 1, None), ("W", 1, None), ("Y", 1, None)],
    },
}


class TED(object):
    def __init__(
        self,
        restricted_enzyme: Union[str, Iterable[str]] = "Trypsin/P",
        enzymatic_specificity: Literal["fully", "semi"] = "fully",
        min_pep_len: int = 7,
        max_pep_len: int = 33,
        restricted_enzyme_mc: Union[int, Iterable[int]] = (0, 1, 2),
        prot_nterm_m_rule: Literal["keep", "alt", "cut"] = "alt",
        prot_termini_role: Literal["restricted", "non_restricted", "drop"] = "restricted",
        return_position: bool = True,
        extend_n: int = 0,
    ):
        """
        Theoretical Enzyme Digestion -> TED

        Parameters
        ----------
        restricted_enzyme : Union[str, Iterable[str]], optional
            Enzyme name or a regular expression to define the digestion rules. Currently supported enzymes are Trypsin, Trypsin/P, lysC, and LysC/P, by default "Trypsin/P"
        enzymatic_specificity: Literal["fully", "semi"], optional
            enzymatic specificity, by default "fully"
            note: set to "semi" will only generate digested peptides under semi-specific rule, with NO fully-specific peptides
        min_pep_len : int, optional
            _description_, by default 7
        max_pep_len : int, optional
            _description_, by default 33
        restricted_enzyme_mc : Union[int, Iterable[int]], optional
            This can be int, which results in a range of 1...input, or a tuple/list of expected miss cleavages, by default (0, 1, 2)
        prot_nterm_m_rule : Literal[&quot;keep&quot;, &quot;alt&quot;, &quot;cut&quot;], optional
            If "M" on N-terminal of input sequence:
            - do nothing ("keep")
            - remove this M ("cut")
            - consider both cases for keep and cut ("alt"), by default "alt"
        prot_termini_role : Literal[&quot;restricted&quot;, &quot;non_restricted&quot;, &quot;drop&quot;], optional
            The role of the protein sequence termini, by default "restricted"
            This defines how to handle the protein sequence termini:
            - restricted: The protein termini will be considered as the restricted digestion sites, and in this case any peptides contain the protein termini and have another cut sites that are non-restricted will be regarded as semi-specific
            - non_restricted: The protein termini will be considered as the non-restricted digestion sites, and in this case any peptides contain the protein termini and have another cut sites that are restricted are semi-specific, while the peptides contain the protein termini and have another cut sites that are non-restricted are non-specific
            - drop: The peptides fall into other digestion rules but contain the protein termini will be dropped
            Should be caution because this argument will work together with `prot_nterm_m_rule`, which can make the 2nd position of the protein sequence as a termini when the 1st AA is M.
            Example: when `prot_nterm_m_rule` is "alt" and `prot_termini_rule` is "drop", any peptide starts with the 1st M or 2nd any AA will be dropped
        return_position : bool, optional
            Returns sequences or tuples of (sequence, 1-based index of peptide position), by default True.
            - If True: A list of tuples of seq and site will be returned. [('ADEFHK', 2), ('PQEDAK', 12), ...]
            - If False: A list of seq will be returned. ['ADEFHK', 'PQEDAK' , ...]
        extend_n : int, optional
            Returns n AAs before and after the seq, by default 0
            Set this to 1 to get previous AA and next AA
            - If 0: disable this behavior
            - If >= 1: the n AAs before and after the seq will also be returned together with associated digested peptides.

        TODO exclude some unusual aa if param assigned
        """

        self.restricted_enzyme = restricted_enzyme
        self.enzymatic_specificity = enzymatic_specificity

        self.min_pep_len = min_pep_len
        self.max_pep_len = max_pep_len
        self.restricted_enzyme_mc = restricted_enzyme_mc

        self._cleave_nterm_m, self._optional_nterm_m = None, None
        self.prot_nterm_m_rule = prot_nterm_m_rule

        self.prot_termini_rule = prot_termini_role
        self.return_position = return_position
        self.extend_n = extend_n

    def _parse_enzyme(self, enzyme):
        if isinstance(enzyme, str):
            enzyme = (enzyme,)
        elif isinstance(enzyme, (list, tuple, set)):
            pass
        else:
            raise ValueError(f"The input enzyme can not be parsed: {enzyme}")
        for each_enzyme in enzyme:
            if isinstance(each_enzyme, str):
                enzyme_property = Enzymes.get(each_enzyme, None)
                if enzyme_property is None:
                    print(
                        f"The input enzyme is not in the pre-defined enzyme collection.\n"
                        f"This input {each_enzyme} will be used as a regular expression."
                    )
                    self._enzyme_names.append("")
                    self._enzyme_rules.append(each_enzyme)
                else:
                    self._enzyme_names.append(each_enzyme)
                    self._enzyme_rules.append(enzyme_property["RE"])
            else:
                raise ValueError(f"The input enzyme {each_enzyme} is not a string")

    @property
    def restricted_enzyme(self):
        """
        Restricted enzyme used for digestion
        """
        return self._enzyme_names, self._enzyme_rules

    @restricted_enzyme.setter
    def restricted_enzyme(self, enzyme):
        self._enzyme_names, self._enzyme_rules = [], []
        self._parse_enzyme(enzyme)

    @property
    def restricted_digestion_rules(self):
        return self._enzyme_rules

    @property
    def restricted_enzyme_mc(self) -> tuple[int, ...]:
        """
        Allowed missed cleavage of restriction enzyme
        """
        return self._mc

    @restricted_enzyme_mc.setter
    def restricted_enzyme_mc(self, mc: Union[int, tuple[int], list[int]]):
        if isinstance(mc, (tuple, list)):
            if all(isinstance(v, int) for v in mc):
                self._mc = tuple(mc)
            else:
                raise TypeError("Miss cleavage should be an int or a tuple/list of int")
        elif isinstance(mc, int):
            if mc < 0:
                raise ValueError("When mc is an int, it should be a non-negative integer.")
            self._mc = tuple(range(0, mc + 1))
        else:
            try:
                self._mc = (int(mc),)
            except (TypeError, ValueError) as e:
                raise TypeError("Miss cleavage should be an int or a tuple/list of int") from e

    @staticmethod
    def _parse_prot_nterm_m_rule(prot_nterm_m_rule):
        match prot_nterm_m_rule:
            case 0 | "keep":
                cleavage_nterm_m = False
                need_optional_nterm_m = False
            case 1 | "alt":
                cleavage_nterm_m = True
                need_optional_nterm_m = True
            case 2 | "cut":
                cleavage_nterm_m = True
                need_optional_nterm_m = False
            case _:
                raise ValueError("The input of `prot_nterm_m_rule` should be in [0, 1, 2, 'keep', 'alt', 'cut']")
        return cleavage_nterm_m, need_optional_nterm_m

    @property
    def prot_nterm_m_rule(self):
        return self._prot_nterm_m_rule

    @prot_nterm_m_rule.setter
    def prot_nterm_m_rule(self, prot_nterm_m_rule: Literal["keep", "alt", "cut"] = "alt"):
        self._prot_nterm_m_rule = prot_nterm_m_rule
        self._cleave_nterm_m, self._optional_nterm_m = self._parse_prot_nterm_m_rule(prot_nterm_m_rule)

    @property
    def return_position(self):
        return self._return_position

    @return_position.setter
    def return_position(
        self,
        return_position: bool = True,
    ):
        if not isinstance(return_position, bool):
            raise TypeError(f"The input of return_type should be bool, now: {return_position}")
        self._return_position = return_position

    @property
    def extend_n(self):
        return self._extend_n

    @extend_n.setter
    def extend_n(self, extend_n: int):
        if isinstance(extend_n, int):
            if extend_n < 0:
                raise ValueError("The input of extend_n should be a non-negative integer")
            else:
                pass
        else:
            raise TypeError(f"The input of extend_n should be an integer, now: {extend_n} with a type {type(extend_n)}")
        self._extend_n = extend_n

    def __str__(self):
        msg = f"""\
Theoretical Enzyme Digestion
===========================
Restricted enzyme: {self._enzyme_names}
Restricted enzyme rule: {self._enzyme_rules}
Enzymatic type: {self.enzymatic_specificity}
Peptide length range: {self.min_pep_len} - {self.max_pep_len}
Missed cleavage of restricted digestion: {self.restricted_enzyme_mc}
Protein N-terminal M rule: {self.prot_nterm_m_rule}
Protein termini rule: {self.prot_termini_rule}
Return position: {self.return_position}
Extend n: {self.extend_n}"""
        return msg

    def _generate_pedestal_sites(self, seq, in_seq_len):
        # Find cleavage sites of the restricted enzyme as pedestals
        pedestal_sites = np.array(
            flatten_list([[_.end() for _ in re.finditer(rule, seq)] for rule in self.restricted_digestion_rules]),
            dtype=np.int32,
        )
        pos1_mc_offset = False
        match self.prot_termini_rule:
            case "restricted":
                if seq[0] == "M":
                    if self.prot_nterm_m_rule == "alt":
                        pedestal_sites = np.hstack((0, 1, pedestal_sites, in_seq_len))
                        pos1_mc_offset = True
                    elif self.prot_nterm_m_rule == "keep":
                        pedestal_sites = np.hstack((0, pedestal_sites, in_seq_len))
                    elif self.prot_nterm_m_rule == "cut":
                        pedestal_sites = np.hstack((1, pedestal_sites, in_seq_len))
                else:
                    pedestal_sites = np.hstack((0, pedestal_sites, in_seq_len))
            case "non_restricted" | "drop":
                # should first remove the last position if it is included as a pedestal site
                pedestal_sites = pedestal_sites[pedestal_sites != in_seq_len]
                # For following, remove cut site index 1 between AA1 and AA2, because it might fall into the restricted enzyme rule
                if seq[0] == "M":
                    if self.prot_nterm_m_rule == "alt":
                        pedestal_sites = pedestal_sites[pedestal_sites != 1]
                    elif self.prot_nterm_m_rule == "keep":
                        pass
                    elif self.prot_nterm_m_rule == "cut":
                        pedestal_sites = pedestal_sites[pedestal_sites != 1]
                else:
                    pass
        pedestal_sites = np.unique(pedestal_sites)
        return pedestal_sites, pos1_mc_offset

    def _extract_peptides_from_compositions(self, seq, compositions, add_info=None):
        results = []
        for site_1, site_2 in compositions:
            one_seq = seq[site_1:site_2]
            others = []
            if self.return_position:
                others.append(site_1 + 1)
            if self.extend_n:
                if site_1 < self.extend_n:
                    prev_seq = "_" * (self.extend_n - site_1) + seq[:site_1]
                else:
                    prev_seq = seq[site_1 - self.extend_n : site_1]
                back_seq = seq[site_2 : site_2 + self.extend_n]
                if len(back_seq) < self.extend_n:
                    back_seq = back_seq + "_" * (self.extend_n - len(back_seq))
                others.extend([prev_seq, back_seq])
            if add_info is not None:
                others.append(add_info)
            if others:
                results.append((one_seq, *others))
            else:
                results.append(one_seq)
        return results

    def _do_fully_restricted_digestion(self, seq, add_info=None):
        in_seq_len = len(seq)

        pedestal_sites, pos1_mc_offset = self._generate_pedestal_sites(seq, in_seq_len)

        dist_mat = np.triu(pedestal_sites.reshape(1, -1) - pedestal_sites.reshape(-1, 1), k=1)
        mc_restricted_dist_mat = dist_mat * functools.reduce(
            np.logical_or,
            (False, *((np.triu(np.tril(dist_mat, k=(mc + 1)), k=(mc + 1)) > 0) for mc in self.restricted_enzyme_mc)),
        )
        idxs = np.where(
            np.logical_and(mc_restricted_dist_mat >= self.min_pep_len, mc_restricted_dist_mat <= self.max_pep_len)
        )
        compositions = pedestal_sites[np.vstack(idxs).T]

        # this will only happen when input sequence has M on N-terminal and `prot_nterm_m_rule` is "alt" and `prot_termini_rule` is "restricted"
        if pos1_mc_offset:
            compositions = compositions[compositions[:, 0] != 0]

            mc_restrains = np.zeros_like(dist_mat[0], dtype=np.bool_)
            mc_restrains[
                list(filter(lambda _idx: _idx < mc_restrains.size, (mc + 2 for mc in self.restricted_enzyme_mc)))
            ] = True
            mc_restricted_dists = dist_mat[0] * mc_restrains
            idxs = np.where(
                np.logical_and(mc_restricted_dists >= self.min_pep_len, mc_restricted_dists <= self.max_pep_len)
            )[0]
            if idxs.size > 0:
                if idxs[0].size != 0:
                    compositions = np.vstack((np.vstack((np.zeros_like(idxs), dist_mat[0][idxs])).T, compositions))

        return self._extract_peptides_from_compositions(seq, compositions, add_info=add_info)

    def _do_semi_restricted_digestion(self, seq, add_info=None):
        in_seq_len = len(seq)

        pedestal_sites, pos1_mc_offset = self._generate_pedestal_sites(seq, in_seq_len)

        scanning_sites = np.arange(in_seq_len + 1)
        scanning_sites = scanning_sites[~np.isin(scanning_sites, pedestal_sites)]

        match self.prot_termini_rule:
            case "restricted":
                if seq[0] == "M":
                    if self.prot_nterm_m_rule == "alt":
                        # when M is kept, the 1st position is still be regarded as a pedestal site, so add 1 back. the problem of same position of pedestal and scanning sites will be solved by the peptide length == 0
                        # this might be wrong if restricted enzyme rule has cut after M
                        scanning_sites = np.hstack((1, scanning_sites))
                    elif self.prot_nterm_m_rule == "keep":
                        pass  # termini has been excluded by pedestal_sites
                    elif self.prot_nterm_m_rule == "cut":
                        scanning_sites = scanning_sites[scanning_sites != 0]  # also exclude already cut M
                else:
                    pass  # termini has been excluded by pedestal_sites
            case "non_restricted":
                if seq[0] == "M":
                    if self.prot_nterm_m_rule == "alt":
                        pass
                    elif self.prot_nterm_m_rule == "keep":
                        pass
                    elif self.prot_nterm_m_rule == "cut":
                        scanning_sites = scanning_sites[scanning_sites != 0]
                else:
                    pass
            case "drop":
                scanning_sites = scanning_sites[scanning_sites != in_seq_len]
                if seq[0] == "M":
                    if self.prot_nterm_m_rule == "alt":
                        scanning_sites = scanning_sites[scanning_sites != 1]
                    elif self.prot_nterm_m_rule == "keep":
                        scanning_sites = scanning_sites[scanning_sites != 0]
                    elif self.prot_nterm_m_rule == "cut":
                        scanning_sites = scanning_sites[(scanning_sites != 0) & (scanning_sites != 1)]
                else:
                    scanning_sites = scanning_sites[scanning_sites != 0]

        dist_mat = pedestal_sites.reshape(1, -1) - scanning_sites.reshape(-1, 1)
        direction_mat = np.sign(dist_mat)
        center_idxs = np.apply_along_axis(np.searchsorted, axis=1, arr=direction_mat, v=0).reshape(-1, 1)
        mcs = np.array(self.restricted_enzyme_mc)
        available_pedestal_idxs = np.hstack((center_idxs + mcs, center_idxs - 1 - mcs))

        composition_idxs = np.vstack(
            (
                np.repeat(np.arange(scanning_sites.size), mcs.size * 2),
                available_pedestal_idxs.ravel(),
            )
        ).T
        composition_idxs = composition_idxs[
            np.logical_and(composition_idxs[:, 1] >= 0, composition_idxs[:, 1] < pedestal_sites.size)
        ]

        # this will only happen when input sequence has M on N-terminal and `prot_nterm_m_rule` is "alt" and `prot_termini_rule` is "restricted"
        if pos1_mc_offset:
            # because any pedestal site on position 1 can be regarded as a restricted cut site, so copy those pairs and set the pedestal site to 0
            pos1_pairs = composition_idxs[composition_idxs[:, 1] == 1]
            pos1_pairs[:, 1] = 0
            composition_idxs = np.vstack((pos1_pairs, composition_idxs[composition_idxs[:, 1] != 0]))

        compositions = np.sort(
            np.vstack(
                (scanning_sites[composition_idxs[:, 0]].ravel(), pedestal_sites[composition_idxs[:, 1]].ravel())
            ).T,
            axis=1,
        )
        _diff = compositions[:, 1] - compositions[:, 0]
        compositions = compositions[
            np.logical_and(
                _diff >= self.min_pep_len,
                _diff <= self.max_pep_len,
            )
        ]
        compositions = compositions[np.lexsort((compositions[:, 1], compositions[:, 0]))]

        return self._extract_peptides_from_compositions(seq, compositions, add_info=add_info)

    def _do_semi_restricted_digestion_by_iter_seq(self, seq, add_info=None):
        in_seq_len = len(seq)

        pedestal_sites, _ = self._generate_pedestal_sites(seq, in_seq_len)
        compositions = []
        for pedestal_site_idx, pedestal_site in enumerate(pedestal_sites):
            for site in range(max(0, pedestal_site - self.max_pep_len), max(0, pedestal_site - self.min_pep_len + 1)):
                if site == pedestal_site:
                    continue
                mc = pedestal_site_idx - np.searchsorted(pedestal_sites, site)
                if (seq[0] == "M") and (self.prot_nterm_m_rule == "alt") and (site == 1):
                    mc -= 1
                if mc not in self.restricted_enzyme_mc:
                    continue
                if site == 0:
                    continue
                if seq[0] == "M":
                    if (self.prot_nterm_m_rule == "cut") and (site == 1):
                        continue
                if site in pedestal_sites:
                    if (seq[0] == "M") and (self.prot_nterm_m_rule == "alt") and (site == 1):
                        pass
                    else:
                        continue
                compositions.append((site, pedestal_site))
            for site in range(
                min(in_seq_len, pedestal_site + self.min_pep_len), min(in_seq_len, pedestal_site + self.max_pep_len + 1)
            ):
                if site == pedestal_site:
                    continue
                mc = max(0, (np.searchsorted(pedestal_sites, site) - pedestal_site_idx - 1))
                if (seq[0] == "M") and (pedestal_site == 0) and (self.prot_nterm_m_rule == "alt"):
                    mc -= 1
                if mc not in self.restricted_enzyme_mc:
                    continue
                if site == in_seq_len:
                    continue
                if site in pedestal_sites:
                    continue
                compositions.append((pedestal_site, site))

        return self._extract_peptides_from_compositions(seq, compositions, add_info=add_info)

    def digest(self, seq, add_info=None):
        """
        All sites refered in this function point to the cleavage sites, with 0-based index. That is, a n-length protein sequence can have n+1 sites from 0 to n (includes n- and c-term).

        Returns
        -------
        list or list of tuples
        1. The basic return is a list of digested peptides
        2. If `return_position` is True, the return will be a list of tuples of seq and site, e.g. [('ADEFHK', 2), ('PQEDAK', 12), ...]
        3. If `extend_n` is not False, the return will be a list of tuples of seq, prev_seq, and back_seq, e.g. n=5 results in [('ADEFHK', '____M', 'QASAC'), ('PQEDAK', 'PQEDA', 'K____'), ...]
        4. If `add_info` is not None, the return will be a list of tuples of seq and add_info, e.g. [('ADEFHK', 'info1'), ('PQEDAK', 'info2'), ...]

        The combination of above additional fields has an order of (seq, site, prev_seq, back_seq, add_info)
        """
        seq = seq.replace("\n", "").replace(" ", "")

        match self.enzymatic_specificity:
            case "fully":
                return self._do_fully_restricted_digestion(seq, add_info=add_info)
            case "semi":
                return self._do_semi_restricted_digestion_by_iter_seq(seq, add_info=add_info)
            case _:
                raise ValueError("The enzymatic specificity should be 'fully' or 'semi'")

    def __call__(self, seq, add_info=None):
        return self.digest(seq, add_info=add_info)


@njit(cache=True)
def _nb_any_axis1(x):
    """
    Numba compatible version of np.any(x, axis=1)
    """
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@njit
def _calc_statisfied_mat(
    all_cleavage_sites: np.ndarray,
    restricted_sites: np.ndarray,
    min_pep_len: int,
    max_pep_len: int,
    max_restricted_enzyme_mc: int,
    prot_termini_as_non_restricted: bool,
    toggle_nterm_m: int,
    seq_first_char: str,
):
    dist_mat = all_cleavage_sites.reshape(-1, 1) - restricted_sites.reshape(1, -1)
    abs_dist_mat = np.abs(dist_mat)
    statisfied_mat = np.logical_and((abs_dist_mat >= min_pep_len), (abs_dist_mat <= max_pep_len))
    statisfied_mat[restricted_sites, :] = False

    if prot_termini_as_non_restricted:
        if (toggle_nterm_m == 2) and (seq_first_char == "M"):
            statisfied_mat[0, :] = False
    else:
        statisfied_mat[0, :] = False
        statisfied_mat[-1, :] = False
        if (toggle_nterm_m == 2) and (seq_first_char == "M"):
            statisfied_mat[1, :] = False

    restrict_sites_positions = _nb_any_axis1(dist_mat == 0)
    cumsum_restrict_sites = np.cumsum(restrict_sites_positions)

    restrict_enzyme_mc_mat = np.abs(
        cumsum_restrict_sites.reshape(-1, 1) - cumsum_restrict_sites[restricted_sites].reshape(1, -1)
    )
    statisfied_mat = np.logical_and(
        statisfied_mat,
        (
            ((dist_mat < 0) & (restrict_enzyme_mc_mat <= (max_restricted_enzyme_mc + 1)))
            | ((dist_mat > 0) & (restrict_enzyme_mc_mat <= max_restricted_enzyme_mc))
        ),
    )
    return statisfied_mat


def get_semi_enzymatic_digestion(
    seq: str | tuple[str, str],
    restricted_enzyme_rule: str = "[KR]",
    min_pep_len: int = 7,
    max_pep_len: int = 30,
    max_restricted_enzyme_mc: int = 1,
    prot_termini_as_non_restricted: bool = False,
    toggle_nterm_m: int = 1,
    add_info: None | tuple | list = None,
) -> tuple:
    """
    Perform semi-enzymatic digestion on a given sequence.

    An experimental version for generating semi-enzymatic peptides. Should use TED for common.

    Parameters
    ----------
    seq : str | tuple[str]
        "sequence" or ("protein_name", "sequence").
        The input sequence to be digested. It can be a string or a tuple of two strings where the first string represents the protein name and the second string represents the sequence.
    restrict_enzyme_rule : str, optional
        The regular expression rule defining the restriction enzyme used for digestion, by default '[KR]'.
    min_pep_len : int, optional
        The minimum length of the resulting peptides, by default 7.
    max_pep_len : int, optional
        The maximum length of the resulting peptides, by default 30.
    max_restrict_enzyme_mc : int, optional
        The maximum number of occurrences of the restriction enzyme within a peptide, by default 1.
    allow_prot_termini : bool, optional
        Whether to regard protein termini as non-restricted enzymatic sites. If True, a termini plus a restricted site will be considered as a semi-enzymatic site, by default False.
    toggle_nterm_m: int, optional
        The option for handling N-terminal methionine (M) residues.
        - 0: No exclusion of N-terminal M residues.
        - 1: Include results from both excluding and keeping N-terminal M residues.
        - 2: Exclude N-terminal M residues.
        The default value is 1.
        When this option is set to 1, `allow_prot_termini` will have no effect because N-terminal M is considered as the termini of the protein but it can also be excluded.
    add_info : None | tuple | list, optional
        Additional information to be included in the resulting peptides. It can be None, a tuple, or a list, by default None.

    Returns
    -------
    tuple
        A tuple with 4 elements in default, or 5 elements when additional information is given (when `seq` is a tuple with protein name given, and when `add_info` is not None).
        - The first element is the peptide sequence.
        - The second element is the starting position of the peptide (1-indexed).
        - The third element is the residue before the peptide, "-" if at terminal.
        - The fourth element is the residue after the peptide, "-" if at terminal.
        - If input `seq` is a tuple as (prot, seq), the fifth element will be the protein name.
        - Any inputs from `add_info` will be appended to the tuple.
    """
    if isinstance(seq, tuple):
        prot, seq = seq
        if add_info is None:
            add_info = (prot,)
        else:
            add_info = (prot, *add_info)

    all_cleavage_sites = np.arange(len(seq) + 1)
    restrict_sites = np.asarray([_.end() for _ in re.finditer(restricted_enzyme_rule, seq)])
    if len(restrict_sites) == 0:
        return []

    statisfied_mat = _calc_statisfied_mat(
        all_cleavage_sites,
        restrict_sites,
        min_pep_len,
        max_pep_len,
        max_restricted_enzyme_mc,
        prot_termini_as_non_restricted,
        toggle_nterm_m,
        seq[0],
    )

    peps = []
    for each_other_site, each_rest_site_idx in np.stack(np.where(statisfied_mat)).T:
        each_rest_site = restrict_sites[each_rest_site_idx]
        left_site, right_site = (
            (each_other_site, each_rest_site)
            if (each_other_site < each_rest_site)
            else (each_rest_site, each_other_site)
        )
        pep = seq[left_site:right_site]

        n_m1 = "_" if (left_site == 0) else seq[left_site - 1]
        c_p1 = "_" if (right_site == len(seq)) else seq[right_site]

        if add_info is None:
            peps.append((pep, left_site + 1, n_m1, c_p1))
        else:
            peps.append((pep, left_site + 1, n_m1, c_p1, *add_info))

    return peps


def test_get_semi_tryptic_digestion():
    seq = "MFRRLTFARESEEKKS"
    expected_results = set(
        [
            "FRR",
            "RLT",
            "RLTF",
            "RLTFA",
            "LTF",
            "LTFA",
            "LTFARE",
            "LTFARES",
            "TFAR",
            "FAR",
            "ARESEEK",
            "RESEEK",
            "ESE",
            "ESEE",
            "SEEK",
            "SEEKK",
            "EEK",
            "EEKK",
            "EKK",
        ]
    )
    real_results = set(
        _[0]
        for _ in get_semi_enzymatic_digestion(
            seq,
            min_pep_len=3,
            max_pep_len=7,
            max_restricted_enzyme_mc=1,
            prot_termini_as_non_restricted=False,
            toggle_nterm_m=1,
        )
    )
    if real_results != expected_results:
        print(f"{(set(expected_results) - real_results)=}")
        print(f"{(real_results - set(expected_results))=}")
        raise ValueError("Test failed")

    expected_results.update({"MFR", "MFRR"})
    real_results = set(
        _[0]
        for _ in get_semi_enzymatic_digestion(
            seq,
            min_pep_len=3,
            max_pep_len=7,
            max_restricted_enzyme_mc=1,
            prot_termini_as_non_restricted=True,
            toggle_nterm_m=1,
        )
    )
    if real_results != expected_results:
        print(f"{(set(expected_results) - real_results)=}")
        print(f"{(real_results - set(expected_results))=}")
        raise ValueError("Test failed")


def test_ted_and_semi_func():
    seq = "MFRARLTFARESEEKKSK"
    expected_results = set(
        [
            "MFRA",
            "FRA",
            "ARL",
            "ARLT",
            "ARLTF",
            "ARLTFA",
            "FRAR",
            "RAR",
            "LTF",
            "LTFA",
            "LTFARE",
            "LTFARES",
            "RLTFAR",
            "TFAR",
            "FAR",
            "ESE",
            "ESEE",
            "ARESEEK",
            "RESEEK",
            "SEEK",
            "EEK",
            "SEEKK",
            "EEKK",
            "EKK",
        ]
    )
    ted = TED(
        restricted_enzyme="Trypsin/P",
        enzymatic_specificity="semi",
        min_pep_len=3,
        max_pep_len=7,
        restricted_enzyme_mc=1,
        prot_nterm_m_rule="alt",
        # prot_termini_role="drop",
        return_position=True,
        extend_n=1,
    )
    result = set([_[0] for _ in ted(seq)])
    if result != expected_results:
        print(f"{(set(expected_results) - result)=}")
        print(f"{(result - set(expected_results))=}")
        raise ValueError("Test failed")

    semi_func_result = set(
        _[0]
        for _ in get_semi_enzymatic_digestion(
            seq,
            min_pep_len=3,
            max_pep_len=7,
            max_restricted_enzyme_mc=1,
            prot_termini_as_non_restricted=False,
            toggle_nterm_m=1,
        )
    )
    expected_results.remove("MFRA")
    expected_results.remove("FRA")
    if semi_func_result != expected_results:
        print(f"{(set(expected_results) - semi_func_result)=}")
        print(f"{(semi_func_result - set(expected_results))=}")
        raise ValueError("Test failed")


if __name__ == "__main__":
    test_get_semi_tryptic_digestion()
    test_ted_and_semi_func()
