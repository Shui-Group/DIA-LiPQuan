from dataclasses import dataclass

from typing import Literal


@dataclass
class AbstractDescConfig:
    """Base configuration class for description summary."""


@dataclass
class AbstractMissingValueHandler:
    """Base configuration class for missing value handlers."""


@dataclass
class AbstractPairwiseMissingValueHandler(AbstractMissingValueHandler):
    """Base configuration class for pairwise missing value handlers."""


@dataclass
class AbstractGroupMissingValueHandler(AbstractMissingValueHandler):
    """Base configuration class for group missing value handlers."""


@dataclass
class AbstractHypoTestConfig:
    """Base configuration class for hypothesis test."""


@dataclass
class AbstractTestAdjustConfig:
    """Base configuration class for test adjustment."""


_T_CompareScope = Literal["pairwise", "group", "all"]
