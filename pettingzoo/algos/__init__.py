# This file is used to import all the algorithms in the playground
from .iql import IQL
from .siql import SIQL
from .psiql import PSIQL
from .spsiql import SPSIQL
from .ippo import IPPO
from .sippo import SIPPO
from .mappo import MAPPO
from .smappo import SMAPPO
from .acsppo import ACSPPO
from .sacsppo import SACSPPO
from .base import BaseMARLAlgo

from .safety_calculator import SafetyCalculator

classes = [
    "IQL",
    "SIQL",
    "PSIQL",
    "SPSIQL",
    "IPPO",
    "SIPPO",
    "MAPPO",
    "SMAPPO",
    "ACSPPO",
    "SACSPPO",
    "BaseMARLAlgo",
    "SafetyCalculator"
]

SHIELDED_ALGOS = {
    "SIQL": SIQL, 
    "SPSIQL": SPSIQL, 
    "SIPPO": SIPPO, 
    "SMAPPO": SMAPPO, 
    "SACSPPO": SACSPPO,
}

UNSHIELDED_ALGOS = {
    "IQL": IQL, 
    "PSIQL": PSIQL, 
    "IPPO": IPPO, 
    "MAPPO": MAPPO, 
    "ACSPPO": ACSPPO,
    "PSPPO": ACSPPO,
}

ALL_ALGORITHMS = {**UNSHIELDED_ALGOS, **SHIELDED_ALGOS}