from __future__ import annotations

from enum import Enum
from typing import List, Dict, Any, TypedDict

import numpy as np


class TransformationMatrixParameters(str, Enum):
    DATA_DIRECTORY = "dataDirectory"
    INSTANT_TO_PLOT = "instantToPlot"


class InitialStateParameters(Enum):
    TEMPERATURE = "temperature"
    ONE_MODE_SQUEEZING = "oneModeSqueezing"
    TWO_MODE_SQUEEZING = "twoModeSqueezing"


class InitialStateDictType(TypedDict):
    temperature: float
    oneModeSqueezing: float
    twoModeSqueezing: float


class TransformationMatrixDictType(TypedDict):
    dataDirectory: str
    instantToPlot: int


class GeneralOptionsParameters(Enum):
    NUM_MODES = "numModes"
    PLOTS_DIRECTORY = "plotsDirectory"
    DATA_DIRECTORY = "dataDirectory"
    BASE_DIRECTORY = "baseDirectory"
    PARALLELIZE = "parallelize"


class GeneralOptionsDictType(TypedDict):
    numModes: int
    plotsDirectory: str
    dataDirectory: str
    baseDirectory: str
    parallelize: bool


class MeasurementParameters(Enum):
    TYPE = "type"
    MODES_TO_APPLY = "modesToApply"
    TYPE_OF_STATE = 'typeOfState'
    RESULTS = "results"
    EXTRA_DATA = "extraData"


class MeasurementDictType(TypedDict):
    type: str
    modesToApply: List[int]
    typeOfState: int
    results: Dict[int, np.ndarray]
    extraData: Dict[int, Any]


class TypeOfMeasurement(Enum):
    FullLogNeg = "fullLogNeg"
    HighestOneByOne = "highestOneByOne"
    OneByOneForAGivenMode = "oneByOneForAGivenMode"
    OddVSEven = "oddVSEven"
    SameParity = "sameParity"
    OccupationNumber = "occupationNumber"
