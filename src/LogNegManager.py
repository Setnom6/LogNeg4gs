from __future__ import annotations

import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, get_type_hints

import numpy as np
from scipy.linalg import sqrtm

import src.qgt as qgt
from src.PartnerMethods import extractMinimalHawkingPartner
from .CompleteSimulation import CompleteSimulation
from .Measurements import Measurements
from .PlotsManager import PlotsManager
from .TypesAndParameters import InitialStateDictType, TransformationMatrixDictType, \
    TransformationMatrixParameters, InitialStateParameters, TypeOfMeasurement, GeneralOptionsParameters, \
    GeneralOptionsDictType, MeasurementParameters, MeasurementDictType


class LogNegManager:
    """
    Class to manage a whole process of in-out process via a Bogoliubov transformation and a entanglement measurement on the states.
    """

    def __init__(self, generalOptions: Dict, transformationDict: Dict, initialStates: List[Dict]):
        """
        Builds an environment to manage different initial states, apply the desired Bogoliubov transformation on them, obtain the specified
        entanglement measure and plot the results.

        Parameters
        """
        listOfSimulations = []

        generalOptions, transformationDict, initialStates = self._validateEssentialDicts(generalOptions,
                                                                                         transformationDict,
                                                                                         initialStates)
        numModes = generalOptions[GeneralOptionsParameters.NUM_MODES.value]
        if isinstance(initialStates, dict):
            listOfSimulations.append(CompleteSimulation(numModes, initialStates, transformationDict=transformationDict))
            listOfSimulations[0].performTransformation()
        else:
            listOfSimulations.append(
                CompleteSimulation(numModes, initialStates[0], transformationDict=transformationDict))
            listOfSimulations[0].performTransformation()
            transformationMatrix = listOfSimulations[0].transformationMatrix
            for i in range(1, len(initialStates)):
                listOfSimulations.append(
                    CompleteSimulation(numModes, initialStates[i], directParseOfTM=transformationMatrix))
                listOfSimulations[i].performTransformation()

        self.measurements = Measurements(parallelize=generalOptions[GeneralOptionsParameters.PARALLELIZE.value])

        self.listOfSimulations = listOfSimulations
        self.initialStatesOptions = initialStates
        self.dictTransformationMatrix = transformationDict
        self.dictGeneralOptions = generalOptions

    def measureEntanglement(self, measurementDict: Dict) -> Dict:
        measurementDict = self._validateMeasurementDict(measurementDict)
        measurementType = measurementDict[MeasurementParameters.TYPE.value]
        modesToApply = measurementDict[MeasurementParameters.MODES_TO_APPLY.value]
        typeOfState = measurementDict[MeasurementParameters.TYPE_OF_STATE.value]

        for i, simulation in enumerate(self.listOfSimulations):
            stateToMeasure = simulation.inState if typeOfState == 0 else simulation.outState
            if measurementType == TypeOfMeasurement.HighestOneByOne.value:
                measurementDict[MeasurementParameters.RESULTS.value][i], \
                    measurementDict[MeasurementParameters.EXTRA_DATA.value][i] = self.measurements.selectMeasurement(
                    measurementType, stateToMeasure, modesToApply)

            elif measurementType == TypeOfMeasurement.HawkingPartner.value:
                if typeOfState == 0:
                    warnings.warn("Hawking-Partner measurement only implemented for out states.")
                    stateToMeasure = simulation.outState
                if abs(self.initialStatesOptions[i][InitialStateParameters.TEMPERATURE.value]) > 1e-3:
                    warnings.warn(
                        "Hawking-Partner formula only works for initial pure states. Computing usual Full log neg...")
                    measurementDict[MeasurementParameters.RESULTS.value][i] = self.measurements.selectMeasurement(
                        TypeOfMeasurement.FullLogNeg.value, stateToMeasure, modesToApply)

                else:
                    measurementDict[MeasurementParameters.RESULTS.value][i] = self.measurements.hawkingPartner(
                        self.callHawkingPartnerCreator, simulation.inState, simulation.transformationMatrix,
                        modesToApply)
            else:
                measurementDict[MeasurementParameters.RESULTS.value][i] = self.measurements.selectMeasurement(
                    measurementType, stateToMeasure, modesToApply)

        dict_saved = self.saveData(measurementDict)
        return dict_saved

    @staticmethod
    def callHawkingPartnerCreator(stateToApply: qgt.Gaussian_state, transformationMatrix: np.ndarray,
                                  modeA: int) -> np.ndarray:
        """
        Callable function which prepares the full Bogoliubov transformation on the state given.
        It assumes that 'stateToApply' is the initial state with some feature (squeezing) but before the main transformation.
        The main transformation is given directly as 'transformationMatrix'.
        """
        V = stateToApply.V
        SInitial = sqrtm(V)
        bogoliubovTransformation = transformationMatrix @ SInitial
        newBogoliubovTransformation = extractMinimalHawkingPartner(bogoliubovTransformation, modeA)
        return newBogoliubovTransformation

    def saveData(self, measurementDict: MeasurementDictType) -> Dict:
        dict_to_save = {
            "generalOptions": self.dictGeneralOptions.copy(),
            "initialStates": self.initialStatesOptions.copy(),
            "transformationMatrix": self.dictTransformationMatrix.copy(),
            "measurement": measurementDict.copy()
        }

        baseDirectory = self.dictGeneralOptions[GeneralOptionsParameters.BASE_DIRECTORY.value]
        dataDirectory = self.dictGeneralOptions[GeneralOptionsParameters.DATA_DIRECTORY.value]
        data_dir = os.path.join(baseDirectory, dataDirectory)
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        measurementType = measurementDict[MeasurementParameters.TYPE.value]
        filename = f"logneg_{measurementType}_{timestamp}.npz"
        save_path = os.path.join(data_dir, filename)

        np.savez_compressed(save_path, **dict_to_save)

        return dict_to_save

    def loadData(self, filename: str) -> Dict:
        baseDirectory = self.dictGeneralOptions[GeneralOptionsParameters.BASE_DIRECTORY.value]
        dataDirectory = self.dictGeneralOptions[GeneralOptionsParameters.DATA_DIRECTORY.value]
        file_path = os.path.join(baseDirectory, dataDirectory, filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        with np.load(file_path, allow_pickle=True) as data:
            dict_loaded = {
                "generalOptions": data["generalOptions"].item(),
                "initialStates": data["initialStates"],
                "transformationMatrix": data["transformationMatrix"].item(),
                "measurement": data["measurement"].item()
            }

        return dict_loaded

    @staticmethod
    def plotResults(dict_of_results: Dict):
        pm = PlotsManager(dict_of_results)
        pm.plotResults()

    @staticmethod
    def ensure_directory_exists(directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _validateEssentialDicts(
            self,
            generalOptions: Dict,
            transformationDict: Dict,
            initialStates: List[Dict]
    ) -> Tuple[GeneralOptionsDictType, TransformationMatrixDictType, List[InitialStateDictType]]:

        # Verify unknown keys
        valid_general_keys = set(get_type_hints(GeneralOptionsDictType).keys())
        valid_transformation_keys = set(get_type_hints(TransformationMatrixDictType).keys())
        valid_initial_state_keys = set(get_type_hints(InitialStateDictType).keys())

        extra_general_keys = set(generalOptions.keys()) - valid_general_keys
        if extra_general_keys:
            warnings.warn(
                f"Non recognized keys in generalOptions: {extra_general_keys}. "
                "These keys will be ignored.",
                UserWarning
            )

        extra_transformation_keys = set(transformationDict.keys()) - valid_transformation_keys
        if extra_transformation_keys:
            warnings.warn(
                f"Non recognized keys in transformationDict: {extra_transformation_keys}. "
                "These keys will be ignored.",
                UserWarning
            )

        for i, initialStateDict in enumerate(initialStates):
            extra_initial_keys = set(initialStateDict.keys()) - valid_initial_state_keys
            if extra_initial_keys:
                warnings.warn(
                    f"Non recognized keys in initialStates[{i}]: {extra_initial_keys}. "
                    "These keys will be ignored.",
                    UserWarning
                )

        # Create valid dicts
        copyGeneralOptions = {
            GeneralOptionsParameters.NUM_MODES.value: generalOptions.get(GeneralOptionsParameters.NUM_MODES.value, 128),
            GeneralOptionsParameters.PLOTS_DIRECTORY.value: generalOptions.get(
                GeneralOptionsParameters.PLOTS_DIRECTORY.value, "./plots/128-plots/"),
            GeneralOptionsParameters.DATA_DIRECTORY.value: generalOptions.get(
                GeneralOptionsParameters.DATA_DIRECTORY.value, "./data/128-plots/"),
            GeneralOptionsParameters.BASE_DIRECTORY.value: generalOptions.get(
                GeneralOptionsParameters.BASE_DIRECTORY.value, "./"
            ),
            GeneralOptionsParameters.PARALLELIZE.value: generalOptions.get(GeneralOptionsParameters.PARALLELIZE.value,
                                                                           False)
        }

        copyTransformationDict = {
            TransformationMatrixParameters.DATA_DIRECTORY.value: transformationDict.get(
                TransformationMatrixParameters.DATA_DIRECTORY.value, "./sims-128/"),
            TransformationMatrixParameters.INSTANT_TO_PLOT.value: transformationDict.get(
                TransformationMatrixParameters.INSTANT_TO_PLOT.value, -1)
        }

        copyInitialStates = []
        for initialStateDict in initialStates:
            copyInitialStates.append({
                InitialStateParameters.TEMPERATURE.value: initialStateDict.get(InitialStateParameters.TEMPERATURE.value,
                                                                               0.0),
                InitialStateParameters.ONE_MODE_SQUEEZING.value: initialStateDict.get(
                    InitialStateParameters.ONE_MODE_SQUEEZING.value, 0.0),
                InitialStateParameters.TWO_MODE_SQUEEZING.value: initialStateDict.get(
                    InitialStateParameters.TWO_MODE_SQUEEZING.value, 0.0)
            })

        return copyGeneralOptions, copyTransformationDict, copyInitialStates

    def _validateMeasurementDict(self, measurementDict: Dict) -> MeasurementDictType:
        valid_keys = set(get_type_hints(MeasurementDictType).keys())
        extra_keys = set(measurementDict.keys()) - valid_keys
        if extra_keys:
            warnings.warn(
                f"on recognized keys in measurementDict: {extra_keys}. "
                "These keys will be ignored.",
                UserWarning
            )

        # Convert modesToApply to 0 index notation
        modesToApply = measurementDict.get(MeasurementParameters.MODES_TO_APPLY.value, None)
        if modesToApply is not None:
            modesToApply = [i - 1 for i in modesToApply]

        return {
            MeasurementParameters.TYPE.value: measurementDict.get(MeasurementParameters.TYPE.value,
                                                                  TypeOfMeasurement.FullLogNeg.value),
            MeasurementParameters.MODES_TO_APPLY.value: modesToApply,
            MeasurementParameters.TYPE_OF_STATE.value: measurementDict.get(MeasurementParameters.TYPE_OF_STATE.value,
                                                                           1),
            MeasurementParameters.RESULTS.value: dict(),
            MeasurementParameters.EXTRA_DATA.value: dict()
        }
