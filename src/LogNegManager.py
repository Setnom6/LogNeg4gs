
import os
import numpy as np
from .CompleteSimulation import CompleteSimulation
from .Measurements import Measurements, TypeOfMeasurement
from .PlotsManager import PlotsManager
from datetime import datetime

class LogNegManager:

    def __init__(self, generalOptions: dict, transformationDict: dict, initialStates):
        listOfSimulations = []
        numModes = generalOptions.get("numModes", 128)
        parallelize = generalOptions.get("parallelize", False)
        if isinstance(initialStates, dict):
            listOfSimulations.append(CompleteSimulation(numModes, transformationDict, initialStates))
            listOfSimulations[0].performTransformation()
        else:
            listOfSimulations.append(CompleteSimulation(numModes, transformationDict, initialStates[0]))
            listOfSimulations[0].performTransformation()
            transformationMatrix = listOfSimulations[0].transformationMatrix
            for i in range(1, len(initialStates)):
                listOfSimulations.append(CompleteSimulation(numModes, {}, initialStates[i], directParseOfTM=transformationMatrix))
                listOfSimulations[i].performTransformation()

        self.measurements = Measurements(parallelize=parallelize)
        self.listOfSimulations = listOfSimulations
        self.initialStatesOptions = initialStates
        self.dictTransformationMatrix = transformationDict
        self.dictGenerealOptions = generalOptions
        self.baseDirectory = generalOptions.get("base_directory", "./")

    def measureInitialStatesEntanglement(self, measurementType: str, modesToApply = None):
        results = {}
        extraData = {}
        for i, simulation in enumerate(self.listOfSimulations):
            if measurementType == TypeOfMeasurement.HighestOneByOne.value:
                results[i], extraData[i] = self.measurements.selectMeasurement(measurementType, simulation.inState, modesToApply)
            else:
                results[i] = self.measurements.selectMeasurement(measurementType, simulation.inState, modesToApply)

        dict_saved = self.saveData(results, measurementType, modesToApply, 0, extraData=extraData)
        return dict_saved

    def measureFinalStatesEntanglement(self, measurementType: str, modesToApply = None):
        results = {}
        extraData = {}
        for i, simulation in enumerate(self.listOfSimulations):
            if measurementType == TypeOfMeasurement.HighestOneByOne.value:
                results[i], extraData[i] = self.measurements.selectMeasurement(measurementType, simulation.outState,
                                                                               modesToApply)
            else:
                results[i] = self.measurements.selectMeasurement(measurementType, simulation.outState, modesToApply)

        dict_saved = self.saveData(results, measurementType, modesToApply, 1, extraData=extraData)
        return dict_saved

    def saveData(self, results: dict, measurementType: str, modesToApply=None, typeOfState = 1, extraData=None):
        dict_to_save = {
            "generalOptions": self.dictGenerealOptions,
            "initialStates": self.initialStatesOptions,
            "transformationMatrix": self.dictTransformationMatrix,
            "measurement": {
                    "type": measurementType,
                    "modesToApply": modesToApply,
                    "results": results,
                    "typeOfState": typeOfState # 0 for instate 1 for outstate
            }
        }

        if extraData is not None:
            dict_to_save["measurement"]["extraData"] = extraData

        data_dir = os.path.join(self.baseDirectory, self.dictGenerealOptions.get("data_directory"))
        os.makedirs(data_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"logneg_{measurementType}_{timestamp}.npz"
        save_path = os.path.join(data_dir, filename)

        np.savez_compressed(save_path, **dict_to_save)

        return dict_to_save

    def loadData(self, filename: str):
        file_path = os.path.join(self.baseDirectory, "data", filename)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' does not exist.")

        with np.load(file_path, allow_pickle=True) as data:
            dict_loaded = {
                    "numModes": data["numModes"].item(),
                    "initialStates": data["initialStates"].item(),
                    "transformationMatrix": data["transformationMatrix"].item(),
                    "measurement": data["measurement"].item()
            }

        return dict_loaded

    @staticmethod
    def plotResults(dict_of_results):
        pm = PlotsManager(dict_of_results)
        pm.plotResults()

    @staticmethod
    def ensure_directory_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)



