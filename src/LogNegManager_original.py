import os
import re
from datetime import datetime
from enum import Enum
from itertools import combinations
from math import comb
from typing import Dict, Any, List, Tuple, Optional

import matplotlib as mpl
import numpy as np
import pylab as pl
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm

import qgt
from partnerMethods import extractHawkingPartner, get_symplectic_from_covariance, obtainHPTwoModeTransformation



class TypeOfData(Enum):
    FullLogNeg = "fullLogNeg"
    FullLogNegBefore = "fullLogNegBefore"
    HighestOneByOne = "highestOneByOne"
    OneByOneForAGivenMode = "oneByOneForAGivenMode"
    OneVSTwoForAGivenMode = "oneVSTwoForAGivenMode"
    OddVSEven = "oddVSEven"
    SameParity = "sameParity"
    OccupationNumber = "occupationNumber"
    LogNegDifference = "logNegDifference"
    JustSomeModes = "justSomeModes"
    HawkingPartner = "hawkingPartner"


class LogNegManager:
    inState: Dict[int, qgt.Gaussian_state]
    outState: Dict[int, qgt.Gaussian_state]
    MODES: int
    kArray: np.ndarray
    instantToPlot: int
    arrayParameters: np.ndarray
    transformationMatrix: np.ndarray
    plottingInfo: Dict[str, Any]

    def __init__(self, dataDirectory: str, initialStateType: InitialState, MODES: int, instantToPlot: int,
                 arrayParameters: np.ndarray = None, temperature: Optional[float] = None,
                 squeezingIntensity: Optional[float] = None, parallelize: bool = False):
        """
        Constructor for the LogNegManager class

        Parameters:
        dataDirectory: str
            Directory where the data of the transformation matrix is stored (alphas and betas)
        initialStateType: InitialState
            Type of initial state to be used
        MODES: int
            Number of modes of the system
        instantToPlot: int
            Instant of the transformation matrix to be used (usually the last one)
        arrayParameters: np.ndarray
            Array of parameters to be used in the initial state, each parameter will mean a different initial state. It depends on the initialStateType
            - Vacuum: None
            - Thermal: Array of temperatures
            - OneModeSqueezed: Array of one mode squeezing intensities
            - TwoModeSqueezed: Array of two mode squeezing intensities (applied pairwise)
            - OneModeSqueezedFixedTemp: Array of one mode squeezing intensities in a thermal bath with fixed T
            - TwoModeSqueezedFixedTemp: Array of two mode squeezing intensities in a thermal bath with fixed T
            - ThermalFixedOneModeSqueezing: Array of temperatures for initial states being one mode squeezing with a fixed squeezing intensity
        """
        self._temperature = None
        self._squeezing = None
        if temperature is not None:
            self.setTemperature(temperature)
        if squeezingIntensity is not None:
            self.setSqueezing(squeezingIntensity)
        self.plottingInfo = dict()
        self.MODES = MODES
        self.instantToPlot = instantToPlot
        if initialStateType is not InitialState.Vacuum:
            self.arrayParameters = arrayParameters
        else:
            self.arrayParameters = None
        self.transformationMatrix = self._constructTransformationMatrix(dataDirectory)
        self.inState = self._createInState(initialStateType)
        self.outState = dict()
        self.parallelize = parallelize


    def computeOneVSTwoForAGivenMode(self, mode: int, inState: bool = False) -> Dict[int, np.ndarray]:
        """
        Computes the logarithmic negativity between a given mode and all unordered pairs of other modes.

        Parameters:
        mode: int
            Mode to be used as partA.
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState.

        Returns:
        Dict[int, np.ndarray]
            Dictionary where each key corresponds to a state index.
            Each value is an array with shape (n_pairs,), containing the logarithmic negativities between `mode` and each pair (i,j) with i < j ≠ mode.
        """

        mode -= 1  # Ajustar a índice 0
        stateToApply = self.inState if inState else self.outState

        # Todas las combinaciones (i, j) con i < j ≠ mode
        allPairs = [(i, j) for i, j in combinations(range(self.MODES), 2)
                    if mode not in (i, j)]

        nPairs = len(allPairs)
        lognegDict = {i: np.zeros(nPairs) for i in range(1, self.plottingInfo["NumberOfStates"] + 1)}

        def task(stateIndex, pairIndex, i, j):
            return lambda: (
                stateIndex,
                pairIndex,
                stateToApply[stateIndex].logarithmic_negativity([mode], [i, j])
            )

        tasks = [task(stateIndex, pairIndex, i, j)
                 for stateIndex in range(1, self.plottingInfo["NumberOfStates"] + 1)
                 for pairIndex, (i, j) in enumerate(allPairs)]

        results = self._execute(tasks)

        for stateIndex, pairIndex, value in results:
            lognegDict[stateIndex][pairIndex] = value

        return lognegDict



    def computeHawkingPartnerLogNeg(self, inState: bool = False, specialModes: List[int] = None) -> Dict[
        int, np.ndarray]:
        """
        Computes the full logarithmic negativity between the hawking mode and its partner (each HP basis is independent from the others).
        That is, for each mode, computes the logarithmic negativity taking that Hawking mode as partA and the partner as partB, tracing out the other modes.

        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        specialModes: list of int
            Particular modes to consider

        Returns:
        dict[int, np.ndarray]
            Dictionary with the Hawking Partner logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the HP logarithmic negativity for each mode.
            (state i, HP log neg of mode j -> fullLogNeg[i][j]) Only works for 1 state i
        """

        if self.plottingInfo["NumberOfStates"] != 1:
            raise ValueError("Hawking-partner calculation only works for 1 simulation at once")

        if self.plottingInfo["InStateName"] in [InitialState.Thermal.value, InitialState.OneModeSqueezedFixedTemp.value,
                                                InitialState.ThermalFixedOneModeSqueezing.value,
                                                InitialState.TwoModeSqueezedFixedTemp.value]:
            raise AttributeError("Hawking-partner calculation only works for pure states")

        results = self.obtainHPLogNegForListOfModes(specialModes, inState)

        HPLogNeg = {i + 1: np.zeros(len(specialModes)) for i in range(self.plottingInfo["NumberOfStates"])}

        for mode, logNeg in results:
            HPLogNeg[1][specialModes.index(mode)] = logNeg

        return HPLogNeg

    def getFigureName(self, plotsRelativeDirectory: str, typeOfData: TypeOfData, date: str = "",
                      beforeTransformation: bool = False) -> str:
        """
        Assuming the plotsRelativeDirectory is defined from the location of the Jupyter Notebook, the standarized figure name is returned

        Parameters:
        plotsRelativeDirectory: str
            Relative directory where the plots are stored
        typeOfData: TypeOfData
            Type of data to be plotted
        date: str
            Date to be added to the figure name
        beforeTransformation: bool
            If True, the figure name will have "Before" before the typeOfData in the name
        
        Returns:
        str
            Figure name with the standarized format
        """
        before = "Before" if beforeTransformation else ""
        figureName = "./{}{}_{}{}_instant_{}_numOfPlots_{}_date_{}.pdf".format(plotsRelativeDirectory, typeOfData.value,
                                                                               self.plottingInfo["InStateName"], before,
                                                                               self.instantToPlot,
                                                                               self.plottingInfo["NumberOfStates"],
                                                                               date)
        return figureName

    def getFileName(self, dataRelativeDirectory: str, typeOfData: TypeOfData, date: str = "",
                    beforeTransformation: bool = False) -> List[str]:
        """
        Assuming the plotsRelativeDirectory is defined from the location of the Jupyter Notebook, 
        the standarized files names where the plot data is stored is returned

        Parameters:
        dataRelativeDirectory: str
            Relative directory where the data is stored
        typeOfData: TypeOfData
            Type of data to be plotted
        date: str
            Date to be added to the figure name
        beforeTransformation: bool
            If True, the figure name will have "Before" before the typeOfData in the name
        
        Returns:
        List[str]
            List of files names with the standarized format. Each element of the list corresponds to a different state (number equals number of arrayParameters)
        """
        before = "Before" if beforeTransformation else ""
        filesNames = []
        for index in range(1, self.plottingInfo["NumberOfStates"] + 1):
            fileName = "./{}{}_{}{}_instant_{}_numOfPlots_{}_date_{}".format(dataRelativeDirectory, typeOfData.value,
                                                                             self.plottingInfo["InStateName"], before,
                                                                             self.instantToPlot,
                                                                             self.plottingInfo["NumberOfStates"], date)
            filesNames.append(fileName)
        return filesNames

    def saveData(self, dataRelativeDirectory: str, data: Dict[int, np.ndarray], typeOfData: TypeOfData, date: str = "",
                 beforeTransformation: bool = False) -> None:
        """
        Method to save the data in the files with the standarized format.
        At the moment it only saves the data if is of type: FullLogNeg, HighestOneByOne, OddVSEven, SameParity, OccupationNumber, Difference

        Parameters:
        dataRelativeDirectory: str
            Relative directory where the data is stored
        data: Dict[int, np.ndarray]
            Dictionary with the data to be saved
        typeOfData: TypeOfData
            Type of data to be saved
        date: str
            Date to be added to the figure name
        beforeTransformation: bool
            If True, the file name will have "Before" before the typeOfData in the name, as it has data from the initial state
        
        Returns:
            None
        """
        filesNamesToSave = self.getFileName(dataRelativeDirectory, typeOfData, date, beforeTransformation)

        for index, fileName in enumerate(filesNamesToSave):

            if typeOfData == TypeOfData.OneByOneForAGivenMode or typeOfData == TypeOfData.OneVSTwoForAGivenMode:
                raise NotImplementedError("For {} data one have to save manually".format(typeOfData.value))

            if typeOfData == TypeOfData.HighestOneByOne:
                dataToSave = np.zeros((2, self.MODES))
                dataToSave[0, :] = data[index + 1][0]
                dataToSave[1, :] = data[index + 1][1]
            else:
                if self.arrayParameters is not None:
                    dataToSave = np.zeros(self.MODES + 1)
                    dataToSave[0] = self.arrayParameters[index] if self.arrayParameters is not None else 0
                    dataToSave[1:] = data[index + 1]
                else:
                    dataToSave = data[index + 1]

            np.savetxt("{}_plotNumber_{}.txt".format(fileName, index + 1), dataToSave)

    def loadData(self, dataRelativeDirectory: str, typeOfData: TypeOfData, beforeTransformation: bool = False) -> Dict[
        int, np.ndarray]:
        """
        Method to load the data from the files with the standarized format.
        At the moment it only loads the data if is of type: FullLogNeg, HighestOneByOne, OddVSEven,SameParity, OccupationNumber, Difference.

        WARNING: At the moment it only loads the data for the last instant of the transformation matrix. It may happen that
        last time a simulation was run with all the same definitions (initialState, instant, number of states, etc) but with different parameters.
        In that case this method will fail and the data will have to be loaded manually.


        Parameters:
        dataRelativeDirectory: str
            Relative directory where the data is stored
        typeOfData: TypeOfData
            Type of data to be loaded
        beforeTransformation: bool
            If True, the data loaded will be from the initial state, as it has data from the initial state

        Returns:
        Dict[int, np.ndarray]
            Dictionary with the data loaded
        """
        data = dict()

        filesNames = self.getFileName(dataRelativeDirectory, typeOfData, date="",
                                      beforeTransformation=beforeTransformation)

        allFiles = [f"./{dataRelativeDirectory}{file}" for file in os.listdir(f"./{dataRelativeDirectory}")]

        matchingFiles = [file for file in allFiles if any(re.search(pattern, file) for pattern in filesNames)]
        matchingFiles.sort(reverse=True)

        numberOfStates = self.plottingInfo["NumberOfStates"]
        selectedFiles = matchingFiles[:numberOfStates]
        selectedFiles.sort()

        for index, fileName in enumerate(selectedFiles):
            dataFile = np.loadtxt(fileName)
            if typeOfData == TypeOfData.OneByOneForAGivenMode or typeOfData == TypeOfData.OneVSTwoForAGivenMode:
                return dict()

            elif typeOfData == TypeOfData.HighestOneByOne:
                data[index + 1] = np.zeros((2, self.MODES))
                data[index + 1][0] = dataFile[0, :]
                data[index + 1][1] = dataFile[1, :]
            else:
                if self.arrayParameters is not None:
                    arrayParameter = dataFile[0]
                    if arrayParameter != self.arrayParameters[index]:
                        print(
                            "WARNING: Array parameter not found in the array parameters used to generate the data for {}".format(
                                typeOfData))
                        return dict()
                    data[index + 1] = dataFile[1:]
                else:
                    data[index + 1] = dataFile

        return data

    def checkIfDataExists(self, dataRelativeDirectory: str, typeOfData: TypeOfData,
                          beforeTransformation: bool = False) -> bool:
        """
        Checks if the data for the given parameters exists in the directory.
        At the moment is not very efficient as it loads all the data and then checks if it is empty.
        """

        dataLoaded = self.loadData(dataRelativeDirectory, typeOfData, beforeTransformation=beforeTransformation)
        return len(dataLoaded) > 0

    def performComputations(self, listOfWantedComputations: List[TypeOfData], plotsDataDirectory: str,
                            tryToLoad: bool = True, specialModes: List[int] = []):
        results = {
            "logNegArray": None,
            "logNegArrayBefore": None,
            "highestOneToOneValue": None,
            "highestOneToOnePartner": None,
            "occupationNumber": None,
            "logNegEvenVsOdd": None,
            "logNegSameParity": None,
            "oneToOneGivenModes": None,
            "oneVSTwoForAGivenModes": None,
            "logNegDifference": None,
            "justSomeModes": None,
            "hawkingPartner": None,
        }

        if (self.plottingInfo["InStateName"] == InitialState.OneModeSqueezedFixedTemp.value
                or self.plottingInfo["InStateName"] == InitialState.ThermalFixedOneModeSqueezing.value
                or self.plottingInfo["InStateName"] == InitialState.TwoModeSqueezedFixedTemp.value):
            tryToLoad = False

        for computation in listOfWantedComputations:
            loadData = tryToLoad and self.checkIfDataExists(plotsDataDirectory, computation)

            if loadData:
                print("Loading data for: ", computation.value)

                if computation == TypeOfData.FullLogNeg:
                    results["logNegArray"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.HighestOneByOne:
                    oneToOneData = self.loadData(plotsDataDirectory, computation)
                    results["highestOneToOneValue"] = oneToOneData[1][0]
                    results["highestOneToOnePartner"] = oneToOneData[1][1]

                elif computation == TypeOfData.OccupationNumber:
                    results["occupationNumber"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.OddVSEven:
                    results["logNegEvenVsOdd"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.SameParity:
                    results["logNegSameParity"] = self.loadData(plotsDataDirectory, computation)

                elif computation == TypeOfData.LogNegDifference:
                    results["logNegDifference"] = self.loadData(plotsDataDirectory, computation)
                    results["logNegArray"] = self.loadData(plotsDataDirectory, TypeOfData.FullLogNeg)
                    results["logNegArrayBefore"] = self.loadData(plotsDataDirectory, TypeOfData.FullLogNegBefore)

            else:
                date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                print("Computing data for: ", computation.value)

                if computation == TypeOfData.FullLogNeg:
                    results["logNegArray"] = self.computeFullLogNeg()
                    self.saveData(plotsDataDirectory, results["logNegArray"], computation, date)

                elif computation == TypeOfData.HighestOneByOne:
                    if self.plottingInfo["NumberOfStates"] == 1:
                        results["highestOneToOneValue"], results[
                            "highestOneToOnePartner"] = self.computeHighestOneByOne()
                        oneToOneDict = {
                            1: np.array([results["highestOneToOneValue"], results["highestOneToOnePartner"]])}
                        self.saveData(plotsDataDirectory, oneToOneDict, computation, date)
                    else:
                        print("Highest one by one not computed for this initial state (more than one initial state)")

                elif computation == TypeOfData.OccupationNumber:
                    results["occupationNumber"] = self.computeOccupationNumber()
                    self.saveData(plotsDataDirectory, results["occupationNumber"], TypeOfData.OccupationNumber, date)

                elif computation == TypeOfData.OddVSEven:
                    results["logNegEvenVsOdd"] = self.computeOddVSEven()
                    self.saveData(plotsDataDirectory, results["logNegEvenVsOdd"], TypeOfData.OddVSEven, date)

                elif computation == TypeOfData.SameParity:
                    results["logNegSameParity"] = self.computeSameParity()
                    self.saveData(plotsDataDirectory, results["logNegSameParity"], TypeOfData.SameParity, date)

                elif computation == TypeOfData.LogNegDifference:
                    results["logNegDifference"] = self.computeLogNegDifference(results["logNegArray"])
                    self.saveData(plotsDataDirectory, results["logNegDifference"], TypeOfData.LogNegDifference, date)
                    results["logNegArray"] = self.computeFullLogNeg()
                    self.saveData(plotsDataDirectory, results["logNegArray"], TypeOfData.FullLogNeg, date)
                    results["logNegArrayBefore"] = self.computeFullLogNeg(inState=True)
                    self.saveData(plotsDataDirectory, results["logNegArrayBefore"], TypeOfData.FullLogNegBefore, date)

                elif computation == TypeOfData.OneByOneForAGivenMode:
                    if self.plottingInfo["NumberOfStates"] == 1:
                        oneToOneGivenModes = dict()
                        oneToOneGivenModes[1] = np.zeros((len(specialModes), self.MODES))
                        for index, mode in enumerate(specialModes):
                            oneToOneGivenModes[1][index] = self.computeOneByOneForAGivenMode(mode)[1]
                        results["oneToOneGivenModes"] = oneToOneGivenModes
                    else:
                        print("For more than one initial state OneByOne for a list of modes is not computed")

                elif computation == TypeOfData.OneVSTwoForAGivenMode:
                    if self.plottingInfo["NumberOfStates"] == 1:
                        oneVsTwoGivenModes = dict()
                        oneVsTwoGivenModes[1] = np.zeros((len(specialModes), comb(self.MODES - 1, 2)))
                        for index, mode in enumerate(specialModes):
                            oneVsTwoGivenModes[1][index] = self.computeOneVSTwoForAGivenMode(mode)[1]
                        results["oneVSTwoForAGivenModes"] = oneVsTwoGivenModes
                    else:
                        print("For more than one initial state OneVsTwo for a list of modes is not computed")


                elif computation == TypeOfData.JustSomeModes:
                    results["justSomeModes"] = self.computeFullLogNeg(specialModes=specialModes)

                elif computation == TypeOfData.HawkingPartner:
                    if results["logNegArray"] is None:
                        results["logNegArray"] = self.computeFullLogNeg(specialModes=specialModes)

                    results["hawkingPartner"] = self.computeHawkingPartnerLogNeg(specialModes=specialModes)
        return results

    def plotFullLogNeg(self, logNegArray, plotsDirectory, saveFig=True, specialModes=None):
        if logNegArray is not None:
            if specialModes is not None:
                karray = [idx + 1 for idx in specialModes]
                numberOfModes = len(specialModes)
            else:
                karray = self.kArray
                numberOfModes = self.MODES

            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LN${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                   self.plottingInfo["Magnitude"][index],
                                                   self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None
                pl.loglog(karray, logNegArray[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(numberOfModes / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.FullLogNeg, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotHighestOneByOne(self, highestOneToOneValue, highestOneToOnePartner, logNegArray, plotsDirectory,
                            saveFig=True):
        if highestOneToOneValue is not None and highestOneToOnePartner is not None:
            if logNegArray is None:
                logNegArray = self.computeFullLogNeg()
            pl.figure(figsize=(12, 6))
            pl.loglog(self.kArray[:], highestOneToOneValue, label=r"Strongest one to one $LN$", alpha=0.5, marker='.',
                      markersize=8, linewidth=0.2)
            if logNegArray is not None:
                pl.loglog(self.kArray[:], logNegArray[1][:], label=r"Full $LN$", alpha=0.5, marker='.', markersize=8,
                          linewidth=0.2)

            y_values_highest = np.concatenate(
                (highestOneToOneValue, logNegArray[1][:] if logNegArray is not None else []))
            y_values_Full = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_values = np.concatenate([y_values_highest, y_values_Full])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            if len(self.kArray) == len(highestOneToOneValue) == len(highestOneToOnePartner):
                for i, txt in enumerate(highestOneToOnePartner):
                    pl.annotate(txt + 1, (self.kArray[i], highestOneToOneValue[i]), textcoords="offset points",
                                xytext=(0, 10), ha='center')
            else:
                raise ValueError("The lengths of k_array, maxValues, and maxPartners do not match.")

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.HighestOneByOne, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotOccupationNumber(self, occupationNumber, plotsDirectory, saveFig=True):
        if occupationNumber is not None:
            pl.figure(figsize=(12, 6))
            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$n${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                  self.plottingInfo["Magnitude"][index],
                                                  self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None
                pl.loglog(self.kArray[:], occupationNumber[index + 1], label=label, alpha=0.5, marker='.', markersize=8,
                          linewidth=0.2)

            y_values = np.concatenate(
                [occupationNumber[index + 1] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$n$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OccupationNumber, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotOddVsEven(self, logNegEvenVsOdd, logNegArray, plotsDirectory, saveFig=True):
        if logNegEvenVsOdd is not None:
            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LN$ Odd vs Even{}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                               self.plottingInfo["Magnitude"][index],
                                                               self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "$LN$ Odd vs Even"
                pl.loglog(self.kArray[:], logNegEvenVsOdd[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values_EvenOdd = np.concatenate(
                [logNegEvenVsOdd[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])

            if logNegArray is not None:
                if self.plottingInfo["NumberOfStates"] == 1:
                    pl.loglog(self.kArray[:], logNegArray[1][:], label=r"Full $LN$", alpha=0.5, marker='.',
                              markersize=8, linewidth=0.2)

                y_values_Full = np.concatenate(
                    [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
                y_values = np.concatenate([y_values_EvenOdd, y_values_Full])

            else:
                y_values = y_values_EvenOdd

            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OddVSEven, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotSameParity(self, logNegSameParity, logNegArray, plotsDirectory, saveFig=True):
        if logNegSameParity is not None:
            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LN$ Same Parity {}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                                self.plottingInfo["Magnitude"][index],
                                                                self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "$LN$ Same Parity"
                pl.loglog(self.kArray[:], logNegSameParity[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values_SameParity = np.concatenate(
                [logNegSameParity[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])

            if logNegArray is not None:
                if self.plottingInfo["NumberOfStates"] == 1:
                    pl.loglog(self.kArray[:], logNegArray[1][:], label=r"Full $LN$", alpha=0.5, marker='.',
                              markersize=8, linewidth=0.2)

                y_values_Full = np.concatenate(
                    [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
                y_values = np.concatenate([y_values_SameParity, y_values_Full])

            else:
                y_values = y_values_SameParity

            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.SameParity, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotOneByOneForGivenMode(self, oneToOneGivenModes, specialModes, plotsDirectory, plotsDataDirectory,
                                 saveFig=True, saveData=True):
        if oneToOneGivenModes is not None:
            pl.figure(figsize=(12, 6))

            for index, mode in enumerate(specialModes):
                label = r"$LN$ {} vs each other".format(mode)
                pl.loglog(self.kArray[:], oneToOneGivenModes[1][index][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values = np.concatenate(
                [oneToOneGivenModes[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min = np.min(y_values)
            y_max = np.max(y_values)

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"] + "{}${: .2f}{}$".format(
                    self.plottingInfo["MagnitudeName"], self.plottingInfo["Magnitude"][0],
                    self.plottingInfo["MagnitudeUnits"]) if self.plottingInfo["Magnitude"][
                                                                0] != "" else "", fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OneByOneForAGivenMode, date)
                pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')

            if saveData:
                fileName = self.getFileName(plotsDataDirectory, TypeOfData.OneByOneForAGivenMode, date)[
                    0]  # Asegurarse de que fileName sea una cadena de texto
                dataToSave = np.zeros((self.MODES + 2, len(specialModes)))
                for index, mode in enumerate(specialModes):
                    arrayParameter = 0
                    if self.arrayParameters is not None:
                        arrayParameter = self.arrayParameters[0]
                    dataToSave[0, index] = arrayParameter
                    dataToSave[1, index] = mode
                    dataToSave[2:, index] = oneToOneGivenModes[1][index]

                os.makedirs(os.path.dirname(fileName), exist_ok=True)

                for index, mode in enumerate(specialModes):
                    np.savetxt("{}_plotNumber_{}.txt".format(fileName, index + 1), dataToSave[:, index])

    def plotOneVsTwoForAGivenMode(self, oneVsTwoForGivenMode, specialModes, plotsDirectory,
                                  plotsDataDirectory, saveFig=True, saveData=True):
        if oneVsTwoForGivenMode is not None and len(specialModes) == 1:
            print(oneVsTwoForGivenMode)

            allPairs = [(i, j) for i, j in combinations(range(self.MODES), 2)
                        if specialModes[0] - 1 not in (i, j)]
            mode = specialModes[0]
            dataVector = oneVsTwoForGivenMode[1][0]
            matrix = np.full((self.MODES, self.MODES), np.nan)

            for index, (i, j) in enumerate(allPairs):
                value = dataVector[index]
                scalarValue = float(value) if np.ndim(value) != 0 else value
                matrix[i, j] = scalarValue
                matrix[j, i] = scalarValue

            # Mask lower triangle
            mask = np.tril(np.ones_like(matrix, dtype=bool))

            # Create figure
            pl.figure(figsize=(8, 6))

            # Validar los datos antes de calcular vmin y vmax
            valid_data = matrix[~np.isnan(matrix) & (matrix > 0)]
            if valid_data.size > 0:
                vmin = np.min(valid_data)
                vmax = np.max(valid_data)
            else:
                vmin = 1e-10  # Valor predeterminado para evitar errores
                vmax = 1  # Valor predeterminado para evitar errores

            # Log normalization para valores positivos
            norm = LogNorm(vmin=vmin, vmax=vmax)

            # Heatmap sin etiquetas automáticas
            sns.heatmap(matrix, mask=mask, cmap='viridis', norm=norm, square=True,
                        cbar_kws={'label': r"$LogNeg$ (log scale)"}, rasterized=False,
                        xticklabels=False, yticklabels=False)

            # Apply hardcoded limits if requested
            xMin, xMax, yMin, yMax = 1, self.MODES, 1, self.MODES
            applyLimits = True
            if applyLimits:
                xMin, xMax = 0, 50
                yMin, yMax = 0, 50
                pl.xlim(xMin, xMax)
                pl.ylim(yMin, yMax)

            # Set custom ticks espaciados
            stepX = max(1, xMax // 10)
            stepY = max(1, yMax // 10)
            ticksX = np.arange(0, xMax, stepX)
            ticksY = np.arange(0, yMax, stepY)
            tick_labels = ticksX + 1
            pl.xticks(ticksX + 0.5, tick_labels, rotation=0)
            pl.yticks(ticksY + 0.5, tick_labels, rotation=0)

            pl.title(f"LogNeg of mode {mode + 1} vs all pairs (excluding {mode + 1})" + "{}${: .2f}{}$".format(
                self.plottingInfo["MagnitudeName"], self.plottingInfo["Magnitude"][0],
                self.plottingInfo["MagnitudeUnits"]) if self.plottingInfo["Magnitude"][
                                                            0] != "" else "")
            pl.xlabel("Mode i")
            pl.ylabel("Mode j")

            # Add top 3 points if enabled
            applyPoints = True
            if applyPoints:
                maskedMatrix = np.triu(matrix, k=1)
                validIndices = [(i, j) for i in range(maskedMatrix.shape[0])
                                for j in range(maskedMatrix.shape[1])
                                if not np.isnan(maskedMatrix[i, j]) and
                                (not applyLimits or (xMin <= j < xMax and yMin <= i < yMax))]

                topCoords = sorted(validIndices, key=lambda x: maskedMatrix[x[0], x[1]], reverse=True)[:3]

                colors = ['red', 'blue', 'green']
                handles = []

                for idx, (i, j) in enumerate(topCoords):
                    color = colors[idx % len(colors)]
                    # Mostrar índices 1-based en la leyenda
                    handle = pl.Line2D([0], [0], marker='o', color='w', label=f'({i + 1},{j + 1})',
                                       markerfacecolor=color, markersize=8)
                    handles.append(handle)
                    pl.plot(j + 0.5, i + 0.5, 'o', color=color)

                pl.legend(handles=handles, title="Top LogNeg (i,j)", loc='upper right', fontsize=8, title_fontsize=9)

            pl.tight_layout()

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.OneVSTwoForAGivenMode, date)
                pl.savefig(figureName)

            if saveData:
                fileNameBase = self.getFileName(plotsDataDirectory, TypeOfData.OneVSTwoForAGivenMode, date)[0]
                os.makedirs(os.path.dirname(fileNameBase), exist_ok=True)
                np.savetxt(f"{fileNameBase}_mode_{mode}.txt", matrix)

            pl.close()

    def plotLogNegDifference(self, differenceArray, logNegArray, logNegArrayBefore, plotsDirectory, saveFig=True):
        if differenceArray is not None:
            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                label = r"$LNAfter-LNBefore${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                                 self.plottingInfo["Magnitude"][index],
                                                                 self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None
                pl.loglog(self.kArray[:], differenceArray[index + 1][:], label=label, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

                labelLogNeg = r"$LNAfter${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                              self.plottingInfo["Magnitude"][index],
                                                              self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None

                pl.loglog(self.kArray[:], logNegArray[index + 1][:], label=labelLogNeg, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

                labelLogNegBefore = r"$LNBefore${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                                     self.plottingInfo["Magnitude"][index],
                                                                     self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else None

                pl.loglog(self.kArray[:], logNegArrayBefore[index + 1][:], label=labelLogNegBefore, alpha=0.5,
                          marker='.', markersize=8, linewidth=0.2)

            y_values_logNegAfter = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_values_logNegBefore = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])

            y_min = np.min([np.min(y_values_logNegAfter), np.min(y_values_logNegBefore)])
            y_max = np.max([np.max(y_values_logNegAfter), np.max(y_values_logNegBefore)])

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(self.MODES / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if label is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.LogNegDifference, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def plotHawkingPartner(self, hawkingPartnerArray, logNegArray, specialModes, plotsDirectory, saveFig=True):
        if logNegArray is not None and hawkingPartnerArray is not None:
            karray = [idx + 1 for idx in specialModes]
            numberOfModes = len(specialModes)

            pl.figure(figsize=(12, 6))

            for index in range(self.plottingInfo["NumberOfStates"]):
                labelFull = r"$FullLN${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                           self.plottingInfo["Magnitude"][index],
                                                           self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "FullLN"

                labelHP = r"$Hawk-P LN${}${:.2f}{}$".format(self.plottingInfo["MagnitudeName"],
                                                            self.plottingInfo["Magnitude"][index],
                                                            self.plottingInfo["MagnitudeUnits"]) if \
                    self.plottingInfo["Magnitude"][index] != "" else "Hawk-P LN"

                pl.loglog(karray, logNegArray[index + 1][:], label=labelFull, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)
                pl.loglog(karray, hawkingPartnerArray[index + 1][:], label=labelHP, alpha=0.5, marker='.',
                          markersize=8, linewidth=0.2)

            y_values_fullLog = np.concatenate(
                [logNegArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min_fl = np.min(y_values_fullLog)
            y_max_fl = np.max(y_values_fullLog)

            y_values_hp = np.concatenate(
                [hawkingPartnerArray[index + 1][:] for index in range(self.plottingInfo["NumberOfStates"])])
            y_min_hp = np.min(y_values_hp)
            y_max_hp = np.max(y_values_hp)

            y_min = np.min([y_min_hp, y_min_fl])
            y_max = np.max([y_max_hp, y_max_fl])

            if y_min <= 0:
                y_min = 1e-8
            else:
                y_min = 10 ** np.floor(np.log10(y_min))

            if y_max <= 0:
                y_max = 1
            else:
                y_max = 10 ** np.ceil(np.log10(y_max))

            x_max = np.ceil(numberOfModes / 100) * 100

            pl.xlim(1, x_max)
            pl.ylim(y_min, y_max)
            pl.xlabel(r"$I$", fontsize=20)
            pl.ylabel(r"$LogNeg(I)$", fontsize=20)
            pl.grid(linestyle="--", color='0.9')
            legend = None
            if labelFull is not None:
                legend = pl.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
            mpl.rc('xtick', labelsize=16)
            mpl.rc('ytick', labelsize=16)

            pl.tight_layout()
            if "title" in self.plottingInfo:
                pl.suptitle(self.plottingInfo["title"], fontsize=20)

            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            if saveFig:
                figureName = self.getFigureName(plotsDirectory, TypeOfData.FullLogNeg, date)
                if legend:
                    pl.savefig(figureName, bbox_extra_artists=(legend,), bbox_inches='tight')
                else:
                    pl.savefig(figureName, bbox_inches='tight')

    def generatePlots(self, results, plotsDirectory, plotsDataDirectory, specialModes, listOfWantedComputations,
                      saveFig=True):
        logNegArray = results.get("logNegArray")
        logNegArrayBefore = results.get("logNegArrayBefore")
        highestOneToOneValue = results.get("highestOneToOneValue")
        highestOneToOnePartner = results.get("highestOneToOnePartner")
        occupationNumber = results.get("occupationNumber")
        logNegEvenVsOdd = results.get("logNegEvenVsOdd")
        logNegSameParity = results.get("logNegSameParity")
        oneToOneGivenModes = results.get("oneToOneGivenModes")
        oneVSTwoForAGivenModes = results.get("oneVSTwoForAGivenModes")
        differenceArray = results.get("logNegDifference")
        justSomeModes = results.get("justSomeModes")
        hawkingPartner = results.get("hawkingPartner")

        if TypeOfData.FullLogNeg in listOfWantedComputations and logNegArray is not None:
            self.plotFullLogNeg(logNegArray, plotsDirectory, saveFig=saveFig)

        if TypeOfData.OccupationNumber in listOfWantedComputations and occupationNumber is not None:
            self.plotOccupationNumber(occupationNumber, plotsDirectory, saveFig=saveFig)

        if TypeOfData.OddVSEven in listOfWantedComputations and logNegEvenVsOdd is not None:
            self.plotOddVsEven(logNegEvenVsOdd, None, plotsDirectory, saveFig=saveFig)

        if TypeOfData.SameParity in listOfWantedComputations and logNegEvenVsOdd is not None:
            self.plotSameParity(logNegSameParity, None, plotsDirectory, saveFig=saveFig)

        if TypeOfData.OneByOneForAGivenMode in listOfWantedComputations and oneToOneGivenModes is not None:
            self.plotOneByOneForGivenMode(oneToOneGivenModes, specialModes, plotsDirectory, plotsDataDirectory,
                                          saveFig=saveFig, saveData=True)

        if TypeOfData.OneVSTwoForAGivenMode in listOfWantedComputations and oneVSTwoForAGivenModes is not None:
            self.plotOneVsTwoForAGivenMode(oneVSTwoForAGivenModes, specialModes, plotsDirectory, plotsDataDirectory,
                                           saveFig=saveFig, saveData=True)

        if TypeOfData.LogNegDifference in listOfWantedComputations and differenceArray is not None:
            if logNegArray is not None and logNegArrayBefore is not None:
                self.plotLogNegDifference(differenceArray, logNegArray, logNegArrayBefore, plotsDirectory,
                                          saveFig=saveFig)

        if TypeOfData.HighestOneByOne in listOfWantedComputations and highestOneToOnePartner is not None and highestOneToOneValue is not None:
            self.plotHighestOneByOne(highestOneToOneValue, highestOneToOnePartner, logNegArray, plotsDirectory,
                                     saveFig=saveFig)

        if TypeOfData.JustSomeModes in listOfWantedComputations and justSomeModes is not None:
            self.plotFullLogNeg(justSomeModes, plotsDirectory, saveFig=saveFig, specialModes=specialModes)

        if TypeOfData.HawkingPartner in listOfWantedComputations and hawkingPartner is not None:
            self.plotHawkingPartner(hawkingPartner, logNegArray, specialModes, plotsDirectory, saveFig=saveFig)

    def obtainHawkingPartner(self, modeA: int = 0, inState: bool = False, atol: float = 1e-8):
        """
        Extracts the basis transformation that maps the original OUT basis to a new one
        where mode 0 is the Hawking mode, mode 1 is the partner, and the rest are reordered.

        Args:
            modeA: mode to be considered the Hawking one
            inState: wheter to obtain the partner for the inState or the outState

        Returns:
             newBogoliubovTransformation: Bogoliubov transformation in the new ordered basis from IN basis -> new OUT basis
             changeOfBasis: 2N x 2N transformation from the original OUT basis → new OUT basis
             statetoInBasis: Gaussian State in the in Basis to perform the newBogo on it (it works for states with thermal noise)
        """

        if self.plottingInfo["NumberOfStates"] != 1:
            raise ValueError("Hawking-partner calculation only works for 1 simulation at once")

        if InitialState.Thermal.value == self.plottingInfo["InStateName"]:
            SInitial = np.eye(2 * self.MODES)
            statetoInBasis = self.inState[1]

        elif InitialState.OneModeSqueezedFixedTemp.value == self.plottingInfo["InStateName"]:
            intensity_array = [self.arrayParameters[0] for _ in range(0, self.MODES)]
            auxiliarInitialState = qgt.elementary_states("squeezed", intensity_array)
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(auxiliarInitialState.V), 0)
            temperature = 0.694554 * self._temperature  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
            n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                        range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)]
            statetoInBasis = qgt.elementary_states("thermal", n_vector)

        elif InitialState.TwoModeSqueezedFixedTemp.value == self.plottingInfo["InStateName"]:
            auxiliarInitialState = qgt.Gaussian_state("vacuum", self.MODES)
            for j in range(0, self.MODES, 2):
                auxiliarInitialState.two_mode_squeezing(self.arrayParameters[0], 0, [j, j + 1])
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(auxiliarInitialState.V), 0)
            temperature = 0.694554 * self._temperature  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
            n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                        range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)]
            statetoInBasis = qgt.elementary_states("thermal", n_vector)

        elif InitialState.ThermalFixedOneModeSqueezing.value == self.plottingInfo["InStateName"]:
            intensity_array = [self._squeezing for _ in range(0, self.MODES)]
            auxiliarInitialState = qgt.elementary_states("squeezed", intensity_array)
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(auxiliarInitialState.V), 0)
            temperature = 0.694554 * self.arrayParameters[
                0]  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
            n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperature) - 1.0) for i in
                        range(0, self.MODES)] if temperature > 0 else [0 for i in range(0, self.MODES)]
            statetoInBasis = qgt.elementary_states("thermal", n_vector)

        else:
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(self.inState[1].V), 0)
            statetoInBasis = qgt.Gaussian_state("vacuum", self.MODES)

        if not inState:
            bogoliubovTransformation = self.transformationMatrix @ SInitial
        else:
            bogoliubovTransformation = SInitial

        newBogoliubovTransformation, changeOfBasis = extractHawkingPartner(bogoliubovTransformation, modeA, atol=atol)

        return newBogoliubovTransformation, changeOfBasis, statetoInBasis

    def obtainHawkingPartnerAsATwoModesState(self, modeA: int = 0, inState: bool = False, atol: float = 1e-8):
        """
        Obtains a two_modes system where mode 0 is the Hawking mode and mode 1 is the partner

        Args:
            modeA: mode to be considered the Hawking one
            inState: wheter to obtain the partner for the inState or the outState

        Returns:
             hawkingPartnerState: Gaussian State where mode 0 is the Hawking mode and mode 1 is the partner
        """

        if self.plottingInfo["NumberOfStates"] != 1:
            raise ValueError("Hawking-partner calculation only works for 1 simulation at once")

        if InitialState.Thermal.value == self.plottingInfo["InStateName"] or InitialState.Thermal.value == \
                self.plottingInfo["OutStateName"]:
            self._temperature = self.arrayParameters[0]

        if (InitialState.Thermal.value == self.plottingInfo["InStateName"] or
                InitialState.TwoModeSqueezedFixedTemp.value == self.plottingInfo["InStateName"] or
                InitialState.OneModeSqueezedFixedTemp.value == self.plottingInfo["InStateName"] or
                InitialState.ThermalFixedOneModeSqueezing.value == self.plottingInfo["InStateName"]):

            raise AttributeError("Hawking partner reduced state only works for pure initial states (no temp)")

        else:
            SInitial = qgt.BasisChange(get_symplectic_from_covariance(self.inState[1].V), 0)
            statetoInBasis = qgt.Gaussian_state("vacuum", 2)

        if not inState:
            bogoliubovTransformation = self.transformationMatrix @ SInitial
        else:
            bogoliubovTransformation = SInitial

        newBogoliubovTransformation = obtainHPTwoModeTransformation(bogoliubovTransformation, modeA, atol=atol)
        statetoInBasis.apply_Bogoliubov_unitary(newBogoliubovTransformation)

        return statetoInBasis

    """def obtainHPLogNegForListOfModes(self, specialModes: List[int], inState: bool = False, atol: float = 1e-8):
        def makeHPLogNegTask(mode):
            def task():
                newOutState = self.obtainHawkingPartnerAsATwoModesState(mode, inState=inState)
                logNegPartner = newOutState.logarithmic_negativity([0], [1])
                return mode, logNegPartner

            return task

        tasks = [makeHPLogNegTask(mode) for mode in specialModes]
        return self._execute(tasks)"""

    def obtainHPLogNegForListOfModes(self, specialModes: List[int], inState: bool = False, atol: float = 1e-8):
        def makeHPLogNegTask(mode):
            def task():
                newBogoTrans, _, newOutState = self.obtainHawkingPartner(mode, inState=inState)
                newOutState.apply_Bogoliubov_unitary(newBogoTrans)
                logNegPartner = newOutState.logarithmic_negativity([0], [1])
                return mode, logNegPartner

            return task

        tasks = [makeHPLogNegTask(mode) for mode in specialModes]
        return self._execute(tasks)
