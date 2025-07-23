
from joblib import Parallel, delayed


import src.qgt as qgt

import numpy as np

from enum import Enum

class TypeOfMeasurement(Enum):
    FullLogNeg = "fullLogNeg"
    HighestOneByOne = "highestOneByOne"
    OneByOneForAGivenMode = "oneByOneForAGivenMode"
    OddVSEven = "oddVSEven"
    SameParity = "sameParity"
    OccupationNumber = "occupationNumber"

class Measurements:
    
    def __init__(self, parallelize):
        self.parallelize = parallelize

    def _execute(self, tasks):
        if self.parallelize:
            return Parallel(n_jobs=5)(delayed(t)() for t in tasks)
        else:
            return [t() for t in tasks]

    def selectMeasurement(self, typeOfMeasurement: str, stateToApply: qgt.Gaussian_state, modesToConsider: list =None):
        if typeOfMeasurement == TypeOfMeasurement.FullLogNeg.value:
            logNegArray = self.fullLogNeg(stateToApply, modesToConsider)
        elif typeOfMeasurement == TypeOfMeasurement.HighestOneByOne.value:
            logNegArray = self.highestOneByOne(stateToApply)
        elif typeOfMeasurement == TypeOfMeasurement.OneByOneForAGivenMode.value:
            logNegArray = self.oneByOneForAGivenMode(stateToApply, modesToConsider[0])
        elif typeOfMeasurement == TypeOfMeasurement.OddVSEven.value:
            logNegArray = self.oddVSEven(stateToApply)
        elif typeOfMeasurement == TypeOfMeasurement.SameParity.value:
            logNegArray = self.sameParity(stateToApply)
        elif typeOfMeasurement == TypeOfMeasurement.OccupationNumber.value:
            logNegArray = self.occupationNumber(stateToApply)
        else:
            raise NotImplementedError("The measurement type is not supported.")

        return logNegArray

    def fullLogNeg(self, stateToApply: qgt.Gaussian_state, modesToConsider: list = None):
        """
                Computes the full logarithmic negativity for the states.
                That is, for each mode, computes the logarithmic negativity taking that mode as partA and all the others as partB.

                Parameters:
                inState: bool
                    If True, the logarithmic negativity is computed for the inState, otherwise for the outState

                specialModes: list of int
                    Particular modes to consider

                Returns:
                dict[int, np.ndarray]
                    Dictionary with the full logarithmic negativity for each state. (indexes 1, 2, ...)
                    Each element of the dictionary is an array with the full logarithmic negativity for each mode.
                    (state i, full log neg of mode j -> fullLogNeg[i][j])
                """
        
        MODES = stateToApply.N_modes
        if modesToConsider is None:
            numberOfModes = MODES
            modesToConsider = [idx for idx in range(numberOfModes)]
        else:
            numberOfModes = len(modesToConsider)

        fullLogNeg = np.zeros(numberOfModes)

        def task(i1):
            return lambda: (
                i1,
                stateToApply.logarithmic_negativity([i1], [x for x in range(MODES) if x != i1])
            )

        tasks = [task(i1) for i1 in modesToConsider]
        results = self._execute(tasks)

        for i1, value in results:
            i1Index = modesToConsider.index(i1)
            fullLogNeg[i1Index] = value

        return fullLogNeg

    def highestOneByOne(self, stateToApply: qgt.Gaussian_state):
        """
        Computes the highest one-to-one logarithmic negativity for each mode and the partner mode that gives the highest value.

        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        Returns:
        Tuple[np.ndarray, np.ndarray]
            Tuple with two arrays:
            - First array: highest values of the one-to-one logarithmic negativity for each mode
            - Second array: partner mode that gives the highest value for each mode

        """
        modeCount = stateToApply.N_modes

        def task(i1):
            def inner():
                values = {}
                for i2 in range(modeCount):
                    if i1 == i2:
                        values[i2] = 0.0
                    else:
                        values[i2] = stateToApply.logarithmic_negativity([i1], [i2])
                maxPartner = max(values, key=values.get)
                return values[maxPartner], maxPartner

            return inner

        tasks = [task(i1) for i1 in range(modeCount)]
        results = self._execute(tasks)
        maxValues, maxPartners = zip(*results)
        return np.array(maxValues), np.array(maxPartners)


    def oneByOneForAGivenMode(self, stateToApply: qgt.Gaussian_state, modeIndex: int):
        """
        Computes the one-to-one logarithmic negativity for a given mode with all the others.

        Parameters:
        mode: int
            Mode to be used as partA
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        Returns:
        Dict[int, np.ndarray]
            Dictionary with the one-to-one logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the one-to-one logarithmic negativity for the given mode with each other mode.
            (state i, one-to-one log neg between given mode and mode j -> lognegarrayOneByOne[i][j])
        """
        MODES = stateToApply.N_modes
        lognegarrayOneByOne = np.zeros(MODES)

        def task(i2):
            return lambda: (
                i2,
                0.0 if modeIndex == i2 else stateToApply.logarithmic_negativity([modeIndex], [i2])
            )

        tasks = [task(i2) for i2 in range(MODES)]

        results = self._execute(tasks)

        for i2, value in results:
            lognegarrayOneByOne[i2] = value

        return lognegarrayOneByOne


    def oddVSEven(self, stateToApply):
        """
        Computes the logarithmic negativity for the even modes vs the odd modes and vice versa.
        That is, for each mode, computes the logarithmic negativity taking that mode as partA,
        if the mode is even, then partB is all the odd modes, if the mode is odd, then partB is all the even modes.

        Parameters:
        inState: bool
            If True, the logarithmic negativity is computed for the inState, otherwise for the outState

        Returns:
        Dict[int, np.ndarray]
            Dictionary with the logarithmic negativity for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the logarithmic negativity for each mode.
            (state i, log neg of mode j -> lognegarrayOneByOne[i][j])
        """
        MODES = stateToApply.N_modes
        evenFirstModes = np.arange(0, MODES - 1, 2)
        oddFirstModes = np.arange(1, MODES, 2)
        logNegEvenVsOdd = np.zeros(MODES)

        def task(mode):
            def inner():
                partA = [mode]
                if mode in evenFirstModes:
                    partB = [x for x in oddFirstModes if x != mode]
                else:
                    partB = [x for x in evenFirstModes if x != mode]
                value = stateToApply.logarithmic_negativity(partA, partB)
                return mode, value

            return inner

        tasks = [task( mode) for mode in range(MODES)]

        results = self._execute(tasks)

        for mode, value in results:
            logNegEvenVsOdd[mode] = value

        return logNegEvenVsOdd


    def sameParity(self, stateToApply: qgt.Gaussian_state):
        """
                Computes the logarithmic negativity for the even modes vs the rest of even modes. The same for the odd modes.
                That is, for each mode, computes the logarithmic negativity taking that mode as partA,
                if the mode is even, then partB is the rest of the even modes, if the mode is odd, then partB is the rest of the odd modes.

                Parameters:
                inState: bool
                    If True, the logarithmic negativity is computed for the inState, otherwise for the outState

                Returns:
                Dict[int, np.ndarray]
                    Dictionary with the logarithmic negativity for each state. (indexes 1, 2, ...)
                    Each element of the dictionary is an array with the logarithmic negativity for each mode.
                    (state i, log neg of mode j -> lognegarrayOneByOne[i][j])
        """
        MODES = stateToApply.N_modes
        evenFirstModes = np.arange(0, MODES - 1, 2)
        oddFirstModes = np.arange(1, MODES, 2)
        logNegEvenVsOdd = np.zeros(MODES)

        def task(mode):
            def inner():
                partA = [mode]
                if mode in evenFirstModes:
                    partB = [x for x in evenFirstModes if x != mode]
                else:
                    partB = [x for x in oddFirstModes if x != mode]
                value = stateToApply.logarithmic_negativity(partA, partB)
                return mode, value

            return inner

        tasks = [task(mode) for mode in range(MODES)]

        results = self._execute(tasks)

        for mode, value in results:
            logNegEvenVsOdd[mode] = value

        return logNegEvenVsOdd

    def occupationNumber(self, stateToApply):
        """
        Computes the occupation number for each mode of the state.

        Parameters:
        inState: bool
            If True, the occupation number is computed for the inState, otherwise for the outState

        Returns:
        Dict[int, np.ndarray]
            Dictionary with the occupation number for each state. (indexes 1, 2, ...)
            Each element of the dictionary is an array with the occupation number for each mode.
            (state i, occupation number of mode j -> occupationNumber[i][j])
        """
        return stateToApply.occupation_number().flatten()
    
    