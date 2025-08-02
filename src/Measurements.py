from __future__ import annotations

import copy
from typing import List, Union, Tuple, Callable

import numpy as np
from joblib import Parallel, delayed

import src.qgt as qgt
from .TypesAndParameters import TypeOfMeasurement


class Measurements:
    """
    Class to perform entanglement and other measurments on given states
    """

    def __init__(self, parallelize: bool):
        """
        Builds the Measurements manager with the instruction to parallelize or not the processes
        """
        self.parallelize = parallelize

    def _execute(self, tasks):
        if self.parallelize:
            return Parallel(n_jobs=5)(delayed(t)() for t in tasks)
        else:
            return [t() for t in tasks]

    def selectMeasurement(self, typeOfMeasurement: str, stateToApply: qgt.Gaussian_state,
                          modesToConsider: List[int] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if typeOfMeasurement == TypeOfMeasurement.FullLogNeg.value:
            logNegArray = self.fullLogNeg(stateToApply, modesToConsider)
        elif typeOfMeasurement == TypeOfMeasurement.HighestOneByOne.value:
            logNegArray = self.highestOneByOne(stateToApply, modesToConsider)
        elif typeOfMeasurement == TypeOfMeasurement.OneByOneForAGivenMode.value:
            logNegArray = self.oneByOneForAGivenMode(stateToApply, modesToConsider[0])
        elif typeOfMeasurement == TypeOfMeasurement.OddVSEven.value:
            logNegArray = self.oddVSEven(stateToApply, modesToConsider)
        elif typeOfMeasurement == TypeOfMeasurement.SameParity.value:
            logNegArray = self.sameParity(stateToApply, modesToConsider)
        elif typeOfMeasurement == TypeOfMeasurement.OccupationNumber.value:
            logNegArray = self.occupationNumber(stateToApply, modesToConsider)
        elif typeOfMeasurement == TypeOfMeasurement.HawkingPartner.value:
            raise RuntimeWarning("HawkingPartner has to be used through specific method in LogNegManager")
        else:
            raise NotImplementedError("The measurement type is not supported.")

        return logNegArray

    def fullLogNeg(self, stateToApply: qgt.Gaussian_state, modesToConsider: list[int] = None):
        """
            Computes the full logarithmic negativity for the states.
            That is, for each mode, computes the logarithmic negativity taking that mode as partA and all the others as partB.
            If modesToConsider is given, it only iterates over the given modes.

            Parameters:
            stateToApply: qgt.Gaussian_state
                The state to consider for the measurement

            modesToConsider: list of int
                Particular modes to consider as partA en the bipartition. If None, it computes the fullLogNeg for every mode

            Returns:
            np.ndarray
                array with the FullLogNeg with the size of modesToConsider or the size of the total number of modes.
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

    def highestOneByOne(self, stateToApply: qgt.Gaussian_state, modesToConsider: List[int] = None):
        """
        Computes the highest one-to-one logarithmic negativity for each mode and the partner mode that gives the highest value.
        That is, for each mode it computes the LN between that mode as subsystem A and each of the rest of the modes as subsystem B, one by one.
        It keeps the highest value encountered and the corresponding partner.
        If modesToConsider is given, it only iterates over the given modes.

        Parameters:
            stateToApply: qgt.Gaussian_state
                The state to consider for the measurement

            modesToConsider: list of int
                Particular modes to consider as partA en the bipartition. If None, it computes the highestOneByOne for every mode

        Returns:
        Tuple[np.ndarray, np.ndarray]
            Tuple with two arrays:
            - First array: highest values of the one-to-one logarithmic negativity for each mode
            - Second array: partner mode that gives the highest value for each mode

        """
        MODES = stateToApply.N_modes
        if modesToConsider is None:
            numberOfModes = MODES
            modesToConsider = [idx for idx in range(numberOfModes)]

        def task(i1):
            def inner():
                values = {}
                for i2 in range(MODES):
                    if i1 == i2:
                        values[i2] = 0.0
                    else:
                        values[i2] = stateToApply.logarithmic_negativity([i1], [i2])
                maxPartner = max(values, key=values.get)
                return values[maxPartner], maxPartner

            return inner

        tasks = [task(i1) for i1 in modesToConsider]
        results = self._execute(tasks)
        maxValues, maxPartners = zip(*results)
        return np.array(maxValues), np.array(maxPartners)

    def oneByOneForAGivenMode(self, stateToApply: qgt.Gaussian_state, modeIndex: int):
        """
        Computes the one-to-one logarithmic negativity for a given mode with all the others.
        That is, it computes the LN between that mode as subsystem A and each of the rest of the modes as subsystem B, one by one.

        Parameters:
        stateToApply: qgt.Gaussian_state
            The state to consider for the measurement
        modeIndex: int
            Mode to be used as partA

        Returns:
        np.ndarray
            Array with the one-to-one logarithmic negativity for the given mode with each other mode.
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

    def oddVSEven(self, stateToApply: qgt.Gaussian_state, modesToConsider: List[int] = None):
        """
        Computes the logarithmic negativity for the even modes vs the odd modes and vice versa.
        That is, for each mode, computes the logarithmic negativity taking that mode as partA,
        if the mode is even, then partB is all the odd modes, if the mode is odd, then partB is all the even modes.

        Parameters:
        stateToApply: qgt.Gaussian_state
                The state to consider for the measurement

        modesToConsider: list of int
                Particular modes to consider as partA en the bipartition. If None, it computes the oddVSEven for every mode

        Returns:
        np.ndarray
            Array with the OddVSEven logarithmic negativity for each mode.
        """

        MODES = stateToApply.N_modes
        if modesToConsider is None:
            numberOfModes = MODES
            modesToConsider = [idx for idx in range(numberOfModes)]

        evenFirstModes = np.arange(0, MODES - 1, 2)
        oddFirstModes = np.arange(1, MODES, 2)
        logNegEvenVsOdd = np.zeros_like(modesToConsider)

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

        tasks = [task(mode) for mode in modesToConsider]

        results = self._execute(tasks)

        for mode, value in results:
            logNegEvenVsOdd[mode] = value

        return logNegEvenVsOdd

    def sameParity(self, stateToApply: qgt.Gaussian_state, modesToConsider: List[int] = None):
        """
        Computes the logarithmic negativity for the even modes vs the odd modes and vice versa.
        That is, for each mode, computes the logarithmic negativity taking that mode as partA,
        if the mode is even, then partB is the rest of the even modes, if the mode is odd, then partB is the rest of the odd modes.


        Parameters:
        stateToApply: qgt.Gaussian_state
                The state to consider for the measurement

        modesToConsider: list of int
                Particular modes to consider as partA en the bipartition. If None, it computes the sameParity for every mode

        Returns:
        np.ndarray
            Array with the sameParity logarithmic negativity for each mode.
        """
        MODES = stateToApply.N_modes
        if modesToConsider is None:
            numberOfModes = MODES
            modesToConsider = [idx for idx in range(numberOfModes)]

        evenFirstModes = np.arange(0, MODES - 1, 2)
        oddFirstModes = np.arange(1, MODES, 2)
        logNegEvenVsOdd = np.zeros_like(modesToConsider)

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

        tasks = [task(mode) for mode in modesToConsider]

        results = self._execute(tasks)

        for mode, value in results:
            logNegEvenVsOdd[mode] = value

        return logNegEvenVsOdd

    def occupationNumber(self, stateToApply: qgt.Gaussian_state, modesToConsider: List[int] = None):
        """
        Computes the occupation number for each mode of the state.

        Parameters:
        stateToApply: qgt.Gaussian_state
                The state to consider for the measurement

        modesToConsider: list of int
                Particular modes to consider as partA en the bipartition. If None, it computes the occupation number for every mode

        Returns:
        np.ndarray
            Array with the occupation number for each mode.
        """
        state = copy.deepcopy(stateToApply)
        state.only_modes(modesToConsider)
        return state.occupation_number().flatten()

    def hawkingPartner(
            self,
            obtainHawkingPartner: Callable[[qgt.Gaussian_state, np.ndarray, int, str], np.ndarray],
            stateToApply: qgt.Gaussian_state,
            transformationMatrix: np.ndarray,
            modesToConsider: List[int] = None,
            criterion: str = None,
    ):
        MODES = stateToApply.N_modes
        if modesToConsider is None:
            numberOfModes = MODES
            modesToConsider = [idx for idx in range(numberOfModes)]

        logNegHawkingPartner = np.zeros(len(modesToConsider))

        def task(mode):
            newBogoTrans = obtainHawkingPartner(stateToApply, transformationMatrix, mode, criterion)
            newInitialState = qgt.Gaussian_state("vacuum", 2)
            newInitialState.apply_Bogoliubov_unitary(newBogoTrans)
            logNegPartner = newInitialState.logarithmic_negativity([0], [1])
            return mode, logNegPartner

        tasks = [lambda mode=mode: task(mode) for mode in modesToConsider]
        results = self._execute(tasks)

        for mode, value in results:
            logNegHawkingPartner[modesToConsider.index(mode)] = value

        return logNegHawkingPartner
