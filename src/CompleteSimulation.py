from __future__ import annotations

import warnings

import numpy as np
import pylab as pl

import src.qgt as qgt
from .TypesAndParameters import (InitialStateParameters, InitialStateDictType,
                                 TransformationMatrixParameters, TransformationMatrixDictType)


class CompleteSimulation:
    """
    Class which encapsulates an initial state, a transformation matrix, and the final state.
    """

    inState: qgt.Gaussian_state
    outState: qgt.Gaussian_state
    transformationMatrix: np.ndarray

    def __init__(self, numModes: int, initialStateDict: InitialStateDictType,
                 transformationDict: TransformationMatrixDictType = None, directParseOfTM: np.ndarray = None):
        """
        Constructor for the CompleteSimulation class, it assigns all the atributes except from the outPut state which is computed only
        if the method 'performTransformation' is called.

        Parameters:
        numModes: (int) 
            number of modes considered in the simulation 
        transformationDict: (dict) 
            data to construct the transformation matrix from a folder with the alphas and betas 
        initialStateDict: (dict) 
            specifications for the initial state over which the transformation will be performed
        directParseOfTM: (np.ndarray)
            transformation matrix in terms of Bogoliubov coefficientes
        """

        self.MODES = numModes
        self.kArray = [i for i in range(1, numModes + 1)]
        if directParseOfTM is not None:
            self.setTransformationMatrix(directParseOfTM)
        else:
            self.transformationMatrix = self._constructTransformationMatrix(transformationDict)
        self.inState = self._createInState(initialStateDict)
        self.outState = None

    def _constructTransformationMatrix(self, transformationDict: TransformationMatrixDictType) -> np.ndarray:
        """
        Constructs the transformation matrix from the data stored in the directory.
        The data should be stored in files named as:
        alpha-nMODES-i.txt
        beta-nMODES-i.txt
        where i is the mode index

        Parameters:
        transformationDict: (dict) 
            data to construct the transformation matrix from a folder with the alphas and betas 

        Returns:
        np.ndarray
            Transformation matrix constructed from the data using the formula (eq 38 paper):
            Smatrix = ((alpha_11 beta*_11 alpha_21 beta*_21 ...) , (beta_11 alpha*_11 beta_21 alpha*_21 ...), ...)
        """
        directory = transformationDict[TransformationMatrixParameters.DATA_DIRECTORY.value]
        instantToPlot = transformationDict[TransformationMatrixParameters.INSTANT_TO_PLOT.value]

        if directory == "":
            warnings.warn("No data directory specified")
            return np.zeros(shape=(self.MODES, self.MODES), dtype=np.complex128)

        solsalpha = dict()
        solsbeta = dict()

        a = np.arange(1, self.MODES + 1)
        dir = directory
        for i in a:
            solsalpha[i] = pl.loadtxt(dir + "alpha-n" + str(self.MODES) + "-" + str(i) + ".txt")
            solsbeta[i] = pl.loadtxt(dir + "beta-n" + str(self.MODES) + "-" + str(i) + ".txt")

        # We now save the data in complex arrays
        time = len(solsbeta[1][:, 0])
        self.kArray = np.arange(1, self.MODES + 1)
        calphas_array = np.zeros((time, self.MODES, self.MODES), dtype=np.complex128)
        cbetas_array = np.zeros((time, self.MODES, self.MODES), dtype=np.complex128)
        for t in range(0, time):
            for i1 in range(0, self.MODES):
                for i2 in range(0, self.MODES):
                    calphas_array[t, i2, i1] = solsalpha[i2 + 1][t, 1 + 2 * i1] + solsalpha[i2 + 1][t, 2 + 2 * i1] * 1j
                    cbetas_array[t, i2, i1] = solsbeta[i2 + 1][t, 1 + 2 * i1] - solsbeta[i2 + 1][t, 2 + 2 * i1] * 1j
        # Label i2 corresponds to in MODES and i1 to out MODES

        # We now save the array at time we are interested in given by the variable "instant"
        self.instantToPlot = min(instantToPlot, time - 1)
        calphas_tot_array = np.zeros((self.MODES, self.MODES), dtype=np.complex128)
        cbetas_tot_array = np.zeros((self.MODES, self.MODES), dtype=np.complex128)
        calphas_tot_array = calphas_array[self.instantToPlot, :, :]
        cbetas_tot_array = cbetas_array[self.instantToPlot, :, :]

        # For our simulations
        Smatrix = np.zeros((2 * self.MODES, 2 * self.MODES), dtype=np.complex128)

        # Constructing the Smatrix out of the alpha and beta complex dicts
        # If we write A_out = Smatrix * A_in, see Eq. 39 of our paper, then
        # Smatrix = ((alpha*_11 -beta*_11 alpha*_12 -beta*_12 ...) , (-beta_11 alpha_11 -beta_12 alpha_12 ...), ...)
        # time = 5
        i = 0
        for i1 in range(0, self.MODES):
            j = 0
            for i2 in range(0, self.MODES):
                Smatrix[i, j] = calphas_tot_array[i2, i1]
                j = j + 1
                Smatrix[i, j] = np.conjugate(cbetas_tot_array[i2, i1])
                j = j + 1
            i = i + 1
            j = 0
            for i2 in range(0, self.MODES):
                Smatrix[i, j] = cbetas_tot_array[i2, i1]
                j = j + 1
                Smatrix[i, j] = np.conjugate(calphas_tot_array[i2, i1])
                j = j + 1
            i = i + 1

        return Smatrix

    def setTransformationMatrix(self, transformationMatrix: np.ndarray) -> None:
        """
        In case the data directory is not provided, one can add directly the transformation matrix
        """

        if transformationMatrix.shape[0] != 2 * self.MODES:
            raise AttributeError("Transformation Matrix should match the number of modes")

        """if not self.checkSymplectic(transformationMatrix):
            raise ValueError("Transformation Matrix should be symplectic")"""

        self.transformationMatrix = transformationMatrix

    def invertTransformationMatrix(self) -> None:
        """
        Given that A_in = S @ A_out, this computes the inverse Bogoliubov transformation matrix
        using the relation:

            S^{-1} = Omega^{-1} @ S^† @ Omega

        where S^† is the conjugate transpose of S.
        """

        omegaMatrix = self.inState.Omega  # Assumes shape (2N, 2N)
        S = self.transformationMatrix  # The original matrix: A_in = S @ A_out
        S_dagger = S.conj().T  # Conjugate transpose: S^†

        self.transformationMatrix = np.linalg.inv(omegaMatrix) @ S_dagger @ omegaMatrix

    @staticmethod
    def checkSymplectic(matrix) -> bool:
        return qgt.Is_Sympletic(matrix, 1)

    def _createInState(self, initialStateDict: InitialStateDictType) -> qgt.Gaussian_state:
        """
        Creates the initial state to be used in the calculations.

        Parameters:
        initialStateDict: (dict) 
            specifications for the initial state over which the transformation will be performed

        Returns:
            qgt.Gaussian_state
            Initial State with temperature and squeezing intensity (one or two modes) specified.
        """

        temperature = initialStateDict[InitialStateParameters.TEMPERATURE.value]
        temperatureGoodUnits = 4.37 * temperature  # For L = 0.01 m, we go from T(Kelvin) to T(Planck) by T(P) = kb*L*T(K)/(c*hbar)
        oneModeSqueezing = initialStateDict[InitialStateParameters.ONE_MODE_SQUEEZING.value]
        twoModeSqueezing = initialStateDict[InitialStateParameters.TWO_MODE_SQUEEZING.value]

        if temperature == 0.0 and oneModeSqueezing == 0.0 and twoModeSqueezing == 0.0:
            return qgt.Gaussian_state("vacuum", self.MODES)

        elif temperature > 0.0 and oneModeSqueezing == 0.0 and twoModeSqueezing == 0.0:
            n_vector = [1.0 / (np.exp(np.pi * self.kArray[i] / temperatureGoodUnits) - 1.0) for i in
                        range(0, self.MODES)] if temperatureGoodUnits > 0 else [0 for i in range(0, self.MODES)]
            return qgt.elementary_states("thermal", n_vector)

        elif temperature == 0.0 and oneModeSqueezing > 0.0 and twoModeSqueezing == 0.0:
            intensity_array = [oneModeSqueezing for i in range(0, self.MODES)]
            return qgt.elementary_states("squeezed", intensity_array)

        elif temperature == 0.0 and oneModeSqueezing == 0.0 and twoModeSqueezing > 0.0:
            state = qgt.Gaussian_state("vacuum", self.MODES)
            for j in range(0, self.MODES, 2):
                state.two_mode_squeezing(twoModeSqueezing, 0, [j, j + 1])
            return state

        elif temperature > 0.0 and oneModeSqueezing > 0.0 and twoModeSqueezing == 0.0:
            intensity_array = [oneModeSqueezing for _ in range(0, self.MODES)]
            state = qgt.elementary_states("squeezed", intensity_array)
            n_vector = np.array([1.0 / (np.exp(np.pi * self.kArray[i] / temperatureGoodUnits) - 1.0) for i in
                                 range(0, self.MODES)] if temperatureGoodUnits > 0 else [0 for _ in
                                                                                         range(0, self.MODES)])
            state.add_thermal_noise(n_vector)
            return state

        elif temperature > 0.0 and oneModeSqueezing == 0.0 and twoModeSqueezing > 0.0:
            state = qgt.Gaussian_state("vacuum", self.MODES)
            for j in range(0, self.MODES, 2):
                state.two_mode_squeezing(twoModeSqueezing, 0, [j, j + 1])
            n_vector = np.array([1.0 / (np.exp(np.pi * self.kArray[i] / temperatureGoodUnits) - 1.0) for i in
                                 range(0, self.MODES)] if temperatureGoodUnits > 0 else [0 for i in
                                                                                         range(0, self.MODES)])
            state.add_thermal_noise(n_vector)
            return state

        else:
            raise ValueError("Unrecognized inStateName")

    def performTransformation(self) -> None:
        """
        Performs the transformation of the initial states using the transformation matrix,
        assuming it is given in terms of the bogoliubov coefficients.
        """
        if self.transformationMatrix is None:
            raise Exception("Transformation matrix not initialized")

        if self.inState is None:
            raise Exception("Initial state not initialized")

        outState = self.inState.copy()
        outState.apply_Bogoliubov_unitary(self.transformationMatrix)

        self.outState = outState
