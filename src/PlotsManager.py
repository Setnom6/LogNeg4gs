from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

from src.TypesAndParameters import MeasurementDictType, GeneralOptionsDictType, MeasurementParameters, \
    InitialStateParameters, InitialStateDictType, TypeOfMeasurement, GeneralOptionsParameters


class PlotsManager:

    def __init__(self, dict_of_results: Dict):
        self.generalOptions: GeneralOptionsDictType = dict_of_results['generalOptions']
        self.listInitialStatesOptions: List[InitialStateDictType] = dict_of_results['initialStates']
        self.measurementDict: MeasurementDictType = dict_of_results['measurement']

        self.correspondence_measurements = {
            TypeOfMeasurement.FullLogNeg.value: "Full Log Neg",
            TypeOfMeasurement.HighestOneByOne.value: "Highest One-by-One Log Neg",
            TypeOfMeasurement.OddVSEven.value: "Odd VS Even Log Neg",
            TypeOfMeasurement.SameParity.value: "Same Parity Log Neg",
            TypeOfMeasurement.OccupationNumber.value: "Occupation Number",
            TypeOfMeasurement.OneByOneForAGivenMode.value: "One-By-One Log Neg",
            TypeOfMeasurement.HawkingPartner.value: "Hawking Partner Log Neg",
            TypeOfMeasurement.Comparison.value: "Difference between measurements"
        }

    def plotResults(self):
        modesToApply = self.measurementDict[MeasurementParameters.MODES_TO_APPLY.value]
        typeOfMeasurement = self.measurementDict[MeasurementParameters.TYPE.value]
        results = self.measurementDict[MeasurementParameters.RESULTS.value]
        typeOfState = self.measurementDict[MeasurementParameters.TYPE_OF_STATE.value]
        extraData = self.measurementDict[MeasurementParameters.EXTRA_DATA.value]

        numberOfModes = self.generalOptions[GeneralOptionsParameters.NUM_MODES.value]
        karray = [idx + 1 for idx in range(1, numberOfModes + 1)]
        if modesToApply is not None:
            if (typeOfMeasurement != TypeOfMeasurement.OneByOneForAGivenMode.value
                    and typeOfMeasurement != TypeOfMeasurement.HighestOneByOne.value):
                karray = [idx + 1 for idx in modesToApply]
                numberOfModes = len(modesToApply)

        plt.figure(figsize=(12, 6))

        y_values = []
        for i, initialStateDict in enumerate(self.listInitialStatesOptions):
            temperature = initialStateDict.get(InitialStateParameters.TEMPERATURE.value, 0.0)
            oneModeSqueezing = initialStateDict.get(InitialStateParameters.ONE_MODE_SQUEEZING.value, 0.0)
            twoModeSqueezing = initialStateDict.get(InitialStateParameters.TWO_MODE_SQUEEZING.value, 0.0)
            label = f"T = {temperature:.2f} K, r1 = {oneModeSqueezing:.2f}, r2 = {twoModeSqueezing:.2f}"

            logNegArray = results[i]
            y_values.extend(list(logNegArray))
            plt.loglog(karray, logNegArray, label=label, alpha=0.5, marker='.', markersize=8, linewidth=0.2)

            if typeOfMeasurement == TypeOfMeasurement.HighestOneByOne.value:
                if extraData[i] is not None:
                    for i, txt in enumerate(extraData[i]):
                        plt.annotate(txt + 1, (karray[i], logNegArray[i]), textcoords="offset points",
                                     xytext=(0, 10), ha='center')

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

        plt.xlim(1, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel(r"$I$", fontsize=20)

        ylabel = r"$n$" if TypeOfMeasurement.OccupationNumber.value == typeOfMeasurement else r"$LogNeg(I)$"
        plt.ylabel(ylabel, fontsize=20)
        legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0., fontsize=16)
        rc('xtick', labelsize=16)
        rc('ytick', labelsize=16)
        plt.grid()
        plt.tight_layout()

        statesTypes = "In States" if typeOfState == 0 else "Out States"
        title = self.correspondence_measurements[typeOfMeasurement] + " for " + statesTypes
        if typeOfMeasurement == TypeOfMeasurement.OneByOneForAGivenMode.value:
            title = self.correspondence_measurements[typeOfMeasurement] + " for " + statesTypes + ' for ' + str(
                modesToApply[0] + 1) + ' state'
            
        if typeOfMeasurement == TypeOfMeasurement.Comparison:
            measure1 = extraData[0]
            state1 = "In" if extraData[1] == 0 else "Out"
            measure2 = extraData[2]
            state2 = "In" if extraData[3] == 0 else "Out"
            title = self.correspondence_measurements[typeOfMeasurement] + f"({measure1},{state1}) and ({measure2},{state2})"
        plt.suptitle(title, fontsize=20)

        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        plotsDirectory = self.generalOptions[GeneralOptionsParameters.PLOTS_DIRECTORY.value]
        os.makedirs(plotsDirectory, exist_ok=True)
        figureName = typeOfMeasurement + "_" + date + ".pdf"
        figurePath = os.path.join(plotsDirectory, figureName)
        plt.savefig(figurePath, bbox_extra_artists=(legend,), bbox_inches='tight')
