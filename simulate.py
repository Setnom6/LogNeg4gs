import json
import os
import sys
import time
from src.LogNegManager import LogNegManager

# Get the configuration file from the command line arguments
if len(sys.argv) < 2:
    print("Usage: python simulate.py <config.json>")
    sys.exit(1)

config_file_path = sys.argv[1]

# Load configuration from the provided config file
with open(config_file_path) as config_file:
    config = json.load(config_file)

# Obtain dicts
generalOptions = config["general"]
transformationMatrixOptions = config["transformationMatrix"]
initialStates = config["initialStates"]
measurements = config["measurements"]

# Start the timer
start_time = time.time()

# Initialize the simulation

LNManager = LogNegManager(generalOptions=generalOptions, transformationDict=transformationMatrixOptions,
                          initialStates=initialStates)


# Perform the computations
dict_to_plot = {}
for measurement in measurements:
    if measurement["typeOfState"] == 0:
        dict_to_plot = LNManager.measureInitialStatesEntanglement(measurement["type"], measurement.get("modesToApply", None))
    else:
        dict_to_plot = LNManager.measureFinalStatesEntanglement(measurement["type"], measurement.get("modesToApply", None))

LNManager.plotResults(dict_to_plot)

# Stop the timer
end_time = time.time()
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Total computation time: {elapsed_time:.2f} seconds")