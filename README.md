# LogNeg4gs
LogNeg4gs (Logarithmic Negativity for Gaussian States) is a suite of methods which compute different bipartitions of LogNeg for a given symplectic transformation matrix which works for several kind of initial states.

# Requisites

- The module qgt.py was obtained from https://github.com/Setnom6/Quantum-Gaussian-Information-Toolbox-v2 in its July 2025 version.
- The creation of the Bogolibov transformation matrix is built taking into account the structure of the data obtained as an output in simulations from DyNCHE-toolbox (https://github.com/jaon-ugr/DyNCHE-toolbox). If any other transformation matrix given in terms of the Bogoliubov coefficientes wants to be used it still possible.
- The following modules are also necessary: pylab, numpy, enum, typing, os, datetime, re, matplotlib

# Workflow

The suite have three main objects:

1) Transformation Matrix: can be given in terms of the alphas and betas coefficients of the Bogoliubov transformation stored with the format of DyNCHE-toolbox simulations or can be introduced directly as an 2Nx2N numpy ndarray. It is the main object of study, which will be introduce the correlations in the system and stablish the number of modes in play.
2) Initial states: for a given transformation matrix an initial state is necessary to be elected. Initial States will be of type qgt.Gaussian_state but the user only has to indicate the ```temperature```, the ```oneModeSqueezing``` factor or the ```twoModeSqueezing``` factor of this initial state. For the same transformation matrix, a list of initial states can be parsed at once.
3) Measurement: once one have an Initial State and a transformation matrix to perform on it, is the time to select which kind of entanglement measurement want to take from them. The specifications are explained below.

The user will be only communicate with ```LogNegManager```, a class intended to manage the initial parameters and options which can handle the different measurements, store the results and plot them. 

To simplify the process one can also just use the script ```simulate.py``` as:

```sh
python simulate.py config.json
```

## Configuration file
The configuration file must have the following shape

```json
{
  "general": {
    "numModes": 128,
    "plots_directory": "./plots/128-plots/",
    "data_directory": "./data/128-data/",
    "parallelize": true
  },
  "transformationMatrix": {
    "DataDirectory": "./sims-128/",
    "InstantToPlot": 15
  },

  "initialStates":
    [
      {"Temperature":  0.0,
      "OneModeSqueezing": 0.0,
        "TwoModeSqueezing": 0.0},
      {"Temperature":  1.0,
      "OneModeSqueezing": 2.0,
        "TwoModeSqueezing": 0.0}
    ]
  ,

  "measurements": [
    {"type": "fullLogNeg",
      "modesToApply": [1,4,27,39],
    "typeOfState":  1},
    {"type": "oddVSEven",
    "typeOfState":  0}
  ]
}
```

### Parameters

- ```"general""``` (dict)
  - ```numModes``` (int): Number of modes in the system, should match the transformation matrix dimensions (2 numModes x 2 numModes)
  - ```plots_directory``` (str): Directory path indicating where to save the figures created
  - ```data_directory``` (str): Directory path indicating where to save the data created
  - ```parallelize``` (bool): Whether to parallelize the measurements or not. Currently it works with joblib (take that into account if you are working with SLURM)

- ```transformationMatrix``` (dict)
  - ```DataDirectory``` (str): Directory to search for data of the transformation matrix in the structure of DyNCHE-toolbox simulations output
  - ```InstantToPlot``` (int): As DyNCHE-toolbox simulations usually carry out more than one instant of time, this value select which one to choose among them

- ```initialStates``` (list) Each element in the list is a dict indicating the parameters for an initial state
    - ```temperature``` (float): Temperature, in Kelvin, of the initial state to consider it thermal.
    - ```oneModeSqueezing``` (float): Squeezing intensity of each mode of the initial state
    - ```twoModeSqueezing``` (float): Squeezing intensity of each pair of modes which will suffer a two mode squeezing prior the main transformation

- ```measurements``` (list) Eache element in the list is a dict indicating which kind of simulation to carry out
  - ```type``` (str): Type of measurement, see the supported types below.
  - ```modesToApply``` (list): List of ints indicating which modes to consider, only works for some types of measurements
  - ```typeOfState``` (int): Whether to apply the measurement to the initial state before the transformation (0) or to the out state after the transformation (1)

### Supported types of measurements

At the moment the following measurements can be done:

- ```fullLogNeg```: Computes the LN between each mode and the rest of the modes 1x(N-1) bipartitions. If ```modesToApply``` is introduced, it only consider as subpart A that selected modes (one by one)
- ```oddVSEven```: Computes the LN between each mode and the rest of the modes if they do not share parity in the ordering of the modes
- ```sameParity```: Computes the LN between each mode and the rest of the modes which share parity with it
- ```oneByOneForAGivenMode```: Computes the LN between the selected mode in ```modesToApply``` (it has to be just one mode in the list) and each of the remain modes taking them one by one
- ```highestOneByOne```: Gives the LN between each mode and the mode which couples the most with it, indicating which one is
- ```occupationNumber```: Gives the occupation number of each mode
