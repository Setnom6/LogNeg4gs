# LogNeg4gs

LogNeg4gs (Logarithmic Negativity for Gaussian States) is a suite of methods which compute different bipartitions of
LogNeg for a given symplectic transformation matrix which works for several kinds of initial Gaussian states.

This repository contains the code used in the preprint available at: . All the physical definitions, background, and derivations required to understand the methods and interpretation of the results can be found there.

# Repository structure

```plaintext
LogNeg4gs/
├── README.md
├── simulate.py
├── src/
│   └── LogNegManager.py   
│   └── CompleteSimulation.py
│   └── Measurements.py
│   └── PartnerMethods.py
│   └── PlotsManager.py
│   └── TypesAndParameters.py
│   └── qgt.py   
├── example_data/
│   └── 10modes/    
│   └── data_info.md
├── tutorials/
│   ├── basic_tutorial.ipynb
│   ├── critical_temperature.ipynb
│   └── hawking_partner.ipynb
├── requirements.txt
├── LICENSE
└── configuration_files/
    └── config_example.json
```


# Requisites

- The module qgt.py was obtained from https://github.com/Setnom6/Quantum-Gaussian-Information-Toolbox-v2 in its July
  2025 version.
- The creation of the Bogolibov transformation matrix is built taking into account the structure of the data obtained as
  an output in simulations from DyNCHE-toolbox (https://github.com/jaon-ugr/DyNCHE-toolbox). If any other transformation
  matrix, given in terms of the Bogoliubov coefficientes, wants to be used it is still possible. For more info on how to
  proceed see ```exmaple_data/data_info.md```.
- The following modules are also necessary: numpy, scipy, joblib, matplotlib.

# Workflow

The suite have three main objects:

1) Transformation Matrix: can be given in terms of the alphas and betas coefficients of the Bogoliubov transformation
   stored with the format of DyNCHE-toolbox simulations or can be introduced directly as an 2Nx2N numpy ndarray. It is
   the main object of study, which will introduce the correlations in the system and stablish the number of modes in
   play.
2) Initial states: for a given transformation matrix an initial state is necessary to be selected. Initial States will
   be
   of type qgt.Gaussian_state but the user only has to indicate the ```temperature```, the ```oneModeSqueezing``` factor
   or the ```twoModeSqueezing``` factor of this initial state. For the same transformation matrix, a list of initial
   states can be parsed at once.
3) Measurement: once one have an Initial State and a transformation matrix to perform on it, is the time to select which
   kind of entanglement measurement one wants to take from them. The specifications are explained below.

The user will communicate only with ```LogNegManager```, a class intended to manage the initial parameters and
options which can handle the different measurements, store the results and plot them.

To simplify the process one can also just use the script ```simulate.py``` as:

```sh
python simulate.py config.json
```

## Configuration file

The configuration file must have the following shape

```json
{
  "generalOptions": {
    "numModes": 10,
    "plotsDirectory": "./plots/10modes-plots/",
    "dataDirectory": "./data/10modes-data/",
    "baseDirectory": "./",
    "parallelize": true
  },
  "transformationMatrix": {
    "dataDirectory": "./example_data/10modes/",
    "instantToPlot": -1
  },
  "initialStates": [
    {
      "temperature": 0.0,
      "oneModeSqueezing": 0.0,
      "twoModeSqueezing": 0.0
    },
    {
      "temperature": 1.0,
      "oneModeSqueezing": 2.0,
      "twoModeSqueezing": 0.0
    }
  ],
  "measurements": [
    {
      "type": "fullLogNeg",
      "modesToApply": [
        1,
        4,
        27,
        39
      ],
      "typeOfState": 1
    },
    {
      "type": "oddVSEven",
      "typeOfState": 0
    }
  ]
}
```

### Parameters

- ```"generalOptions""``` (dict)
    - ```numModes``` (int): Number of modes in the system, should match the transformation matrix dimensions (2 numModes
      x 2 numModes)
    - ```plotsDirectory``` (str): Directory path indicating where to save the figures created
    - ```dataDirectory``` (str): Directory path indicating where to save the data created
    - ```baseDirectory``` (str): Directory to use as a relative path
    - ```parallelize``` (bool): Whether to parallelize the measurements or not. Currently it works with joblib (take
      that into account if you are working with SLURM)

- ```transformationMatrix``` (dict)
    - ```dataDirectory``` (str): Directory to search for data of the transformation matrix in the structure of
      DyNCHE-toolbox simulations output
    - ```instantToPlot``` (int): As DyNCHE-toolbox simulations usually carry out more than one instant of time, this
      value select which one to choose among them
    - ```precomputedMatrix``` (np.ndarray): If one wants to use a precomputed np.ndarray transformation matrix.

- ```initialStates``` (list) Each element in the list is a dict indicating the parameters for an initial state
    - ```temperature``` (float): Temperature, in Kelvin, of the initial state to consider it thermal.
    - ```oneModeSqueezing``` (float): Squeezing intensity of each mode of the initial state
    - ```twoModeSqueezing``` (float): Squeezing intensity of each pair of modes which will suffer a two mode squeezing
      prior the main transformation

- ```measurements``` (list) Each element in the list is a dict indicating which kind of simulation to carry out
    - ```type``` (str): Type of measurement, see the supported types below.
    - ```modesToApply``` (list): List of ints indicating which modes to consider, only works for some types of
      measurements
    - ```typeOfState``` (int): Whether to apply the measurement to the initial state before the transformation (0) or to
      the out state after the transformation (1)
    - ```specialInfo``` (Any): Some measurements require more parameters to configure

### Supported types of measurements

At the moment the following measurements can be done:

- ```fullLogNeg```: Computes the LN between each mode and the rest of the modes 1x(N-1) bipartitions. If
  ```modesToApply``` is introduced, it only consider as subpart A that selected modes (one by one).
- ```oddVSEven```: Computes the LN between each mode and the rest of the modes if they do not share parity in the
  ordering of the modes. It supports ```modesToApply``` parameter.
- ```sameParity```: Computes the LN between each mode and the rest of the modes which share parity with it. It supports
  ```modesToApply``` parameter.
- ```oneByOneForAGivenMode```: Computes the LN between the selected mode in ```modesToApply``` (it will use only the
  element 0 of the list) and each of the remain modes taking them one by one.
- ```highestOneByOne```: Gives the LN between each mode and the mode which couples the most with it, indicating which
  one is. It supports ```modesToApply``` parameter.
- ```occupationNumber```: Gives the occupation number of each mode. It supports ```modesToApply``` parameter.
- ```hawkingPartner```: For each mode it computes its Partner and computes the LN betweem them. It supports
  ```modesToApply``` parameter. Only works for initial pure states (without temperature). There are two ways of compute
  the partner following the Hotta-Schutzhold-Unruh formula, "B1" or "B2" which can be selected with the parameters
  ```specialInfo```.


## Tutorials

The folder `tutorials/` contains some Jupyter notebooks illustrating how to use the package for typical configurations, custom measurements, and visualization of results. These notebooks are a good starting point to explore the capabilities of the code.


## Authors and Citation

This repository is maintained by J. M. Montes-Armenteros.
