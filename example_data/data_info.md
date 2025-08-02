# Data procedence

The colection of alpha and beta coefficients used to construct the Bogoliubov transformation matrix for $N=10$ modes was
obtained from DyNCHE-toolbox (https://github.com/jaon-ugr/DyNCHE-toolbox). It corresponds to a simulation of the
dynamics of a field on a box with moving boundaries.

In particular, this case corresponds to a simulation where one of the boundaries remains at rest while the
other accelerates and stops, symmetrically, at a final position. The concrete parameters are $\kappa =33$
and $\epsilon=0.375$ (see arXiv preprint for more info.)

# Data generation

The particular parameters for the simulation ($\kappa$, $\epsilon$ and initial and final times) have to be changed in
the ```exec.c``` file from DyNCHE-toolbox. Then, to generate a simulation with specific $N$ modes a ```script.sh``` file
has to be execute with the following structure (example for $N=10$)

```bash
    #!/bin/sh
  make clean
  make
  echo alpha-n11-1.txt beta-n11-1.txt norm-n11-1.txt 1 11 1 0.0 10.0 | ./exec
  echo alpha-n11-2.txt beta-n11-2.txt norm-n11-2.txt 1 11 2 0.0 10.0 | ./exec
  echo alpha-n11-3.txt beta-n11-3.txt norm-n11-3.txt 1 11 3 0.0 10.0 | ./exec
  echo alpha-n11-4.txt beta-n11-4.txt norm-n11-4.txt 1 11 4 0.0 10.0 | ./exec
  echo alpha-n11-5.txt beta-n11-5.txt norm-n11-5.txt 1 11 5 0.0 10.0 | ./exec
  echo alpha-n11-6.txt beta-n11-6.txt norm-n11-6.txt 1 11 6 0.0 10.0 | ./exec
  echo alpha-n11-7.txt beta-n11-7.txt norm-n11-7.txt 1 11 7 0.0 10.0 | ./exec
  echo alpha-n11-8.txt beta-n11-8.txt norm-n11-8.txt 1 11 8 0.0 10.0 | ./exec
  echo alpha-n11-9.txt beta-n11-9.txt norm-n11-9.txt 1 11 9 0.0 10.0 | ./exec
  echo alpha-n11-10.txt beta-n11-10.txt norm-n11-10.txt 1 11 10 0.0 10.0 | ./exec
  echo alpha-n11-11.txt beta-n11-11.txt norm-n11-11.txt 1 11 11 0.0 10.0 | ./exec
```

# Data structure

All data coming from DyNCHE-toolbox and generated with a script as the one above has the follwoing structure

```plaintext
data-folder/
|--alpha-n#MODES#-1.txt
|-- ...
|--alpha-n#MODES#-MODES.txt
|--beta-n#MODES#-1.txt
|-- ...
|--beta-n#MODES#-MODES.txt
```

Each file contains all the alpha (or beta) coefficients for a given out mode (given by the last number in the name of
the file) for all times considered. Inside each ```.txt```, apart from the first 4 rows, each row correspond to a given
time (expressed in the 0 element of the row). From elements $1$ to $2N$ there are the alpha (or beta) coefficients for
each in mode which contributes to the selected out mode, J. They are ordered as Re(alpha_1J), Im(alpha_1J),..., Re(
alpha_NJ), Im(alpha_NJ).

# Reconstruction of the Bogoliubov transformation matrix

In order to use a particular transformation matrix coming from this structured data, one have to instantiate the
constructor of ```LogNegManager```, among other parameters, with a dictionary containing the path of the stored data as
```"dataDirectory"``` and the instant of time tu use (usually the latest one with "-1") as ```"instantToPlot"```. This
will reconstruct the transformation matrix and apply it to the selected initial states. The reconstruction of the matrix
is made following Eq. 19 from the preprint.

# Initialize with a different Bogoliubov transformation

If your transformation data is not structured as in DyNCHE-toolbox, you can still use ```LogNegManager``` class. To do
that, add to the transformation matrix dictionary a new key called ```"precomputedMatrix"``` with your ```np.ndarray```
transformation matrix in terms of the Bogoliubov coefficients.