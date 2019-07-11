# How a well-adapting immune system remembers

<img src="https://raw.githubusercontent.com/andim/paper-bayesimmune/master/fig1/fig1.png" width="400" alt='Figure 1' title="Sketch of a model of immune repertoire dynamics as a sequential inference process about a time-varying pathogen distribution.">

This repository contains the source code associated with the manuscript

Mayer, Balasubramanian, Walczak, Mora: [How a well-adapting immune system remembers](https://doi.org/10.1073/pnas.1812810116), PNAS 2019

It allows reproduction of all figures of the manuscript and also provides the simulation code.

## Installation requirements

The code uses Python 2.7+.

A number of standard scientific python packages are needed for the numerical simulations and visualizations. An easy way to install all of these is to install a Python distribution such as [Anaconda](https://www.continuum.io/downloads). 

- [numpy](http://github.com/numpy/numpy/)
- [scipy](https://github.com/scipy/scipy)
- [pandas](http://github.com/pydata/pandas)
- [matplotlib](http://github.com/matplotlib/matplotlib)

Additionally the code also relies on the following two packages:

- [projgrad](https://github.com/andim/projgrad)
- [palettable](https://github.com/jiffyclub/palettable)

## Structure/running the code

Every folder contains a file `plot.py` which needs to be run to produce the figures. For a number of figures cosmetic changes were done in inkscape as a postprocessing step. In these cases the figures will not be reproduced precisely. 

## Contact

If you run into any difficulties running the code, feel free to contact us at `andimscience@gmail.com`.

## License

The source code is freely available under an MIT license. The plots are licensed under a Creative commons attributions license (CC-BY).
