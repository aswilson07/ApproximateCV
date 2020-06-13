# Code for Approximate Cross-validation: Guarantees for Model Assessment and Selection

This repository contains the code used to generate the figures in "Approximate Cross-validation: Guarantees for Model Assessment and Selection" by Ashia Wilson, Maximilian Kasy, and Lester Mackey (https://arxiv.org/abs/2003.00617).

## Figure 1

To generate the plots in Figure 1 (tested with R 3.6.3):

* Navigate to the `r_code` directory
* Execute `R -f ACV_multiple_minima.R` to generate the plots

## Figure 2

To generate the plots in Figure 2 (tested with Python 3.7.4):

* Navigate to the `Stephenson_Code` directory
* Execute `python runLambdaCalibrationExps.py` to run each experiment
* Run the notebook `hyperparameterSelectionExps.ipynb` to generate the plots

Note: This code was adapted from code provided by Will Stephenson. See `Stephenson_Code/README.txt` for more details.

## Figure 3

To generate the plots in Figure 3 (tested with MATLAB R2020a):

* Compile QUIC using the mex compiler (e.g., > mex -llapack QUIC.C QUIC-mex.C -output QUIC.mexa64)
* Execute `ACV_QUIC_experiment.m` to run the experiment for each dataset
* Run the file `ACV_Quic_plots.m` to generate relative error and time of each experiment
* `QUIC_Plots.ipynb` generates the plots in the Figure 3, where the values in the notebook are calculated using `ACV_Quic_plots.m`

Note: This code was adapted form code provided by Ḿatyas A. Sustik (code at http://www.cs.utexas.edu/~sustik/QUIC/.) The QUIC README file included in the QUIC folder is also helpful.
