# Code for Approximate Cross-validation: Guarantees for Model Assessment and Selection

This repository contains the code used to generate the figures in "Approximate Cross-validation: Guarantees for Model Assessment and Selection" by Ashia Wilson, Maximilian Kasy, and Lester Mackey (https://arxiv.org/abs/2003.00617).



## Figure 2

To generate the plots in Figure 2 (tested with Python 3.7.4):
-Navigate to the `Stephenson_Code` directory
-Execute `python runLambdaCalibrationExps.py` to run each experiment
-Run the notebook `hyperparameterSelectionExps.ipynb` to generate the plots

## Figure 3

To generate the plots in Figure 3 (tested with MATLAB R2020a):
-compile QUIC using the mex compiler (e.g. > mex -llapack QUIC.C QUIC-mex.C -output QUIC.mexa64)
- Execute `ACV_QUIC_experiment.m' to run the experiment for each dataset
-Run the file `ACV_Quic_plots.m' to generate relative error and time of each experiment
-'QUIC_Plots.ipynb' generates the plots in the Figure 3, where the values in the notebook are calculated using 'ACV_Quic_plots.m'

Notes: the QUIC README file produced by sustik@cs.utexas.edu included in the QUIC folder is also helpful fo
