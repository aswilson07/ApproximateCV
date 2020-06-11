This code was adapted from an early version of the code of Will Stephenson. The latest version of his code can be found at: https://bitbucket.org/wtstephe/sparse_appx_cv/src/master/ 

Only the following files were edited to introduce an unoptimized implementation of ProxACV:
-cmdExperiments.py
--Added flags for running IJ (the shorthand for ACVIJ) and Prox (the shorthand for ProxACV)
-expUtils.py
--Accommodated the loading of Prox results
-hyperparameterSelectionExps.ipynb
--Updated plots
-models.py
--Added Prox approximation
-retrainingPlans.py
--Added Prox option
-runLambdaCalibrationExps.py
--Updated choices of number of datapoints N, parameter setting D, support size S, and regularization parameters lambda

The text below represents the original code documentation provided by Will Stephenson and is not relevant for recreating the experiments in "Approximate Cross-validation: Guarantees for Model Assessment and Selection."

The main external dependency is autograd. Depending on your Python installation, you may have to use pip to install a few other standard packages (like multiprocessing and dill).

All of the experiments can be run using the script cmdExperiments.py. We give a couple examples here. To run the bcTCGA experiment from the paper, first download the dataset from the link given in the bcTCGADatasetGenerator class in datasets.py. Then from the same directory containing cmdExperiments.py, create an output/ directory and run the command:

python cmdExperiments.py --dataset bcTCGADatasetGenerator --model LinearRegressionModel --regularization L1 --use_fit_L1 --lambdaCoeff 1.5 --B 536

The output will be saved as a pickle file in the output/ directory. Inside the pickle is a single dictionary containing all the saved results. You can also automatically loop over synthetic datasets of different size. For example, the following will run approximate and exact CV for 10 different l2-regularized logistic regression tasks of size N=100 to N=1000, with D fixed at 5:

python cmdExperiments.py --dataset SyntheticLogisticDatasetGenerator --model LogisticRegressionModel --regularization L2 --minNtrain 100 --maxNtrain 1000 --numNtrains 10 --NtoD const --D 5


We now list some relevant parameters to cmdExperiments.py.  (they're listed in the format --parameter : options1, option2,...,optionN : description)

--dataset : (many options) : Selects the dataset to run on. You can specify any class name included in datasets.py that inherits from DatasetGenerator (e.g., "--dataset SyntheticLogisticDatasetGenerator").

--model : (many options) : You can specify the name of any class in models.py that inherits from GeneralizedLinearModel (e.g., "--model LogisticRegressionModel").

--regularization : L1, L2, smoothedL1, None : determines the kind of regularization to use. smoothedL1 forms a smooth approximation to L1 (--alpha controls the amount of smoothness in the approximation; see the description in section 4 of the paper). If using L1, you should provide the --use_fit_L1 option (tells the program to use the solver we included in fit_L1.py, rather than the standard scipy.optimize, which will not work with the non-differentiable L1 loss)

--lambdaScaling : sqrtND, const : Controls how the regularization parameter, lambda, scales with N and D. sqrtND sets it to the float value provided in --lambdaCoeff times np.sqrt(np.log(D) * N) (which is the standard rate for linear and logistic regression); const sets it exactly equal to --lambdaCoeff.

--NtoD : const, scaling : If running on a synthetic dataset of multiple sizes, this allows you to control how the dimension grows with N. const fixes D to whatever is set in the --D argument, and scaling sets D to np.ceil(N/10).

--minNtrain : (any integer >= 1) : If running on a synthetic dataset (i.e., anything of the form Synthetic*DatasetGenerator), cmdExperiments will automatically run over multiple values of N. --minNtrain sets the smallest value of N. --maxNtrain is the max value of N, and --numNtrain is the number of values inbetween (spaced evenly in log-space).

--B : (any integer >= 1) : We randomly select --B datapoints and only run exact / appx. CV for holding out those datapoints (e.g., in the bcTCGA example above, we specify --B 536 because the dataset has N=536, and we want to run CV over the whole dataset). Note that the default for this parameter is 30.


--nCores : (any integer >= 0) : Can parallelize rounds of exact/approximate CV across multiple cores using multiprocessing (all of our timings are reported using --nCores 1). Note that if your dataset size needs >= 4GB of RAM, the multiprocessing library will crash. In this case, you should use --nCores 0, which will call a special function retrainingPlans.leave_k_out_cv(), which has no calls to multiprocessing (as opposed to retrainingPlans.leave_k_out_cv_parellel()).
