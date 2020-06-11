import os
import subprocess
import numpy as np

import expUtils


# support size of true parameter; only the first S entries will be nonzero
S = 75

# Settings of N, D and lambda (the amount of regularization) that you want to
#  run over
Ns = np.array([150,])
Ds = np.array([150])
lams = np.logspace(np.log10(0.01), np.log10(2.0), 15)
lams = np.concatenate((np.logspace(np.log10(0.000001), np.log10(0.001), 4), lams))
nCores = 4

for D in Ds:
  for N in Ns:
    for lam in lams:
      cmd = 'python cmdExperiments.py --dataset SyntheticLogisticDatasetGenerator --model LogisticRegressionModel --regularization L1 --use_fit_L1 --minNtrain %d --maxNtrain %d --numNtrains 1 --lambdaCoeff %f --B 100 --upTo %d --NtoD const --D %d --k 1 --Ntrials 1 --tag testtest --nCores %d --runNS --runIJ --runExactCV --runProx' % (N, N+1, lam, S, D, nCores)
      subprocess.call(cmd, shell=True)

      
