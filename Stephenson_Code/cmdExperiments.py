import autograd.numpy as np
import argparse
import autograd
import pickle
import copy
import time
import os

import Params
import retrainingPlans
import datasets
import expUtils
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--Ntrials', type=int, default=1)
parser.add_argument('--nCores', type=int, default=1)
parser.add_argument('--upTo', type=int, default=-1)
parser.add_argument('--Xrank', type=int, default=-1)
parser.add_argument('--lowRankNoise', type=float, default=0.0)
parser.add_argument('--solveRank', type=int, default=-1)
parser.add_argument('--every', type=int, default=1)
parser.add_argument('--B', type=int, default=30)
parser.add_argument('--k', type=int, default=1)
parser.set_defaults(use_fit_L1=False)
parser.add_argument('--use_fit_L1', dest='use_fit_L1', action='store_true')
parser.set_defaults(use_glmnet=False)
parser.add_argument('--use_glmnet', dest='use_glmnet', action='store_true')
parser.add_argument('--lambdaCoeff', type=str, default='1.0')
parser.add_argument('--lambdaScaling', type=str, default='sqrtND')
parser.add_argument('--NtoD', type=str, default='const')
parser.add_argument('--D', type=int, default=2)
parser.add_argument('--modelName', type=str)
parser.add_argument('--datasetName', type=str)
parser.add_argument('--regularization', type=str, default='None')
parser.add_argument('--Xscaling', type=str, default='None')
parser.add_argument('--tol', type=float, default='1e-10')
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--initPath', type=str, default=None)

parser.set_defaults(runExactCV=False)
parser.add_argument('--runExactCV', dest='runExactCV', action='store_true')
parser.add_argument('--no-runExactCV', dest='runExactCV', action='store_false')

# Argument to save params from initial fit (i.e., save \hat\theta)
parser.set_defaults(saveInitParams=True)
parser.add_argument('--saveInitParams', dest='saveInitParams',
                    action='store_true')

parser.set_defaults(runNS=False)
parser.add_argument('--runNS', dest='runNS', action='store_true')
parser.add_argument('--no-runNS', dest='runNS', action='store_false')

parser.set_defaults(runIJ=False)
parser.add_argument('--runIJ', dest='runIJ', action='store_true')
parser.add_argument('--no-runIJ', dest='runIJ', action='store_false')

parser.set_defaults(runProx=True)
parser.add_argument('--runProx', dest='runProx', action='store_true')
parser.add_argument('--no-runProx', dest='runProx', action='store_false')

# Special flags for cutting up real datasets
parser.set_defaults(smallNsmallDDataset=False)
parser.add_argument('--smallNsmallDDataset', dest='smallNsmallDDataset',
                    action='store_true')
parser.set_defaults(smallNDataset=False)
parser.add_argument('--smallNDataset', dest='smallNDataset',
                    action='store_true')
parser.set_defaults(smallDDataset=False)
parser.add_argument('--smallDDataset', dest='smallDDataset',
                    action='store_true')


parser.add_argument('--numNtrains', type=int, default=10)
parser.add_argument('--minNtrain', type=int, default=100)
parser.add_argument('--maxNtrain', type=int, default=15000)

parser.add_argument('--name', type=str, default=None)
args = parser.parse_args()

# Parse lambdaCoeff into a float. This is convoluted b/c I want to call this
#  from bash and don't know how to use bash very well.
if 'e' in args.lambdaCoeff:
  base, exponent = args.lambdaCoeff.split('e')
  lambdaCoeff = float(base)**(float(exponent))
else:
  lambdaCoeff = float(args.lambdaCoeff)



# Do stuff relating to given arguments
if args.NtoD == 'const':
  #NtoD = lambda N: 2
  NtoD = lambda N, D=args.D: D
elif args.NtoD == 'scaling':
  NtoD = lambda N: int(np.ceil(N/10))
elif args.NtoD == '2scaling':
  NtoD = lambda N: int(np.ceil(N*2))
elif args.NtoD == '1scaling':
  NtoD = lambda N: int(np.ceil(N))
elif args.NtoD == 'scalingOver5':
  NtoD = lambda N: int(np.ceil(N/5))
elif args.NtoD == 'squared':
  NtoD = lambda N: N**2

if args.datasetName.count('Synthetic') > 0:
  Ntrains = np.around(np.logspace(np.log10(args.minNtrain),
                                np.log10(args.maxNtrain),
                                  args.numNtrains), 0)
  Ntrains = Ntrains.astype(np.int32)
else:
  Ntrains = [0,]

if args.upTo == -1:
  upTo = None
else:
  upTo = args.upTo
if args.Xrank == -1:
  Xrank = None
else:
  Xrank = args.Xrank


## Create the filename to save results to
if args.use_fit_L1:
  L1str = '-fitL1'
else:
  L1str = ''
if args.name is None:
  name = expUtils.makeName(args.datasetName,
                           args.Xrank,
                           args.upTo,
                           L1str,
                           lambdaCoeff,
                           args.regularization,
                           args.k,
                           args.B,
                           alpha=args.alpha,
                           tag=args.tag,
                           solveRank=args.solveRank,
                           lowRankNoise=args.lowRankNoise)
else:
  name = args.name
outputPath = 'output/error_scaling_experiments-%s.pkl' % name


print(outputPath)
results = expUtils.loadResultsDict(outputPath)

dataset_generator = eval('datasets.%s()' % args.datasetName)
model_type = eval('models.%s' % args.modelName)

## Main loop. Basically run CV for all datasets.
for nn, Ntrain in enumerate(Ntrains):
  D = NtoD(Ntrain)
  if Ntrain in results and D+1 in results[Ntrain]:
      continue
  print('Starting', Ntrain, D)
  for trial in range(args.Ntrials):
    train, test = dataset_generator.get_dataset(data_seed=123456+trial,
                                                param_seed=123456+trial,
                                                D=D,
                                                Ntrain=Ntrain,
                                                Ntest=10000,
                                                upTo=upTo,
                                                Xrank=Xrank,
                                                every=args.every,
                                                Xscaling=args.Xscaling,
                                                lowRankNoise=args.lowRankNoise,
                                  smallNsmallDDataset=args.smallNsmallDDataset,
                                                small=args.smallNDataset,
                                                smallD=args.smallDDataset)
    print(train.X.shape)

    # Setup model
    #w = vb.ArrayParam(name='w', shape=(train.X.shape[1],))
    #params = vb.ModelParamsDict('params')
    #params.push_param(w)
    params = Params.Param(train.X.shape[1])
    model = model_type(train,
                       params,
                       example_weights=np.ones(train.X.shape[0]),
                       test_data=test,
                       regularization=None)

    if train.truth is not None:
      init = train.truth['theta'] * train.truth['scaling']
    elif args.initPath is not None:
      init = np.loadtxt(args.initPath)
    else:
      init = np.random.normal(scale=0.01, size=train.X.shape[1])
    model.params.set_free(init)

    if args.lambdaScaling == 'sqrtND':
      lam = lambdaCoeff * np.sqrt(np.log(train.X.shape[1]-1) * train.X.shape[0])
    elif args.lambdaScaling == 'const':
      lam = lambdaCoeff

    if args.regularization == 'L1':
      model.regularization = lambda x: lam * np.abs(x).sum()
      model.L1Lambda = lam
    elif args.regularization == 'L2':
      model.L2Lambda = lam
      model.regularization = lambda x: lam * np.linalg.norm(x)**2
    elif args.regularization == 'smoothedL1':
      model.regularization = lambda x: lam * (np.log(1+np.exp(args.alpha*x)) +
                                              np.log(1+np.exp(-args.alpha*x))).sum() / args.alpha
    elif args.regularization == 'None':
      model.regularization = lambda x: 0.0

    # Initial fit to get \hat\theta
    model.fit(use_fit_L1=args.use_fit_L1,
              use_glmnet=args.use_glmnet,
              extra_precision=True)
    if not args.use_fit_L1: # fit twice for extra precision, unless you're using
                                #   fitL1.
      model.fit(use_fit_L1=args.use_fit_L1,
                use_glmnet=args.use_glmnet,
                extra_precision=True)
    if args.use_glmnet:
      model.fit(use_fit_L1=True,
                extra_precision=True)

    if args.saveInitParams:
      np.savetxt('output/initParams-%s.txt' % name,
                 model.params.get_free())
      print('Done saving params')

    if args.regularization == 'L1':
      non_fixed_dims = np.where(model.params.get_free() != 0)[0]
    else:
      non_fixed_dims = None
    w0 = model.params.get_free()
    model.params.set_free(w0)

    expUtils.runCVAndLogResults(results, model, train, test,
                                k=1,
                                hold_outs='stochastic',
                                B=args.B,
                                nCores=args.nCores,
                                use_cvxpy=False,
                                cvxpy_tol=False,
                                non_fixed_dims=non_fixed_dims,
                                use_glmnet=args.use_glmnet,
                                use_fit_L1=args.use_fit_L1,
                                extra_precision=False,
                                runExactCV=args.runExactCV,
                                runNS=args.runNS,
                                runIJ=args.runIJ,
                                runProx=args.runProx,
                                is_cv=True,
                                rank=args.solveRank)

    # End inner loop over trials; save current results to disk.
    f = open(outputPath, 'wb')
    pickle.dump(results, f)
    f.close()
