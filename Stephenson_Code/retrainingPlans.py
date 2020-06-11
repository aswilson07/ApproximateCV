'''
retrainingPlans holds the code to do one run of bootstrap or CV
  on a given model (e.g. see the bootstrap() function below). This is all
  done in parallel (you can manually adjust the number of cores in the various
  functions below).  Really this stuff is a wrapper around
  model.retrain_with_weights().


An random note on the parallelization used in some of the functions below:

The pathos.multiprocessing library (pip install pathos) seems to be a *huge*
improvement over the built-in multiprocessing library.  It uses dill instead of
pickle (pip install dill) to pass stuff to workers, which prevents 5 million
"this object is not pickleable" errors (you can't pickle lambda functions, you
can't pickle autograd nodes) by the fact that dill can serialize almost anything.

Also something I learned: itertools.cycle(['hello']) is a generator that
always returns 'hello' (useful for multiprocessing when you want to give
each thread the argument 'hello' without doing ['hello' for i in range(10000000)])
'''



import numpy as np
from tqdm import tqdm
import time
import itertools
import matplotlib.pyplot as plt
import datasets
import joblib
import copy
import pathos

import models

def leave_k_out_cv(model, k, method="IJ",
                   hold_outs="exact",
                   B=3000, log_dir=None, save_each=False,
                   seed=1234, non_fixed_dims=None, is_cv=False, **kwargs):

  '''
  Performs leave k out CV on any model (and its associated data) either via
  brute force or via approximations
  '''
  np.random.seed(seed)
  N = model.training_data.N
  D = model.params.get_free().shape[0]

  if hold_outs != "exact":
    held_out_sets = []
    for b in range(B):
      held_out_sets.append(np.random.choice(np.arange(N), size=k,
        replace=False))
  else:
    held_out_sets = itertools.combinations(np.arange(N), k)
  count = 0
  total_error = 0
  num_sets = 0

  all_params = np.empty((len(held_out_sets),D))
  w0 = model.params.get_free().copy()

  # Compute needed derivatives if doing approximation
  if method == 'IJ':
    model.compute_dParams_dWeights(np.ones(N), non_fixed_dims=non_fixed_dims,**kwargs)
  elif method == 'NS':
    if is_cv and model.is_a_glm:
      kwargs['non_fixed_dims'] = non_fixed_dims
      model.compute_loocv_rank_one_updates(**kwargs)
  elif method == 'Prox':
    model.compute_dParams_dWeightsProx(np.ones(N), non_fixed_dims=non_fixed_dims,**kwargs)

  for i, held_out_set in enumerate(held_out_sets):
    num_sets += 1
    weights = np.ones(N)
    held_out_idxs = list(held_out_set)
    weights[held_out_idxs] = 0.0

    if method=="IJ":
      kwargs['non_fixed_dims'] = non_fixed_dims
      params = model.retrain_with_weights(weights, doIJAppx=True, **kwargs)
    elif method == 'Prox':
      params = model.retrain_with_weights(weights, doProxAppx=True, **kwargs)
    elif method == 'exact':
      params = model.retrain_with_weights(weights, **kwargs)
    elif method == 'NS':
      kwargs['non_fixed_dims'] = non_fixed_dims
      if is_cv and model.is_a_glm:
        kwargs['is_cv'] = True
      params = model.retrain_with_weights(weights,
                                          doIJAppx=False,
                                          doProxAppx = False,
                                          doNSAppx=True,
                                          **kwargs)
    else:
      print('\nI dont recognize this method in leave_k_out_cv ',method,'\n')

    all_params[i,:] = params
    held_out_X = model.training_data.X[held_out_idxs]
    held_out_Y = model.training_data.Y[held_out_idxs]
    total_error += model.get_error(datasets.create_dataset(
      held_out_X, held_out_Y, copy=False))
    #count += k
    count += 1
    model.params.set_free(w0)
  # TODO: we probably want to compute variance
  return total_error * 1.0 / count, all_params


def leave_k_out_cv_parallel(model, k, method="IJ",
                            hold_outs="exact", nCores=6,
                            B=3000, log_dir=None, save_each=False,
                            seed=1234, non_fixed_dims=None, is_cv=False, **kwargs):
  '''
  Performs leave k out CV on any model (and its associated data) either via
  brute force or via approximations

  hold_outs:
    exact = exhaustively compute hold outs
    stochastic = take B number of samples

  method:
    exact = retrain model each time
    IJ = use a linear approximation instead of retraining model

  '''
  np.random.seed(seed)
  N = model.training_data.N
  D = model.params.get_free().shape[0]

  if hold_outs != "exact":
    held_out_sets = []
    for b in range(B):
      held_out_sets.append(np.random.choice(np.arange(N), size=k,
        replace=False))
  else:
    held_out_sets = list(itertools.combinations(np.arange(N), k))
  count = 0
  total_error = 0
  num_sets = 0
  pool = pathos.multiprocessing.ProcessingPool(nCores)
  
  def run_me(held_out_set, model, approx, approxProx, runSlowAppxCV,
             kwargs, log_dir=None, save_each=False, non_fixed_dims=None,):
    model = copy.deepcopy(model)
    w0 = model.params.get_free().copy()
    Ntrain = model.training_data.X.shape[0]
    held_out_idxs = list(held_out_set)
    weights = np.ones(Ntrain)
    weights[held_out_idxs] = 0
    label = hash(".".join([str(idx) for idx in held_out_idxs]))
    params = model.retrain_with_weights(weights,
                                        doProxAppx=approxProx,
                                        doIJAppx=approx,
                                        doNSAppx=runSlowAppxCV,
                                        log_dir=log_dir,
                                        label=label,
                                        non_fixed_dims=non_fixed_dims,
                                        **kwargs)
    #params[inds] = 0.0
    held_out_X = model.training_data.X[held_out_idxs]
    held_out_Y = model.training_data.Y[held_out_idxs]
    total_error = model.get_error(datasets.create_dataset(
      held_out_X, held_out_Y, copy=False))
    return total_error, params

  if method == 'IJ':
    approx = True
    approxProx =False
    runSlowAppxCV = False
    model.compute_dParams_dWeights(np.ones(N),
                                   non_fixed_dims=non_fixed_dims,
                                   **kwargs)
  elif method == 'Prox':
    approxProx =True
    runSlowAppxCV = False
    approx = False
    model.compute_dParams_dWeightsProx(np.ones(N),
                                       non_fixed_dims=non_fixed_dims,
                                       **kwargs)
  elif method == 'NS':
    runSlowAppxCV = True
    approx = False
    approxProx = False
    if is_cv and model.is_a_glm:
      model.compute_loocv_rank_one_updates(non_fixed_dims=non_fixed_dims,
                                           **kwargs)
      kwargs['is_cv'] = True
  elif method == 'exact':
    runSlowAppxCV = False
    approx = False
    approxProx = False
  else:
    print('\n\n I dont recognize this method %s in leave_k_out_cv_parallel\n\n',method)

  if non_fixed_dims is not None:
    inds = np.setdiff1d(np.arange(D), non_fixed_dims)
  else:
    inds = []

  test_data = model.test_data
  model.test_data = None

  rets = pool.map(run_me,
                  held_out_sets,
                  itertools.cycle([model]),
                  itertools.cycle([approx]),
                  itertools.cycle([approxProx]),                  
                  itertools.cycle([runSlowAppxCV]),
                  itertools.cycle([kwargs]),
                  itertools.cycle([None]),
                  itertools.cycle([False]),
                  itertools.cycle([non_fixed_dims]))
  model.test_data = test_data

  pool.clear()
  pool.restart()
  pool.close()
  cv_errors = [ret[0] for ret in rets]
  params = [ret[1] for ret in rets]
  return np.array(cv_errors).mean(), np.array(params)

def bootstrap(model, B, sample_bootstrap_weights,
              method='exact', seed=None,
              gs=None, gIDs=None, hessian_scaling=None, S1=None, S2=None,
              nCores=6):
  '''
  Passing in gs and gIDs allows you to bootstrap just some fixed functions of the
    parameters. method='approx' will use stochastic inverse hessian vector products
    to compute the needed derivatives. hessian_scaling, S1, and S2 are parameters
    relating to the stochastic inverse hvp
  '''
  np.random.seed(seed)
  N = model.training_data.X.shape[0]
  K = model.get_params().shape[0]
  bootstrap_weights = np.zeros((B,N))
  bootstrap_params = np.zeros((B,K))

  # You may have pre-computed this, but in order to be fair on timing this
  #   function, we recompute it here.
  deriv_computation_time = 0.0
  if method == 'IJ':
    if gs is None:
      start = time.time()
      model.compute_dParams_dWeights(np.ones(N), **kwargs)
      deriv_computation_time = time.time() - start
    else:
      for idx in range(len(gs)):
        model.compute_dgParams_dWeights(gs[idx], gID=idx,
                                        hessian_scaling=hessian_scaling,
                                        S1=S1, S2=S2, **kwargs)


  for b in range(B):
    bootstrap_weights[b] = sample_bootstrap_weights()

  pool = pathos.multiprocessing.ProcessingPool(nCores)
  if method == "exact":
    run_me = lambda weights, model=model: model.retrain_with_weights(weights,
                                                                  doIJAppx=False,
                                                                  doProxAppx=False,
                                                                  gs=gs,
                                                                  gIDs=gIDs)

  elif method == "IJ":
    run_me = lambda weights, model=model: model.retrain_with_weights(weights,
                                                                  doIJAppx=True,
                                                                  doProxAppx=False,                                                                     
                                                                  gs=gs,
                                                                  gIDs=gIDs)
  elif method == "Prox":
    run_me = lambda weights, model=model: model.retrain_with_weights(weights,
                                                                  doIJAppx=False,
                                                                  doProxAppx=False,                                                                     
                                                                  gs=gs,
                                                                  gIDs=gIDs)
    
  bootstrapped_samples = pool.map(run_me, bootstrap_weights)
  pool.clear()
  pool.restart()
  pool.close()
  return bootstrap_weights, np.array(bootstrapped_samples), deriv_computation_time
  #return bootstrap_weights, bootstrap_params


def bootstrap_specific_functions(model, B, sample_bootstrap_weights, gs,
                                 method='exact', seed=None, nCores=6):
  '''
  Forms bootstrap samples of specific functions gs = [g1(theta), g2(theta), ...].
  For method='approx', directly computes dgParams_dWeights for each g using
     stochastic inverse hessian vector products
  '''
  np.random.seed(seed)
  N = model.training_data.X.shape[0]
  K = model.get_params().shape[0]
  numG = len(gs)
  bootstrap_weights = np.zeros((B,N))
  bootstrap_vals = np.zeros((B,numG))
  gIDs = np.arange(numG)

  # You may have pre-computed this, but in order to be fair on timing this
  #   function, we recompute it here.
  if method == 'IJ':
    for idx, g in enumerate(gs):
      mode.compute_dgParams_dWeights(g, gID=gIDs[idx],
                                     hessian_scaling=10000.0,
                                     S1=1, S2=2*N,
                                     **kwargs)

  for b in range(B):
    bootstrap_weights[b] = sample_bootstrap_weights()

  pool = pathos.multiprocessing.ProcessingPool(nCores)
  if method == "exact":
    run_me = lambda weights, model=model: model.retrain_with_weights(weights,
                                                                     gs=gs,
                                                                     gIDs=gIDs)

  elif method == "IJ":
    run_me = lambda weights, model=model: model.retrain_with_weights(weights,
                                                                   doIJAppx=True,
                                                                   gs=gs,
                                                                   gIDs=gIDs)
  bootstrapped_samples = pool.map(run_me, bootstrap_weights)
  pool.clear()
  pool.restart()
  pool.close()
  return bootstrap_weights, np.array(bootstrapped_samples)
  #return bootstrap_weights, bootstrap_params


def repeat_synthetic(data_getter, model, N,
                     reps=1000, init_seed=1234, nCores=6):
  '''
  Repeatedly resamples a new synthetic dataset and learns its parameters.
  Gives a ground truth to compare bootstrap to.

  data_getter should take as input a seed and return a (training_data, test_data)
  pair (actually the test_data is unused, so it can really return anything there)
  '''

  D = model.training_data.X.shape[1]
  params = []
  '''
  # Serialized version of this code
  for i in range(reps):
    training_data, _ = data_generator.get_dataset(Ntrain=N,
                                            Ntest=0,
                                            seed=init_seed+i,
                                            D=D)
    w = vb.ArrayParam(name='w', shape=(training_data.D+1,))
    params = vb.ModelParamsDict('params')
    params.push_param(w)
    params.set_free(np.random.normal(size=training_data.D+1))

    # Fit model on actual data (all weights = 1) and assess error
    model = models.LogisticRegressionModel(
        training_data, params, example_weights=np.ones(training_data.N)) # number of training examples
    model.fit()
    #model.training_data = copy.deepcopy(dataset)
    #model.fit()
    ret.append(model.params.get_free().copy())
  '''


  pool = pathos.multiprocessing.ProcessingPool(nCores)
  def run_me(data_generator, model, N, seed):
    model.training_data = data_getter(seed)[0]
    model.fit()
    return model.params.get_free()
  params = pool.map(run_me,
                    itertools.cycle([data_getter]),
                    itertools.cycle([model]),
                    itertools.cycle([N]),
                    np.arange(reps)+init_seed)
                    #[data_getter for i in range(reps)] ,
                    #[model for i in range(reps)],
                    #[N for i in range(reps)],
                    #np.arange(reps)+init_seed)
  pool.clear()
  pool.restart()
  pool.close()
  return np.array(params)
