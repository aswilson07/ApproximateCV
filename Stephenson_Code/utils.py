'''
Maybe a better name for this is misc.py ...

Contains code for running experiments and saving their results (wrapper around
  functions from retrainingPlans.py), as well as some other random utilities 
'''



import autograd.numpy as np
import time
import retrainingPlans
import pickle
import copy
import os
from tqdm import tqdm
import models
import scipy.spatial
import autograd
import matplotlib.pyplot as plt

def sigmoid(X, w, bias=0):
  return 1./(1 + np.exp(-(np.dot(X, w) + bias)))

def sigmoid_only(Y):
  return 1./(1+ np.exp(-Y))

def whiten_data(X):
  '''
  Returns copy of dataset with zero mean and identity covariance
  '''
  X = X.copy()
  X = X - X.mean(axis=0)[np.newaxis,:]
  stds = np.sqrt(np.var(X, axis=0))[np.newaxis,:]
  stds[np.where(stds == 0)] = 1.0
  X = X / stds
  return X

def get_percentile_calibration(true_params, bs_runs,
                                 interval_coverage=90):
  '''
  Currently checks percentile estimates over each dimension of the parameters
  independently (so we only have to compute precentiles over 1-D things)

  true_params should be a D dimensional array
  bootstrap_samples should be a list of B x D arrays, where B is the number of
    bootstrap samples.
  interval_coverage specifies the size of the interval around the median;
    i.e. 95 corresponds to the interval [2.5%, 97.5%]
  '''
  D = true_params.shape[0]
  nExp = len(bs_runs)
  in_range = np.zeros((nExp,D), dtype=np.bool)
  tail = (100-interval_coverage)/2.0

  for n in range(nExp):
    lower = np.percentile(bs_runs[n]['multinomial']['bootstrap_params_appx'], tail, axis=0)
    upper = np.percentile(bs_runs[n]['multinomial']['bootstrap_params_appx'], 100-tail, axis=0)
    for d in range(D):
      in_range[n,d] = (lower[d] < true_params[d]) and (true_params[d] < upper[d])
  return in_range

def get_predictive_percentile_calibration(runs, percentile,
                                          method='exact', max_Ntest=50):
  model = models.PoissonRegressionModel(None,
                                        None,
                                        example_weights=None,
                                        test_data=None)
  in_interval = []
  for run in runs:
    if method == 'exact':
      bootstrap_samples = run['multinomial']['bootstrap_params_exact']
    elif method == 'appx':
      bootstrap_samples = run['multinomial']['bootstrap_params_appx']

    test_data = run['test_data']
    test_data.Y = test_data.Y[:max_Ntest]
    test_data.X = test_data.X[:max_Ntest]
    test_data.N = test_data.X.shape[0]
    model.test_data = test_data
    Ntest = test_data.N
    
    for n in range(Ntest):
      sampled_ys = model.get_predictive_distribution(bootstrap_samples,
                                                     model.test_data.X[n],
                                                     Nsamples=100)
      true_y = model.test_data.Y[n]
      lower = np.percentile(sampled_ys, tail)
      upper = np.percentile(sampled_ys, 100-tail)

      in_interval_for_run[n] = (lower <= true_y) and (true_y <= upper)
    in_interval.append(in_interval_for_run)
  return in_interval

def get_likelihood_based_interval(thetas, interval_coverage, model):
  '''
  thetas should be a list of free parameters for the model
  q should be in [0,100]

  Returns list of thetas, starting with lowest model.eval_objective(theta)
  up to the qth percentile.
  '''
  #fn_vals = [model.eval_objective(theta) for theta in thetas]
  #sorted_inds = np.array(np.argsort(fn_vals))
  #thresh = int(np.floor(thetas.shape[0]*(q/100.0)))
  #return thetas[sorted_inds[:thresh]], np.array(fn_vals)[sorted_inds]

  fn_vals = np.array([model.eval_objective(theta) for theta in thetas])

  upper = np.percentile(fn_vals, interval_coverage+(100-interval_coverage)/2)
  lower = np.percentile(fn_vals, (100-interval_coverage)/2)
  inds = np.where(np.logical_and(lower <= fn_vals, fn_vals <= upper))
  return fn_vals[inds]
  
def get_percentile_calibration_likelihood(filenames, model,
                                          interval_coverage=90,
                                          method='IJ',
                                          inds=None):
  '''
  Currently checks percentile estimates over each dimension of the parameters
  independently (so we only have to compute precentiles over 1-D things)

  true_params should be a D dimensional array

  interval_coverage specifies the size of the interval around the median;
    i.e. 95 corresponds to the interval [2.5%, 97.5%]
  '''
  nExp = len(filenames)
  in_range = np.zeros(nExp, dtype=np.bool)
  for n in range(nExp):
    bs_run = load_run(filenames[n])
    true_params = bs_run['multinomial']['truth']['theta']
    
    if method == 'linear_approx':
      bs_samples = bs_run['multinomial']['bootstrap_params_appx'].copy()
    elif method == 'exact':
      bs_samples = bs_run['multinomial']['bootstrap_params_exact'].copy()

    model.training_data = bs_run['train_data']

    if inds is not None:
      fixed_inds = np.setdiff1d(np.arange(true_params.shape[0]), inds)
      bs_samples[:,fixed_inds] = np.mean(bs_samples, axis=0)[fixed_inds]

    true_like = model.eval_objective(true_params)
    likelihood_interval = get_likelihood_based_interval(bs_samples,
                                                        interval_coverage,
                                                        model)
    if (likelihood_interval.min() <= true_like and
        true_like <= likelihood_interval.max()):
      in_range[n] = True
    else:
      in_range[n] = False
      

    '''
    if inds is None:
      conv_hull = scipy.spatial.Delaunay(params_in_interval)
      in_range[n] = conv_hull.find_simplex(true_params) > 0
    else:
      if len(inds) > 1:
        conv_hull = scipy.spatial.Delaunay(params_in_interval[:,inds])
        in_range[n] = conv_hull.find_simplex(true_params[inds]) > 0
      else:
        in_range[n] = ((params_in_interval[:,inds].min() < true_params[inds][0]) and
                       (params_in_interval[:,inds].max() > true_params[inds][0]))

    '''
  return in_range


def get_percentile_calibration_normal(true_params, model, bs_runs,
                                          percentile=90):
  K = true_params.shape[0]
  mahalanobis_thresh = scipy.stats.chi2.isf(1-percentile/100., df=K)
  
  in_range = np.zeros(len(bs_runs), dtype=np.bool)

  for rr, run in enumerate(bs_runs):
    model.training_data = run['train_data']
    model.params.set_free(run['multinomial']['w0'])
    model.example_weights = np.ones(run['train_data'].X.shape[0])
    
    cov = lunch_estimator(model) / model.training_data.X.shape[0]
    precision = np.linalg.inv(cov)
    params0 = model.params.get_free()
    mahalanobis = (true_params-params0).T.dot(precision).dot(true_params-params0)
    in_range[rr] = mahalanobis < mahalanobis_thresh
  return in_range
  
def lunch_estimator(model):
  '''
  Huber sandwich estimator
  '''
  X = model.training_data.X.copy()
  Y = model.training_data.Y.copy()
  N = X.shape[0]
  D = model.params.get_free().shape[0]

  gradTgrad = np.zeros((D,D))
  grad_sum = np.zeros(D)
  hess_sum = np.zeros((D,D))
  w0 = model.params.get_free().copy()

  for n in range(N):
    model.training_data.X = X[n].reshape(1,-1)
    model.training_data.Y = Y[n].reshape(1,-1)
    grad_n = autograd.jacobian(model.eval_objective)(model.params.get_free())
    model.params.set_free(w0)
    gradTgrad += np.outer(grad_n, grad_n)
    grad_sum += grad_n
    hess_sum += autograd.hessian(model.eval_objective)(model.params.get_free())
    model.params.set_free(w0)
    
  est_fisher = np.linalg.inv(hess_sum / N)
  est_score_score = (gradTgrad - np.outer(grad_sum, grad_sum)) / N
  model.training_data.X = X
  model.training_data.Y = Y
  return est_fisher.dot(est_score_score).dot(est_fisher)

def gen_low_rank_X(N, D, rank,
                   basis_vectors=None, lowRankNoise=0.0, rotate=True):
  import scipy.stats
  rank = min(rank,D)
  
  if basis_vectors is None:
    # Generate random set of D-dimensional orthonormal vectors
    basis_vectors = np.zeros((rank,D))
    if rotate:
      U = scipy.stats.special_ortho_group.rvs(D)
    else:
      U = np.eye(D)
    for r in range(rank):
      basis_vectors[r,r] = 1.0
      basis_vectors[r] = U.dot(basis_vectors[r])

  coeffs = np.random.normal(size=(N,rank), scale=np.sqrt(D/rank))
  #coeffs = np.random.normal(size=(N,rank), scale=1.0)

  if lowRankNoise == 0.0:
    return coeffs.dot(basis_vectors), basis_vectors
  else:
    noise = np.random.normal(size=(N,D), scale=lowRankNoise)
    return coeffs.dot(basis_vectors) + noise, basis_vectors
