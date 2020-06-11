'''
Most stuff in here inherits from GeneralizedLinearModel (really this doesn't
  necessarily have to be a GLM..), which provides functionality of
  retrain_with_weights() and fit().

Inheriting classes like PoissonRegressionModel mainly need to provide
  eval_objective()
'''
import autograd.numpy as np
import copy
import scipy
import autograd
import scipy
import time
from math import sqrt
from scipy import linalg

#import LinearResponseVariationalBayes as vb
import utils
from operator import mul
import fitL1

from collections import namedtuple

def soft_thresh(x, l):
  return np.sign(x) * np.maximum(np.abs(x) - l, 0.)


class GeneralizedLinearModel(object):
  def __init__(self, training_data, params_init, example_weights,
               test_data=None, regularization=None):
    if regularization is None:
      regularization = lambda x: 0.0
    self.regularization = regularization

    self.training_data = training_data
    self.test_data = test_data
    self.example_weights = copy.deepcopy(example_weights)

    self.params = copy.deepcopy(params_init)

    self.dParams_dWeights = None
    self.dParams_dWeightsProx = None
    self.dgParams_dWeights = {}
    self.is_a_glm = True

    self.L2Lambda = None


  def eval_objective(self, free_params):
    pass

  def get_params(self):
    return self.params.get_free()

  def fit(self, warm_start=True, label=None, save=False,
          use_fit_L1=False, tol=1e-15, **kwargs):
    '''
    Trains the model
    '''
    if False and hasattr(self, 'objective_gradient'):
      eval_objective_grad = self.objective_gradient
    else:
      eval_objective_grad = autograd.grad(self.eval_objective)

    eval_objective = lambda theta: self.eval_objective(theta)
    eval_objective_hess = autograd.hessian(self.eval_objective)
    eval_objective_hvp = autograd.hessian_vector_product(self.eval_objective)

    #if not warm_start:
    #  self.params.set_free(np.random.normal(size=self.training_data.D+1))
    if use_fit_L1:
      fitL1.fit_L1(self, **kwargs)
    else:
      opt_res = scipy.optimize.minimize(eval_objective,
                                        jac=eval_objective_grad,
                                        hessp=eval_objective_hvp,
                                        hess=eval_objective_hess,
                                        x0=copy.deepcopy(self.params.get_free()),
                                        method='trust-ncg',
                                        tol=tol,
                                        options={
                                          'initial_trust_radius':0.1,
                                          'max_trust_radius':1,
                                          'gtol':tol,
                                          'disp':False,
                                          'maxiter':100000
                                        })

      self.params.set_free(opt_res.x)
      if np.linalg.norm(opt_res.jac) > .01:
        print('Got grad norm', np.linalg.norm(opt_res.jac))





  # TODO: can we rewrite to avoid rewriting the instance var each time?
  def weighted_model_objective(self, example_weights, free_params):
    ''' The actual objective that we differentiate '''
    self.example_weights = example_weights
    return self.eval_objective(free_params)

  def compute_gradients(self, weights):
    if self.is_a_glm:
      self.compute_derivs()
      grads = (self.D1 * weights)[np.newaxis,:] * self.training_data.X.copy().T
    else:
      dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
      d2Obj_dParamsdWeights = autograd.jacobian(dObj_dParams, argnum=0)
      array_box_go_away = self.params.get_free().copy()
      cur_weights = self.example_weights.copy()

      grads = d2Obj_dParamsdWeights(some_example_weights,
                                    self.params.get_free())
      self.params.set_free(array_box_go_away)
    return grads

  def compute_dParams_dWeights(self, some_example_weights,
                               non_fixed_dims=None, rank=-1, **kwargs):
    '''
    sets self.jacobian = dParams_dxn for each datapoint x_n
    rank = -1 uses a full-rank matrix solve (i.e. np.linalg.solve on the full
      Hessian). A positive integer uses a low rank approximation in
      inverse_hessian_vector_product

    '''
    if non_fixed_dims is None:
      non_fixed_dims = np.arange(self.params.get_free().shape[0])
    if len(non_fixed_dims) == 0:
      self.dParams_dWeights = np.zeros((0,some_example_weights.shape[0]))
      return

    dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
    d2Obj_dParams2 = autograd.jacobian(dObj_dParams, argnum=1)
    d2Obj_dParamsdWeights = autograd.jacobian(dObj_dParams, argnum=0)

    # Have to re-copy this into self.params after every autograd call, as
    #  autograd turns self.params.get_free() into an ArrayBox (whereas we want
    #  it to be a numpy array)
    #array_box_go_away = self.params.get_free().copy()
    #cur_weights = self.example_weights.copy()

    start = time.time()
    grads = self.compute_gradients(some_example_weights)

    self.dParams_dWeights = -self.inverse_hessian_vector_product(grads,
                                                non_fixed_dims=non_fixed_dims,
                                                rank=rank).T
    #self.params.set_free(array_box_go_away)
    #self.example_weights = cur_weights
    self.non_fixed_dims = non_fixed_dims

  def compute_dParams_dWeightsProx(self, some_example_weights, **kwargs):
    '''
    Stores how parameters change when example weights change for Prox
    '''
    grads = self.compute_gradients(some_example_weights)

    self.dParams_dWeightsProx = self.prox_newton_step(grads).T

  def prox_newton_step(self,grad):
    '''
    Returns transpose of matrix with 
      columns = -bhat + argmin_b <grad loss_{-i}, b - bhat> + (1/2) (b-bhat) H_i (b-bhat) + lam||b||_1
              = -bhat + argmin_b <grad loss_{-i} - H_i*bhat, b> + (1/2) b H_i b + lam||b||_1
      where bhat is the training estimator and H is the leave-one-out Hessian evaluated at bhat. 
    Note the returned matrix will be of size grad[non_fixed_dims,:].shape
    '''
    X = self.training_data.X[:,:-1]
    D = X.shape[1]
    N = X.shape[0]
    print("D = {}, N = {}".format(D, N))
    print("grad.shape = {}".format(grad.shape))
    self.compute_derivs()
    Xfull = self.training_data.X
    H_SS = Xfull.T.dot(Xfull*self.D2[:,np.newaxis])
    # Minimize <g, b> + (1/2) b H b + lam||b||_1 
    # with g = grad loss_{-i} - H*bhat 
    # for bhat the training estimator
    params = self.params.get_free()
    Hparams = H_SS.dot(params)
    # Form grad loss_{-i} by summing over all other datapoint gradients
    # Warm start optimization at bhat
    return np.column_stack([self.fista(H_SS - np.outer(Xfull[i],Xfull[i]) * self.D2[i], 
                                       np.delete(grad,i,axis=1).sum(axis=1) - Hparams + Xfull[i] * Xfull[i].dot(params) * self.D2[i],
                                       self.L1Lambda, 300, params) - params 
                            for i in range(grad.shape[1])]).T

  def fista(self, H, g, lam, maxit, x0 = None):
    # Returns argmin_b <g, b> + (1/2) b H b + lam||b||_1
    # 
    # Args:
    #   x0 - optional initial value of optimization vector
    start = time.time()
    if x0 is None:
      x = np.zeros(H.shape[1])
    else:
      x = x0.copy()
    pobj = []
    t = 1
    z = x.copy()
    L = np.linalg.norm(H)
    for iter in range(int(maxit)):
      xold = x.copy()
      z = z - (g + H.dot(z))/L
      x = soft_thresh(z, lam / L)
      t0 = t
      t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
      z = x + ((t0 - 1.) / t) * (x - xold)
      this_pobj = 0.5 * (H.dot(x).dot(x)) + g.dot(x) + lam * linalg.norm(x, 1)
      if iter % 100 == 0:
        print(this_pobj)
      #pobj.append(this_pobj))
      #pobj.append((time.time() - time0, this_pobj))
    print("Fit fista in {}s".format(time.time() - start))
    #times, pobj = map(np.array, zip(*pobj))
    return x


  def compute_dgParams_dWeights(self, g, gID,
                                hessian_scaling=10000.0, S1=None, S2=None):
    '''
    Poorly named function that computes \frac{d g(\thetaw)}{d w} for some
      specific real-valued function g(theta)
    The derivatives are evaluated at the passed in weights and theta = whatever
      the current params are
    gID is whatever you want g to be called; model.dgParams_dWeights[gID] will
      store the derivatives that are computed here
    '''
    theta = self.params.get_free().copy()
    dg_dtheta = autograd.grad(g)
    dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
    d2Obj_dParamsdWeights = autograd.jacobian(dObj_dParams, argnum=0)

    cur_weights = self.example_weights.copy()
    d2Obj = d2Obj_dParamsdWeights(self.example_weights, self.params.get_free())
    self.example_weights = cur_weights
    dg = dg_dtheta(theta)
    self.params.set_free(theta)

    dgTimesHinv = self.hessian_inverse_vector_product(dg,
                                                  method='stochastic',
                                                      hessian_scaling=hessian_scaling,
                                                      S1=S1,
                                                      S2=S2)
    self.dgParams_dWeights[gID] = -dgTimesHinv.dot(d2Obj)



  def retrain_with_weights(self, new_example_weights,
                           doIJAppx=False, doNSAppx=False, doProxAppx = False,
                           label=None,
                           gs=None, gIDs=None,
                           is_cv=False,
                           non_fixed_dims=None,
                           **kwargs):
    '''
    in_place: updates weights and params based on the new data

    If you've pre-computed dgParams_dWeights for various functions g, you can
      pass them in as a list (as well as their gID values) to only "retrain"
      these functions of the parameters.

    Can do things a bit more efficiently if it's cross-validation; you actually
    don't need to multiply (KxN) times (Nx1) vector; just select the components
    that have been left out
    '''
    if doIJAppx: # i.e. infinitesimal jackknife approx
      delta_example_weights = new_example_weights - self.example_weights
      if is_cv and False:
        left_out_inds = np.where(delta_example_weights == 0)
        new_params = self.params.get_free()
        new_params += self.dParams_dWeights[:,left_out_inds].sum(axis=1)
      else:
        if non_fixed_dims is None:
          new_params = self.params.get_free() - self.dParams_dWeights.dot(
            delta_example_weights)
        else:
          new_params = self.params.get_free()
          new_params[non_fixed_dims] += self.dParams_dWeights.dot(
            delta_example_weights)
    elif doProxAppx:
      delta_example_weights = new_example_weights - self.example_weights
      new_params = self.params.get_free() - self.dParams_dWeightsProx.dot(
        delta_example_weights)
    elif doNSAppx: # i.e., Newton step based approx
      if is_cv and self.is_a_glm: # Can do rank-1 update
        n = np.where(new_example_weights != 1)[0]
        new_params = self.params.get_free().copy()
        new_params[non_fixed_dims] += self.loocv_rank_one_updates[n,:].squeeze()
      else:
        self.compute_dParams_dWeights(new_example_weights,
                                      non_fixed_dims=non_fixed_dims)
        delta_example_weights = new_example_weights - self.example_weights
        new_params = self.params.get_free().copy()
        new_params[non_fixed_dims] += self.dParams_dWeights.dot(
          delta_example_weights)
    else: # non-approximate: re-fit the model
      curr_params = copy.copy(self.params.get_free())
      curr_example_weights = copy.copy(self.example_weights)
      self.example_weights = new_example_weights
      self.fit(**kwargs)
      new_fit_params = copy.copy(self.params.get_free())

      if gs is None:
        new_params = new_fit_params
      else:
        for idx, g in enumerate(gs):
          new_params[idx] = g(new_fit_params)

    #if not in_place:
    #  self.example_weights = curr_example_weights
    #  self.params.set_free(curr_params)
    #else:
    #  self.params.set_free(new_params)
    self.params.set_free(new_params)

    return new_params

  def predict_probability(self, X):
    pass

  def get_error(self, test_data, metric):
    pass

  def get_single_datapoint_hessian(self, n):
    X = self.training_data.X
    Y = self.training_data.Y
    weights = self.example_weights
    self.training_data.X = X[n].reshape(1,-1)
    self.training_data.Y = Y[n]
    self.example_weights = np.ones(1)
    array_box_go_away = copy.copy(self.params.get_free())
    dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
    d2Obj_dParams2 = autograd.jacobian(dObj_dParams, argnum=1)
    hess_n = d2Obj_dParams2(self.example_weights, self.params.get_free())
    self.params.set_free(array_box_go_away)

    self.training_data.X = X
    self.training_data.Y = Y
    self.example_weights = weights
    return hess_n

  def get_single_datapoint_hvp(self, n, vec):
    '''
    Returns Hessian.dot(vec), where the Hessian is the Hessian of the objective
       function with just datapoint n
    '''
    X = self.training_data.X
    Y = self.training_data.Y
    weights = self.example_weights
    self.training_data.X = X[n].reshape(1,-1)
    self.training_data.Y = Y[n]
    self.example_weights = np.ones(1)

    array_box_go_away = copy.copy(self.params.get_free())
    eval_hvp = autograd.hessian_vector_product(self.weighted_model_objective,
                                               argnum=1)
    hess_n_dot_vec = eval_hvp(self.example_weights, self.params.get_free(), vec)

    self.params.set_free(array_box_go_away)
    self.training_data.X = X
    self.training_data.Y = Y
    self.example_weights = weights
    return hess_n_dot_vec

  def get_all_data_hvp(self, vec):
    '''
    Returns Hessian.dot(vec), where the Hessian is the Hessian of the objective
       function with all the data.
    '''
    array_box_go_away = copy.copy(self.params.get_free())
    eval_hvp = autograd.hessian_vector_product(self.weighted_model_objective,
                                               argnum=1)
    hvp = eval_hvp(self.example_weights, self.params.get_free(), vec)

    self.params.set_free(array_box_go_away)
    return hvp

  def compute_hessian(self):
    dObj_dParams = autograd.jacobian(self.weighted_model_objective, argnum=1)
    d2Obj_dParams2 = autograd.jacobian(dObj_dParams, argnum=1)
    array_box_go_away = self.params.get_free().copy()
    hessian = d2Obj_dParams2(self.example_weights, self.params.get_free())
    self.params.set_free(array_box_go_away)
    self.hessian = hessian

  def compute_restricted_hessian_and_dParamsdWeights(self, dims, weights,
                                                     comp_dParams_dWeights=True):
    '''
    Computes the dims.shape[0] by dims.shape[0] Hessian only along the entries
    in dims (used when using l_1 regularization)
    '''
    theta0 = self.params.get_free()

    # Objective to differentiate just along the dimensions specified
    def lowDimObj(weights, thetaOnDims, thetaOffDims, invPerm):
      allDims = np.append(dims, offDims)
      thetaFull = np.append(thetaOnDims, thetaOffDims)[invPerm]
      return self.weighted_model_objective(weights, thetaFull)

    offDims = np.setdiff1d(np.arange(self.params.get_free().shape[0]), dims)
    thetaOnDims = theta0[dims]
    thetaOffDims = theta0[offDims]

    # lowDimObj will concatenate thetaOnDims, thetaOffDims, then needs to
    #  un-permute them into the original theta.
    allDims = np.append(dims, offDims)
    invPerm = np.zeros(theta0.shape[0], dtype=np.int32)
    for i, idx in enumerate(allDims):
      invPerm[idx] = i

    evalHess = autograd.hessian(lowDimObj, argnum=1)
    array_box_go_away = self.params.get_free().copy()

    restricted_hess = evalHess(weights,
                               thetaOnDims,
                               thetaOffDims,
                               invPerm)
    self.params.set_free(theta0)

    dObj_dParams = autograd.jacobian(lowDimObj, argnum=1)
    d2Obj_dParamsdWeights = autograd.jacobian(dObj_dParams, argnum=0)

    if comp_dParams_dWeights:
      restricted_dParamsdWeights = d2Obj_dParamsdWeights(weights,
                                                         thetaOnDims,
                                                         thetaOffDims,
                                                         invPerm)
      return restricted_hess, restricted_dParamsdWeights
    else:
      return restricted_hess



  def hessian_inverse_vector_product(self, vec, hessian_scaling,
                                     S1=None, S2=None, method='stochastic'):
    '''
    From Agarwal et. al. "Second-order stochastic optimization for machine
       learning in linear time." 2017.

    Not clear that this provides good accuracy in a reasonable amount of time.
    '''
    N = self.training_data.X.shape[0]
    D = vec.shape[0]
    if S1 is None and S2 is None:
      S1 = int(np.ceil(np.sqrt(N)/10))
      S2 = int(np.ceil(10*np.sqrt(N)))

    hivpEsts = np.zeros((S1,D))
    for ii in range(S1):
      hivpEsts[ii] = vec
      for n in range(1,S2):
        idx = np.random.choice(N)
        #H_n_prod_prev = self.get_single_datapoint_hvp(idx, hivpEsts[ii]) * N
        #H_n_prod_prev /= hessian_scaling
        H_n_prod_prev = self.get_all_data_hvp(hivpEsts[ii]) / hessian_scaling
        hivpEsts[ii] = vec + hivpEsts[ii] - H_n_prod_prev
    return np.mean(hivpEsts, axis=0) / hessian_scaling

  def stochastic_hessian_inverse(self, hessian_scaling, S1=None, S2=None):
    '''
    From Agarwal et. al. "Second-order stochastic optimization for machine
       learning in linear time." 2017.

    Not clear that this provides good accuracy in a reasonable amount of time.
    '''
    self.compute_derivs()
    X = self.training_data.X
    N = self.training_data.X.shape[0]
    D = self.params.get_free().shape[0]
    if S1 is None and S2 is None:
      S1 = int(np.sqrt(N)/10)
      S2 = int(10*np.sqrt(N))

    if self.regularization is not None:
      evalRegHess = autograd.hessian(self.regularization)
      paramsCpy = self.params.get_free().copy()
      regHess = evalRegHess(self.params.get_free())
      regHess[-1,-1] = 0.0
      self.params.set_free(paramsCpy)

    hinvEsts = np.zeros((S1,D,D))
    for ii in range(S1):
      hinvEsts[ii] = np.eye(D)
      for n in range(1,S2):
        idx = np.random.choice(N)
        H_n = np.outer(X[idx],X[idx]) * self.D2[idx] * N + regHess

        if np.linalg.norm(H_n) >= hessian_scaling*0.9999:
          from IPython import embed; np.set_printoptions(linewidth=150); embed()
          print(np.linalg.norm(H_n))
        #H_n = self.get_single_datapoint_hessian(idx) * N
        H_n /= hessian_scaling
        hinvEsts[ii] = np.eye(D) + (np.eye(D) - H_n).dot(hinvEsts[ii])
    return np.mean(hinvEsts, axis=0) / hessian_scaling




  def inverse_hessian_vector_product(self, b, rank=-1, non_fixed_dims=None):
    '''
    Returns transpose of matrix with columns H^{-1}.dot(b[:,i]), where H is the Hessian
      evaluated at the current model.params.get_free(). Note the returned
      vectors will be of size b[non_fixed_dims,:].shape
    If rank is given as an integer, uses special code under the assumption
      that the model is a GLM with low rank. **Note** that this appraoch assumes
      that the regularizer is model.L2Lambda*np.linalg.norm(theta[:-1])**2
      (i.e., l2 regularization with lambda > 0, but no regularization on the
      bias term)
    Method for low-rank is from Tropp et. al 2017, "Fixed-rank approximation of
      a positive-semidefinite matrix from streaming data"
    '''
    X = self.training_data.X[:,:-1]
    D = X.shape[1]
    N = X.shape[0]
    self.compute_derivs()
    if rank == -1:
      print('GOING FULL')

      start = time.time()
      if non_fixed_dims.shape[0] < D or self.L2Lambda is None:
        H_SS = \
          self.compute_restricted_hessian_and_dParamsdWeights(non_fixed_dims,
                                                    np.ones(N),
                                                    comp_dParams_dWeights=False)

        hivps = np.linalg.solve(H_SS, b[non_fixed_dims,:]).T
      else:
        Xfull = self.training_data.X
        H_SS = Xfull.T.dot(Xfull*self.D2[:,np.newaxis])
        l2Hessian = np.eye(D+1) * self.L2Lambda*2
        l2Hessian[-1,-1] = 0.0
        H_SS += l2Hessian
        hivps = np.linalg.solve(H_SS, b).T
    elif rank > 0:
      print('GOING LOW')
      K = min(rank + 2, D)

      # Sketching-based approach to finding SVD of X.T.dot(np.diag(D2)).dot(X)
      omega = np.linalg.qr(np.random.normal(size=(D,K)))[0]
      sketch = np.zeros((D,K))
      for n in range(N):
        sketch += np.outer(X[n] * self.D2[n], X[n].T.dot(omega))
      nu = 1e-10 * np.linalg.norm(sketch.ravel())
      sketch += nu * omega
      B = omega.T.dot(sketch)
      C = np.linalg.cholesky((B + B.T)/2)
      E = np.linalg.solve(C, sketch.T).T # == sketch.dot(np.linalg.inv(C.T))
      U, S, V = np.linalg.svd(E,
                              full_matrices=False) #E = U.dot(np.diag(S)).dot(V)
      # A := U.dot(np.diag(S**2)).dot(U.T) ~= X.T.dot(np.diag(self.D2)).dot(X)
      U = U[:,:K]

      # Solve (U.dot(S**2).dot(U.T) + lam*np.eye(D)) without explicitly forming
      #  U.dot(S**2).dot(U.T) (which would cost D^2 memory and time)
      z = (X * self.D2[:,np.newaxis]).sum(axis=0)
      D2 = self.D2.sum()
      lam = 2*self.L2Lambda
      Ainvz = z/lam - np.dot(U, np.dot(np.diag(1.0 / (lam + lam**2/(S**2 - nu))),
                                 U.T.dot(z)))
      Ainvz = np.tile(Ainvz.reshape(-1,1), (1,b.shape[1]))
      Ainvb = b[:-1]/lam - np.dot(U, np.dot(np.diag(1.0 / (lam + lam**2/(S**2 - nu))),
                                      U.T.dot(b[:-1])))
      E = D2 - np.dot(Ainvz.T, z)
      hivps = np.append(Ainvb + Ainvz * (z.T.dot(Ainvb)/E)[np.newaxis,:] - Ainvz*b[-1]/E,
                        ((-z.T.dot(Ainvb) + b[-1])/E).reshape(1,-1),
                        axis=0).T

      #Aappx = U.dot(np.diag(S**2 - nu)).dot(U.T)
      #regDiag = np.eye(D) * lam
      ##regDiag[-1,-1] = 0.0
      #Happx = Aappx + regDiag
      #Happx = np.append(Happx, z.reshape(-1,1), axis=1)
      #Happx = np.append(Happx, np.append(z, D2).reshape(1,-1), axis=0)
      #hivpsExact = np.linalg.solve(Happx, b)
    return hivps

  def compute_loocv_rank_one_updates(self, non_fixed_dims=None, rank=-1,
                                     **kwargs):
    '''
    When the model is a GLM and you're doing approximate LOOCV with the Newton
    step approximation, rank one matrix inverse updates allow you to use only
    O(D^3), rather than O(ND^3) computation.
    '''
    X = self.training_data.X
    N = X.shape[0]

    if non_fixed_dims is None:
      non_fixed_dims = np.arange(self.params.get_free().shape[0])
    if len(non_fixed_dims) == 0:
      self.loocv_rank_one_updates = np.zeros((N,0))
      return

    X_S = X[:,non_fixed_dims]
    hivps = self.inverse_hessian_vector_product(X.T, rank=rank,
                                                non_fixed_dims=non_fixed_dims)
    X_S_hivps = np.einsum('nd,nd->n', X_S, hivps)
    updates = 1 + (self.D2 * X_S_hivps) / (1 - self.D2 * X_S_hivps)
    self.loocv_rank_one_updates = (updates*self.D1)[:,np.newaxis] * hivps


class PoissonRegressionModel(GeneralizedLinearModel):
  '''
  Poisson regression with:
           y_n ~ Poi( log(1 + e^{<x_n, \theta>}) )
  '''
  def __init__(self, *args, **kwargs):
    super(PoissonRegressionModel, self).__init__(*args, **kwargs)

  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params = self.params['w'].get()
    Y = self.training_data.Y
    params_x = np.dot(self.training_data.X, params)
    M = np.maximum(params_x, 0.0)
    lam = np.log(np.exp(0-M) + np.exp(params_x-M)) + M
    ret = Y*np.log(lam + 1e-15) - lam
    ll = (-(ret*self.example_weights).sum())
    return ll + self.regularization(params[:-1])
      
  def eval_loss(self,free_params):
    self.params.set_free(free_params)
    params = self.params['w'].get()
    Y = self.training_data.Y
    params_x = np.dot(self.training_data.X, params)
    M = np.maximum(params_x, 0.0)
    lam = np.log(np.exp(0-M) + np.exp(params_x-M)) + M
    ret = Y*np.log(lam + 1e-15) - lam
    ll = (-(ret*self.example_weights).sum())
    return self.regularization(params[:-1])

  def get_error(self, test_data, metric="mse"):
    if metric == "mse":
      X = test_data.X
      Y = test_data.Y
      params = self.params['w'].get()
      params_x = np.dot(X, params)
      stacked = np.stack([params_x, np.zeros(params_x.shape[0])], axis=0)
      lam = scipy.special.logsumexp(stacked, axis=0)
      Yhat = lam
      return np.mean((Yhat-Y)**2)


class ExponentialPoissonRegressionModel(GeneralizedLinearModel):
  '''
  Poisson regression with:
           y_n ~ Poi( e^{<x_n, \theta>} )
  '''
  def __init__(self, *args, **kwargs):
    self.L1Lambda = None
    super(ExponentialPoissonRegressionModel, self).__init__(*args, **kwargs)

  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params = self.params['w'].get()
    Y = self.training_data.Y
    params_x_bias = np.dot(self.training_data.X, params)
    ret = Y*params_x_bias - np.exp(params_x_bias)
    ll = (-(ret*self.example_weights).sum())

    return ll + self.regularization(params[:-1])

  def fit(self, warm_start=True, label=None, save=False,
          use_glmnet=False, use_cvxpy=False, cvxpy_tol=1e-4,
          **kwargs):
    '''
    Note: use_cvxpy or use_glmnet only works with CV weights (i.e. all 0 or 1)
    '''
    if not use_cvxpy and not use_glmnet:
      super(ExponentialPoissonRegressionModel, self).fit(warm_start,
                                                         label,
                                                         save,
                                                         **kwargs)

    elif use_glmnet:
      from glmnet_py import glmnet
      lambdau = np.array([self.L1Lambda,])
      inds = self.example_weights.astype(np.bool)
      x = self.training_data.X[inds,:].copy()
      y = self.training_data.Y[inds].copy().astype(np.float)
      fit = glmnet(x=x,
                   y=y,
                   family='poisson',
                   standardize=False,
                   lambdau=lambdau,
                   thresh=1e-20,
                   maxit=10e4,
                   alpha=1.0,
      )

    elif use_cvxpy:
      import cvxpy
      D = self.params.get_free().shape[0]
      Ntrain = self.training_data.X.shape[0]
      theta = cvxpy.Variable(D, value=self.params.get_free())
      weights = self.example_weights

      X = self.training_data.X
      Y = self.training_data.Y
      obj = cvxpy.Minimize(-Y*cvxpy.multiply(weights,(X*(theta)))
         + cvxpy.sum(cvxpy.multiply(weights, cvxpy.exp(X*(theta))))
         + self.L1Lambda * cvxpy.sum(cvxpy.abs((theta)[:-1]))
      )
      problem = cvxpy.Problem(obj)
      problem.solve(solver=cvxpy.SCS, normalize=True,
                    eps=cvxpy_tol, verbose=False, max_iters=2000)
      if (problem.status == 'infeasible_inaccurate' or
          problem.status == 'unbounded_inaccurate'):
        problem.solve(solver=cvxpy.SCS, normalize=True,
                      eps=cvxpy_tol, verbose=False, max_iters=2000)
        problem.solve(solver=cvxpy.SCS, normalize=True,
                      eps=cvxpy_tol, verbose=False, max_iters=10000)
      try:
        self.params.set_free(theta.value)
      except:
        print('Bad problem?', problem.status)

  def compute_derivs(self):
    '''
    For use from fitL1.py.
    '''
    Y = self.training_data.Y
    params = self.params.get_free()
    #exp_params_X = np.exp(self.training_data.X.dot(self.params['w'].get()))
    exp_params_X = np.exp(self.training_data.X.dot(params))
    self.D1 = -(Y - exp_params_X)
    self.D2 = -(-exp_params_X)

  def get_error(self, test_data, metric="mse"):
    if metric == "mse":
      X = test_data.X
      Y = test_data.Y
      params = self.params['w'].get()
      params_x_bias = np.dot(X, params)
      lam = np.exp(params_x_bias)
      Yhat = lam
      return np.mean((Yhat-Y)**2)


class LogisticRegressionModel(GeneralizedLinearModel):

  def __init__(self, *args, **kwargs):
    super(LogisticRegressionModel, self).__init__(*args, **kwargs)

  def fit(self, warm_start=True, label=None, save=False,
          use_glmnet=False, use_cvxpy=False, cvxpy_tol=1e-4,
          **kwargs):
    '''
    Note: use_glmnet only works with CV weights (i.e. all 0 or 1)
    '''

    if not use_glmnet:
      super(LogisticRegressionModel, self).fit(warm_start,
                                               label,
                                               save,
                                               **kwargs)

    elif use_glmnet:
      from glmnet_py import glmnet
      lambdau = np.array([self.L1Lambda / self.training_data.X.shape[0],])
      inds = self.example_weights.astype(np.bool)
      x = self.training_data.X[inds,:-1].copy()
      y = self.training_data.Y[inds].copy().astype(np.float)
      y[np.where(y==-1)] = 0.0
      fit = glmnet(x=x,
                   y=y,
                   family='binomial',
                   standardize=True,
                   lambdau=lambdau,
                   thresh=1e-10,
                   maxit=10e3,
                   alpha=1.0,
      )
      self.params.set_free(np.append(fit['beta'], 0))
      return

  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params = self.params['w'].get()
    X = self.training_data.X
    Y = self.training_data.Y
    paramsXY = -Y * (np.dot(X, params))
    M = np.maximum(paramsXY, 0)
    log_likelihood = -(np.log(np.exp(0-M) + np.exp(paramsXY-M)) + M)
    return ( -(log_likelihood*self.example_weights).sum() +
             self.regularization(params[:-1]) )

 # def eval_loss(self, free_params):
#      self.params.set_free(free_params)
#      params = self.params['w'].get()
#      X = self.training_data.X
#      Y = self.training_data.Y
#      paramsXY = -Y * (np.dot(X, params))
#      M = np.maximum(paramsXY, 0)
#      log_likelihood = -(np.log(np.exp(0-M) + np.exp(paramsXY-M)) + M)
#      return ( -(log_likelihood*self.example_weights).sum() )

  def predict_probability(self, X):
    return utils.sigmoid(X, self.params.get_free())

  def predict_target(self, X):
    probs = self.predict_probability(X)
    probs[np.where(probs > .5)] = 1
    probs[np.where(probs <= .5)] = -1
    return probs

  def compute_derivs(self):
    '''
    For use from fitL1.py
    '''
    Y = self.training_data.Y
    params = self.params.get_free()
    exp_params_XY = np.exp(Y *
              self.training_data.X.dot(params))
    self.D1 = -Y/ (1 + exp_params_XY)
    self.D2 = -Y*self.D1 - (self.D1)**2

  def get_error(self, test_data, metric='log_likelihood'):
    if metric == "accuracy":
      # change Y_Test to 01 if required
      return 1.0 * np.where(
        self.predict_target(test_data.X) != test_data.Y)[0].shape[0] / test_data.N
    elif metric == 'log_likelihood':
      train_data = self.training_data
      weights = self.example_weights
      self.training_data = test_data
      self.example_weights = np.ones(test_data.X.shape[0])
      nll = self.eval_objective(self.params.get_free())
      nll -= self.regularization(self.params.get_free()[:-1])
      self.training_data = train_data
      return nll / test_data.X.shape[0]

class LinearRegressionModel(GeneralizedLinearModel):
  def __init__(self, *args, **kwargs):
    super(LinearRegressionModel, self).__init__(*args, **kwargs)

  def eval_objective(self, free_params):
    '''
    Objective that we minimize; \sum_n w_n f(x_n, \theta) + ||\theta||_2
    '''
    self.params.set_free(free_params)
    params = self.params['w'].get()
    params_x = np.dot(self.training_data.X, params)
    sq_error = (self.training_data.Y - params_x)**2 * self.example_weights
    return sq_error.sum() + self.regularization(params[:-1])

  def get_error(self, test_data, metric="mse"):
    if metric == "mse":
      Yhat = np.dot(test_data.X, self.params.get_free())
      Y = test_data.Y
      return np.mean((Yhat - Y)**2)

  def compute_derivs(self):
    Y = self.training_data.Y
    params_x = self.training_data.X.dot(self.params.get_free())
    self.D1 = -2*(Y - params_x)
    self.D2 = 2*np.ones(Y.shape[0])

  def fit(self, warm_start=True, label=None, save=False,
          use_glmnet=False, **kwargs):
    '''
    Note: this only works with CV weights (i.e. all 0 or 1)
    '''
    if not use_glmnet:
      super(LinearRegressionModel, self).fit(warm_start,
                                             label,
                                             save,
                                             **kwargs)

    elif use_glmnet:
      from glmnet_py import glmnet
      inds = self.example_weights.astype(np.bool)
      x = self.training_data.X[inds,:].copy()
      y = self.training_data.Y[inds].copy().astype(np.float)
      lambdau = np.array([self.L1Lambda/(2*x.shape[0]),])

      fit = glmnet(x=x[:,:-1],
                   y=y,
                   family='gaussian',
                   standardize=True,
                   lambdau=lambdau,
                   thresh=1e-10,
                   maxit=10e4,
                   alpha=1.0,
      )
      self.params.set_free(np.append(np.squeeze(fit['beta']), fit['a0']))

class ProbitRegressionModel(GeneralizedLinearModel):
  def __init__(self, *args, **kwargs):
    super(ProbitRegressionModel, self).__init__(*args, **kwargs)

  def eval_objective(self, free_params):
    self.params.set_free(free_params)
    params_no_bias = self.params['w'].get()[:-1]
    bias = self.params['w'].get()[-1]
    y_x_params = self.training_data.Y * (
      np.dot(self.training_data.X, params_no_bias) + bias)

    log_likelihood = \
                autograd.scipy.stats.norm.logcdf(y_x_params) * self.example_weights
    return -(log_likelihood).sum() + self.regularization(params_no_bias)

  def predict_probability(self, X):
    params_no_bias = self.params.get_free()[:-1]
    bias = self.params.get_free()[-1]
    return autograd.scipy.stats.norm.cdf(X.dot(params_no_bias) + bias)

  def predict_target(self, X):
    probs = self.predict_probability(X)
    probs[np.where(probs > .5)] = 1
    probs[np.where(probs <= .5)] = -1
    return probs

  def get_error(self, test_data, metric="log_likelihood"):
    if metric == "accuracy":
      # change Y_Test to 01 if required
      return np.where(
        self.predict_target(test_data.X) != test_data.Y)[0].shape[0] / test_data.N
    elif metric == 'log_likelihood':
      train_data = self.training_data
      weights = self.example_weights
      self.training_data = test_data
      self.example_weights = np.ones(test_data.X.shape[0])
      nll = self.eval_objective(self.params.get_free())
      self.training_data = train_data
      return nll / test_data.X.shape[0]
