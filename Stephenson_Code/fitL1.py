
import numpy as np


def fit_L1(model, eps=1e-8, max_iters=7500, extra_precision=False,
           intermediate_output=None, do_active_set=True,
           **kwargs):
  if extra_precision:
    max_iters *= 4
    eps = 1e-14

  #import cProfile
  #pr = cProfile.Profile()
  #pr.enable()
  
  N = model.training_data.N
  D = model.training_data.D
  X = model.training_data.X
  weights = model.example_weights
  L1Lambda = model.L1Lambda
  theta = model.params.get_free()
  is_converged = False
  iters = 0

  active_set_on = False
  active_set = None
  active_set_converged = False

  while not is_converged:
    model.compute_derivs()
    XD1 = X * model.D1[:,np.newaxis]
    XD2 = X * model.D2[:,np.newaxis]
    varX_D2 = (X**2 * (model.D2 * weights)[:,np.newaxis]).sum(axis=0)
    theta_outer_old = theta.copy()
    prev_support = (theta_outer_old != 0)
    theta_old_X = X.dot(theta_outer_old)
    theta_old_X_XD1_XD2_weights = weights.dot(-theta_old_X[:,np.newaxis] * XD2 + XD1)
    XD2_weights = weights[:,np.newaxis] * XD2

    for coord_decent_iter in range(5):

      theta_inner_old = theta.copy()
      if active_set_on and active_set is not None:
        dims_to_iterate_over = active_set
      else:
        dims_to_iterate_over = range(D+1)
        active_set_converged = False

        
      inds = np.where(theta_inner_old != 0)[0]
      X_theta = X[:,inds].dot(theta_inner_old[inds])
      
      for d in dims_to_iterate_over:
        #theta_tmp = theta.copy()
        #theta_tmp[d] = 0
        #d_left_out = X[:,inds].dot(theta_tmp[inds])
        if theta[d] == 0.0:
          d_left_out = X_theta
        else:
          d_left_out = X_theta - X[:,d]*theta[d]
     
        numerator = ( (XD2_weights[:,d] * d_left_out).sum()  
                      + theta_old_X_XD1_XD2_weights[d] )
 
        if d < D: # Only first D params have L1 regularization (last one is bias)
          numerator_sign = np.sign(numerator)
          numerator = numerator_sign * np.maximum(0,
                                                  np.abs(numerator) - L1Lambda)
        theta[d] = -numerator / (varX_D2[d])
        
        if theta[d] != theta_inner_old[d]:
          inds = np.where(theta != 0)[0]
          X_theta = X[:,inds].dot(theta[inds])

          
      # Break out of coordinate descent and form a fresh quadratic approximation
      if np.linalg.norm(theta_inner_old - theta) <= 1e-8:
        break

    if iters % 10 == 0:
      print(iters, np.count_nonzero(theta))
    iters += 1
    model.params['w'].set(theta.copy())


    if intermediate_output is not None:
      np.savetxt(intermediate_output, theta)
      

    if np.linalg.norm(theta - theta_outer_old) <= eps:
      if not active_set_on:
        is_converged = True
        print('converged on', np.linalg.norm(theta-theta_outer_old))
      else:
        #print('converged on w/ active set', np.linalg.norm(theta-theta_outer_old),
        #     active_set, active_set_on)
        active_set_on = False
        active_set = None
        active_set_converged = True
    if iters > max_iters:
      print('\n\n Finishing on max iters', np.linalg.norm(theta-theta_outer_old), '\n\n')
      is_converged = True

      
    if (np.all(prev_support == (theta != 0)) and not active_set_converged
        and do_active_set):
      active_set_on = True
      active_set = np.where(theta != 0)[0]

  model.params.set_free(theta)
  #pr.disable()
