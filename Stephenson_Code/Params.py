'''
Standin for LinearResponseVariationalBayes.py, which allows for transforming
  between constrained / unconstrained representations of parameters. Since
  we're just using unconstrained GLM parameters here so far, this doesn't
  really matter.
The reason for switching out is getting around the scipy / autograd dependencies   in LinearResponseVariationalBayes.

These classes are entirely useless. They just imitate the basic behavior we need
  without requiring scipy / autograd.
'''

import numpy as np

class Param(object):
  def __init__(self, size):
    self.w = SubParam(np.zeros(size))

  def __getitem__(self, key):
    return self.__dict__[key]

  def get_free(self):
    return np.copy(self.w.get())

  def set_free(self, val):
    assert(val.shape == self.w.get().shape)
    self.w.set(val)
    

class SubParam(object):
  def __init__(self, val):
    #self.val = np.copy(val)
    self.val = val

  def get(self):
    return self.val

  def set(self, val):
    #self.val = np.copy(val)
    self.val = val
