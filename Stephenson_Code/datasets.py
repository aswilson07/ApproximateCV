'''
Code for generating new datasets. Everything in here should inherit from
 the Dataset class, which really means it needs to provide a get_dataset() method.
'''


import numpy as np
import pandas as pd
import utils
from copy import deepcopy
from datetime import datetime
import random
import os
import itertools
import scipy.stats

class Dataset(object):
    def __init__(self, X, Y, D, N,
                 negative_label=-1, classification=True, copy=True, truth=None,
                 ):
        '''
        truth should be a dictionary of the model's true parameters
        '''
        self.X = deepcopy(X) if copy else X
        self.Y = deepcopy(Y) if copy else Y
        self.D = D
        self.N = N
        if classification:
            self.negative_label = negative_label
            self.Y01 = Y.copy()
            if negative_label != 0:
                self.Y01[np.where(self.Y == negative_label)] = 0

        self.truth = truth


def create_dataset(X, Y, copy=True):
    return Dataset(X, Y, X.shape[1], X.shape[0], copy=copy)


class DatasetGenerator(object):
    def __init__(self):
        pass
  
class SyntheticDatasetGenerator(DatasetGenerator):
    def __init__(self):
        pass

    def get_theta(self, D,
                  upTo=None, every=1, sigma_theta=1.0, param_seed=1234,):
        theta = np.zeros(D)
        bias = 0.0
        np.random.seed(param_seed)
        if upTo is None:
          numNonzero = int(np.ceil(D/every))
          theta[::every] = np.random.normal(scale=sigma_theta,
                                            size=numNonzero)
        else:
          upTo = np.minimum(upTo, D)
          theta[:upTo] = np.random.normal(scale=sigma_theta, size=upTo)
        theta = np.append(theta, bias)
        return theta

    def get_X(self, N, D,
              Xrank=None,
              lowRankNoise=0.0,
              rotateLowRank=False,
              basis_vectors=None,
              normalize=False):
        '''
        Generic setup for generating X (assuming the data is for a GLM)
        Assumes you have already set the seed you want
        '''
        if Xrank is None:
          X = np.random.normal(loc=0, size=(N,D))
          basis_vectors = None
        else:
          X, basis_vectors = utils.gen_low_rank_X(N, D,
                                                  rank=Xrank,
                                                  rotate=rotateLowRank,
                                                  lowRankNoise=lowRankNoise,
                                                  basis_vectors=basis_vectors)
        if normalize:
          pass
        X = np.append(X, np.ones((N,1)), axis=1)
        return X, basis_vectors

    def get_Y(self, theta, X):
      pass
    
    def get_dataset(self,
                    Ntrain=1000, Ntest=10000, D=4,
                    data_seed=1234,
                    every=10,
                    Xscaling='None',
                    param_seed=1234,
                    upTo=None,
                    Xrank=None,
                    rotateLowRank=False,
                    lowRankNoise=0.0,
                    **kwargs):

      np.random.seed(param_seed)
      theta = self.get_theta(D,
                             upTo=upTo,
                             every=every,
                             param_seed=param_seed)

      np.random.seed(data_seed)
      Xtrain, basis_vectors = self.get_X(Ntrain, D,
                                         Xrank=Xrank,
                                         lowRankNoise=lowRankNoise,
                                         rotateLowRank=rotateLowRank,
                                         basis_vectors=None,
                                         normalize=False)
      Xtest, basis_vectors = self.get_X(Ntest, D,
                                        Xrank=Xrank,
                                        lowRankNoise=lowRankNoise,
                                        rotateLowRank=rotateLowRank,
                                        basis_vectors=basis_vectors,
                                        normalize=False)

      Ytrain = self.get_Y(theta, Xtrain)
      Ytest = self.get_Y(theta, Xtest)

      if Xscaling == 'columns':
        scalings = np.sqrt(np.var(Xtrain, axis=0))[:-1]
        Xtrain[:,:-1] /= scalings[np.newaxis,:]
        Xtest[:,:-1] /= scalings[np.newaxis,:]
      trainDataset = Dataset(Xtrain, Ytrain, D, Ntrain, classification=False,
                             truth={'theta':theta, 'scaling':1.0})
      testDataset = Dataset(Xtest, Ytest, D, Ntest, classification=False,
                            truth={'theta':theta, 'scaling':1.0})
      return trainDataset, testDataset

          

class SyntheticLogisticDatasetGenerator(SyntheticDatasetGenerator):
    def __init__(self):
        super(SyntheticLogisticDatasetGenerator, self).__init__()

    def get_Y(self, theta, X):
      Y = np.zeros(X.shape[0])
      for n in range(X.shape[0]):
        Y[n] = np.random.binomial(1,
                                  p=utils.sigmoid(theta, X[n]))
      Y[np.where(Y == 0)] = -1
      return Y
    
    def get_theta(self, D,
                  upTo=None, every=1, sigma_theta=1.0, param_seed=1234,):
      return super(SyntheticLogisticDatasetGenerator, self).get_theta(D,
                                                                    upTo,
                                                                    every,
                                                                    1.0,
                                                                    param_seed)
      
class SyntheticClfnGaussianFeaturesDatasetGenerator(DatasetGenerator):
    def __init__(self):
        super(SyntheticClfnGaussianFeaturesDatasetGenerator, self).__init__()

    def get_dataset(self, Ntrain=1000, D=4, method="logreg", 
                    negative_label=-1, Ntest=0, seed=1234, filepath_X=None,
                    filepath_Y=None):

        if filepath_X is not None and filepath_Y is not None:
          X = np.loadtxt(filepath_X, delimiter=",")
          Y = np.loadtxt(filepath_Y, delimiter=",")
          negative_label = min(Y)
          return Dataset(X, Y, X.shape[1], X.shape[0], negative_label), None
        
        np.random.seed(seed)

        # draw means and variances
        means = np.zeros(D)
        variances = np.random.random(D)

        X = np.zeros((Ntrain, D))
        for cc in range(D):
          X[:, cc] = np.random.normal(loc=means[cc], scale=variances[cc],
            size=Ntrain)

        if method == "logreg":
            true_params = np.random.randint(10, size=D)
            bias = np.random.random() * np.random.randint(10)
            noise = np.random.random() * 5

            Y = np.dot(X, true_params) + bias

            Y = np.random.binomial(1, p=utils.sigmoid_only(Y))

            Y[np.where(Y == 0)] = negative_label
            return Dataset(X, Y, D, Ntrain, negative_label), None

class SyntheticLinearDatasetGenerator(SyntheticDatasetGenerator):
    def __init__(self):
        super(SyntheticLinearDatasetGenerator, self).__init__()

    def get_Y(self, theta, X):
      return X.dot(theta) + np.random.normal(size=X.shape[0])
    
    def get_theta(self, D, upTo=None, every=1, sigma_theta=1.0, param_seed=1234,):
      return super(SyntheticLinearDatasetGenerator, self).get_theta(D,
                                                                    upTo,
                                                                    every,
                                                                    3.0,
                                                                    param_seed)
    

class ZillowDatasetGenerator(DatasetGenerator):
    def __init__(self, filepath='zillow_train'):
        self.filepath = filepath
        super(ZillowDatasetGenerator, self).__init__()

    def get_dataset(self):
        X = pd.read_csv(self.filepath + "_X.csv", delimiter=',').as_matrix()
        Y = pd.read_csv(self.filepath + "_Y.csv", delimiter=',').as_matrix()
        D = X.shape[1]
        return Dataset(X, Y, D, X.shape[0], classification=False)

class CounterfitDatasetGenerator(DatasetGenerator):
    def __init__(self, filepath='counterfit.txt'):
        self.filepath = filepath
        super(CounterfitDatasetGenerator, self).__init__()

    def get_dataset(self):
        data = np.genfromtxt(self.filepath, skip_header=1, delimiter=',')
        X = data[:,:4]
        Y = data[:,4]
        Y[np.where(Y == 0)] = -1
        D = X.shape[1]
        negative_label = -1
        return Dataset(X, Y, D, X.shape[0], negative_label)

class BankDatasetGenerator(DatasetGenerator):
    def __init__(self, filepath='Dataset/bank-full-X.csv'):
        self.filepath = filepath
        super(CounterfitDatasetGenerator, self).__init__()

    def get_dataset(self):
        X = np.loadtxt('Dataset/bank-full-X.csv')
        Y = np.loadtxt('Dataset/bank-full-Y.csv')
        X = X[:,:-1] # the last feature makes it have 100% accuracy, which is boring!
        D = X.shape[0]

        return Dataset(X, Y, D, X.shape[0], negative_label)

class MNISTDatasetGenerator(DatasetGenerator):
    def __init__(self, filepath='./../data/mnist/binary_task.pkl'):
        self.filepath = filepath
        super(MNISTDatasetGenerator, self).__init__()

    def get_dataset(self):
        import sys
        sys.path.append('../data/mnist')
        import unpackMnist
        X, Y, X_test, Y_test = unpackMnist.loadBinaryTask(self.filepath)
        N = X.shape[0]
        D = X.shape[1]
        #X = logisticRegression.whitenData(X)
        #Xtest = logisticRegression.whitenData(Xtest)

        return Dataset(X, Y, D, X.shape[0]), \
        Dataset(X_test, Y_test, D, X_test.shape[0])


class SyntheticPoissonDatasetGenerator(SyntheticDatasetGenerator):
  def __init__(self):
    super(SyntheticPoissonDatasetGenerator, self).__init__()

  def get_Y(self, theta, X):
    return np.random.poisson(np.log(1+np.exp(X.dot(theta))))


class SyntheticExponentialPoissonDatasetGenerator(SyntheticDatasetGenerator):
  def __init__(self):
    super(SyntheticExponentialPoissonDatasetGenerator, self).__init__()
    
  def get_Y(self, theta, X):
    return np.random.poisson(np.exp(X.dot(theta)))

  
class SyntheticNegativeBinomialDatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(SyntheticNegativeBinomialDatasetGenerator, self).__init__()

  def get_dataset(self, Ntrain=1000, Ntest=10000, D=4, seed=1234):
    np.random.seed(seed)
    every = int(np.ceil(D/10))
    theta = np.zeros(D)
    theta[::every] = np.linspace(.1,2,D/every)

    Xtrain = np.random.normal(loc=0, scale=1, size=(Ntrain,D))
    Ytrain = np.random.negative_binomial(n=np.log(1 + np.exp(Xtrain.dot(theta))),
                                         p=.3)
    Xtest = np.random.normal(loc=0, scale=1, size=(Ntest,D))
    Ytest = np.random.negative_binomial(n=np.log(1 + np.exp(Xtest.dot(theta))),
                                         p=.3)

    trainDataset = Dataset(Xtrain, Ytrain, D, Ntrain, classification=False,
                           truth={'theta':theta, 'scaling':1.0})
    testDataset = Dataset(Xtest, Ytest, D, Ntest, classification=False,
                          truth={'theta':theta, 'scaling':1.0})
    return trainDataset, testDataset

class BikeShareDatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(BikeShareDatasetGenerator, self).__init__()


  def get_dataset(self, Ntrain=16000,
                  fpath='../data/bikeshare/hour.csv', seed=None):

    
    f = open(fpath, 'r')
    header = f.readline().split(',')
    X = []
    Y = []
    #base_date = datetime.strptime('2011-01-01')
    #date_fmt = '%Y-%m-%d'

    for line in f:
      split = line.split(',')
      Y.append(int(split[-1]))
      x = np.zeros(12)
      x[0] = split[6]
      x[1] = split[8]
      x[1+int(split[9])] = 1
      x[6:] = [float(val) for val in split[10:16]]
      #days = (datetime.strptime(split[1]) - base_date).days
      X.append(x)
    f.close()

    np.random.seed(1234)
    X = np.array(X)
    Y = np.array(Y)
    perm = np.random.permutation(X.shape[0])
    X = X[perm,:]
    X = utils.whiten_data(X)
    Y = Y[perm]
    return (Dataset(X[:Ntrain], Y[:Ntrain], D=12, N=Ntrain, classification=False),
            Dataset(X[Ntrain:], Y[Ntrain:], D=12, N=len(X)-Ntrain, classification=False))


class MouseBrainDatasetGenerator(DatasetGenerator):
    def __init__(self, filepath='../data/filtered_mouse_brain_data.txt'):
        self.filepath = filepath
        super(MouseBrainDatasetGenerator, self).__init__()

    def get_dataset(self):
        if not os.path.exists(self.filepath):
          self.create_dataset()

        Y = np.loadtxt(self.filepath)
        return (Dataset(X=Y, Y=None,
                       D=Y.shape[1], N=Y.shape[0], classification=False),
                None)
      
    def create_dataset(self):
        '''
        Download and extract the Gene / cell matrix(filtered) file from:
        https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/neuron_9k
        into the data/ directory of this repo.
        '''
        path = '../data/filtered_gene_bc_matrices/mm10/matrix.mtx'
        Y = scipy.sparse.csr_matrix.todense(scipy.io.mmread(path)).T

        # Throw away genes that are expressed in <40% of cells and cells with
        #  gene counts less than 3000 ... these numbers are kind of arbitrary 
        C = Y.shape[0]
        genes_to_keep = np.where(np.count_nonzero(Y, axis=0) > .4*C)[1]
        Y = Y[:,genes_to_keep]
        cells_to_keep = np.where(Y.sum(axis=1)[:,0] > 3000)[0]
        Y = Y[cells_to_keep,:]
        np.savetxt('../data/filtered_mouse_brain_data.txt', Y)

        
class SmallMouseBrainDatasetGenerator(DatasetGenerator):
  def __init__(self, filepath='../data/small_filtered_mouse_brain_data.txt'):
    self.filepath = filepath
    super(SmallMouseBrainDatasetGenerator, self).__init__()
    
  def get_dataset(self, C, G, num_zero, seed=1234, regenerate_data=False):
    np.random.seed(seed)
    if (not os.path.exists(self.filepath)) or (regenerate_data is True):
      Y = np.loadtxt('../data/filtered_mouse_brain_data.txt')
      cells = np.random.choice(Y.shape[0], size=C, replace=False)
      Y = Y[cells,:]
      genes = np.random.choice(Y.shape[1], size=G, replace=False)
      Y = Y[:,genes]
      np.savetxt('../data/small_filtered_mouse_brain_truth.txt', Y)
      
      # Zero out a few entries
      cells = np.random.choice(Y.shape[0], num_zero, replace=True)
      genes = np.random.choice(Y.shape[1], num_zero, replace=True)
      Y[cells,genes] = 0
      np.savetxt(self.filepath, Y)
      
    Y = np.loadtxt(self.filepath)
    truth = np.loadtxt('../data/small_filtered_mouse_brain_truth.txt')
    return (Dataset(X=Y, Y=None, truth={'Y':truth},
                    D=Y.shape[1], N=Y.shape[0], classification=False),
            None)

class SyntheticSCRNADatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(SyntheticSCRNADatasetGenerator, self).__init__()


  def make_sparse_cov_mtx(self, G, num_nonzero):
    num_nonzero = int(num_nonzero)
    cov = np.eye(G)
    cov *= 3
    for i in range(4):
      vec = np.zeros(G)
      inds = np.random.choice(G, size=num_nonzero, replace=False)
      for ind in inds:
        vec[ind] = np.random.normal(loc=3, scale=3)
      cov += np.outer(vec, vec)

    return cov

  def get_dataset(self, G, C, seed=None,
                  params_seed=1234, data_seed=1234, num_zero=0):
    if seed is not None:
      params_seed = seed
      data_seed = seed

    np.random.seed(params_seed)
    cov = self.make_sparse_cov_mtx(G, np.maximum(np.ceil(G/4), 3))
    mus = np.abs(np.random.multivariate_normal(mean=10*np.ones(G),
                                               cov=cov,
                                               size=C))
    # Make random correllations among the genes
    #mus = np.zeros((C,G))
    #inds = itertools.cycle(np.arange(G))
    #t = np.linspace(0,1,G)
    #np.random.seed(params_seed)
    #coeff = np.abs(np.random.normal(loc=10, scale=5))
    #bias = np.abs(np.random.normal(loc=20))
    #for c in range(C):
    #  mus[c] = t * coeff + bias

    np.random.seed(params_seed)
    betas = np.random.gamma(1, 10, size=G)
    np.random.seed(data_seed)
    lambdas = np.random.gamma(shape=(betas[np.newaxis,:]*mus).T,
                              scale=(1./betas[np.newaxis,:]).T).T
    np.random.seed(data_seed)
    X_full = np.random.poisson(lambdas.T).T
    if num_zero > 0:
      cells = np.random.choice(C, size=num_zero, replace=True)
      genes = np.random.choice(G, size=num_zero, replace=True)
      X = X_full.copy()
      X[cells,genes] = 0
    else:
      X = X_full.copy()
      
    truth = {'betas':betas, 'mus':mus, 'lambdas':lambdas, 'Y':X_full,
             'cov':cov}

    return (Dataset(X=X, Y=None, D=G, N=C, classification=False, truth=truth),
            None)
                              
    

class SyntheticScImputeDatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(SyntheticScImputeDatasetGenerator, self).__init__()

  def get_dataset(self, G, C, seed=None,
                  params_seed=1234, data_seed=1234, num_zero=0):
    np.random.seed(seed)
    #lambdas = np.random.uniform(size=G)
    lambdas = np.random.beta(a=1, b=5, size=G)
    alphas = np.random.gamma(3, scale=1, size=G)
    betas = np.random.gamma(10, size=G)
    mus = np.abs(np.random.normal(loc=20, size=G))
    sigmas = np.random.gamma(2, size=G)

    X = np.zeros((C,G))
    for c in range(C):
      for g in range(G):
        z = np.random.choice([0,1], p=[lambdas[g],1-lambdas[g]])
        if z == 0:
          X[c,g] = np.random.gamma(shape=alphas[g], scale=1./betas[g])
        elif z == 1:
          X[c,g] = np.random.normal(loc=mus[g], scale=sigmas[g]**2)
    X = np.exp(X) - 1.01
    X = np.floor(np.abs(X))
    
    truth = {'betas':betas, 'mus':mus, 'alphas':alphas, 'Y':X,
             'sigmas':sigmas, 'lambdas':lambdas}      
    return (Dataset(X=X, Y=None, D=G, N=C, classification=False, truth=truth),
            None)
     


class CensusDatasetGenerator(DatasetGenerator):
    def __init__(self, filepath='../data/adult.csv'):
      self.filepath = filepath
      super(CensusDatasetGenerator, self).__init__()

    def get_dataset(self, one_hot_encode=True):
      # adapted from Kaggle kernel: https://www.kaggle.com/bananuhbeatdown/multiple-ml-techniques-and-analysis-of-dataset
      data = pd.read_csv(self.filepath, delimiter=',')

      # remove rows where occupation is unknown
      data = data[data.occupation != '?']
      raw_data = data[data.occupation != '?']

      # value to predict
      data['over50K'] = np.where(data.income == '<=50K', 0, 1)

      if not one_hot_encode:
        # create numerical columns representing the categorical data
        data['workclass_num'] = data.workclass.map(
          {'Private':0, 'State-gov':1, 'Federal-gov':2, 'Self-emp-not-inc':3, 
          'Self-emp-inc':4, 'Local-gov':5, 'Without-pay':6})
        data['marital_num'] = data['marital.status'].map(
          {'Widowed':0, 'Divorced':1, 'Separated':2, 'Never-married':3, 
          'Married-civ-spouse':4, 'Married-AF-spouse':4, 
          'Married-spouse-absent':5})
        data['race_num'] = data.race.map(
          {'White':0, 'Black':1, 'Asian-Pac-Islander':2, 'Amer-Indian-Eskimo':3, 
          'Other':4})
        data['sex_num'] = np.where(data.sex == 'Female', 0, 1)
        data['rel_num'] = data.relationship.map(
          {'Not-in-family':0, 'Unmarried':0, 'Own-child':0, 
          'Other-relative':0, 'Husband':1, 'Wife':1})
        X = data[['workclass_num', 'education.num', 'marital_num', 'race_num', 
          'sex_num', 'rel_num', 'capital.gain', 'capital.loss']].as_matrix()
      else:
        ohe_data = pd.get_dummies(data, columns=["workclass", "marital.status", 
          "occupation", "relationship", "race", "sex", "native.country",
          "education"], drop_first=True, dummy_na=True)
        ohe_data = ohe_data.drop(labels=["income", "over50K"], axis=1)
        print(ohe_data.columns)
        X = ohe_data.as_matrix()
      Y = data.over50K.as_matrix()
      D = X.shape[1]

      Y[np.where(Y == 0)] = -1
      return Dataset(X, Y, D, X.shape[0], classification=False)
  


class SyntheticProbitDatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(SyntheticProbitDatasetGenerator, self).__init__()

  def get_dataset(self, Ntrain=1000, Ntest=10000, D=4, data_seed=1234, every=10,
                  param_seed=1234, negative_label=-1, upTo=None, Xrank=None):
    np.random.seed(param_seed)
    theta = np.zeros(D)
    if upTo is None:
      theta[::every] = np.random.normal(scale=.5, size=int(np.ceil(D/every)))
    else:
      theta[:upTo] = np.random.normal(scale=.5, size=upTo)
    bias = 0.0

    np.random.seed(data_seed)
    if Xrank is None:
      Xtrain = np.random.normal(loc=0, scale=1.0, size=(Ntrain,D))
      Xtest = np.random.normal(loc=0, scale=1.0, size=(Ntest,D))
      Xtrain /= np.linalg.norm(Xtrain, axis=1)[:,np.newaxis]
      Xtest /= np.linalg.norm(Xtest, axis=1)[:,np.newaxis]
    else:
      Xtrain, basis_vectors = utils.gen_low_rank_X(Ntrain, D,
                                      rank=Xrank, basis_vectors=None)
      Xtest, _ = utils.gen_low_rank_X(Ntest, D,
                                      rank=Xrank, basis_vectors=basis_vectors)

    
    cdf_vals = scipy.stats.norm.cdf(Xtrain.dot(theta) + bias)    
    Ytrain = np.random.binomial(n=1, p=cdf_vals)
    Ytrain[np.where(Ytrain==0)] = -1
    cdf_vals = scipy.stats.norm.cdf(Xtest.dot(theta) + bias)
    Ytest = np.random.binomial(n=1, p=cdf_vals)
    Ytest[np.where(Ytest==0)] = -1

    true_theta = np.append(theta, bias)
    trainDataset = Dataset(Xtrain, Ytrain, D, Ntrain, classification=False,
                           truth={'theta':true_theta})
    testDataset = Dataset(Xtest, Ytest, D, Ntest, classification=False,
                          truth={'theta':true_theta})
    return trainDataset, testDataset

class GisetteDatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(DatasetGenerator, self).__init__()


  def get_dataset(self, **kwargs):
    filepath = '../data/gisette'
    dataPath = os.path.join(filepath, 'gisette_train.txt')
    labelsPath = os.path.join(filepath, 'train_labels.txt')
    Y = np.loadtxt(labelsPath)
    X = np.loadtxt(dataPath)
    X = utils.whiten_data(X)
    good_features = np.setdiff1d(np.arange(X.shape[1]),
                                 np.where(np.var(X, axis=0) == 0)[0])
    X = X[:,good_features]
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)

    return Dataset(X, Y, X.shape[1]-1, X.shape[0], classification=True), None


class P53DatasetGenerator(DatasetGenerator):
  '''
  Classification task from:
  https://archive.ics.uci.edu/ml/datasets/p53+Mutants
  '''
  def __init__(self):
    self.filepath = '../data/p53'
    super(DatasetGenerator, self).__init__()

  def get_dataset(self, small=False, smallNsmallDDataset=False,
                  Ntrain=None, seed=1234, **kwargs):
    if small is True:
      dataPath = os.path.join(self.filepath, 'parsedX_small.txt')
      labelsPath = os.path.join(self.filepath, 'parsedY_small.txt')
    elif smallNsmallDDataset is True:
      dataPath = os.path.join(self.filepath, 'parsedX_smallNsmallD.txt')
      labelsPath = os.path.join(self.filepath, 'parsedY_smallNsmallD.txt')
    else:
      dataPath = os.path.join(self.filepath, 'parsedX.txt')
      labelsPath = os.path.join(self.filepath, 'parsedY.txt')
    
    Y = np.loadtxt(labelsPath)
    X = np.loadtxt(dataPath)

    if Ntrain is not None and Ntrain != 0:
      np.random.seed(seed)
      positiveLocs = np.where(Y == 1)
      negativeLocs = np.where(Y == -1)
      remainder = Ntrain - positiveLocs[0].shape[0]
      negativeSubsample = np.random.choice(negativeLocs[0], remainder,
                                           replace=False)
      
      X = np.append(X[positiveLocs[0],:], X[negativeSubsample,:], axis=0)
      Y = np.append(Y[positiveLocs[0]], Y[negativeSubsample], axis=0)

    X = utils.whiten_data(X)
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)
    return Dataset(X, Y, X.shape[1]-1, X.shape[0], classification=True), None
  
  def parse_dataset(self):
    path = os.path.join(self.filepath, 'K9.data')
    rawX = np.genfromtxt(path, delimiter=',', dtype=np.str_)
    rawLabels = rawX[:,-2]
    #rawX = rawX[:,:-2]
    missing = (rawX == '?')
    X = rawX[np.where(missing.sum(axis=1) == 0)[0],:-2].astype(np.float32)
    rawX = None
    
    rawLabels = rawLabels[np.where(missing.sum(axis=1) == 0)]
    labels = np.ones(rawLabels.shape[0]) * -1
    labels[np.where(rawLabels == 'active')] = 1.0
    
    np.savetxt(os.path.join(self.filepath, 'parsedY.txt'), labels)
    np.savetxt(os.path.join(self.filepath, 'parsedX.txt'), X)
  

  def make_small_dataset(self):
    Yfull = np.loadtxt(os.path.join(self.filepath, 'parsedY.txt'))
    Xfull = np.loadtxt(os.path.join(self.filepath, 'parsedX.txt'))

    tokeep = np.random.choice(np.arange(Yfull.shape[0]), 10000, replace=False)
    np.savetxt(os.path.join(self.filepath, 'parsedY_small.txt'),
               Yfull[tokeep])
    np.savetxt(os.path.join(self.filepath, 'parsedX_small.txt'),
               Xfull[tokeep,:])

    
class bcTCGADatasetGenerator(DatasetGenerator):
  '''
  Linear regression task from:
  hhttp://myweb.uiowa.edu/pbreheny/data/bcTCGA.html
  '''
  def __init__(self):
    self.filepath = '../data/bcTCGA'
    super(DatasetGenerator, self).__init__()


  def get_dataset(self, small=False, smallNsmallDDataset=False, smallD=False,
                  Ntrain=None, seed=1234, **kwargs):


    if not smallD:
      dataPath = os.path.join(self.filepath, 'X_clean.txt')
    else:
      dataPath = os.path.join(self.filepath, 'X_smallD.txt')
      
    labelsPath = os.path.join(self.filepath, 'Y_clean.txt')
    Y = np.loadtxt(labelsPath)
    X = np.loadtxt(dataPath)
    X = utils.whiten_data(X)
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)
    return Dataset(X, Y, X.shape[1]-1, X.shape[0], classification=True), None
  
class BasketballDatasetGenerator(DatasetGenerator):
  '''
  Linear regression task from
  "Estimating an NBA player’s impact on his team’s chances of winning"
    (Deshpande and Jensen, 2016)
  '''
  def __init__(self):
    self.filepath = '../data/basketball'
    super(DatasetGenerator, self).__init__()

  def get_dataset(self, small=False, seed=1234, **kwargs):

    if small is True:
      dataPath = os.path.join(self.filepath, 'X_5k_1234.txt')
      labelsPath = os.path.join(self.filepath, 'Y_5k_1234.txt')
    else:
      dataPath = os.path.join(self.filepath, 'X.txt')
      labelsPath = os.path.join(self.filepath, 'Y.txt')

    
    Y = np.loadtxt(labelsPath)
    X = np.loadtxt(dataPath)
    bad_dims = np.where(np.var(X, axis=0) == 0)
    good_dims = np.setdiff1d(np.arange(X.shape[1]), bad_dims)
    X = X[:,good_dims]

    
    X = utils.whiten_data(X)
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)
    return Dataset(X, Y, X.shape[1]-1, X.shape[0], classification=False), None
  
class RCV1DatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(DatasetGenerator, self).__init__()


  def get_dataset(self, **kwargs):
    filepath = '../data/rcv1'
    dataPath = os.path.join(filepath, 'X-N=full-D=30k.txt')
    labelsPath = os.path.join(filepath, 'Y-N=full-D=30k.txt')
    Y = np.loadtxt(labelsPath)
    X = np.loadtxt(dataPath)
    X = utils.whiten_data(X)
    good_features = np.setdiff1d(np.arange(X.shape[1]),
                                 np.where(np.var(X, axis=0) == 0)[0])
    X = X[:,good_features]
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)

    return Dataset(X, Y, X.shape[1]-1, X.shape[0], classification=True), None

class SyntheticDualALOO(SyntheticDatasetGenerator):
  def __init__(self):
    super(SyntheticDualALOO, self).__init__()

  def get_Y(self, theta, X):
    Y = np.zeros(X.shape[0])

    Y_tmp = X.dot(theta) + np.random.normal(scale=np.sqrt(0.25),
                                            size=X.shape[0])
    Y = np.sign(Y_tmp) * np.sqrt(np.abs(Y_tmp))
    return Y
    
  def get_theta(self, D, upTo=None, every=1, sigma_theta=1.0, param_seed=1234,):
    theta = np.zeros(D)
    theta[:upTo] = np.random.normal(size=upTo)
    return np.append(theta, 0.0)
  
  def get_X(self, N, D,
            Xrank=None,
            lowRankNoise=0.0,
            rotateLowRank=False,
            basis_vectors=None,
            normalize=False,
            upTo=5):
    '''
      Generic setup for generating X (assuming the data is for a GLM)
      Assumes you have already set the seed you want
    '''
    if Xrank is None:
      X = np.random.normal(loc=0, size=(N,D), scale=np.sqrt(1/60))
      basis_vectors = None
    else:
      X, basis_vectors = utils.gen_low_rank_X(N, D,
                                              rank=Xrank,
                                              lowRankNoise=lowRankNoise,
                                              rotate=rotateLowRank,
                                              basis_vectors=basis_vectors,
                                              sigma=0.0)
    X = np.append(X, np.ones((N,1)), axis=1)
    return X, basis_vectors

  
class DualALOODatasetGenerator(DatasetGenerator):
  '''
  Meant to replicate dataset from Wang et. al. "Approximate leave-one-out for fast
    parameter tuning in high dimensions" 2018.
  '''
  def __init__(self):
    super(DatasetGenerator, self).__init__()

  def get_dataset(self, **kwargs):
    filepath = '../data'
    dataPath = os.path.join(filepath, 'dualALOO-N=300-D=200_X.txt')
    labelsPath = os.path.join(filepath, 'dualALOO-N=300-D=200_Y.txt')
    Y = np.loadtxt(labelsPath)
    X = np.loadtxt(dataPath)
    X = utils.whiten_data(X)
    good_features = np.setdiff1d(np.arange(X.shape[1]),
                                 np.where(np.var(X, axis=0) == 0)[0])
    X = X[:,good_features]
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)

    return Dataset(X, Y, X.shape[1]-1, X.shape[0], classification=True), None

  
class E2006DatasetGenerator(DatasetGenerator):
  def __init__(self):
    super(DatasetGenerator, self).__init__()
 

  def get_dataset(self, **kwargs):
    filepath = '../data/E2006-tfidf'
    #dataPath = os.path.join(filepath, 'X-N=full-D=last60k.h5')
    #f = h5py.File(dataPath, 'r')
    #X = np.array(f['X'])
    #Y = np.array(f['Y'])
    dataPath = os.path.join(filepath, 'X-N=full-D=last60k.txt')
    labelPath = os.path.join(filepath, 'Y-N=full.txt')
    X = np.loadtxt(dataPath)
    Y = np.loadtxt(labelPath)
    X = utils.whiten_data(X)
    X = np.append(X, np.ones((X.shape[0],1)), axis=1)

    return Dataset(X, Y, X.shape[1]-1, X.shape[0], classification=True), None
