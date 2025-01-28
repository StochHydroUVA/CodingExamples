import math
import numpy as np
import scipy.stats as ss
import lmoments3 as lm
import matplotlib.pyplot as plt
from scipy.optimize import brentq as root
import scipy.integrate as integrate
import scipy.special as special

class Distribution:
  def __init__(self):
    self.xbar = None
    self.var = None
    self.skew = None
    self.kurtosis = None
    self.L1 = None
    self.L2 = None
    self.T3 = None
    self.T4 = None

  def findMoments(self, data):
    self.xbar = np.mean(data)
    self.var = np.var(data, ddof=1)
    self.skew = ss.skew(data, bias=False)
    self.kurtosis = ss.kurtosis(data, bias=False)

  def findLmoments(self, data):
    self.L1, self.L2, self.T3, self.T4 = lm.lmom_ratios(data, nmom=4)

class LogNormal(Distribution):
  def __init__(self):
    super().__init__()
    self.mu = None
    self.sigma = None
    self.tau = None

  def fit(self, data, method, npars):
    assert method == 'MLE' or method == 'MOM' or method == "Lmom","method must = 'MLE','MOM', or 'Lmom'"
    assert npars == 2 or npars == 3,"npars must = 2 or 3"

    self.findMoments(data)
    self.findLmoments(data)
    if method == 'MLE':
      if npars == 2:
        shape, loc, scale = ss.lognorm.fit(data, floc=0)
      elif npars == 3:
        shape, loc, scale = ss.lognorm.fit(data)

      self.mu = np.log(scale)
      self.sigma = shape
      self.tau = loc
    elif method == 'MOM':
      if npars == 2:
        self.sigma = np.sqrt(np.log(1+self.var/self.xbar**2))
        self.mu = np.log(self.xbar) - 0.5*self.sigma**2
        self.tau = 0
      elif npars == 3:
        self.sigma = root(lambda x: (np.exp(3*x**2)-3*np.exp(x**2)+2) / (np.exp(x**2)-1)**(3/2) - self.skew,
                   0.01, np.std(np.log(data),ddof=1))
        self.mu = 0.5 * (np.log(self.var / (np.exp(self.sigma**2)-1)) - self.sigma**2)
        self.tau = self.xbar - np.exp(self.mu + 0.5*self.sigma**2)
    elif method == 'Lmom':
      if npars == 2:
        self.tau = 0
        self.sigma = root(lambda x: special.erf(x/2) - self.L2/self.L1, 0.01, self.L2)
        self.mu = np.log(self.L1) - 0.5*self.sigma**2
      elif npars == 3:
        self.sigma = root(lambda x: self.T3 - (6/np.sqrt(math.pi)) * \
                 integrate.quad(lambda y: special.erf(y/np.sqrt(3))*np.exp(-y**2), 0, x/2)[0] \
                 / math.erf(x/2),
                 0.01, self.L2)
        self.mu = np.log(self.L2 / math.erf(self.sigma/2)) - 0.5*self.sigma**2
        self.tau = self.L1 - np.exp(self.mu + 0.5*self.sigma**2)

  def findReturnPd(self, T):
    q_T = ss.lognorm.ppf(1-1/T, self.sigma, self.tau, np.exp(self.mu))
    return q_T

  def plotHistPDF(self, data, min, max, title):
    x = np.arange(min, max,(max-min)/100)
    f_x = ss.lognorm.pdf(x, self.sigma, self.tau, np.exp(self.mu))

    plt.hist(data, density=True)
    plt.plot(x,f_x)
    plt.xlim([min, max])
    plt.title(title)
    plt.xlabel('Flow')
    plt.ylabel('Probability Density')
    plt.show()