import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import spline

# the desired probability distribution's function (not normalised) 
def p(x):
  return np.exp(-1*x**4) * (2 + np.sin(5*x) + np.sin(-2*x*x))

# generate plot of p
f = []
x_range = []
for x in np.arange(-2.0, 2.0, 0.01):
  f.append(p(x))
  x_range.append(x)
  
plt.plot(x_range,f)
plt.show()

sigma_space  = [0.05,1,50]
num_samples = 150000
seeds=[0]
seed_index=0
x_0 = -1
samples = []
next_samples= []
for sigma_index in range(len(sigma_space)):
  samples.append([])
  next_samples.append([])
  sigma = sigma_space[sigma_index]
    
  samples[sigma_index].append(x_0)
  next_samples[sigma_index].append(x_0)
  #seed the random number generator
  np.random.seed(seeds[seed_index])
  burning_period=10000
  x_i = x_0
  for i in range(num_samples+burning_period):
    #generate candidate from prposal distribution - a gaussian centered at current state 
    x_candidate = np.random.normal(x_i,sigma)
    if(i>burning_period):
      next_samples[sigma_index].append(x_candidate)
    #calculate acceptance probability. Since gaussian is symmetric q(x_i|x_candidate) = q(x_candidate|x_i)
    a = min(1.0,p(x_candidate)/p(x_i))
    u = np.random.uniform()
    if(u<=a):
      #accept the proposal
      x_i = x_candidate
    if(i>burning_period):
      samples[sigma_index].append(x_i)
# to plot the markov chain of samples
#   plt.plot(range(len(samples[sigma_index][seed_index])),samples[sigma_index][seed_index])
#   plt.show()
#   plt.hist(samples[sigma_index],bins=10000)
#   plt.show()
  xt=[]
  for i in range (num_samples):
    xt.append(i)
  plt.plot(xt,next_samples[sigma_index])
  plt.show()