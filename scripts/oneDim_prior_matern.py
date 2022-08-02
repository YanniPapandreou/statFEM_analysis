from dolfin import *
set_log_level(LogLevel.ERROR)
import numpy as np
import time
import pickle

# import required functions from oneDim
from statFEM_analysis.oneDim import mean_assembler, kernMat, cov_assembler
from scipy import integrate
from scipy.linalg import sqrtm
from tqdm import tqdm

# set up true mean and f_bar
μ_star = Expression('0.5*x[0]*(1-x[0])',degree=2)
f_bar = Constant(1.0)

# set up kernel functions for forcing f
σ_f = 0.1
κ = 4

def c_f(x,y):
    return (σ_f**2)*np.exp(-κ*np.abs(x-y))

def k_f(x):
    return (σ_f**2)*np.exp(-κ*np.abs(x))

# set up true cov function for solution
# compute inner integral over t
def η(w,y):
    I_1 = integrate.quad(lambda t: t*c_f(w,t),0.0,y)[0]
    I_2 = integrate.quad(lambda t: (1-t)*c_f(w,t),y,1.0)[0]
    return (1-y)*I_1 + y*I_2

# use this function eta and compute the outer integral over w
def c_u(x,y):
    I_1 = integrate.quad(lambda w: (1-w)*η(w,y),x,1.0)[0]
    I_2 = integrate.quad(lambda w: w*η(w,y),0.0,x)[0]
    return x*I_1 + (1-x)*I_2

# set up reference grid
N = 51
grid = np.linspace(0,1,N)

# get the true cov matrix and its square root
print(f"Starting to compute true cov matrix and its square root.")
C_true = kernMat(c_u,grid,True,False)
tol = 1e-6
C_true_sqrt = np.real(sqrtm(C_true))
rel_error = np.linalg.norm(C_true_sqrt @ C_true_sqrt - C_true)/np.linalg.norm(C_true)
assert rel_error <= tol
print(f"Finished computation of true cov mat and sqrt.")

# set up function to compute fem_prior
def fem_prior(h,f_bar,k_f,grid):
    J = int(np.round(1/h))
    μ = mean_assembler(h,f_bar)
    Σ = cov_assembler(J,k_f,grid,False,True)
    return μ,Σ

# function to compute cov error needed for the approximate wasserstein
def compute_cov_diff(C1,C_true,C_true_sqrt,tol=1e-10):
    N = C_true.shape[0]
    C12 = C_true_sqrt @ C1 @ C_true_sqrt
    C12_sqrt = np.real(sqrtm(C12))
    rel_error = np.linalg.norm(C12_sqrt @ C12_sqrt - C12)/np.linalg.norm(C12)
    assert rel_error < tol
    h = 1/(N-1)
    return h*(np.trace(C_true) + np.trace(C1) - 2*np.trace(C12_sqrt))

def W(μ_fem,μ_true,Σ_fem,Σ_true,Σ_true_sqrt):
    mean_error = errornorm(μ_true,μ_fem,'L2')
    cov_error = compute_cov_diff(Σ_fem,Σ_true,Σ_true_sqrt)
    cov_error = np.sqrt(np.abs(cov_error))
    error = mean_error + cov_error
    return error

# set up range of h values to use
h_range_tmp = np.linspace(0.25,0.02,100)
h_range = 1/np.unique(np.round(1/h_range_tmp))
np.round(h_range,2)

start = time.time()
errors = []
for h in tqdm(h_range):
    # obtain the fem prior for this value of h
    μ, Σ = fem_prior(h,f_bar,k_f,grid)
    # compute the error between the true and fem prior
    error = W(μ,μ_star,Σ,C_true,C_true_sqrt)
    # append to errors
    errors.append(error)

end = time.time()
print(f"time elapsed: {end - start}")

results = {'h_range': h_range, 'errors': errors}
with open('results/oneDim_prior_matern_results', 'wb') as f:
    pickle.dump(results, f)