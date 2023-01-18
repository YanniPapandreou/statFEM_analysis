from dolfin import *
set_log_level(LogLevel.ERROR)
import numpy as np
import numba
import ot
import time
import pickle

# import required functions from oneDim
from statFEM_analysis.oneDim import mean_assembler, kernMat, cov_assembler, sample_gp

from scipy import integrate
from scipy.linalg import sqrtm

from tqdm import tqdm

from statFEM_analysis.maxDist import wass

# set up true mean
@numba.jit
def m_u(x):
    return 0.5*x*(1-x)

# set up mean and kernel functions
σ_f = 0.1
κ = 4

# @numba.jit
# def m_f(x):
#     return 1.0

@numba.jit
def c_f(x,y):
    return (σ_f**2)*np.exp(-κ*np.abs(x-y))

@numba.jit
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


print("Starting to simulate trajectories from true prior.")
start = time.time()
n_sim = 1000
grid = np.linspace(0,1,100)
np.random.seed(235)
u_sim = sample_gp(n_sim, m_u, c_u, grid, par = False, trans = False, tol = 1e-8)
end = time.time()
print(f"Finished simulation, time elapsed: {end - start}")

max_true = u_sim.max(axis=0)

# create statFEM sampler function
def statFEM_sampler(n_sim, grid, h, f_bar, k_f, par = False, trans = True, tol=1e-9):
    # get length of grid
    d = len(grid)
    
    # get size of FE mesh
    J = int(np.round(1/h))

    # get statFEM mean function
    μ_func = mean_assembler(h, f_bar)
    
    # evaluate this on the grid
    μ = np.array([μ_func(x) for x in grid]).reshape(d,1)
    
    # get statFEM cov mat on grid
    Σ = cov_assembler(J, k_f, grid, parallel=par, translation_inv=trans)
    
    # construct the cholesky decomposition Σ = GG^T
    # we add a small diagonal perturbation to Σ to ensure it
    # strictly positive definite
    G = np.linalg.cholesky(Σ + tol * np.eye(d))

    # draw iid standard normal random vectors
    Z = np.random.normal(size=(d,n_sim))

    # construct samples from GP(m,k)
    Y = G@Z + np.tile(μ,n_sim)

    # return the sampled trajectories
    return Y

f_bar = Constant(1.0)

# set up range of h values to use
h_range_tmp = np.linspace(0.25,0.02,100)
h_range = 1/np.unique(np.round(1/h_range_tmp))
np.round(h_range,2)

start = time.time()
errors = []
###################
n_bins = 100
##################
np.random.seed(3252)
for h in tqdm(h_range):
    # sample trajectories from statFEM prior for current h value
    sim = statFEM_sampler(n_sim,grid,h,f_bar,k_f)
    # get max
    max_sim = sim.max(axis=0)
    # compute error
    error = wass(max_true,max_sim,n_bins)
    # append to errors
    errors.append(error)

end = time.time()
print(f"time elapsed: {end - start}")

results =  {'h_range': h_range, 'errors': errors, 'u_sim': u_sim, 'max_true': max_true}
with open('results/oneDim_prior_max_matern_results', 'wb') as f:
    pickle.dump(results, f)