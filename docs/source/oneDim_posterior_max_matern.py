from dolfin import *
set_log_level(LogLevel.ERROR)
import numpy as np
import numba
import time
import pickle

from scipy import integrate

from tqdm import tqdm

# import required functions from oneDim
from statFEM_analysis.oneDim import mean_assembler, cov_assembler, kernMat, m_post, gen_sensor, MyExpression, m_post_fem_assembler, c_post, c_post_fem_assembler, sample_gp

from statFEM_analysis.maxDist import wass

# set up mean and kernel functions
σ_f = 0.1
κ = 4

@numba.jit
def m_f(x):
    return 1.0

@numba.jit
def c_f(x,y):
    return (σ_f**2)*np.exp(-κ*np.abs(x-y))

@numba.jit
def k_f(x):
    return (σ_f**2)*np.exp(-κ*np.abs(x))

# mean of forcing for use in FEniCS
f_bar = Constant(1.0)

# true prior solution mean
μ_true = Expression('0.5*x[0]*(1-x[0])',degree=2)

# set up true prior cov function for solution
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

def u_quad(x,f,maxiter=50):
    I_1 = integrate.quadrature(lambda w: w*f(w), 0.0, x,maxiter=maxiter)[0]
    I_2 = integrate.quadrature(lambda w: (1-w)*f(w),x, 1.0,maxiter=maxiter)[0]
    return (1-x)*I_1 + x*I_2

N = 41
grid = np.linspace(0,1,N)

s = 10 # number of sensors
# create sensor grid
Y = np.linspace(0.01,0.99,s)[::-1] 
# get true prior covariance on sensor grid
C_true_s = kernMat(c_u,Y.flatten())
# create function to compute vector mentioned above
def c_u_vect(x):
    return np.array([c_u(x,y_i) for y_i in Y])

# set up function to compute fem_prior
def fem_prior(h,f_bar,k_f,grid):
    J = int(np.round(1/h))
    μ = mean_assembler(h,f_bar)
    Σ = cov_assembler(J,k_f,grid,False,True)
    return μ,Σ

# set up function to compute statFEM posterior
def fem_posterior(h,f_bar,k_f,ϵ,Y,v_dat,grid):
    J = int(np.round(1/h))
    m_post_fem = m_post_fem_assembler(J,f_bar,k_f,ϵ,Y,v_dat)
    μ = MyExpression()
    μ.f = m_post_fem
    Σ = c_post_fem_assembler(J,k_f,grid,Y,ϵ,False,True)
    return μ,Σ

ϵ_list = [0.0001,0.0005,0.001,0.01]

def statFEM_posterior_sampler(n_sim, grid, h, f_bar, k_f, ϵ, Y, v_dat, par = False, trans = True, tol = 1e-9):
    # get length of grid
    d = len(grid)
    
    # get size of FE mesh
    J = int(np.round(1/h))
    
    # get statFEM posterior mean function
    m_post_fem = m_post_fem_assembler(J,f_bar,k_f,ϵ,Y,v_dat)
    μ_func = MyExpression()
    μ_func.f = m_post_fem
    
    # evaluate this on the grid
    μ = np.array([μ_func(x) for x in grid]).reshape(d,1)
    
    # get statFEM posterior cov mat on grid
    Σ = c_post_fem_assembler(J,k_f,grid,Y,ϵ,False,True)
    
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

# set up range of h values to use
h_range_tmp = np.linspace(0.25,0.02,100)
h_range = 1/np.unique(np.round(1/h_range_tmp))
np.round(h_range,2)

results = {}
start = time.time()
np.random.seed(345346)
###################
n_bins = 150
##################
tol = 1e-10
n_sim = 1000
for ϵ in tqdm(ϵ_list,desc='Eps loop'):
    # generate sensor data
    v_dat = gen_sensor(ϵ,m_f,k_f,Y,u_quad,grid,maxiter=400)
    # get true B mat required for posterior
    B_true = (ϵ**2)*np.eye(s) + C_true_s

    # set up true posterior mean
    def true_mean(x):
        return m_post(x,μ_true,c_u_vect,v_dat,Y,B_true)
    
    # set up true posterior covariance
    def c_post_true(x,y):
        return c_post(x,y,c_u,Y,B_true)
    
    u_sim = sample_gp(n_sim, true_mean, c_post_true, grid,True,False)
    
    max_true = u_sim.max(axis=0)
    
    errors = []
    
    for h in tqdm(h_range):
        # sample trajectories from statFEM prior for current h value
        sim = statFEM_posterior_sampler(n_sim, grid, h, f_bar, k_f, ϵ, Y, v_dat)
        # get max
        max_sim = sim.max(axis=0)
        # compute error
        error = wass(max_true,max_sim,n_bins)
        # append to errors
        errors.append(error)
        
    results[ϵ] = (errors,u_sim)

end = time.time()
print(f"time elapsed: {end - start}")

results['ϵ_list'] = ϵ_list
results['h_range'] = h_range

with open('results/oneDim_posterior_max_matern_results', 'wb') as f:
    pickle.dump(results, f)