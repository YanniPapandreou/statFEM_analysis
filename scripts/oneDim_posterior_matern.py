import time
import pickle
from dolfin import *
set_log_level(LogLevel.ERROR)
import numpy as np
import numba

from scipy import integrate
from scipy.linalg import sqrtm

from tqdm import tqdm

# import required functions from oneDim
from statFEM_analysis.oneDim import mean_assembler, cov_assembler, kernMat, m_post, gen_sensor, MyExpression, m_post_fem_assembler, c_post, c_post_fem_assembler

# set up mean and kernel functions
σ_f = 0.1
κ = 4

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
print("Computing true prior covariance mat on sensor grid")
C_true_s = kernMat(c_u,Y.flatten())
print("Finished computing true prior covariance mat on sensor grid")
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

# function to compute cov error
def compute_cov_diff(C_fem,C_true,C_true_sqrt,tol=1e-10):
    N = C_true.shape[0]
    C12 = C_true_sqrt @ C_fem @ C_true_sqrt
    C12_sqrt = np.real(sqrtm(C12))
    rel_error = np.linalg.norm(C12_sqrt @ C12_sqrt - C12)/np.linalg.norm(C12)
    assert rel_error < tol
    h = 1/(N-1)
    return h*(np.trace(C_true) + np.trace(C_fem) - 2*np.trace(C12_sqrt))


def W(μ_fem_s,μ_true_s,Σ_fem_s,Σ_true_s,Σ_true_s_sqrt,J_norm):
    mean_error = errornorm(μ_true_s,μ_fem_s,'L2',mesh=UnitIntervalMesh(J_norm))
    cov_error = compute_cov_diff(Σ_fem_s,Σ_true_s,Σ_true_s_sqrt)
    cov_error = np.sqrt(np.abs(cov_error))
    error = mean_error + cov_error
    return error

#hide_input
h_range_tmp = np.linspace(0.25,0.025,100)
h_range = 1/np.unique(np.round(1/h_range_tmp))
# print h_range to 2 decimal places
print('h values: ' + str(np.round(h_range,3))+'\n')
# noise levels to use
ϵ_list = [0.0001/2,0.0001,0.01,0.1]
print('ϵ values: ' + str(ϵ_list))
J_norm = 40


set_log_level(LogLevel.ERROR)

start = time.time()
results = {}
np.random.seed(42)
tol = 0.05 # tolerance for computation of posterior cov sqrt
for i, ϵ in enumerate(ϵ_list):
    # generate sensor data
    v_dat = gen_sensor(ϵ,m_f,k_f,Y,u_quad,grid,maxiter=300)
    
    # get true B mat required for posterior
    B_true = (ϵ**2)*np.eye(s) + C_true_s
    
    # set up true posterior mean
    def true_mean(x):
        return m_post(x,μ_true,c_u_vect,v_dat,Y,B_true)
    μ_true_s = MyExpression()
    μ_true_s.f = true_mean
    
    # set up true posterior covariance
    def c_post_true(x,y):
        return c_post(x,y,c_u,Y,B_true)
    Σ_true_s = kernMat(c_post_true,grid.flatten())
    Σ_true_s_sqrt = np.real(sqrtm(Σ_true_s))
    rel_error = np.linalg.norm(Σ_true_s_sqrt @ Σ_true_s_sqrt - Σ_true_s) / np.linalg.norm(Σ_true_s)
    if rel_error >= tol:
        print('ERROR')
        break
    
    # loop over the h values and compute the errors 
    # first create a list to hold these errors
    res = []
    for h in tqdm(h_range,desc=f'#{i+1} epsilon, h loop', position=0, leave=True):
        # get statFEM posterior mean and cov mat
        μ_fem_s, Σ_fem_s = fem_posterior(h,f_bar,k_f,ϵ,Y,v_dat,grid)
        # compute the error
        error = W(μ_fem_s,μ_true_s,Σ_fem_s,Σ_true_s,Σ_true_s_sqrt,J_norm)
        # store this in res
        res.append(error)
    
    # store ϵ value with errors in the dictionary
    results[ϵ] = res

end = time.time()
print(f"time elapsed: {end - start}")

results['h_range'] = h_range
with open('results/oneDim_posterior_matern_results', 'wb') as f:
    pickle.dump(results, f)
