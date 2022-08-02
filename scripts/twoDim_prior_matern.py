import time
import pickle
from dolfin import *
set_log_level(LogLevel.ERROR)
import numpy as np
import numba

# import required functions from twoDim
from statFEM_analysis.twoDim import mean_assembler, kernMat, cov_assembler
from scipy.linalg import sqrtm
from tqdm import tqdm

# set up mean and kernel functions
f_bar = Constant(1.0)

σ_f = 0.1
κ = 4

@numba.jit
def c_f(x,y):
    return (σ_f**2)*np.exp(-κ*np.abs(x-y))

@numba.jit
def k_f(x):
    return (σ_f**2)*np.exp(-κ*np.abs(x))

def fem_prior(h,f_bar,k_f,grid):
    J = int(np.round(1/h))
    μ = mean_assembler(h,f_bar)
    Σ = cov_assembler(J,k_f,grid,False,True)
    return μ,Σ

N = 41
x_range = np.linspace(0,1,N)
grid = np.array([[x,y] for x in x_range for y in x_range])

def compute_cov_diff(C1,C2,tol=1e-10):
    N = np.sqrt(C1.shape[0])
    C1_sqrt = np.real(sqrtm(C1))
    rel_error_1 = np.linalg.norm(C1_sqrt @ C1_sqrt - C1)/np.linalg.norm(C1)
    assert rel_error_1 < tol
    
    C12 = C1_sqrt @ C2 @ C1_sqrt
    C12_sqrt = np.real(sqrtm(C12))
    rel_error_12 = np.linalg.norm(C12_sqrt @ C12_sqrt - C12)/np.linalg.norm(C12)
    assert rel_error_12 < tol
    
    hSq = (1/(N-1))**2
    return hSq*(np.trace(C1) + np.trace(C2) - 2*np.trace(C12_sqrt))

def W(μ_1,μ_2,Σ_1,Σ_2):
    mean_error = errornorm(μ_1,μ_2,'L2')
    cov_error = compute_cov_diff(Σ_1,Σ_2)
    cov_error = np.sqrt(np.abs(cov_error))
    error = mean_error + cov_error
    return error

def refine(h,n,f_bar,k_f,grid):
    # set up empty lists to hold h-values and errors (this being the ratios)
    h_range = []
    errors = []
    # get the statFEM prior for h and h/2
    μ_1, Σ_1 = fem_prior(h,f_bar,k_f,grid)
    μ_2, Σ_2 = fem_prior(h/2,f_bar,k_f,grid)
    # compute the distance between these and store in numerator variable
    numerator = W(μ_1,μ_2,Σ_1,Σ_2)
    # succesively refine the mesh by halving and do this n times
    for i in tqdm(range(n), desc="inner loop", position=1, leave=False):
    # for i in range(n):
        # print(f"inner loop iteration #{i} out of {n}")
        # store mean and cov for h/2 in storage for h
        μ_1, Σ_1 = μ_2, Σ_2 
        # in storage for h/2 store mean and cov for h/4
        μ_2, Σ_2 = fem_prior(h/4,f_bar,k_f,grid)
        # compute the distance between the priors for h/2 and h/4
        # and store in denominator variable
        denominator = W(μ_1,μ_2,Σ_1,Σ_2)
        # compute the ratio and store in error
        error = numerator/denominator
        # append the current value of h and the ratio
        h_range.append(h)
        errors.append(error)
        # store denominator in numerator and halve h
        numerator = denominator
        h = h/2
    # return the list of h-values together with the ratios for these values
    return h_range,errors

my_list = [(0.25,4),(0.2,3),(0.175,3),(0.22,4),(0.176,3),(0.177,3),(0.178,3),(0.179,3),(0.18,3),(0.21,3),(0.215,3),(0.1,2),(0.3,4),(0.31,4),(0.315,4),(0.24,4),(0.245,4),(0.25,4),(0.14,2),(0.16,2),(0.15,2),(0.18,2)]

set_log_level(LogLevel.ERROR)

n_outer=len(my_list)

h_range = []
errors = []
print("Starting computation.")
start = time.time()
# for i,(h,n) in enumerate(my_list):
    # print(f"outer loop iteration #{i} out of {n_outer}")
for (h, n) in tqdm(my_list, desc="outer", position=0):
    h_range_tmp, errors_tmp = refine(h,n,f_bar,k_f,grid)
    h_range.extend(h_range_tmp)
    errors.extend(errors_tmp)

end = time.time()
print("Finished computation.")
print(f"time elapsed: {end - start}")


results = {'my_list': my_list, 'h_range': h_range, 'errors': errors}
with open('results/twoDim_prior_matern_results', 'wb') as f:
    pickle.dump(results, f)