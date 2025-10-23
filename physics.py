import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eig
import itertools
import copy
from tqdm.notebook import trange, tqdm
from scipy import linalg as LA

from tools import sort_eig


def kinetic_constants(i, j, V, B, phi, N):
    """
    Define kinetic constants as function of
    energy landscape (V,B)
    chemical potential (phi)
    """
    kij = float( np.exp( V[j] - B[i][j] ) )
    kji = float( np.exp( V[i] - B[j][i] + phi[i][j] ) )
    if B[i][j] != B[j][i]:
        print('Matrix of barrier is asymmetric')
    return kij, kji

def build_matrix(V, B, phi, N):
    """
    Building the matrix of kinetic constants
    """    
    k = np.zeros([N,N])
    ii = range(0, N)
    jj = range(0, N)
    for i, j in itertools.product(ii, jj):
        if j>i:
            kij, kji = kinetic_constants(i, j, V, B, phi, N)
            k[i][j] = kij
            k[j][i] = kji

    k = normalise_outflow(k, N)
    
    return k

def normalise_outflow(k, N):
    """
    Normalising the outflow in the matrix of kinetic constants
    """
    for i in range(N):
        k[i,i] = -(sum(k[:,i])-np.diagonal(k)[i])
        
    return k


def firstpt(k, p_0, N, end):
    """
    Definition of the first passage time to the final state
    """       
    tmp = []

    k_pinv = pinv(k[0:end, 0:end])
    for j in range(N):
        if j != end:
            e_j = np.zeros(N)
            e_j[j] = 1
            tmp.append( -(k_pinv@p_0[0:end]).T@e_j[0:end] )
        
    fpt = sum(tmp)
    
    # ev_r, evec_r = LA.eig( np.float64(k))
    # if k.shape[0] != k.shape[1]:
    #     print('k is not square')
    # ev_r, evec_r = sort_eig(ev_r, evec_r)

    return fpt
    #return np.abs(evec_r[:,0][end])

def error(k_r, k_w, N, init, end):
    """
    Compute the error as the ratio of steady-states probabilities
    """
    ev_r, evec_r = LA.eig( np.float64(k_r))
    if k_r.shape[0] != k_r.shape[1]:
        print('k_r is not square')
    ev_r, evec_r = sort_eig(ev_r, evec_r)
    
    ev_w, evec_w = LA.eig( np.float64(k_w))
    if k_w.shape[0] != k_w.shape[1]:
        print('k_w is not square')
    ev_w, evec_w = sort_eig(ev_w, evec_w)
    
    if abs(ev_r[0])>1e-9 or abs(ev_w[0])>1e-9:
        print("Eigenvalue(s) larger than 1e-9 (null-space does not exist)", abs(ev_r[0]), abs(ev_w[0]))
        
    err = ( evec_w[:,0][end]/evec_w[:,0][init] ) / ( evec_r[:,0][end]/evec_r[:,0][init] )
    
    return err.real

def error_ss(k_r, k_w, N, init, end):
    """
    Compute the error as the ratio of steady-states probabilities
    """
    ev_r, evec_r = LA.eig( np.float64(k_r))
    if k_r.shape[0] != k_r.shape[1]:
        print('k_r is not square')
    ev_r, evec_r = sort_eig(ev_r, evec_r)
    
    ev_w, evec_w = LA.eig( np.float64(k_w))
    if k_w.shape[0] != k_w.shape[1]:
        print('k_w is not square')
    ev_w, evec_w = sort_eig(ev_w, evec_w)
    
    if abs(ev_r[0])>1e-9 or abs(ev_w[0])>1e-9:
        print("Eigenvalue(s) larger than 1e-9 (null-space does not exist)", abs(ev_r[0]), abs(ev_w[0]))
        
    err = ( evec_w[:,0][end]/evec_w[:,0][init] ) / ( evec_r[:,0][end]/evec_r[:,0][init] )

    #print('p_0 W: ', evec_w[:,0][init], 'p_0 R: ', evec_r[:,0][init])

    return err.real, evec_r[:,0][end]/sum(evec_r[:,0]), evec_w[:,0][end]/sum(evec_w[:,0])


def fitness(k_r, k_w, p_0, N, init, end, t_stall):
    """
    Define the fitness as the time of copying per nucleotide
    with a stalling time t_stall occurring upon error making
    """
    
    err = error(k_r, k_w, N, init, end)

    if t_stall > 0:
        return (1-err)*firstpt(k_r, p_0, N, end) + err*firstpt(k_w, p_0, N, end) + t_stall*err

def fitness_dissipation(k_r, k_w, p_0, N, init, end, t_stall, q):
    """
    Define the fitness as the time of copying per nucleotide
    with a stalling time t_stall occurring upon error making
    A cost to entropy production q*diss is introduced
    """
    
    err = error(k_r, k_w, N, init, end)
    diss = entropy_dissipation(k_r, N, init)

    #return (1-err)*firstpt(k_r, p_0, N, end) + err*firstpt(k_w, p_0, N, end) + t_stall*err + q*diss
    return firstpt(k_r, p_0, N, end) + t_stall*err + q*diss


def entropy_dissipation(k, N, init):
    
    ev, evec = eig(k)
    ev, evec = sort_eig(ev, evec)
    evec[:,0] =  evec[:,0] / evec[:,0][init]
    
    S = []
    ii = range(0, N)
    jj = range(0, N)
    for i, j in itertools.product(ii, jj):
        if i!=j:
            S.append(1/2 * (k[i][j]*evec[:,0][j].real -  k[j][i]*evec[:,0][i].real) \
                     * np.log( k[i][j]*evec[:,0][j].real / (k[j][i]*evec[:,0][i].real) ) )
    
    return sum(S)

def identify_pareto(xy):
    # Count number of items
    population_size = xy.shape[0]

    # Create index for scores on the pareto front
    population_ids = np.arange(population_size)

    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Pareto front
    pareto_front = np.ones(population_size, dtype=bool)
    print(pareto_front)
    # Loop through each item. This will then be compared with all other items
    for i in tqdm(range(population_size)):
        
        # Loop through all other items
        for j in range(population_size):
            
            # Check if our 'i' pint is dominated by out 'j' point
            if all(xy[j] <= xy[i]) and any(xy[j] < xy[i]):
               
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

def scatter(n_scatter, N, V, B, phi, delta, init, end, p_0):
    """
    Create a scatter plot from some initial values of parameters [V,B,phi]
    """
    from tools import max_k

    kij_R = []
    kji_R = []
    fpt_sp = []
    err_sp = []

    for vi,vj,b,m in itertools.product(V,V,B,phi):
        kij_R.append( float( np.exp( vj - b ) ) )
        kji_R.append( float( np.exp( vi - b + m ) ) )

    n_scatter = int(n_scatter)
    for samples in tqdm(range(n_scatter)):
        ku = np.random.choice(np.asarray(kij_R), int(N*(N-1)/2))
        kl = np.random.choice(np.asarray(kji_R), int(N*(N-1)/2))
        
        k_r = np.zeros([N,N])
        ii = range(0, N)
        jj = range(0, N)
        idx_u = 0
        idx_l = 0
        for i, j in itertools.product(ii, jj):
            if i<j:
                k_r[i,j] = ku[idx_u]
                idx_u = idx_u + 1
            if i>j:            
                k_r[i,j] = ku[idx_l]
                idx_l = idx_l + 1
            
        k_r[0,1] = kl[0]
        k_r[0,2] = kl[1]
        k_r[2,1] = kl[2]
        
        k_r = normalise_outflow(k_r, N)
        k_r1 = copy.copy(k_r)/max_k(copy.copy(k_r))
        
        k_w = copy.copy(k_r)    
        for i in range(N):
            for j in range(N):
                if i < int(N/2) and j >= int(N/2):
                    k_w[i,j] = k_w[i,j]*np.exp(delta)
                
        
        k_w = normalise_outflow(k_w, N)
        k_w1 = copy.copy(k_w)/max_k(copy.copy(k_r))

        err_sp.append(error(k_r1, k_w1, N, init, end).item())
        fpt_sp.append(firstpt(copy.copy(k_r1), p_0, N, end).item())

    return err_sp, fpt_sp