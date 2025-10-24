import numpy as np
import random
from random import  uniform
import copy


from tools import bc
from tools import max_k
from physics import build_matrix, build_k_dkp, build_kw_dkp, error

def define_eps(N, delta):
    eps_v = np.zeros([N,1])
    eps_b = np.zeros([N,N])
    for i in range(N):
        if i < int(N/2):
            eps_v[i] = 0
        if i >= int(N/2):
            eps_v[i] = delta
    for i in range(N):
        for j in range(N):
            if i < int(N/2) and j < int(N/2):
                eps_b[i][j] = 0
            if i >= int(N/2) and j >= int(N/2):
                eps_b[i][j] = delta
            if i < int(N/2) and j > int(N/2):
                eps_b[i][j] = 0
            if i > int(N/2) and j < int(N/2):
                eps_b[i][j] = 0
        
    return eps_v, eps_b


def initialise_k(V, B, N, eps_v, eps_b):
    """
    Initialise the kinetic constant matrix
    at equilibrium - detailed balance - phi=0
    """    
    # Define matrix for RIGHT incorporation    
    V_r = np.asarray([random.choice(V) for i in range(N)]).reshape(N, 1)
    B_r = np.zeros([N,N])
    for i in range(N):
        for j in range(i+1,N):
            B_r[i][j] = random.choice(B)
    B_r = B_r + B_r.T
    
    # Initialise at equilibrium: phi = 0
    phi_ = np.zeros([N, N])
    
    k_r1 = build_matrix(V_r, B_r, phi_, N)
    k_r = copy.copy(k_r1)/max_k(copy.copy(k_r1))
    
    # Build matrix for WRONG incorporation
    V_w = np.add(V_r, eps_v)
    B_w = copy.copy(B_r) + eps_b
    
    k_w = build_matrix(V_w, B_w, phi_, N)
    k_w = copy.copy(k_w)/max_k(copy.copy(k_r1))

    return k_r, k_w, V_r, B_r, phi_


def initialise_k_phi(V, B, phi, N, eps_v, eps_b):
    """
    Initialise the kinetic constant matrix
    non-equilibrium 
    """    
    # Define matrix for RIGHT incorporation    
    V_r = np.asarray([random.choice(V) for i in range(N)]).reshape(N, 1)
    B_r = np.zeros([N,N])
    for i in range(N):
        for j in range(i+1,N):
            B_r[i][j] = random.choice(B)
    B_r = B_r + B_r.T

    phi_r = np.zeros([N,N])
    for i in range(N):
        for j in range(i+1,N):
            phi_r[i][j] = random.choice(phi)
    phi_r = phi_r + phi_r.T
    
    k_r1 = build_matrix(V_r, B_r, phi_r, N)
    k_r = copy.copy(k_r1)/max_k(copy.copy(k_r1))
    
    # Build matrix for WRONG incorporation
    V_w = np.add(V_r, eps_v)
    B_w = copy.copy(B_r) + eps_b
    
    k_w = build_matrix(V_w, B_w, phi_r, N)
    k_w = copy.copy(k_w)/max_k(copy.copy(k_r1))

    return k_r, k_w, V_r, B_r, phi_r

def initialise_network(N, delta, init, size, dsize, maxsize):

    eps_v, eps_b = define_eps(N, delta)
    
    p_0 = np.zeros((N,1))
    p_0[init] = 1
    
    end = N-1
    
    V = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)
    B = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)
    phi = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)
    print("Genome:", V, 'Length:', len(V))

    k_r, k_w, V_0, B_0, phi_0 = initialise_k(V, B, N, eps_v, eps_b)
    print('Error at initialisation:', error(k_r, k_w, N, init, end), 'e^{-\Delta}:', np.exp(-delta))
    print('Sum over rows:', sum(k_r) )
    
    return eps_v, eps_b, p_0, end, V, B, phi, k_r, k_w, V_0, B_0, phi_0

def build_matrix_params(V_r, B_r, phi_r, N, eps_v, eps_b):
    """
    Initialise the kinetic constant matrix
    non-equilibrium 
    """    

    k_r1 = build_matrix(V_r, B_r, phi_r, N)
    k_r = copy.copy(k_r1)/max_k(copy.copy(k_r1))

    # Build matrix for WRONG incorporation
    V_w = np.add(V_r, eps_v)
    B_w = copy.copy(B_r) + eps_b

    k_w = build_matrix(V_w, B_w, phi_r, N)
    k_w = copy.copy(k_w)/max_k(copy.copy(k_r1))

    return k_r, k_w



def mutate_k(i, j, V, B, phi, N, eps_v, eps_b, eq, flag, size, dsize):
    """
    Change the kinetic constant matrix according to the
    mutation performed. 
    rr = 1: random mutation
    rr = 0: first-neighbor mutation
    flag = 0: freeze V
    flag = 1: freeze V, B
    flag = 2: freeze B
    flag = 3: Vary all parameters V, B
    eq = 0: enforce equilibrium by setting all driving forces phi=0
    """
    # Mutate matrix for RIGHT incorporation 
    rr = 0
    if rr == 1:
        V_m = np.zeros([N, 1])
        B_m = np.zeros([N, N])
        phi_m = np.zeros([N, N])
        
        V_m[i] = random.choice(V)
        V_m[j] = random.choice(V)
        B_m[i][j] = random.choice(B)
        B_m[j][i] = B_m[i][j]
        phi_m[i][j] = random.choice(phi)
        phi_m[j][i] = phi_m[i][j]
    
    if rr == 0:
        V_m = copy.deepcopy(V)
        B_m = copy.deepcopy(B)
        phi_m = copy.deepcopy(phi)
        
        #V_m[i] = random.choice( [bc(V[i]+1, size), bc(V[i]-1, size)] )
        V_m[i] = bc(V[i] + uniform(-dsize, dsize), size)
        #V_m[j] = random.choice( [bc(V[j]+1, size), bc(V[j]-1, size)] )
        V_m[j] = bc(V[j] + uniform(-dsize, dsize), size)
        #B_m[i][j] = random.choice( [bc(B[i][j]+1, size), bc(B[i][j]-1, size)] )
        B_m[i][j] = bc( B[i][j] + uniform(-dsize, dsize), size )
        B_m[j][i] = B_m[i][j]
        phi_m[i][j] = bc(phi[i][j] + uniform(-dsize, dsize), size)
        phi_m[j][i] = phi_m[i][j]
        
    if eq:
        phi_m = np.zeros([N,N])
        
    if flag == 0:
        V_m = copy.deepcopy(V)
        
    if flag == 1:
        V_m = copy.deepcopy(V)
        B_m = copy.deepcopy(B)
        
    if flag == 2:
        B_m = copy.deepcopy(B)
       
    k_r = build_matrix(V_m, B_m, phi_m, N)
    k_r1 = copy.copy(k_r)/max_k(copy.copy(k_r))
        
    # Mutate matrix for WRONG incorporation
    
    if rr == 0:
        V_mw = np.add(copy.deepcopy(V_m), eps_v)
        B_mw = copy.deepcopy(B_m) + eps_b
    
    k_w = build_matrix(V_mw, B_mw, phi_m, N)
    k_w = copy.copy(k_w)/max_k(copy.copy(k_r))

    return k_r1, k_w, V_m, B_m, phi_m

def mutate_k_dkp(i, j, V, B, phi, N, eq, flag, size, delta, dsize):
    """
    Change the kinetic constant matrix according to the
    mutation performed. 
    rr = 1: random mutation
    rr = 0: first-neighbor mutation
    flag = 0: freeze V
    flag = 1: freeze V, B
    flag = 2: freeze B
    flag = 3: Vary all parameters V, B
    eq = 0: enforce equilibrium by setting all driving forces phi=0
    """
    # Mutate matrix for RIGHT incorporation 
    rr = 0
    
    if rr == 0:
        V_m = copy.deepcopy(V)
        B_m = copy.deepcopy(B)
        phi_m = copy.deepcopy(phi)
        
        V_m[i] = bc(V[i] + uniform(-dsize, dsize), size)
        V_m[j] = bc(V[j] + uniform(-dsize, dsize), size)
        B_m[i][j] = bc( B[i][j] + uniform(-dsize, dsize), size )
        B_m[j][i] = B_m[i][j]
        phi_m[i][j] = bc(phi[i][j] + uniform(-dsize, dsize), size)
        phi_m[j][i] = phi_m[i][j]
        
    if eq:
        phi_m = np.zeros([N,N])
        
    if flag == 0:
        V_m = copy.deepcopy(V)
        
    if flag == 1:
        V_m = copy.deepcopy(V)
        B_m = copy.deepcopy(B)
        
    if flag == 2:
        B_m = copy.deepcopy(B)
       
       
    k_r = build_k_dkp(V_m, B_m, phi_m, N)
    k_r1 = copy.copy(k_r)/max_k(copy.copy(k_r))
        
    # Mutate matrix for WRONG incorporation
    k_w = copy.copy(k_r) 
    k_w = build_kw_dkp(k_w, N, delta)
    k_w = copy.copy(k_w)/max_k(copy.copy(k_r))

    return k_r1, k_w, V_m, B_m, phi_m