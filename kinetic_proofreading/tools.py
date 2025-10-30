import numpy as np
from math import factorial
import itertools
import copy
import random

def process_scatterplot(x, y, ulim_x, ulim_y, percent):
    
    fflatx = [item for sublist in x for item in sublist]
    fflaty = [item for sublist in y for item in sublist]
    flatx = [item for sublist in fflatx for item in sublist]
    flaty = [item for sublist in fflaty for item in sublist]

    idx_x = np.where( np.asarray(flatx) < ulim_x )
    idx_y = np.where( np.asarray(flaty) < ulim_y )

    in_first = set(list(idx_x[0]))
    in_second = set(list(idx_y[0]))
    in_second_but_not_in_first = in_second - in_first

    idx = list(idx_x[0]) + list(in_second_but_not_in_first)
    
    xx = np.asarray(flatx)[idx]
    yy = np.asarray(flaty)[idx]
    xy = np.concatenate([xx, yy]).reshape(2, len(xx)).T
    
    random_idx = random.sample(range(1, int(len(xx))), int(percent*len(xx)))
    
    return xy, random_idx

def sort_eig(ev, evec):
    """
    Sort eigenvalues and eigenvectors from eig
    """
    idx = ev.argsort()[::-1]   
    ev = ev[idx]
    evec = evec[:,idx]
    
    return ev, evec

def binom_coeff(n,k):

    bc = factorial(n) / ( factorial(k)*factorial(n-k) )

    return bc


def max_k(k):
    N = k.shape[0]
    k[range(N), range(N)]=-100
    dup = []
    for j in k:
        for i in j:
            dup.append(i)
            
    return max(dup)

def chain(N, a):
    k = np.zeros([N,N])
    ii = range(0, N)
    jj = range(0, N-1)
    for i, j in itertools.product(ii, jj):
        if i == j+1:
            k[i][j] = a
            
    k = normalise_outflow(k, N)
    
    return k

def bc(value, size):
    
    if value > size:
        return size
    
    if value < -size:
        return -size
    
    if value <= size or value >= -size:
        return value

def moving_average(l, N):
    sum_ = 0
    result = np.zeros((len(l),))

    for i in range(0, N):
        sum_ = sum_ + l[i]
        result[i] = sum_ / (i+1)

    for i in range(N, len(l)):
        sum_ = sum_ - l[i-N] + l[i]
        result[i] = sum_ / N

    return result


def moving_average_log(l, N):

    norm = 0
    sum_ = 0
    result = np.zeros((len(l),))
    
    for i in range(0, N):
        weight = np.log10(N + 1 - i)
        norm = norm + weight
        sum_ = sum_ + l[i]*weight
        
        result[i] = sum_ / norm
        
    for i in range(N, len(l)):
        sum_ = sum_ - l[i-N] + l[i]
        result[i] = sum_ / N
        
    return result

def moving_std(x, x_filtered, N):
    return np.sqrt(moving_average((x-x_filtered)**2, N))


def moving_plot(x, y, N, label=''):
    '''
    !!! Takes as x, y torch.tensor() of shape [length]
    N is the time window on which to do the moving average
    '''
    y_log = np.log10(y, out=np.zeros_like(y), where=(y!=0))
    sort_idx = np.argsort(x)

    y_filtered = moving_average(y[sort_idx], N)
    y_filtered_log = moving_average_log(y_log[sort_idx], N)
    std = moving_std(y[sort_idx], y_filtered, N)
    std_log = moving_std(y_log[sort_idx], y_filtered_log, N)
    
    return sort_idx, y_filtered, y_filtered_log, std, std_log

def save_vars(track_speed, track_error, track_fpt, track_fptw, track_mu, track_S, track_k, track_kw, k_r, k_w, p_0, \
    N, init, end, t_stall, mu, q):

    from physics import fitness_dissipation, error, firstpt, entropy_dissipation


    track_speed.append( fitness_dissipation(k_r, k_w, p_0, N, init, end, t_stall, q).item() )

    track_error.append( error(k_r, k_w, N, init, end).item() )
    track_fpt.append( firstpt(copy.copy(k_r), p_0, N, end).item() )
    track_fptw.append( firstpt(copy.copy(k_w), p_0, N, end).item() )
    track_mu.append( sum(sum(abs(np.triu(mu)))) )
    track_S.append( entropy_dissipation(k_r, N, init) )
    track_k.append( k_r )
    track_kw.append( k_w )