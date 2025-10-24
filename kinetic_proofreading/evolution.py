import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import eig
import itertools
import copy
from tqdm.notebook import trange, tqdm
from scipy import linalg as LA

from physics import *
from network import *
from tools import save_vars

def evolve(N, L, mcsteps, T, t_stall, q, delta, n_samples, size, dsize, maxsize):
    """
    Metropolis evolution algorithm
    """
	equilibrium = 0

	# flag = 0: freeze V
	# flag = 1: freeze V, B
	# flag = 2: freeze B
	# flag = 3: Vary all parameters V, B
	flag = 3

	beta = 1/T

	# Defining the perturbation (quenched): the kinetics difference
	# between R and W is fixed to be at most e^delta (-delta < eps < delta)
	eps_v, eps_b = define_eps(N, delta)

	# Definition of the initial state
	init = 0
	p_0 = np.zeros((N,1))
	p_0[init] = 1

	# Definition of the final state
	end = N-1

	# Definition of genome {V, B, phi}
	# possible values that {V, B, phi} can take: such that e^{V,B,phi} is between {1e-size, 1esize}

	V = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)
	B = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)
	phi = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)

	from tools import binom_coeff
	binom = []
	for n in range(L):
		binom.append( binom_coeff(L,n) )
		

	track_speed_s, track_error_s, track_fpt_s, track_phi_s, track_time_s, track_ent_s, \
	track_k_s, track_fptw_s, track_params_s = [ [] for i in range(9)]
	for samples in range(n_samples):
		
		k_r, k_w, V_0, B_0, phi_0 = initialise_k(V, B, N, eps_v, eps_b)

		track_speed, track_error, track_fpt, track_phi, track_S, track_k, track_kw, params, \
		track_time, track_fptw, track_params = [ [] for i in range(11)]

		# If equilibrium = 1, phi is fixed to 0
		equilibrium = 0

		# flag = 0: freeze V
		# flag = 1: freeze V, B
		# flag = 2: freeze B
		# flag = 3: Vary all parameters V, B
		flag = 3
		
		for t in range(mcsteps):

			if t==0:
				k_new_r = copy.copy(k_r)
				k_new_w = copy.copy(k_w)
				params = copy.copy([V_0, B_0, phi_0])
				save_vars(track_speed, track_error, track_fpt, track_fptw, track_phi, track_S, track_k, track_kw, \
					  k_new_r, k_new_w, p_0, N, L, init, end, t_stall, params[2], q, binom)
				track_time.append(t)
				track_params.append(params)

			f_0 = fitness_dissipation(k_new_r, k_new_w, p_0, N, init, end, t_stall, q)

			i = random.choice(range(init,end+1))
			j = random.choice(range(init,end+1))
			if i != j:
				if equilibrium:
					k_rm, k_wm, V_m, B_m, phi_m = mutate_k(i, j, params[0], params[1], np.zeros([N, N]), \
														  N, eps_v, eps_b, equilibrium, flag, size, dsize)
				if equilibrium == 0:
					k_rm, k_wm, V_m, B_m, phi_m = mutate_k(i, j, params[0], params[1], params[2], \
														  N, eps_v, eps_b, equilibrium, flag, size, dsize)

				f = fitness_dissipation(k_rm, k_wm, p_0, N, init, end, t_stall, q)

				delta_F = f - f_0

				# Metropolis rule
				if delta_F <= 0:

					k_new_r = copy.copy(k_rm)
					k_new_w = copy.copy(k_wm)
					params = copy.copy([V_m, B_m, phi_m])
					save_vars(track_speed, track_error, track_fpt, track_fptw, track_phi, track_S, track_k, track_kw, \
					  k_new_r, k_new_w, p_0, N, L, init, end, t_stall, params[2], q, binom)
					track_time.append(t)
					track_params.append(params)


				else:
					r = random.random()
					if r < np.exp(-beta*delta_F):
						k_new_r = copy.copy(k_rm)
						k_new_w = copy.copy(k_wm)
						params = copy.copy([V_m, B_m, phi_m])
						save_vars(track_speed, track_error, track_fpt, track_fptw, track_phi, track_S, track_k, track_kw, \
						  k_new_r, k_new_w, p_0, N, L, init, end, t_stall, params[2], q, binom)
						track_time.append(t)
						track_params.append(params)

						
		track_error_s.append(track_error)
		track_speed_s.append(track_speed)
		track_fpt_s.append(track_fpt)
		track_fptw_s.append(track_fptw)
		track_phi_s.append(track_phi)
		track_ent_s.append(track_S)
		track_time_s.append(track_time)
		#track_k_s.append(track_k[-1])
		track_k_s.append(track_k)
		track_params_s.append(track_params)

	return track_error_s, track_speed_s, track_fpt_s, track_fptw_s, track_phi_s, track_ent_s, track_time_s, track_params_s


def evolve_from_ic(N, L, mcsteps, T, t_stall, q, delta, n_samples, size, dsize, maxsize, ic):
    """
    Evolve starting from a given initial condition of [V,B,phi]
    """
	equilibrium = 0

	# flag = 0: freeze V
	# flag = 1: freeze V, B
	# flag = 2: freeze B
	# flag = 3: Vary all parameters V, B
	flag = 3

	beta = 1/T

	# Defining the perturbation (quenched): the kinetics difference
	# between R and W is fixed to be at most e^delta (-delta < eps < delta)
	eps_v, eps_b = define_eps(N, delta)

	# Definition of the initial state
	init = 0
	p_0 = np.zeros((N,1))
	p_0[init] = 1

	# Definition of the final state
	end = N-1

	# Definition of genome {V, B, phi}
	# possible values that {V, B, phi} can take: such that e^{V,B,phi} is between {1e-size, 1esize}

	V = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)
	B = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)
	phi = np.asarray(np.arange(-size, maxsize, dsize), dtype=float)

	from tools import binom_coeff
	binom = []
	for n in range(L):
		binom.append( binom_coeff(L,n) )
		

	track_speed_s, track_error_s, track_fpt_s, track_phi_s, track_time_s, track_ent_s, \
	track_k_s, track_fptw_s, track_params_s = [ [] for i in range(9)]
	for samples in tqdm(range(n_samples)):
		
		V_0 = ic[0]
		B_0 = ic[1]
		phi_0 = ic[2]

		k_r, k_w = build_matrix_params(V_0, B_0, phi_0, N, eps_v, eps_b)

		track_speed, track_error, track_fpt, track_phi, track_S, track_k, track_kw, params, \
		track_time, track_fptw, track_params = [ [] for i in range(11)]

		# If equilibrium = 1, phi is fixed to 0
		equilibrium = 0

		# flag = 0: freeze V
		# flag = 1: freeze V, B
		# flag = 2: freeze B
		# flag = 3: Vary all parameters V, B
		flag = 3
		
		for t in range(mcsteps):

			if t==0:
				k_new_r = copy.copy(k_r)
				k_new_w = copy.copy(k_w)
				params = copy.copy([V_0, B_0, phi_0])
				save_vars(track_speed, track_error, track_fpt, track_phi, track_S, track_k, track_kw, \
					  k_new_r, k_new_w, p_0, N, L, init, end, t_stall, params[2], q, binom)
				track_time.append(t)
				track_params.append(params)

			f_0 = fitness_dissipation(k_new_r, k_new_w, p_0, N, init, end, t_stall, q)

			i = random.choice(range(init,end+1))
			j = random.choice(range(init,end+1))
			if i != j:
				if equilibrium:
					k_rm, k_wm, V_m, B_m, phi_m = mutate_k(i, j, params[0], params[1], np.zeros([N, N]), \
														  N, eps_v, eps_b, equilibrium, flag, size, dsize)
				if equilibrium == 0:
					k_rm, k_wm, V_m, B_m, phi_m = mutate_k(i, j, params[0], params[1], params[2], \
														  N, eps_v, eps_b, equilibrium, flag, size, dsize)


				f = fitness_dissipation(k_rm, k_wm, p_0, N, init, end, t_stall, q)

				delta_F = f - f_0

				# Metropolis rule
				if delta_F <= 0:

					k_new_r = copy.copy(k_rm)
					k_new_w = copy.copy(k_wm)
					params = copy.copy([V_m, B_m, phi_m])
					save_vars(track_speed, track_error, track_fpt, track_phi, track_S, track_k, track_kw, \
					  k_new_r, k_new_w, p_0, N, L, init, end, t_stall, params[2], q, binom)
					track_time.append(t)
					track_params.append(params)


				else:
					r = random.random()
					if r < np.exp(-beta*delta_F):
						k_new_r = copy.copy(k_rm)
						k_new_w = copy.copy(k_wm)
						params = copy.copy([V_m, B_m, phi_m])
						save_vars(track_speed, track_error, track_fpt, track_phi, track_S, track_k, track_kw, \
						  k_new_r, k_new_w, p_0, N, L, init, end, t_stall, params[2], q, binom)
						track_time.append(t)
						track_params.append(params)

						
		track_error_s.append(track_error)
		track_speed_s.append(track_speed)
		track_fpt_s.append(track_fpt)
		track_phi_s.append(track_phi)
		track_ent_s.append(track_S)
		track_time_s.append(track_time)
		#track_k_s.append(track_k[-1])
		track_params_s.append(track_params)

	return track_error_s, track_speed_s, track_fpt_s, track_fptw_s, track_phi_s, track_ent_s, track_time_s, track_params_s