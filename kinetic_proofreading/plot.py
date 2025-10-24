import numpy as np
import matplotlib.pyplot as plt

def plot_network(track_k_s, sample, n_samples, N, t_stall, T, flag_mean):

	import networkx as nx
	import itertools
	import copy
	import palettable as pal

	G = nx.DiGraph()
	G.add_nodes_from(list(range(N)))
	ii = range(0, N)
	jj = range(0, N)
	list_edges = []
	for i, j in itertools.product(ii, jj):
	    if i!=j:
	        list_edges.append((i,j))
	G.add_edges_from(list_edges)
	pos = nx.spring_layout(G)
	k_mean = np.zeros([N,N])
	for u, v, d in G.edges(data=True):
	    for sample_idx in range(n_samples):
	        k_mean[u][v] = k_mean[u][v] + track_k_s[sample_idx][u][v]/n_samples

	# Alternative way to add nodes:
	# G.add_node(0, pos=(0,1))
	# G.add_node(1, pos=(1,2))
	# G.add_node(2, pos=(0,2))
	# pos=nx.get_node_attributes(G, 'pos')

	print(G.number_of_nodes(), G.number_of_edges())

	for u, v, d in G.edges(data=True):
		if flag_mean == 0:
			kk = copy.copy(track_k_s[sample][v][u])
		if flag_mean == 1:
			kk = copy.copy(k_mean[v][u])
		d['length'] = kk
		d['weight'] = kk   

	edges, weights = zip(*nx.get_edge_attributes(G, 'length').items())
	arc_weight = nx.get_edge_attributes(G,'weight')

	edge_labels = dict([((u,v,),d['length']) for u,v,d in G.edges(data=True)])
	import matplotlib.cm as cm
	cmap=pal.colorbrewer.sequential.Greys_9.mpl_colormap
	edge_col = [cmap(G[u][v]['length']) for u,v,d in G.edges(data=True)]

	node_color = []
	for i in range(N):
	    if i == 0:
	        node_color.append('gray')
	    if i == N-1:
	        node_color.append('gold')
	    if (i!=0)*(i!=N-1):
	        node_color.append('white')
	        
	nx.draw_networkx(G, pos, width=0.5, node_size=1000, with_labels=True)
	nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, width=0, node_size=1000, with_labels=True)
	nodes.set_edgecolor('black')


	nx.draw_networkx_edges(G, pos, edge_color=edge_col, width=20*np.asarray(weights), arrows=True, \
	                       node_size=600, arrowstyle='->', arrowsize=50)
	#nx.draw_networkx_edge_labels(G, pos, edge_color=edge_col, edge_labels=edge_labels, label_pos=0.3, font_size=15)
	plt.axis('off')
	sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = 0, vmax = 1))
	sm._A = []
	plt.colorbar(sm, fraction=0.02)
	plt.gcf().set_size_inches(8,5)
	plt.savefig('graph_randomnet_copier_N%d_tstall%.2e_T%.2e.pdf' % (N, t_stall, T) )

def plot_single_network(track_k, N, t_stall, T):

	import networkx as nx
	import itertools
	import copy
	import palettable as pal

	G = nx.DiGraph()
	G.add_nodes_from(list(range(N)))
	ii = range(0, N)
	jj = range(0, N)
	list_edges = []
	for i, j in itertools.product(ii, jj):
	    if i!=j:
	        list_edges.append((i,j))
	G.add_edges_from(list_edges)
	pos = nx.spring_layout(G, seed = 1)

	print(G.number_of_nodes(), G.number_of_edges())

	for u, v, d in G.edges(data=True):
		kk = copy.copy(track_k[v][u])
		d['length'] = kk
		d['weight'] = kk

	edges, weights = zip(*nx.get_edge_attributes(G, 'length').items())
	arc_weight = nx.get_edge_attributes(G,'weight')

	edge_labels = dict([((u,v,),d['length']) for u,v,d in G.edges(data=True)])
	import matplotlib.cm as cm
	cmap=pal.scientific.sequential.GrayC_20.mpl_colormap
	cmap=pal.scientific.sequential.LaJolla_20.mpl_colormap
	edge_col = [cmap(G[u][v]['length']) for u,v,d in G.edges(data=True)]

	node_color = []
	for i in range(N):
	    if i == 0:
	        node_color.append('gray')
	    if i == N-1:
	        node_color.append('gold')
	    if (i!=0)*(i!=N-1):
	        node_color.append('white')
	        
	nx.draw_networkx(G, pos, width=0, node_size=1000)
	nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=1000)
	nodes.set_edgecolor('black')


	# nx.draw_networkx_edges(G, pos, edge_color=edge_col, arrows=True, \
	#                        node_size=600, arrowstyle='->', arrowsize=50, width=5*np.asarray(weights),connectionstyle='arc3,rad=0.2')
	nx.draw_networkx_edges(G, pos, edge_color=edge_col, arrows=True, \
	                       node_size=600, arrowstyle='->', arrowsize=50, width=6,connectionstyle='arc3,rad=0.2')
	#nx.draw_networkx_edge_labels(G, pos, edge_color=edge_col, edge_labels=edge_labels, label_pos=0.3, font_size=15)
	plt.axis('off')
	sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=plt.Normalize(vmin = 0, vmax = 1))
	sm._A = []
	plt.colorbar(sm, fraction=0.02, ax=plt.gca())
	plt.gcf().set_size_inches(8,5)
	plt.savefig('graph_randomnet_copier_N%d_tstall%.2e_T%.2e.pdf' % (N, t_stall, T) )


def plot_k_t(time, track_k, track_kw, delta):

	k01 = [track_k[i][0][1] for i in range(np.asarray(track_k).shape[0]) ]
	k10 = [track_k[i][1][0] for i in range(np.asarray(track_k).shape[0]) ]
	k12 = [track_k[i][1][2] for i in range(np.asarray(track_k).shape[0]) ]
	k21 = [track_k[i][2][1] for i in range(np.asarray(track_k).shape[0]) ]
	k02 = [track_k[i][0][2] for i in range(np.asarray(track_k).shape[0]) ]
	k20 = [track_k[i][2][0] for i in range(np.asarray(track_k).shape[0]) ]

	fig, axs = plt.subplots(1,6, figsize=(20, 4))

	time = list(range(1,len(track_k)+1))

	axs[0].plot(time, k01, 'o', color='grey')
	axs[0].set_xlabel('MC steps')
	axs[0].set_ylabel(r'$k_{01}$')
	axs[0].set_yscale('log')
	axs[0].set_xscale('log')
	axs[0].set_ylim([1e-6,1e3])

	axs[1].plot(time, k10, 'o', color='grey')
	axs[1].set_ylabel(r'$k_{10}$')
	axs[1].set_xlabel('MC steps')
	axs[1].set_yscale('log')
	axs[1].set_xscale('log')
	axs[1].set_ylim([1e-6,1e3])

	axs[2].plot(time, k12, 'o', color='grey')
	axs[2].set_xlabel('MC steps')
	axs[2].set_ylabel(r'$k_{12}$')
	axs[2].set_yscale('log')
	axs[2].set_xscale('log')
	axs[2].set_ylim([1e-6,1e3])

	axs[3].plot(time, k21, 'o', color='grey')
	axs[3].set_xlabel('MC steps')
	axs[3].set_ylabel(r'$k_{21}$')
	axs[3].set_xscale('log')
	axs[3].set_yscale('log')
	axs[3].set_ylim([1e-6,1e3])

	axs[4].plot(time, k02, 'o', color='grey')
	axs[4].set_xlabel('MC steps')
	axs[4].set_ylabel(r'$k_{02}$')
	axs[4].set_xscale('log')
	axs[4].set_yscale('log')
	axs[4].set_ylim([1e-6,1e3])

	axs[5].plot(time, k20, 'o', color='grey')
	axs[5].set_xlabel('MC steps')
	axs[5].set_ylabel(r'$k_{20}$')
	axs[5].set_xscale('log')
	axs[5].set_yscale('log')
	axs[5].set_ylim([1e-6,1e3])

	fig.tight_layout()
	#plt.savefig('randomnet_copier_N%d_tstall%d_T_%d.pdf' % (N, t_stall, T) )
	#plt.savefig('randomnet_copier_fpt_arbitrary.pdf' )

	plt.show()

	k01w = [track_kw[i][0][1] for i in range(np.asarray(track_kw).shape[0]) ]
	k10w = [track_kw[i][1][0] for i in range(np.asarray(track_kw).shape[0]) ]
	k12w = [track_kw[i][1][2] for i in range(np.asarray(track_kw).shape[0]) ]
	k21w = [track_kw[i][2][1] for i in range(np.asarray(track_kw).shape[0]) ]
	k02w = [track_kw[i][0][2] for i in range(np.asarray(track_kw).shape[0]) ]
	k20w = [track_kw[i][2][0] for i in range(np.asarray(track_kw).shape[0]) ]

	fig, axs = plt.subplots(1,6, figsize=(20, 4))

	time = list(range(1,len(track_k)+1))

	axs[0].plot(time, np.asarray(k01w), 'o', color='grey')
	axs[0].set_xlabel('MC steps')
	axs[0].set_ylabel(r'$k_{01}^W$')
	axs[0].set_yscale('log')
	axs[0].set_xscale('log')
	xlim = axs[0].get_xlim()
	axs[0].plot(xlim, [np.exp(delta), np.exp(delta)], '--', color = 'blue', linewidth=3, alpha=0.6)
	axs[0].set_ylim([1e-6,1e3])

	axs[1].plot(time, k10w, 'o', color='grey')
	axs[1].set_ylabel(r'$k_{10}^W$')
	axs[1].set_xlabel('MC steps')
	axs[1].set_yscale('log')
	axs[1].set_xscale('log')
	xlim = axs[1].get_xlim()
	axs[1].plot(xlim, [np.exp(delta), np.exp(delta)], '--', color = 'blue', linewidth=3, alpha=0.6)
	axs[1].set_ylim([1e-6,1e3])

	axs[2].plot(time, k12w, 'o', color='grey')
	axs[2].set_xlabel('MC steps')
	axs[2].set_ylabel(r'$k_{12}^W$')
	axs[2].set_yscale('log')
	axs[2].set_xscale('log')
	xlim = axs[2].get_xlim()
	axs[2].plot(xlim, [np.exp(delta), np.exp(delta)], '--', color = 'blue', linewidth=3, alpha=0.6)
	axs[2].set_ylim([1e-6,1e3])

	axs[3].plot(time, k21w, 'o', color='grey')
	axs[3].set_xlabel('MC steps')
	axs[3].set_ylabel(r'$k_{21}^W$')
	axs[3].set_xscale('log')
	axs[3].set_yscale('log')
	xlim = axs[3].get_xlim()
	axs[3].plot(xlim, [np.exp(delta), np.exp(delta)], '--', color = 'blue', linewidth=3, alpha=0.6)
	axs[3].set_ylim([1e-6,1e3])

	axs[4].plot(time, k02w, 'o', color='grey')
	axs[4].set_xlabel('MC steps')
	axs[4].set_ylabel(r'$k_{02}^W$')
	axs[4].set_xscale('log')
	axs[4].set_yscale('log')
	xlim = axs[4].get_xlim()
	axs[4].plot(xlim, [np.exp(delta), np.exp(delta)], '--', color = 'blue', linewidth=3, alpha=0.6)
	axs[4].set_ylim([1e-6,1e3])

	axs[5].plot(time, k20w, 'o', color='grey')
	axs[5].set_xlabel('MC steps')
	axs[5].set_ylabel(r'$k_{20}^W$')
	axs[5].set_xscale('log')
	axs[5].set_yscale('log')
	xlim = axs[5].get_xlim()
	axs[5].plot(xlim, [np.exp(delta), np.exp(delta)], '--', color = 'blue', linewidth=3, alpha=0.6, label=r'$e^{\Delta}$')
	axs[5].set_ylim([1e-6,1e3])
	axs[5].legend(frameon=False)

	fig.tight_layout()
	#plt.savefig('randomnet_copier_N%d_tstall%d_T_%d.pdf' % (N, t_stall, T) )
	#plt.savefig('randomnet_copier_fpt_arbitrary.pdf' )
	plt.show()

def plot_observables(idx_s, delta, track_time_s, track_speed_s, track_error_s, track_ent_s, track_fpt_s, N, t_stall, T, n_samples):

	fig, axs = plt.subplots(2,2, figsize=(10, 8.5), sharex=True)

	for idx in range(n_samples):
	    time = np.asarray(track_time_s[idx])+1
	    
	    if idx == idx_s:
	        axs[0,0].plot(time, track_speed_s[idx], '-o', color='black', markersize=7)
	    else:
	        axs[0,0].plot(time, track_speed_s[idx], '-', color='grey', markersize=3, alpha=0.2)  
	    
	    if idx == idx_s:
	        axs[0,1].plot(time, track_error_s[idx], '-o', color='black', markersize=7)
	    else:
	        axs[0,1].plot(time, track_error_s[idx], '-', color='grey', markersize=3, alpha=0.2)
	        
	    if idx == idx_s:
	        axs[1,0].plot(time, np.asarray(track_fpt_s[idx]), '-o', color='black', markersize=7)
	    else:
	        axs[1,0].plot(time, np.asarray(track_fpt_s[idx]), '-', color='grey', markersize=3, alpha=0.2)


	    if idx == idx_s:
	        axs[1,1].plot(time, track_ent_s[idx], '-o', color='black', markersize=7)
	    else:
	        axs[1,1].plot(time, track_ent_s[idx], '-', color='grey', markersize=3, alpha=0.2)

	axs[0,0].set_ylabel('Time to copy strand')
	axs[0,0].set_yscale('log')
	axs[0,0].set_xscale('log')
	
	#axs[0,0].set_ylim([2e3,1e4])
	axs[0,0].set_xlim([1.5e2,1e4])
	axs[0,0].set_ylim([2e3,0.8e4])


	axs[0,1].set_ylabel(r'Mutation rate')
	axs[0,1].set_yscale('log')
	axs[0,1].set_xscale('log')
	xlim = axs[0,0].get_xlim()
	axs[0,1].plot(xlim, [np.exp(-delta), np.exp(-delta)], '--', color = 'black', linewidth=3, alpha=0.6, label=r'$e^{-\Delta}$')
	axs[0,1].plot(xlim, [np.exp(-(N-int(N/2))*delta), np.exp(-(N-int(N/2))*delta)], ':', color=(0.85, 0.37, 0.01), linewidth=3, \
	              alpha=0.6, label=r'$e^{-%d\Delta}$'%((N-int(N/2))))
	# axs[0,1].set_xlim([1e1,1e3])
	axs[0,1].legend(frameon=False)

	axs[1,1].set_ylabel("Entropy dissipation")
	axs[1,1].set_xlabel('MC steps')
	axs[1,1].set_xscale('log')
	
	if N>3:
		axs[1,1].set_yscale('log')
		axs[1,1].set_ylim([1e-2,1e3])

	axs[1,0].set_ylabel('Time to insert base')
	axs[1,0].set_xlabel('MC steps')
	axs[1,0].set_yscale('log')
	axs[1,0].set_xscale('log')
	
	# axs[1,0].set_xlim([1e1,1e3])
	axs[1,0].set_ylim([0.8e0,2e2])
	# axs[1,0].set_ylim([1e1,1e3])

	fig.tight_layout()
	plt.savefig('randomnet_copier_N%d_tstall%d_T_%d.pdf' % (N, t_stall, T) )
	plt.show()

def plot_errent(idx_s, delta, track_error, track_ent, alpha, N, T, n_samples, t_st):
	
	import matplotlib.ticker as mticker
	f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
	g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
	fmt = mticker.FuncFormatter(g)

	n = len(track_error)
	fig, axs = plt.subplots(1, n, figsize=(14, 4), sharex=True, sharey=True)

	colors = ['#80b1d3', '#fdb462', 'black', '#8dd3c7']
	for ii in range(n):
		for idx in range(n_samples):
		    
		    if idx == idx_s:
		        axs[ii].plot(alpha*np.asarray(track_ent[ii][idx]), track_error[ii][idx], '-o', color=colors[ii], markersize=7)
		    else:
		        axs[ii].plot(alpha*np.asarray(track_ent[ii][idx]), track_error[ii][idx], '-o', color=colors[ii], markersize=6, alpha=0.2)

	for ii in range(n):
		axs[0].set_ylabel(r'Mutation rate')
		axs[0].set_xlabel(r'$\alpha\times$Entropy dissipation')
		axs[ii].set_yscale('log')
		#axs[ii].set_xscale('log')
		xlim = axs[ii].get_xlim()
		axs[ii].plot(xlim, [np.exp(-delta), np.exp(-delta)], '--', color = 'black', linewidth=3, alpha=0.6, label=r'$e^{-\Delta}$')
		axs[ii].plot(xlim, [np.exp(-2*delta), np.exp(-2*delta)], ':', color=(0.85, 0.37, 0.01), linewidth=3, \
		              alpha=0.6, label=r'$e^{-2\Delta}$')
		axs[ii].set_title(r"$t_{{stall}}$={}".format(fmt(t_st[ii])) )
	    
	fig.tight_layout()
	plt.savefig('errent_N%d_T_%d.pdf' % (N, T) )
	plt.show()

