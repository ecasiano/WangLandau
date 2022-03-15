# Wang-Landau simulation

from webbrowser import get
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time

# Define functions

'''--------------------------------------------------------------------------'''

def create_ising_lattice(L):

    ising_lattice = np.random.random((L,L))
    ising_lattice[ising_lattice>=0.5] = 1
    ising_lattice[ising_lattice<0.5] = -1

    return ising_lattice

'''--------------------------------------------------------------------------'''

def get_histogram_edges(E_min,E_max,num_bins):

    edges=np.histogram_bin_edges(np.ones(num_bins),bins=num_bins,
    range=(E_min,E_max))

    return edges

'''--------------------------------------------------------------------------'''

def get_bin_size(edges):

    return edges[1]-edges[0]

'''--------------------------------------------------------------------------'''

def get_index(energy,E_min,bin_size):

    return np.int(np.floor((energy-E_min)/bin_size))

'''--------------------------------------------------------------------------'''

def update_histogram_dos(histogram,dos,visited,E_index,mod_factor,num_bins):

    index = E_index
    # print(E_index,visited)
    if visited[index]==0:
        if index==0:
            refIdx = 1
        elif index==num_bins-1:
            refIdx = index-1
        else:
            if visited[index-1]>0 and visited[index+1]>0:
                refIdx=(index-1) if (dos[index-1]<dos[index+1]) else (index+1)
            elif visited[index-1]==0:
                refIdx = index+1
            elif visited[index+1]==0:
                refIdx = index-1
        dos[index] = dos[refIdx]
        visited[index] = 1
        histogram[:] = 0.0
    
    else:
        dos[index] += mod_factor
        histogram[index] += 1

    # dos[index] += mod_factor
    # histogram[index] += 1
'''--------------------------------------------------------------------------'''

def spin_flip(ising_lattice,L,x_indices,y_indices,dos,mod_factor,histogram,
num_bins,E_min,bin_size,edges,visited):

    "Chooses a spin randomly and proposes to flip it"

    random_spin_index = int(np.random.random()*L**2)

    x = x_indices[random_spin_index]
    y = y_indices[random_spin_index]

    # Compute energy of spin and neighboring bonds before update
    E_old = 0
    E_new = 0
    if x != L-1:
        E_old -= ising_lattice[x,y]*ising_lattice[x+1,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x+1,y]
    else:
        E_old -= ising_lattice[x,y]*ising_lattice[0,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[0,y]

    if y != L-1:
        E_old -= ising_lattice[x,y]*ising_lattice[x,y+1]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,y+1]
    else:
        E_old -= ising_lattice[x,y]*ising_lattice[x,0]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,0]
    
    if x != 0:
        E_old -= ising_lattice[x,y]*ising_lattice[x-1,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x-1,y]
    else:
        E_old -= ising_lattice[x,y]*ising_lattice[L-1,y]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[L-1,y]

    if y != 0:
        E_old -= ising_lattice[x,y]*ising_lattice[x,y-1]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,y-1]
    else:
        E_old -= ising_lattice[x,y]*ising_lattice[x,L-1]
        E_new -= (-1)*ising_lattice[x,y]*ising_lattice[x,L-1]
    
    # Calculate energy difference between new and old configurations
    E_flip = E_new - E_old

    # Compute total energies of old and new configurations
    E_1 = ising_energy(ising_lattice,L)
    E_2 = E_1 + E_flip

    # Get histogram and dos index of old and new energies
    E_1_index = get_index(E_1,E_min,bin_size)
    E_2_index = get_index(E_2,E_min,bin_size)

    # Wang-Landau sampling
    r = np.random.random()

    # print(histogram)
    # print(E_1_index,E_2_index,histogram[E_1_index],histogram[E_2_index])
    # print("")
    # NOTE: With dictionary method, E_i_index will be key's instead, as strings
    # print(histogram,dos,E_1,E_2,E_1_index,E_2_index,"num_bins:",
    #num_bins,visited,edges,bin_size)
    if r < np.exp(dos[E_1_index]-dos[E_2_index]):

        # Flip spin
        ising_lattice[x,y] *= -1
        # print(dos[E_2_index],np.log(mod_factor))

        # Update density of states and histogram
        update_histogram_dos(histogram,dos,visited,E_2_index,mod_factor,
        num_bins)

    else:

        # Update density of states and histogram
        update_histogram_dos(histogram,dos,visited,E_2_index,mod_factor,
        num_bins) 


'''--------------------------------------------------------------------------'''

def ising_energy(ising_lattice, L):
    "Computes energy of LxL Ising configuration"

    E = 0
    for i in range(L):
        for j in range(L):
            # Accumulate 'horizontal' bond
            if i != L-1:
                E -= ising_lattice[i,j]*ising_lattice[i+1,j] 
            else:
                E -= ising_lattice[i,j]*ising_lattice[0,j]
            
             
            # Accumulate 'vertical' bond
            if j != L-1:
                E -= ising_lattice[i,j]*ising_lattice[i,j+1] 
            else:
                E -= ising_lattice[i,j]*ising_lattice[i,0]
    
    return E

'''--------------------------------------------------------------------------'''

def is_histogram_flat_old(histogram,flatness_criterion):

    mean = np.mean(histogram)
    deviations = np.abs(histogram - mean)/mean

    return (deviations < 1-flatness_criterion).all()

'''--------------------------------------------------------------------------'''

def is_histogram_flat_old_2(histogram,flatness_criterion):

    visited_bin_indices = np.where(histogram>0)
    histogram_with_just_the_visited_bins = histogram[visited_bin_indices]
    mean = np.mean(histogram_with_just_the_visited_bins)
    # flatness_reference = flatness_criterion*mean

    deviations = np.abs(histogram_with_just_the_visited_bins - mean)/mean

    return (deviations < 1-flatness_criterion).all()
    # return (histogram_with_just_the_visited_bins<flatness_reference).all()

'''--------------------------------------------------------------------------'''

def is_histogram_flat(histogram,flatness_criterion,visited,num_bins):

    num_visited_bins = 0
    sum_entries = 0

    for i in range(num_bins):
        if visited[i]==1:
            sum_entries += histogram[i]
            num_visited_bins += 1

    flatness_reference = flatness_criterion * sum_entries/num_visited_bins

    num_bins_failing_criterion = 0
    for i in range(num_bins):
        if visited[i]==1:
            if histogram[i] < flatness_reference:
                num_bins_failing_criterion += 1

    return num_bins_failing_criterion==0

'''--------------------------------------------------------------------------'''

# Main
def main():
    np.random.seed(2000)

    # Wang-Landau Sampling inputs
    dim = 2
    flatness_criterion = 0.6
    mod_factor = 1 # actually log of mod_factor: log(f) = log(e^1)
    mod_factor_final = 1.25000000e-06
    mod_factor_reducer = 2
    histogram_check_interval = 10000
    histogram_refresh_interval = 1000000
    bin_size = 1

    L = 4
    total_sites = L**dim
    E_min = -2*total_sites
    E_max = +2*total_sites

    ising_lattice = create_ising_lattice(L)

    # Initialize histogram
    num_bins = 5
    bin_size = (E_max-E_min)/num_bins
    edges = get_histogram_edges(E_min-bin_size/2,E_max+bin_size/2,num_bins)
    bin_size = get_bin_size(edges) # Actual bin_size after recentering
    histogram = np.zeros(num_bins) 
    dos = np.zeros(num_bins) # Actually log(G(E))
    visited = np.zeros(num_bins)
    # print(edges,bin_size)

    # Should update histogram,dos for first energy visited
    E_1 = ising_energy(ising_lattice,L)
    E_1_index = get_index(E_1,E_min,bin_size)
    update_histogram_dos(histogram,dos,visited,E_1_index,mod_factor,num_bins)

    # Get bin centers for plotting
    bin_centers = np.zeros(edges.size-1)
    for i in range(edges.size-1):
        bin_centers[i] = (edges[i+1]+edges[i])/2

    # Initialize arrays that store x,y indices of spin (basically a meshgrid) 
    flattened_spin_indices = np.arange(L**2)
    x_indices,y_indices = np.unravel_index(flattened_spin_indices,(L,L))

    while (mod_factor > mod_factor_final):

        histogram_flat = False

        # if (m%(total_sites*skip)==0 and m>10000):
        while(not(histogram_flat)):
            print(histogram)
            for mc_steps in range(histogram_check_interval):

                spin_flip(ising_lattice,L,x_indices,y_indices,dos,mod_factor,
                histogram,num_bins,E_min,bin_size,edges,visited)

            is_histogram_flat(histogram,flatness_criterion,visited,num_bins)

        # # Plot log(G(E)) iteratively
        # plt.plot(bin_centers,dos)
        # plt.ylabel(r'$\log(g(E))$')
        # plt.xlabel(r'$E$')
        # plt.xlim(E_min,E_max)
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.close()

        print("modification factor: ",mod_factor)
        print("edges: ",edges)
        print("histogram (when it reaches flatness): ",histogram)
        print("log(g(E)): ",dos,"\n")

        # Write dos to disk

        # Reset histogram
        histogram[:] = 0.0

        # Reduce modification factor
        mod_factor /= mod_factor_reducer


if __name__ == "__main__":
    main()