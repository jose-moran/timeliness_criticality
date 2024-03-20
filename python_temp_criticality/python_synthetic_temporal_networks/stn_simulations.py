# This file contains the code for the simulations of the synthetic temporal network model.
# It contains functions to simulate regular STNs, and STNs with heterogeneity in K, and sparsity.
# Some parts of code may be difficult to grasp at first, since numerous NumPy optimizations are used.
# Author: Matthijs Romeijnders, 2023-2024.

import numpy as np
import scipy


def synthetic_temporal_network(B, k, n, T, T_start):
    """Simulate a synthetic temporal network.

    Args:
        B (float): The uniform buffer
        k (int): The number of connections per node
        n (int): The number of nodes
        T (int): The number of time steps
        T_start (int): The number of time steps before we start measuring delays, to decorrelate the data from the initialization.
        
    Returns:
        mean_delay_propagation (np.array): The mean delay propagation over time.
        mean_delays (np.array): The mean delay over time.
    """
    
    # Initialize the several delay arrays used for the simulation and the output.
    delays_last_iteration = np.zeros((n))
    delays_current_iteration = np.zeros((n))
    delays_selected = np.zeros((n*k))
    mean_delays = np.zeros((T-T_start))
    mean_delay_propagation = np.zeros((T-T_start))
    
    # We loop over the timerange, and simulate the network.
    for t in range(0, T):
        # Reset the delays_selected array.
        delays_selected = np.zeros((n*k))
        
        # Prepare exponential delays
        eps = scipy.stats.expon.rvs(size=(n))
        
        # Shuffle the delays from the last iteration k times.
        for i in range(k):
            np.random.shuffle(delays_last_iteration)
            # Reformat them each time into the selected delays array, this ensures exactly k connections per node.
            # While making sure that each node has exactly k connections.
            delays_selected[i*n:(i+1)*n] = delays_last_iteration
            
        # Reformat the delays_selected array into a matrix.
        delays_selected = np.reshape(delays_selected, (n, k))
        
        # Take the maximum delay of each node, and subtract the buffer.
        max_values = np.max(delays_selected, axis=1) - B
        
        # Add the exponential random variable to the delays.
        delays_current_iteration = np.where(max_values < 0, 0, max_values) + eps
        
        # If we are past the initialization phase, we can start measuring the delays.
        if t >= T_start:
            # Calculate the mean delay and the mean delay propagation.
            mean_delays[t-T_start] = np.mean(delays_current_iteration)
            mean_delay_propagation[t-T_start] = np.mean(delays_current_iteration - delays_last_iteration)
            
        # Update the delays_last_iteration array.
        delays_last_iteration = delays_current_iteration
        
    return mean_delay_propagation, mean_delays


def synthetic_temporal_network_heterogeneous_K(B, k, n, T, T_start, heterogeneity_range):
    """Simulate a synthetic temporal network, with heterogeneity in K.

    Args:
        B (float): The uniform buffer
        k (int): The number of connections per node
        n (int): The number of nodes
        T (int): The number of time steps
        T_start (int): The number of time steps before we start measuring delays, to decorrelate the data from the initialization.
        heterogeneity_range (int): The range around k that we allow for heterogeneity in k.
        
    Returns:
        mean_delay_propagation (np.array): The mean delay propagation over time.
        mean_delays (np.array): The mean delay over time.
    """
    if k - heterogeneity_range < 1:
        raise ValueError("Heterogeneity range cannot be larger than or equal to k.")
    
    # Initialize the several delay arrays used for the simulation and the output.
    delays_last_iteration = np.zeros((n))
    delays_current_iteration = np.zeros((n))
    delays_selected = np.zeros((n*k))
    mean_delays = np.zeros((T-T_start))
    mean_delay_propagation = np.zeros((T-T_start))
    
    k_probabilities = np.ones(len(np.arange(k-heterogeneity_range, k+heterogeneity_range+1))) / (2*heterogeneity_range+1)
    
    # We loop over the timerange, and simulate the network.
    for t in range(0, T):
        # Take a random sample from the uniform distribution.
        k_curr = np.random.choice(np.arange(k-heterogeneity_range, k+heterogeneity_range+1), 1, p=k_probabilities)[0]
        # Reset the delays_selected array.
        delays_selected = np.zeros((n*k_curr))
        
        # Prepare exponential delays
        eps = scipy.stats.expon.rvs(size=(n))
        
        # Shuffle the delays from the last iteration k times.
        for i in range(k_curr):
            np.random.shuffle(delays_last_iteration)
            # Reformat them each time into the selected delays array, this ensures exactly k connections per node.
            # While making sure that each node has exactly k connections.
            delays_selected[i*n:(i+1)*n] = delays_last_iteration
            
        # Reformat the delays_selected array into a matrix.
        delays_selected = np.reshape(delays_selected, (n, k_curr))
        
        # Take the maximum delay of each node, and subtract the buffer.
        max_values = np.max(delays_selected, axis=1) - B
        
        # Add the exponential random variable to the delays.
        delays_current_iteration = np.where(max_values < 0, 0, max_values) + eps
        
        # If we are past the initialization phase, we can start measuring the delays.
        if t >= T_start:
            # Calculate the mean delay and the mean delay propagation.
            mean_delays[t-T_start] = np.mean(delays_current_iteration)
            mean_delay_propagation[t-T_start] = np.mean(delays_current_iteration - delays_last_iteration)
            
        # Update the delays_last_iteration array.
        delays_last_iteration = delays_current_iteration
        
    return mean_delay_propagation, mean_delays
    
    
def synthetic_temporal_network_sparsity(B, k, n, T, T_start, sparsity):
    """Simulate a synthetic temporal network, with sparsity.

    Args:
        B (float): The uniform buffer
        k (int): The number of connections per node
        n (int): The number of nodes
        T (int): The number of time steps
        T_start (int): The number of time steps before we start measuring delays, to decorrelate the data from the initialization.
        sparsity (float): The sparsity of the STN, ranges from 0 to 1. 100% sparsity reflects a network where no nodes interact,
                            50% reflects an STN where half the agents are in events each time step, 0% will give a normal STN.
        
    Returns:
        mean_delay_propagation (np.array): The mean delay propagation over time.
        mean_delays (np.array): The mean delay over time.
    """
    
    # Initialize the several delay arrays used for the simulation and the output.
    delays_last_iteration = np.zeros((n))
    delays_current_iteration = np.zeros((n))
    delays_selected = np.zeros((n*k))
    mean_delays = np.zeros((T-T_start))
    mean_delay_propagation = np.zeros((T-T_start))
    
    # We loop over the timerange, and simulate the network.
    for t in range(0, T):
        if sparsity != 0:
            # The probability of a node being selected is 1-sparsity.
            n_selected = np.random.choice([0, 1], size=n, p=[sparsity, 1-sparsity])
            delays_last_iteration_orginal = delays_last_iteration.copy()
        else:
            n_selected = n
            
        # Reset the delays_selected array.
        delays_selected = np.zeros((n*k))
        
        # Prepare exponential delays
        eps = scipy.stats.expon.rvs(size=(n))
        
        # Shuffle the delays from the last iteration k times.
        for i in range(k):
            np.random.shuffle(delays_last_iteration)
            # Reformat them each time into the selected delays array, this ensures exactly k connections per node.
            # While making sure that each node has exactly k connections.
            delays_selected[i*n:(i+1)*n] = delays_last_iteration
            
        # Reformat the delays_selected array into a matrix.
        delays_selected = np.reshape(delays_selected, (n, k))
        
        # Take the maximum delay of each node, and subtract the buffer.
        max_values = np.max(delays_selected, axis=1) - B
        
        if sparsity != 0: 
            # Only propagate the delays of the nodes that are selected.
            delays_current_iteration = np.where(n_selected == 1, np.where(max_values < 0, 0, max_values) + eps, delays_last_iteration_orginal)
        else:
            delays_current_iteration = np.where(max_values < 0, 0, max_values) + eps
        
        # If we are past the initialization phase, we can start measuring the delays.
        if t >= T_start:
            # Calculate the mean delay and the mean delay propagation.
            mean_delays[t-T_start] = np.mean(delays_current_iteration)
            mean_delay_propagation[t-T_start] = np.mean(delays_current_iteration - delays_last_iteration)
            
        # Update the delays_last_iteration array.
        delays_last_iteration = delays_current_iteration
        
    return mean_delay_propagation, mean_delays
       