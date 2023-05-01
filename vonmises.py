import networkx as nx
import numpy as np
import torch

class VonMisesGrid():
    def __init__(self, grid_size):
        num_nodes = grid_size ** 2
        self.num_nodes = num_nodes
        self.big_lambda = self._init_big_lamda(num_nodes)
        self.mu = torch.randn(num_nodes)
        self.kappa = torch.randn(num_nodes)

    def _init_big_lambda(self, grid_size):
        G = nx.grid_2d_graph(grid_size, grid_size)
        mask = nx.adjacency_matrix(G).toarray()
        big_lambda = mask * np.random.randn(grid_size ** 2 , grid_size ** 2)
        zero_inds = np.tril_indices_from(big_lambda,-1)
        big_lambda[zero_inds] = 0        
        big_lambda += big_lambda.T.copy()
        return big_lambda
    
    def _gibbs_sample(self, node):
        a = self.kappa[node] * torch.cos(self.theta[node] - self.mu[node])
        b = torch.reduce_sum(self.big_lambda[node, :] * torch.sin(self.theta - self.mu) * torch.sin(self.theta[node] - self.mu[node]))
        return torch.exp(a + b)
        
    def _all_but(self, array, idx):
        return array[array]