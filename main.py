# Load libraries

# Math
import numpy as np
import pandas as pd

# Import data
import scipy.io

from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6), dpi=80)
plt.rcParams.update({'figure.max_open_warning': 0})
import time

# Import helper functions
from lib.utils import compute_purity
from lib.utils import construct_knn_graph
from lib.louvain_priors import Louvain
from lib.utils import read_priors

# Louvain algorithm
import community
import networkx as nx

# Remove warnings
import warnings
warnings.filterwarnings("ignore")
import copy
import math

if __name__ == '__main__':
    
    data = pd.read_csv('data/pokemon_ground_truth.csv')
    ground_truth_data = copy.deepcopy(data)

    # replace popular NaN with 0
    #data['Popular'] = data['Popular'].fillna(0)
    # replace Type 2 with Unknown
    data['Type 2'] = data['Type 2'].fillna("Unknown")
    # drop unneeded columns (guess abit)
    data = data.drop(['Abilities', 'Name', 'Generation', 'Popular', 'Experience type', 'Number', 'Type 1', 'Type 2', 'GTCluster'], axis=1)

    # one hot encoding for type
    numerical_data = pd.get_dummies(data)

    # min-max [0,1] scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(numerical_data), columns=numerical_data.columns)

    def get_ground_truth_partial_cluster(louvain_communities):
        # returns a list of numerical cluster e.g [0,0,0,1,1,1...] of a smaller list size (since only partial labels are available)
        ground_truth_communities = []
        louvain_partial_communities = []

        for idx, row in ground_truth_data.iterrows():
            if not math.isnan(row['GTCluster']):
                louvain_partial_communities.append(louvain_communities[idx])
                ground_truth_communities.append(row['GTCluster'])
        
        return np.array(ground_truth_communities), np.array(louvain_partial_communities)

    # construct knn graph of pokemon data
    W = construct_knn_graph(data_scaled.to_numpy(),10,dist='euclidean')
    Wnx = nx.from_numpy_array(W)

    custom_Louvain = Louvain()
    coupling_priors, decoupling_priors = read_priors(W.shape[0])
    custom_partition = custom_Louvain.getBestPartition(Wnx, 1.5, coupling_priors, decoupling_priors, 0)
    nc_louvain = len(np.unique( [custom_partition[nodes] for nodes in custom_partition.keys()] ))
    n = len(Wnx.nodes())
    print("")

    # Extract clusters
    Clouv = np.zeros([n])
    clusters = []
    k = 0
    for com in set(custom_partition.values()):
        list_nodes = [nodes for nodes in custom_partition.keys() if custom_partition[nodes] == com]
        Clouv[list_nodes] = k
        k += 1
        clusters.append(list_nodes)

    GTClusters,partial_Clouv_no_priors = get_ground_truth_partial_cluster(Clouv)

    # Baseline accuracy
    Crand = np.random.randint(0,8,[len(GTClusters)])
    acc = compute_purity(Crand,GTClusters,8)
 
    custom_Louvain = Louvain()
    custom_partition = custom_Louvain.getBestPartition(Wnx, 1.5, coupling_priors, decoupling_priors, 50)
    nc_louvain = len(np.unique( [custom_partition[nodes] for nodes in custom_partition.keys()] ))
    n = len(Wnx.nodes())
    

    # Extract clusters
    Clouv = np.zeros([n])
    clusters = []
    k = 0
    for com in set(custom_partition.values()):
        list_nodes = [nodes for nodes in custom_partition.keys() if custom_partition[nodes] == com]
        Clouv[list_nodes] = k
        k += 1
        clusters.append(list_nodes)
    GTClusters,partial_Clouv = get_ground_truth_partial_cluster(Clouv)
    print("baseline purity score: ", acc)
    print("louvain with no-priors purity score: ", compute_purity(partial_Clouv_no_priors, GTClusters, nc_louvain))
    print("louvain with priors purity score: ", compute_purity(partial_Clouv, GTClusters, nc_louvain))