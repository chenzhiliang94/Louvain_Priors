# Load libraries

# Math
import numpy as np
import pandas as pd

# Import data
import scipy.io
import time

# Import helper functions
from lib.utils import compute_purity
from lib.utils import construct_knn_graph
from lib.louvain_priors import Louvain
from lib.utils import read_priors

from community import community_louvain # for checking of accuracy of louvain
import networkx as nx

# Remove warnings
import warnings
warnings.filterwarnings("ignore")
import copy
import math

def get_clusters(custom_partition):
    nc_louvain = len(np.unique( [custom_partition[nodes] for nodes in custom_partition.keys()] ))
    n = len(Wnx.nodes())
    C = np.zeros([n])
    clusters = []
    k = 0
    for com in set(custom_partition.values()):
        list_nodes = [nodes for nodes in custom_partition.keys() if custom_partition[nodes] == com]
        C[list_nodes] = k
        k += 1
        clusters.append(list_nodes)
    return C

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

    # get ground truth from pokemon data
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

    # custom louvain with priors
    custom_Louvain = Louvain()
    coupling_priors, decoupling_priors = read_priors(W.shape[0])
    custom_partition = custom_Louvain.getBestPartition(Wnx, 1, coupling_priors, decoupling_priors, 0)
    nc_louvain_prior = len(np.unique( [custom_partition[nodes] for nodes in custom_partition.keys()] ))

    GTClusters,partial_Clouv_no_priors = get_ground_truth_partial_cluster(get_clusters(custom_partition))

    # Baseline accuracy
    Crand = np.random.randint(0,8,[len(GTClusters)])
    acc = compute_purity(Crand,GTClusters,8)
    
    # Louvain partition with existing library to check for our implementation accuracy
    # partition = community_louvain.best_partition(Wnx)
    # nc_louvain_lib = len(np.unique( [partition[nodes] for nodes in partition.keys()] ))
    # GTClusters,partial_Clouv_lib = get_ground_truth_partial_cluster(get_clusters(partition))
 
    # custom louvain with no priors
    custom_Louvain = Louvain()
    custom_partition = custom_Louvain.getBestPartition(Wnx, 1, coupling_priors, decoupling_priors, 50)
    nc_louvain_no_prior = len(np.unique( [custom_partition[nodes] for nodes in custom_partition.keys()] ))
    GTClusters,partial_Clouv = get_ground_truth_partial_cluster(get_clusters(custom_partition))
    
    print("\n")
    print("baseline purity score: ", acc)
    #print("louvain external library purity score: ", compute_purity(partial_Clouv_lib, GTClusters, nc_louvain_lib))
    print("louvain own code with no-priors purity score : ", compute_purity(partial_Clouv_no_priors, GTClusters, nc_louvain_no_prior))
    print("louvain with priors purity score: ", compute_purity(partial_Clouv, GTClusters, nc_louvain_prior))