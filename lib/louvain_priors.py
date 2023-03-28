from itertools import permutations
from itertools import combinations
from collections import defaultdict
import networkx as nx

import numpy as np

class Louvain(object):
    def __init__(self):
        self.MIN_VALUE = 0.0001
        self.node_weights = {}
        self.coupling_prior = np.zeros((10000, 10000))
        self.decoupling_prior = np.zeros((10000, 10000))
        self.prior_penalty = 1.0
        self.couples = []
        self.decouples = []
        self.all_edge_weights = None
    
    @classmethod
    def updateNodeWeights(self, edge_weights):
        node_weights = defaultdict(float)
        for node in edge_weights.keys():
            node_weights[node] = sum([weight for weight in edge_weights[node].values()])
        return node_weights

    def getBestPartition(self, graph, param=1., coupling_prior=np.zeros((10000, 10000)), decoupling_prior=np.zeros((10000, 10000)), prior_penalty=1.0):
        self.coupling_prior = coupling_prior
        self.decoupling_prior = decoupling_prior
        self.prior_penalty = prior_penalty
        self.couples = self.getCouplingPriorPairs()
        self.decouples = self.getDecouplingPriorPairs()
        node2com, edge_weights = self._createOneNodeCommunity(graph)
        self.all_edge_weights = nx.to_numpy_array(graph)
        partition = node2com.copy()
        node2com = self._runFirstPhase(node2com, edge_weights, param, partition) # map nodes to communities
        best_modularity = self.computeModularity(node2com, edge_weights, param, partition)
        
        partition = node2com.copy()

        new_node2com, new_edge_weights = self._runSecondPhase(node2com, edge_weights)

        partition = self._updatePartition(new_node2com, partition)

        while True:

            new_node2com = self._runFirstPhase(new_node2com, new_edge_weights, param, partition)
            partition = self._updatePartition(new_node2com, partition)
            modularity = self.computeModularity(new_node2com, new_edge_weights, param, partition)
            if abs(best_modularity - modularity) < self.MIN_VALUE:
                break
            #print("current modularity: ", modularity)
            best_modularity = modularity

            _new_node2com, _new_edge_weights = self._runSecondPhase(new_node2com, new_edge_weights)
            new_node2com = _new_node2com
            new_edge_weights = _new_edge_weights
        return partition

    def getDecouplingPriorPairs(self):
      decouple_priors = np.nonzero(self.decoupling_prior)
      result = []
      for row, col in zip(decouple_priors[0], decouple_priors[1]):
        result.append((row, col))
      return result
    def getCouplingPriorPairs(self):
      couple_priors = np.nonzero(self.coupling_prior)
      result = []
      for row, col in zip(couple_priors[0], couple_priors[1]):
        result.append((row, col))
      return result

    def computeModularity(self, node2com, edge_weights, param, partition):
      # param : higher indicates more weight placed on community separation; lower indicates smaller communities
      
        graph_modularity = 0
        all_edge_weights = sum([weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2 # divide by two because each edge added twice

        com2node = defaultdict(list) # mapping between each community as key to list of nodes idx
        for node, com_id in node2com.items():
            com2node[com_id].append(node)
        com2node_all = self._convertToComToNode(partition)
        for com_id, nodes in com2node.items(): # for each community, compute the community modularity

            node_combinations = list(combinations(nodes, 2)) + [(node, node) for node in nodes] # all pair wise nodes in a community
            inner_community_weights = sum([edge_weights[node_pair[0]][node_pair[1]] for node_pair in node_combinations]) # total edge weights within community
            tot = self.getClusterEdgeWeightsTotal(nodes, node2com, edge_weights) # all edge weights from nodes in current community to all other nodes (including other communities)

            prior_penalty = 0
            # priors which are violated lead to reduction in modularity
            for prior in self.decouples:
              if (partition[prior[0]] == com_id and partition[prior[1]] == com_id):
                #print("applied decoupling penalty to modularity")
                prior_penalty +=  self.getInnerCommunityEdgeWeights(prior[0],  self.all_edge_weights, com_id, com2node_all)
              
            for prior in self.couples:
              if (partition[prior[0]] == com_id and partition[prior[1]] == com_id):
                #print("applied coupling penalty to modularity")
                prior_penalty -= self.getInnerCommunityEdgeWeights(prior[0],  self.all_edge_weights, com_id, com2node_all)

            graph_modularity += (0.5 * inner_community_weights / all_edge_weights) - param * ((tot / (2 * all_edge_weights)) ** 2) - ( prior_penalty) # refer to existing paper or https://en.wikipedia.org/wiki/Louvain_method
        return graph_modularity

    def getClusterEdgeWeightsTotal(self, nodes_in_cluster, node2com, edge_weights):
        weight = sum([sum(list(edge_weights[n].values())) for n in nodes_in_cluster])
        return weight

    def getTotalEdgeWeights(self, edge_weights):
      return sum([weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2

    def _updatePartition(self, new_node2com, partition):
        reverse_partition = defaultdict(list)
        for node, com_id in partition.items():
            reverse_partition[com_id].append(node)

        for old_com_id, new_com_id in new_node2com.items():
            for old_com in reverse_partition[old_com_id]:
                partition[old_com] = new_com_id
        return partition

    def _runFirstPhase(self, node2com, edge_weights, param, partition):
        # assign each node to a community (via combining of communities)

        all_edge_weights = self.getTotalEdgeWeights(edge_weights)
        self.node_weights = self.updateNodeWeights(edge_weights)
        status = True
        while status:
            statuses = []
            for node in node2com.keys():
                statuses = []
                community_id = node2com[node]

                # neighbouring nodes of current node
                neigh_nodes = [edge[0] for edge in self.getNeighborNodes(node, edge_weights)]

                max_delta = 0.
                max_com_id = community_id
                communities = {}
                #com2node = self._convertToComToNode(node2com)
                for neigh_node in neigh_nodes:
                    node2com_new = node2com.copy() # new community dictionary representing after merging one community with another (or one node with another)
                    if node2com_new[neigh_node] in communities:
                        continue
                    communities[node2com_new[neigh_node]] = 1
                    node2com_new[node] = node2com_new[neigh_node]

                    previous_community = community_id
                    new_community_to_enter = node2com_new[neigh_node]
                    com2node = self._convertToComToNode(partition)
                    # nodes in this previous community:
                    nodes_leaving = com2node[previous_community]
                    # nodes in this new community:
                    nodes_joining = com2node[new_community_to_enter]
                    #if nodes_leaving == nodes_joining:
                    #  continue

                    change_in_prior_penalty = self._getConflictsWithPriors(nodes_leaving, nodes_joining, previous_community, new_community_to_enter, com2node)
                    #if (change_in_prior_penalty != 0):
                    #  print(change_in_prior_penalty)

                    # compute change in total modularity (see https://en.wikipedia.org/wiki/Louvain_method for exact formulation)
                    # find the merging of community such that the change (reduction) in modularity is the greatest
                    change_modularity = 2 * self.getNodeWeightInCluster(node, node2com_new, edge_weights) - (self.getTotWeight(node, node2com_new, edge_weights) * self.node_weights[node] / all_edge_weights) * param - change_in_prior_penalty
                    
                    if change_modularity > max_delta:
                        max_delta = change_modularity
                        max_com_id = node2com_new[neigh_node]
                node2com[node] = max_com_id # join a new community
                statuses.append(community_id != max_com_id)

            if sum(statuses) == 0:
                break

        return node2com

    def _runSecondPhase(self, node2com, edge_weights):
        # combine communities into a single node

        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda : defaultdict(float))

        for node, com_id in node2com.items():
            com2node[com_id].append(node)
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        nodes = list(node2com.keys())
        node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
        for edge in node_pairs:
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][edge[1]]
        return new_node2com, new_edge_weights

    def getInnerCommunityEdgeWeights(self, node, all_edge_weights, community_idx, com2node):
        return self.prior_penalty
    
    def getTotWeight(self, node, node2com, edge_weights):
        nodes = [n for n, com_id in node2com.items() if com_id == node2com[node] and node != n]

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    def getNeighborNodes(self, node, edge_weights):
        if node not in edge_weights:
            return 0
        return edge_weights[node].items()

    def getNodeWeightInCluster(self, node, node2com, edge_weights):
        neigh_nodes = self.getNeighborNodes(node, edge_weights)
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]
        return weights
    
    def _createOneNodeCommunity(self, graph):
      # form a community of one node each from graph
        node2com = {}
        edge_weights = defaultdict(lambda : defaultdict(float))
        for idx, node in enumerate(graph.nodes()):
            node2com[node] = idx
            for edge in graph[node].items():
                edge_weights[node][edge[0]] = edge[1]["weight"]
        return node2com, edge_weights
    
    def _convertToComToNode(self, node2com):
      com2node = defaultdict(list) # mapping between each community as key to list of nodes idx
      for node, com_id in node2com.items():
          com2node[com_id].append(node)
      return com2node

    def _getConflictsWithPriors(self, nodes_leaving, nodes_joining, prev_com, new_com, com2node):
      
      prior_conflict_cost = 0
      for leaving in nodes_leaving:
        for joining in nodes_joining:
          if self.decoupling_prior[joining][leaving]: # if two decoupled nodes join, apply positive penalty
            prior_conflict_cost += self.getInnerCommunityEdgeWeights(leaving, self.all_edge_weights, new_com, com2node)
          if self.coupling_prior[joining][leaving]: # if two couple nodes join, apply negative penalty
            prior_conflict_cost -= self.getInnerCommunityEdgeWeights(leaving, self.all_edge_weights, new_com, com2node)
  

      return prior_conflict_cost
