import pickle
import gzip
import torch
import torch_geometric
import numpy as np
import os


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, collection_name='collection', root='.'):
        super().__init__(root=None, transform=None, pre_transform=None)
        collection_path = f'{root}/collections/{collection_name}/train_trajectories/'

        self.dataset = []

        for _, directories, _ in os.walk(collection_path):
            for directory in directories:
                for _, _, files in os.walk(collection_path + directory):
                    for file in files:
                        self.dataset.append(f'{collection_path}{directory}/{file}')

    def len(self):
        return len(self.dataset)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.dataset[index], 'rb') as f:
            trajectory = pickle.load(f)

        graphs = []

        for sample in trajectory:

            sample_observation, sample_action, sample_action_set, sample_rewards = sample

            rewards = torch.from_numpy(np.array(list(sample_rewards.values())))
            constraint_features, (edge_indices, edge_features), variable_features = sample_observation
            constraint_features = torch.from_numpy(constraint_features.astype(np.float32))
            edge_indices = torch.from_numpy(edge_indices.astype(np.int64))
            edge_features = torch.from_numpy(edge_features.astype(np.float32)).view(-1, 1)
            variable_features = torch.from_numpy(variable_features.astype(np.float32))

            # We note on which variables we were allowed to branch, the scores as well as the choice
            # taken by strong branching (relative to the candidates)
            candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
            candidate_choice = torch.where(candidates == sample_action)[0][0]

            graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                      candidates, candidate_choice, rewards)

            # We must tell pytorch geometric how many nodes there are, for indexing purposes
            graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
            graphs.append(graph)

        return graphs


class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, candidate_choice, rewards):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = len(candidates)
        self.candidate_choice = candidate_choice
        self.rewards = rewards

    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value)