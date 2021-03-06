import pickle
import gzip
import torch
import torch_geometric
import numpy as np


class GraphDataset(torch_geometric.data.Dataset):
    # Basé sur https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb

    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores, sample_rewards,\
            terminal, next_sample_observation, next_sample_action_set = sample

        rewards = torch.from_numpy(np.array(list(sample_rewards.values())))
        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.from_numpy(constraint_features.astype(np.float32))
        edge_indices = torch.from_numpy(edge_indices.astype(np.int64))
        edge_features = torch.from_numpy(edge_features.astype(np.float32)).view(-1, 1)
        variable_features = torch.from_numpy(variable_features.astype(np.float32))
        terminal = torch.as_tensor(int(terminal))

        # We note on which variables we were allowed to branch, the scores as well as the choice
        # taken by strong branching (relative to the candidates)
        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])
        candidate_choice = torch.where(candidates == sample_action)[0][0]

        if not terminal:
            next_constraint_features, (next_edge_indices, next_edge_features), next_variable_features = next_sample_observation
            next_constraint_features = torch.from_numpy(next_constraint_features.astype(np.float32))
            next_edge_indices = torch.from_numpy(next_edge_indices.astype(np.int64))
            next_edge_features = torch.from_numpy(next_edge_features.astype(np.float32)).view(-1, 1)
            next_variable_features = torch.from_numpy(next_variable_features.astype(np.float32))
            next_candidates = torch.LongTensor(np.array(next_sample_action_set, dtype=np.int32))
        else:
            next_constraint_features = constraint_features.clone()
            next_edge_indices = edge_indices.clone()
            next_edge_features = edge_features.clone()
            next_variable_features = variable_features.clone()
            next_candidates = candidates.clone()

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, next_candidates, candidate_scores, candidate_choice, rewards, terminal, next_constraint_features,
                                  next_edge_indices, next_edge_features, next_variable_features)

        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]

        return graph


class BipartiteNodeData(torch_geometric.data.Data):
    # Basé sur https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb

    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, next_candidates, candidate_scores, candidate_choice, rewards, terminal, next_constraint_features,
                 next_edge_indices, next_edge_features, next_variable_features):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.next_constraint_features = next_constraint_features
        self.next_edge_index = next_edge_indices
        self.next_edge_attr = next_edge_features
        self.next_variable_features = next_variable_features
        self.candidates = candidates
        self.next_candidates = next_candidates
        self.candidate_scores = candidate_scores
        self.nb_candidates = len(candidates)
        self.next_nb_candidates = len(next_candidates)
        self.candidate_choice = candidate_choice
        self.rewards = rewards
        self.terminal = terminal

    def __inc__(self, key, value):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        elif key == 'next_edge_index':
            return torch.tensor([[self.next_constraint_features.size(0)], [self.next_variable_features.size(0)]])
        elif key == 'next_candidates':
            return self.next_variable_features.size(0)
        else:
            return super().__inc__(key, value)
