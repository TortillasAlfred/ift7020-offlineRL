from core import GraphDataset, GNNPolicy
from core.utils import pad_tensor, save_work_done, update_CQL_config_for_job_index

import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric
import glob
from pathlib import Path
from collections import defaultdict
import pickle
import argparse

class CQL:
    # TODO: Logging - Training (and val) loop
    def __init__(self, q_network, gamma=0.99, reward="nb_nodes", alpha=1.0, bc_network=None, target_update_interval=10):
        self.q_network = q_network
        self.target_q_network = copy.deepcopy(q_network)
        self.reward = reward
        self.reward_index, self.reward_sign = self._get_reward_idx_sign(reward)
        self.gamma = gamma
        self.alpha = alpha
        self.bc_network = bc_network
        self.target_update_interval = target_update_interval

        self.n_steps_done = 0

    def get_loss(self, batch):
        batch_size = batch.nb_candidates.shape[0]
        current_state = (batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features, batch.candidates, batch.nb_candidates)
        next_state = (batch.next_constraint_features, batch.next_edge_index, batch.next_edge_attr, batch.next_variable_features, batch.next_candidates, batch.next_nb_candidates)
        
        # DQN Loss
        a_t = F.one_hot(batch.candidate_choice.view(-1), num_classes=batch.nb_candidates.max())
        full_q_t = self._get_network_pred(self.q_network, *current_state)
        q_t = (full_q_t * a_t.float()).sum(dim=1, keepdim=True)

        with torch.no_grad():
            q_tp1 = self._get_network_pred(self.q_network, *next_state)
            next_action = q_tp1.argmax(dim=1)
            next_action = F.one_hot(next_action.view(-1), num_classes=batch.next_nb_candidates.max())
            targ_q_tp1 = self._get_network_pred(self.target_q_network, *next_state)
            q_tp1 = (targ_q_tp1 * next_action.float()).sum(dim=1, keepdim=True)

            r = batch.rewards.view(batch_size, -1)[:, self.reward_index] * self.reward_sign 
            targ_q_t = r + self.gamma * q_tp1 * (1 - batch.terminal)

        dqn_loss = ((targ_q_t - q_t) ** 2).mean()

        # CQL Loss
        lse = torch.logsumexp(full_q_t, dim=1, keepdim=True)

        if self.bc_network:
            with torch.no_grad():
                action_logits = self._get_network_pred(self.bc_network, *current_state)
                action_likelihoods = F.softmax(action_logits, dim=-1)
                data_values = (full_q_t * action_likelihoods).sum(dim=1, keepdim=True)
        else:
            data_values = q_t

        cql_loss = (lse - data_values).mean()

        return dqn_loss, cql_loss, (dqn_loss + self.alpha * cql_loss).mean()

    def _get_network_pred(self, network, constraint_features, edge_index, edge_attr, variable_features, candidates, nb_candidates):
        preds = network(constraint_features, edge_index, edge_attr, variable_features)
        return pad_tensor(preds[candidates], nb_candidates)

    def _get_reward_idx_sign(self, reward):
        if reward == "nb_nodes":
            return 0, -1
        elif reward == "lp_iterations":
            return 1, -1
        elif reward == "time":
            return 2, -1
        else:
            raise NotImplementedError(f"The reward function {reward} is not implemented yet.")

    def update(self):
        self.n_steps_done += 1

        if (self.n_steps_done % self.target_update_interval) == 0:
            self._update_target()

    def _update_target(self):
        with torch.no_grad():
            params = self.q_network.parameters()
            targ_params = self.target_q_network.parameters()
            for p, p_targ in zip(params, targ_params):
                p_targ.data.copy_(p.data)

    def to(self, device):
        self.q_network = self.q_network.to(device)
        self.target_q_network = self.target_q_network.to(device)
        
        if self.bc_network:
            self.bc_network = self.bc_network.to(device)

def do_epoch(learner, data_loader, optimizer, device='cuda'):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_dqn_loss = 0
    mean_cql_loss = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(True):
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            
            dqn_loss, cql_loss, loss = learner.get_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            learner.update()

            mean_loss += loss.item() * batch.num_graphs
            mean_dqn_loss += dqn_loss.item() * batch.num_graphs
            mean_cql_loss += cql_loss.item() * batch.num_graphs

            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_dqn_loss /= n_samples_processed
    mean_cql_loss /= n_samples_processed

    return mean_dqn_loss, mean_cql_loss, mean_loss

def train_gnn_rl(config):
    # Train GNN
    LEARNING_RATE = 3e-4
    NB_EPOCHS = 150
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trajectories_path = f'{config.working_path}/data/collections/{config.collection_name}/{config.trajectories_name}/'
    models_path = f'{config.working_path}/models/'
    results_path = f'{config.working_path}/train_results/'

    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    sample_files = [str(file_path) for file_path in glob.glob(f'{trajectories_path}*')]
    sample_files.sort()
    train_files = sample_files

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, 
                                                   batch_size=config.train_batch_size, 
                                                   num_workers=config.num_workers, 
                                                   pin_memory=True, 
                                                   shuffle=True)

    q_network = GNNPolicy()

    # if load_bc_network:
    #     bc_network = GNNPolicy()
    #     bc_network.load_state_dict(torch.load(f'{path}/trained_params/GCNN_trained_params.pkl'))
    # else:
    #     bc_network = None

    cql = CQL(q_network)
    cql.to(DEVICE)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    prev_min_val_nodes = 1e10 # ~ inf
    n_steps_per_epoch = len(train_loader)
    n_steps_done = 0
    train_results = defaultdict(list)

    for epoch in range(NB_EPOCHS):
        print(f"Epoch {epoch + 1}")

        dqn_loss, cql_loss, train_loss = do_epoch(cql, train_loader, optimizer, device=DEVICE)
        print(f"Train loss: {train_loss:0.3f}, DQN loss : {dqn_loss:0.3f}, CQL loss : {cql_loss:0.3f}")

        # TODO: Whole validation loop

        n_steps_done += n_steps_per_epoch

        train_results["train_loss"].append((n_steps_done, train_loss))
        train_results["train_dqn_loss"].append((n_steps_done, dqn_loss))
        train_results["train_cql_loss"].append((n_steps_done, cql_loss))

        with open(f'{results_path}/{config_name}.pkl', 'wb') as f:
            pickle.dump(train_results, f)

        save_work_done(config.working_path, config.saving_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_instances', type=int, default=1000)
    parser.add_argument('--nb_trajectories', type=int, default=3)
    parser.add_argument('--working_path', type=str, default='.') # This is where we will work from 
    parser.add_argument('--saving_path', type=str, default='.') # This is where we will persistently save the files
    parser.add_argument('--expert_probability', type=float, default=0.0)
    parser.add_argument('--job_index', type=int, default=-1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--reward', type=str, default='nb_nodes')
    parser.add_argument('--use_bc', type=int, default=0) # Fake boolean

    args = parser.parse_args()

    # Recast fake booleans to true booleans
    args.use_bc = bool(args.use_bc)    
    
    config_name, args = update_CQL_config_for_job_index(args)

    # Default vals
    args.collection_name = f'{args.nb_instances}_instances_collection'
    args.trajectories_name = f'{args.nb_trajectories}_trajectories_expert_{args.expert_probability}'

    print("\n\nRunning with the following config :\n")
    for key, value in args.__dict__.items():
        print(f"{key:<30}: {value}")
    print("\n")

    train_gnn_rl(args)