from core import GraphDataset, GNNPolicy
from core.utils import pad_tensor, save_work_done, update_CQL_config_for_job_index, set_seed

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
import os
import numpy as np
import ecole

class CQL:
    def __init__(self, q_network, gamma=0.95, reward="lp-iterations", alpha=1.0, bc_network=None, target_update_interval=1000):
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
        q_t = (full_q_t * a_t.float()).sum(dim=1, keepdim=False)

        with torch.no_grad():
            q_tp1 = self._get_network_pred(self.q_network, *next_state)
            next_action = q_tp1.argmax(dim=1)
            next_action = F.one_hot(next_action.view(-1), num_classes=batch.next_nb_candidates.max())
            targ_q_tp1 = self._get_network_pred(self.target_q_network, *next_state)
            q_tp1 = (targ_q_tp1 * next_action.float()).sum(dim=1, keepdim=False)

            r = batch.rewards.view(batch_size, -1)[:, self.reward_index] * self.reward_sign
            targ_q_t = r + self.gamma * q_tp1 * (1 - batch.terminal)

        dqn_loss = ((targ_q_t - q_t) ** 2).mean()

        # CQL Loss
        lse = torch.logsumexp(full_q_t, dim=1, keepdim=False)

        if self.bc_network:
            with torch.no_grad():
                action_logits = self._get_network_pred(self.bc_network, *current_state)
                action_likelihoods = F.softmax(action_logits, dim=-1)
                data_values = (full_q_t * action_likelihoods).sum(dim=1, keepdim=True)
        else:
            data_values = q_t

        cql_loss = (lse - data_values).mean()

        return dqn_loss, cql_loss, dqn_loss + self.alpha * cql_loss

    def _get_network_pred(self, network, constraint_features, edge_index, edge_attr, variable_features, candidates, nb_candidates):
        preds = network(constraint_features, edge_index, edge_attr, variable_features)
        preds = -F.relu(preds)
        return pad_tensor(preds[candidates], nb_candidates)

    def _get_reward_idx_sign(self, reward):
        if reward == "nb-nodes":
            return 0, -1
        elif reward == "lp-iterations":
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
        for batch in tqdm(data_loader, "train epoch..."):
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

def load_valid_instances(config):
    valid_instances_path = f'{config.working_path}/data/collections/{config.collection_name}/validation_instances/'
    
    Path(f'{valid_instances_path}').mkdir(parents=True, exist_ok=True)
    loaded_instances = []
    for _, _, files in os.walk(valid_instances_path):
        for file in files:
            instance = ecole.scip.Model.from_file(valid_instances_path+file)
            loaded_instances.append(instance)

    return loaded_instances

def test_model_on_instances(model, instances, device, n_runs=1):
    mean_solve_time = 0.0
    mean_nb_nodes = 0.0
    mean_lp_iters = 0.0
    nb_runs_processed = 0

    # We can pass custom SCIP parameters easily
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                        scip_params=scip_parameters,
                                        information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                                                              "time": ecole.reward.SolvingTime().cumsum(),
                                                              "lp_iters": ecole.reward.LpIterations().cumsum()})

    for instance in tqdm(instances, "Processing val instances..."):
        for run in range(n_runs):
            env.seed(run)
            observation, action_set, _, done, info = env.reset(instance)

            while not done:
                with torch.no_grad():
                    observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                           torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device), 
                           torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(device),
                           torch.from_numpy(observation.column_features.astype(np.float32)).to(device))
                    logits = model(*observation)
                    logits = -F.relu(logits)
                    action = action_set[logits[action_set.astype(np.int64)].argmax()]
                    observation, action_set, _, done, info = env.step(action)

            mean_solve_time += info['time']
            mean_nb_nodes += info['nb_nodes']
            mean_lp_iters += info['lp_iters']
            nb_runs_processed += 1

    mean_solve_time /= nb_runs_processed
    mean_nb_nodes /= nb_runs_processed
    mean_lp_iters /= nb_runs_processed

    return mean_solve_time, mean_nb_nodes, mean_lp_iters

def train_gnn_rl(config, config_name):
    set_seed(config.seed)
    
    LEARNING_RATE = 3e-4
    NB_EPOCHS = 70
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

    valid_instances = load_valid_instances(config)

    q_network = GNNPolicy()

    bc_network = None
    if config.use_bc:
        # bc_network = GNNPolicy()
        # bc_network.load_state_dict(torch.load(f'{path}/trained_params/GCNN_trained_params.pkl'))
        raise NotImplementedError("Usage of behaviour cloning network not yet implemented for CQL.")
        

    cql = CQL(q_network, bc_network=bc_network, reward=config.reward)
    cql.to(DEVICE)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    prev_min_val_nodes = 1e10 # ~ inf
    n_steps_per_epoch = len(train_loader)
    n_steps_done = 0
    train_results = defaultdict(list)

    solve_time, nb_nodes, lp_iters = test_model_on_instances(cql.q_network, valid_instances, device=DEVICE)
    print(f'Val solve time : {solve_time:0.3f}, Val nb nodes : {nb_nodes:0.3f}, Val lp iters : {lp_iters:0.3f}')

    train_results["val_nb_nodes"].append((n_steps_done, nb_nodes))
    train_results["val_solve_time"].append((n_steps_done, solve_time))
    train_results["val_lp_iters"].append((n_steps_done, lp_iters))

    for epoch in tqdm(list(range(NB_EPOCHS)), "Processing epochs..."):
        print(f"Epoch {epoch + 1}")

        dqn_loss, cql_loss, train_loss = do_epoch(cql, train_loader, optimizer, device=DEVICE)
        print(f"Train loss: {train_loss:0.3f}, DQN loss : {dqn_loss:0.3f}, CQL loss : {cql_loss:0.3f}")

        solve_time, nb_nodes, lp_iters = test_model_on_instances(cql.q_network, valid_instances, device=DEVICE)
        print(f'Val solve time : {solve_time:0.3f}, Val nb nodes : {nb_nodes:0.3f}, Val lp iters : {lp_iters:0.3f}')

        n_steps_done += n_steps_per_epoch

        train_results["train_loss"].append((n_steps_done, train_loss))
        train_results["train_dqn_loss"].append((n_steps_done, dqn_loss))
        train_results["train_cql_loss"].append((n_steps_done, cql_loss))

        train_results["val_nb_nodes"].append((n_steps_done, nb_nodes))
        train_results["val_solve_time"].append((n_steps_done, solve_time))
        train_results["val_lp_iters"].append((n_steps_done, lp_iters))

        if prev_min_val_nodes > nb_nodes:
            torch.save(cql.q_network.state_dict(), f'{models_path}/{config_name}.pt')

            prev_min_val_nodes = nb_nodes

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
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--reward', type=str, default='lp-iterations')
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

    train_gnn_rl(args, config_name)