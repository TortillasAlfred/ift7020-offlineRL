from core import GraphDataset, GNNPolicy
from core.utils import pad_tensor

import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch_geometric
import glob
from pathlib import Path

class CQL:
    # TODO: Logging - Training (and val) loop - Testing loop
    def __init__(self, q_network, gamma=0.99, reward="nb_nodes", alpha=1.0, bc_network=None, target_update_interval=5000):
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

        dqn_loss = (targ_q_t - q_t) ** 2


        # CQL Loss
        lse = torch.logsumexp(full_q_t, dim=1, keepdim=True)

        if self.bc_network:
            action_logits = self._get_network_pred(self.bc_network, *current_state)
            action_likelihoods = F.softmax(action_logits, dim=-1)
            data_values = (full_q_t * action_likelihoods).sum(dim=1, keepdim=True)
        else:
            data_values = q_t

        cql_loss = lse - data_values


        return (dqn_loss + self.alpha * cql_loss).mean()

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

    n_samples_processed = 0
    with torch.set_grad_enabled(True):
        for batch in tqdm(data_loader):
            batch = batch.to(device)
            
            loss = learner.get_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            learner.update()

            mean_loss += loss.item() * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed

    return mean_loss


if __name__ == '__main__':
    collection_root = '.'
    collection_name = '2_instances_collection'
    load_bc_network = True

    expert_probability = 0.0
    trajectories_name = f'10_trajectories_expert_{expert_probability}'

    # Train GNN
    LEARNING_RATE = 0.001
    NB_EPOCHS = 50
    PATIENCE = 10
    EARLY_STOPPING = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = f'{collection_root}/collections/{collection_name}/{trajectories_name}/'

    sample_files = [str(file_path) for file_path in glob.glob(f'{path}*')]
    sample_files.sort()
    train_files = sample_files

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=4, shuffle=True)

    q_network = GNNPolicy()

    if load_bc_network:
        bc_network = GNNPolicy()
        bc_network.load_state_dict(torch.load(f'{path}/trained_params/GCNN_trained_params.pkl'))
    else:
        bc_network = None

    cql = CQL(q_network, bc_network=bc_network)
    cql.to(DEVICE)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    prev_max_val_acc = 0.0
    n_epochs_no_better_val = 0
    for epoch in range(NB_EPOCHS):
        print(f"Epoch {epoch + 1}")

        train_loss = do_epoch(cql, train_loader, optimizer, device=DEVICE)
        print(f"Train loss: {train_loss:0.3f}")

        # TODO: Whole validation loop
        # valid_loss, valid_acc = process_epoch(q_network, valid_loader, None, device=DEVICE)
        # print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

        # if valid_acc > prev_max_val_acc:
        #     prev_max_val_acc = valid_acc
        #     n_epochs_no_better_val = 0
        # else:
        #     n_epochs_no_better_val += 1
        
        # if n_epochs_no_better_val >= EARLY_STOPPING:
        #     print(f"Early stopping after {epoch + 1} epochs, the last {EARLY_STOPPING} of which without validation improvement.")
        #     break

    Path(f'{path}/trained_params/').mkdir(parents=True, exist_ok=True)
    torch.save(cql.q_network.state_dict(), f'{path}/trained_params/Q_network_trained_params.pkl')
