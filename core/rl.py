from core import GraphDataset, GNNPolicy, process_epoch

import copy
import torch
import torch.nn.functional as F
import torch_geometric
import glob

class CQL:
    # TODO: Logging - Training (and val) loop - Testing loop - Missing hparams (reward, dataset, algo?)
    def __init__(self, q_network, alpha=1.0, bc_network=None, target_update_interval=5000):
        self.q_network = q_network
        self.target_q_network = copy.deepcopy(q_network)
        self.alpha = alpha
        self.bc_network = bc_network
        self.target_update_interval = target_update_interval

        self.n_steps_done = 0

    def get_loss(self, batch):
        # (s, a, s', r, t) = batch

        # Compute DDQN loss

        # Compute CQL loss

        # Return L_DDQN + alpha * L_CQL
        pass

    def _get_ddqn_loss(self, batch):
        # one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        # q_t = (self.forward(obs_t) * one_hot.float()).sum(dim=1, keepdim=True)
        # y = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)
        # loss = _huber_loss(q_t, y)
        # return _reduce(loss, reduction)
        pass

    def _get_cql_loss(self, batch):
        # # compute logsumexp
        # policy_values = self._q_func(obs_t)
        # logsumexp = torch.logsumexp(policy_values, dim=1, keepdim=True)

        # # estimate action-values under data distribution
        # one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        # data_values = (self._q_func(obs_t) * one_hot).sum(dim=1, keepdim=True)

        # return (logsumexp - data_values).mean()
        
        # TODO: don't forget to take into account whether or not we have a bc_network
        pass

    def get_pred(self, batch):
        return self.q_network(batch)

    def update(self):
        self.n_steps_done += 1

        if self.n_steps_done % self.target_update_interval == 0:
            self._update_target()

    def _update_target(self):
        with torch.no_grad():
            params = self.q_network.parameters()
            targ_params = self.target_q_network.parameters()
            for p, p_targ in zip(params, targ_params):
                p_targ.data.copy_(p.data)



if __name__ == '__main__':
    collection_root = '.'
    collection_name = '100_instances_collection'

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
    train_files = sample_files[:int(0.8 * len(sample_files))]
    valid_files = sample_files[int(0.8 * len(sample_files)):]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=4, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=16, shuffle=False)

    policy = GNNPolicy().to(DEVICE)

    observation = train_data[0].to(DEVICE)

    logits = policy(observation.constraint_features, observation.edge_index, observation.edge_attr,
                    observation.variable_features)
    action_distribution = F.softmax(logits[observation.candidates], dim=-1)

    print(action_distribution)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    prev_max_val_acc = 0.0
    n_epochs_no_better_val = 0
    for epoch in range(NB_EPOCHS):
        print(f"Epoch {epoch + 1}")

        train_loss, train_acc = process_epoch(policy, train_loader, optimizer, device=DEVICE)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process_epoch(policy, valid_loader, None, device=DEVICE)
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

        if valid_acc > prev_max_val_acc:
            prev_max_val_acc = valid_acc
            n_epochs_no_better_val = 0
        else:
            n_epochs_no_better_val += 1
        
        if n_epochs_no_better_val >= EARLY_STOPPING:
            print(f"Early stopping after {epoch + 1} epochs, the last {EARLY_STOPPING} of which without validation improvement.")
            break

    torch.save(policy.state_dict(), f'{path}GCNN_trained_params.pkl')
