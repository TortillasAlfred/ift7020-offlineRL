import copy
import torch

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
