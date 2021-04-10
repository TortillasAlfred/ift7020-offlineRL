from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def data_collection_stats_figure(cpu_pct, cpus_pct, ram_pct, ram_used, ram_active, collect_trajectory_times,
                                 collection_name, trajectories_name):
    fig, axs = plt.subplots(3)
    axs[0].plot(cpu_pct, label='cpu_pct')
    axs[0].plot(ram_pct, label='ram_pct')
    cpus_pct = np.asarray(cpus_pct)
    for i in range(cpus_pct.shape[1]):
        axs[0].plot(cpus_pct[:, i], label=f'cpu{i+1}_pct')
    axs[0].set_ylabel('%')
    axs[0].legend()
    axs[1].plot(ram_used, label='ram_used')
    axs[1].plot(ram_active, label='ram_active')
    axs[1].set_ylabel('Gb')
    axs[1].legend()
    axs[2].plot(collect_trajectory_times, label='collect_trajectory_time')
    axs[2].set_ylabel('Seconds')
    axs[2].legend()
    plt.show()


def process_epoch(policy, data_loader, optimizer=None, device='cuda'):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in data_loader:
            batch = batch.to(device)
            # Compute the logits (i.e. pre-softmax activations) according to the policy on the concatenated graphs
            logits = policy(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            # Index the results by the candidates, and split and pad them
            logits = pad_tensor(logits[batch.candidates], batch.nb_candidates)
            # Compute the usual cross-entropy classification loss
            loss = F.cross_entropy(logits, batch.candidate_choice)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            true_scores = pad_tensor(batch.candidate_scores, batch.nb_candidates)
            true_bestscore = true_scores.max(dim=-1, keepdims=True).values

            predicted_bestindex = logits.max(dim=-1, keepdims=True).indices
            accuracy = (true_scores.gather(-1, predicted_bestindex) == true_bestscore).float().mean().item()

            mean_loss += loss.item() * batch.num_graphs
            mean_acc += accuracy * batch.num_graphs
            n_samples_processed += batch.num_graphs

    mean_loss /= n_samples_processed
    mean_acc /= n_samples_processed
    return mean_loss, mean_acc


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    """
    This utility function splits a tensor and pads each split to make them all the same size, then stacks them.
    """
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size - slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output
