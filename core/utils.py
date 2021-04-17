from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from core import GraphDataset, GNNPolicy
import glob
from pathlib import Path
import shutil
import os
from collections import defaultdict
from tqdm import tqdm
import pickle


def data_collection_stats_figure(cpu_pct, cpus_pct, ram_pct, ram_used, ram_active, collect_trajectory_times):
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


def train_gnn(working_path, config_name, collection_name, trajectories_name, train_batch_size, test_batch_size, num_workers):
    # Train GNN
    LEARNING_RATE = 0.001
    NB_EPOCHS = 15
    PATIENCE = 3
    MIN_DELTA = 0.01
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trajectories_path = f'{working_path}/data/collections/{collection_name}/{trajectories_name}/'
    models_path = f'{working_path}/models/'
    results_path = f'{working_path}/train_results/'

    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    sample_files = [str(file_path) for file_path in glob.glob(f'{trajectories_path}*')]
    sample_files.sort()
    train_files = sample_files[:int(0.8 * len(sample_files))]
    valid_files = sample_files[int(0.8 * len(sample_files)):]

    train_data = GraphDataset(train_files)
    train_loader = torch_geometric.data.DataLoader(train_data, batch_size=train_batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    valid_data = GraphDataset(valid_files)
    valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=test_batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    policy = GNNPolicy().to(DEVICE)

    train_results = defaultdict(list)
    best_valid_accuracy = 0.0
    n_epochs_no_improvement = 0
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    n_steps_per_epoch = len(train_loader)
    n_steps_done = 0

    valid_loss, valid_acc = process_epoch(policy, valid_loader, None, device=DEVICE)
    print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

    train_results["val_loss"].append((n_steps_done, valid_loss))
    train_results["val_acc"].append((n_steps_done, valid_acc))

    for epoch in tqdm(list(range(NB_EPOCHS)), "Processing epochs..."):
        print(f"Epoch {epoch + 1}")

        train_loss, train_acc = process_epoch(policy, train_loader, optimizer, device=DEVICE)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

        valid_loss, valid_acc = process_epoch(policy, valid_loader, None, device=DEVICE)
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

        n_steps_done += n_steps_per_epoch

        train_results["train_loss"].append((n_steps_done, train_loss))
        train_results["train_acc"].append((n_steps_done, train_acc))
        train_results["val_loss"].append((n_steps_done, valid_loss))
        train_results["val_acc"].append((n_steps_done, valid_acc))

        if valid_acc >= best_valid_accuracy + MIN_DELTA:            
            torch.save(policy.state_dict(), f'{models_path}/{config_name}.pt')

            # Reinit no_improvement counter and best_valid_accuracy
            n_epochs_no_improvement = 0
            best_valid_accuracy = valid_acc
        else:
            n_epochs_no_improvement += 1
            if n_epochs_no_improvement >= PATIENCE:
                print(f'Validation accuracies stopped improving. Max accuracy reached was {best_valid_accuracy}.' +
                      f'Model weights saved at {models_path}/{config_name}.pt')
                break

    with open(f'{results_path}/{config_name}.pkl', 'wb') as f:
        pickle.dump(train_results, f)


def process_epoch(policy, data_loader, optimizer=None, device='cuda'):
    """
    This function will process a whole epoch of training or validation, depending on whether an optimizer is provided.
    """
    mean_loss = 0
    mean_acc = 0

    n_samples_processed = 0
    with torch.set_grad_enabled(optimizer is not None):
        msg = f"{'train' if optimizer else 'val'} epoch..."
        for batch in tqdm(data_loader, msg):
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

def get_name_for_BC_config(config):
    name = ["BC"]

    name.append(f"expert-proba={config.expert_probability}")

    return "_".join(name)

def save_work_done(working_path, saving_path):
    if working_path == saving_path:
        return

    # Move models
    src_path = os.path.join(working_path, 'models')
    if os.path.exists(src_path):
        dest_path = os.path.join(saving_path, 'models')
        Path(dest_path).mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(src_path):
            full_fname = os.path.join(src_path, fname)
            if os.path.isfile(full_fname):
                shutil.copyfile(full_fname, os.path.join(dest_path, fname))


    # Move training results
    src_path = os.path.join(working_path, 'train_results')
    if os.path.exists(src_path):
        dest_path = os.path.join(saving_path, 'train_results')
        Path(dest_path).mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(src_path):
            full_fname = os.path.join(src_path, fname)
            if os.path.isfile(full_fname):
                shutil.copyfile(full_fname, os.path.join(dest_path, fname))

    # Move testing results
    src_path = os.path.join(working_path, 'test_results')
    if os.path.exists(src_path):
        dest_path = os.path.join(saving_path, 'test_results')
        Path(dest_path).mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(src_path):
            full_fname = os.path.join(src_path, fname)
            if os.path.isfile(full_fname):
                shutil.copyfile(full_fname, os.path.join(dest_path, fname))