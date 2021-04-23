import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core import GNNPolicy
from core.data_collection import ExploreThenStrongBranch
from core.utils import get_testing_more_config_name_for_job_index

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import pickle
import argparse
import os
import numpy as np
import ecole


def load_test_instances(src_path):
    loaded_instances = {}

    for set_name in ["easy", "medium"]:
        set_instances_path = f'{src_path}/{set_name}/'

        Path(f'{set_instances_path}').mkdir(parents=True, exist_ok=True)
        loaded_instances[set_name] = []
        for _, _, files in os.walk(set_instances_path):
            for file in files:
                instance = ecole.scip.Model.from_file(set_instances_path + file)
                loaded_instances[set_name].append(instance)

    return loaded_instances


def test_scip_on_instances(instances, n_runs=5):
    # Basé sur https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
    solve_times = []
    nb_nodes = []
    lp_iters = []

    # We can pass custom SCIP parameters easily
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Configuring(observation_function=None,
                                        information_function={"nb_nodes": ecole.reward.NNodes(),
                                                              "lp_iters": ecole.reward.LpIterations(),
                                                              "time": ecole.reward.SolvingTime()},
                                        scip_params=scip_parameters)

    for instance in tqdm(instances, f"Processing easy instances..."):
        for run in tqdm(list(range(n_runs))):
            env.seed(run)
            env.reset(instance)
            _, _, _, _, info = env.step({})

            solve_times.append(info['time'])
            nb_nodes.append(info['nb_nodes'])
            lp_iters.append(info['lp_iters'])

    mean_solve_time = np.mean(solve_times)
    mean_nb_nodes = np.mean(nb_nodes)
    mean_lp_iters = np.mean(lp_iters)

    std_solve_time = np.std(solve_times)
    std_nb_nodes = np.std(nb_nodes)
    std_lp_iters = np.std(lp_iters)

    results = {}

    results["mean_solve_time"] = mean_solve_time
    results["mean_nb_nodes"] = mean_nb_nodes
    results["mean_lp_iters"] = mean_lp_iters

    results["std_solve_time"] = std_solve_time
    results["std_nb_nodes"] = std_nb_nodes
    results["std_lp_iters"] = std_lp_iters

    return results


def test_cql_model_on_instances(model, instances, device, n_runs=5, is_easy=False):
    # Basé sur https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
    mean_solve_time = 0.0
    mean_nb_nodes = 0.0
    mean_lp_iters = 0.0
    mean_model_preds = 0.0
    nb_runs_processed = 0

    if is_easy:
        mean_sb_acc = 0

    # We can pass custom SCIP parameters easily
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(observation_function=ecole.observation.NodeBipartite(),
                                      scip_params=scip_parameters,
                                      information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                                                            "time": ecole.reward.SolvingTime().cumsum(),
                                                            "lp_iters": ecole.reward.LpIterations().cumsum()})

    if is_easy:
        fsb_env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=1.0),
                                                                    ecole.observation.NodeBipartite()),
                                              scip_params=scip_parameters,
                                              information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                                                                    "time": ecole.reward.SolvingTime().cumsum(),
                                                                    "lp_iters": ecole.reward.LpIterations().cumsum()})

    name = "easy" if is_easy else "medium"
    for instance in tqdm(instances, f"Processing {name} instances..."):
        for run in tqdm(list(range(n_runs))):
            env.seed(run)
            observation, action_set, _, done, info = env.reset(instance)
            sum_pred_values = 0

            while not done:
                with torch.no_grad():
                    observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                                   torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                                   torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(
                                       device),
                                   torch.from_numpy(observation.column_features.astype(np.float32)).to(device))
                    logits = model(*observation)
                    logits = -F.relu(logits)
                    action = action_set[logits[action_set.astype(np.int64)].argmax()]
                    sum_pred_values -= logits[action_set.astype(np.int64)].max().item()

                    observation, action_set, _, done, info = env.step(action)

            mean_solve_time += info['time']
            mean_nb_nodes += info['nb_nodes']
            mean_lp_iters += info['lp_iters']
            mean_model_preds += sum_pred_values

            if is_easy:
                fsb_env.seed(run)
                full_observation, action_set, _, done, info = fsb_env.reset(instance)
                sb_acc = []

                while not done:
                    with torch.no_grad():
                        (scores, _), observation = full_observation

                        observation = (torch.from_numpy(observation.row_features.astype(np.float32)).to(device),
                                       torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(device),
                                       torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1,
                                                                                                                  1).to(
                                           device),
                                       torch.from_numpy(observation.column_features.astype(np.float32)).to(device))
                        logits = model(*observation)
                        logits = -F.relu(logits)
                        action = action_set[logits[action_set.astype(np.int64)].argmax()]
                        sum_pred_values -= logits[action_set.astype(np.int64)].max()

                        sb_action = action_set[scores[action_set].argmax()]
                        sb_acc.append(sb_action == action)

                        full_observation, action_set, _, done, info = fsb_env.step(action)

                mean_sb_acc += np.array(sb_acc).mean()

            nb_runs_processed += 1

    mean_solve_time /= nb_runs_processed
    mean_nb_nodes /= nb_runs_processed
    mean_lp_iters /= nb_runs_processed
    mean_model_preds /= nb_runs_processed

    results = {}

    results["mean_solve_time"] = mean_solve_time
    results["mean_nb_nodes"] = mean_nb_nodes
    results["mean_lp_iters"] = mean_lp_iters
    results["mean_lp_iters_predicted"] = mean_model_preds

    if is_easy:
        mean_sb_acc /= nb_runs_processed
        results["mean_sb_accuracy"] = mean_sb_acc

    return results


def test_fsb_on_instances(instances, n_runs=5):
    # Basé sur https://github.com/ds4dm/ecole/blob/master/examples/branching-imitation.ipynb
    solve_times = []
    nb_nodes = []
    lp_iters = []

    # We can pass custom SCIP parameters easily
    scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

    env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=1.0),
                                                            ecole.observation.NodeBipartite()),
                                      scip_params=scip_parameters,
                                      information_function={"nb_nodes": ecole.reward.NNodes().cumsum(),
                                                            "time": ecole.reward.SolvingTime().cumsum(),
                                                            "lp_iters": ecole.reward.LpIterations().cumsum()})

    for instance in tqdm(instances, f"Processing easy instances..."):
        for run in tqdm(list(range(n_runs))):
            env.seed(run)
            full_observation, action_set, _, done, info = env.reset(instance)

            while not done:
                (scores, _), _ = full_observation
                action = action_set[scores[action_set].argmax()]

                full_observation, action_set, _, done, info = env.step(action)

            solve_times.append(info['time'])
            nb_nodes.append(info['nb_nodes'])
            lp_iters.append(info['lp_iters'])

    mean_solve_time = np.mean(solve_times)
    mean_nb_nodes = np.mean(nb_nodes)
    mean_lp_iters = np.mean(lp_iters)

    std_solve_time = np.std(solve_times)
    std_nb_nodes = np.std(nb_nodes)
    std_lp_iters = np.std(lp_iters)

    results = {}

    results["mean_solve_time"] = mean_solve_time
    results["mean_nb_nodes"] = mean_nb_nodes
    results["mean_lp_iters"] = mean_lp_iters

    results["std_solve_time"] = std_solve_time
    results["std_nb_nodes"] = std_nb_nodes
    results["std_lp_iters"] = std_lp_iters

    return results


def cql_testing(args, config_name, test_instances, device):
    model = GNNPolicy()
    model.load_state_dict(torch.load(f'{args.saving_path}/models/{config_name}.pt'))
    model.eval()
    model = model.to(device)

    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'rb') as f:
        all_results = pickle.load(f)

    # MEDIUM
    medium_results = test_cql_model_on_instances(model, test_instances["medium"], device, is_easy=False)
    all_results["medium"] = medium_results

    # Write to disk
    os.makedirs(f'{args.saving_path}/test_results', exist_ok=True)
    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

        # SB accuracy
    sb_results = test_cql_model_on_instances(model, test_instances["easy"], device, is_easy=True)
    all_results["easy"]["mean_sb_accuracy"] = sb_results["mean_sb_accuracy"]

    # Write to disk
    os.makedirs(f'{args.saving_path}/test_results', exist_ok=True)
    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    # CPU
    model = model.to('cpu')
    cpu_results = test_cql_model_on_instances(model, test_instances["easy"], 'cpu', is_easy=False)

    for key, val in cpu_results.items():
        all_results["easy"][f"{key}_cpu"] = val

    # Write to disk
    os.makedirs(f'{args.saving_path}/test_results', exist_ok=True)
    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


def scip_testing(args, config_name, test_instances):
    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'rb') as f:
        all_results = pickle.load(f)

    medium_results = test_scip_on_instances(test_instances["medium"])
    all_results["medium"] = medium_results

    # Write to disk
    os.makedirs(f'{args.saving_path}/test_results', exist_ok=True)
    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


def fsb_testing(args, config_name, test_instances):
    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'rb') as f:
        all_results = pickle.load(f)

    medium_results = test_fsb_on_instances(test_instances["medium"])
    all_results["medium"] = medium_results

    # Write to disk
    os.makedirs(f'{args.saving_path}/test_results', exist_ok=True)
    with open(f'{args.saving_path}/test_results/{config_name}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


def main(args, config_name):
    test_instances = load_test_instances(args.src_path)

    if "CQL" in config_name:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cql_testing(args, config_name, test_instances, DEVICE)
    elif config_name == "SCIP":
        scip_testing(args, config_name, test_instances)
    elif config_name == "FSB":
        fsb_testing(args, config_name, test_instances)
    else:
        raise NotImplementedError(f"No implementation for {config_name} config testing.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_path', type=str,
                        default='./test_instances')  # This is where we the 'easy' and 'med' folder are located
    parser.add_argument('--saving_path', type=str, default='.')  # This is where we will persistently save the files
    parser.add_argument('--job_index', type=int, default=0)

    args = parser.parse_args()

    config_name = get_testing_more_config_name_for_job_index(args)

    print(f"\n\nCollecting more test results for following config :\n{config_name}\n")

    main(args, config_name)
