import argparse
import pickle
import os
import numpy as np

from collections import defaultdict

class CQLConfig:
    def __init__(self, expert_proba, alpha, seed):
        self.expert_proba = expert_proba
        self.alpha = alpha
        self.seed = seed

    @classmethod
    def from_config_name(cls, config_name):
        _, expert_proba, _, alpha, _, seed  = config_name.split("_")

        expert_proba = expert_proba.split("=")[-1] # Can't cast to float because of 'mixed'
        alpha = float(alpha.split("=")[-1])
        seed = int(seed.split("=")[-1])

        return cls(expert_proba, alpha, seed)

class BaselineConfig:
    def __init__(self, name):
        self.name = name

def load_results_from_path(path):
    results = {}

    for fname in os.listdir(path):
        with open(os.path.join(path, fname), "rb") as f:
            config_results = pickle.load(f)

        config_name = fname.replace(".pkl", "")

        if "CQL" in config_name:
            config = CQLConfig.from_config_name(config_name)
        elif "SCIP" in config_name:
            config = BaselineConfig(config_name)
        elif "FSB" in config_name:
            config = BaselineConfig(config_name)
        else:
            raise NotImplementedError(f"No implementation for loading config named {config_name}")

        results[config] = config_results

    return results

def load_results(saving_path):
    train_results = load_results_from_path(f"{saving_path}/train_results/")
    test_results = load_results_from_path(f"{saving_path}/test_results/")

    return train_results, test_results

def gather_results(saving_path):
    train_results, test_results = load_results(saving_path)

    # Exemple for learning_curve
    train_losses = defaultdict(list)

    for key, res in train_results.items():
        if isinstance(key, CQLConfig):
            run_key = (key.expert_proba, key.alpha)
        else:
            run_key = (key.name)

        # Train results are a list of (n_steps_done, n_epochs_done, val) tuples
        # So we collect
        train_losses[run_key].append([_[2] for _ in res["train_loss"]])

    mean_train_losses = {key: np.mean(vals) for key, vals in train_losses.items()}
    # 95 % CI is at mean +/- 1.98 * std
    ci_train_losses = {key: 1.98 * np.std(vals) for key, vals in train_losses.items()}

    # Pas obligé d'être mean + shade pour le CI. On peut aussi juste
    # plotter la moyenne et les 3 runs en plus pâle. C'est juste
    # esthétique.

    dummy_key = next(iter(mean_train_losses))
    train_steps = [_[0] for _ in train_results[dummy_key]]
    train_epochs = [_[1] for _ in train_results[dummy_key]]

    # On va surement plus vouloir plotter en fonction 
    # du nombre d'epochs que de steps, pcq les steps sont
    # pas fixés pour les datasets mais les epochs oui.

    # N.B. Les steps et les epochs sont pas exactement les mêmes 
    # en valid et en train pcq en valid on a aussi un exemple 
    # avant de faire la première batch.

    # Train results keys:
    #
    # - train_dqn_loss: La loss en entraînement sur l'objectif RL classique
    # - train_cql_loss: La pénalité en entraînement liée à l'explosion des Q-values
    # - train_loss: La loss qui est backpropée (dqn + alpha * cql)
    #
    # Toutes les losses sont des moyennes par item (pas par batch).
    #
    # - val_nb_nodes: Le nb de noeuds en moyenne par instance de valid (1 run).
    # - val_solve_time: Le temps de résolution (s) en moyenne par instance de valid (1 run).
    # - val_lp_iters: Le nb d'itérations LP en moyenne par instance de valid (1 run).

    # Exemple for test performance
    test_nb_nodes = defaultdict(list)

    for _, easy_results in test_results.items():
        # Subdict for easy instance
        for key, res in easy_results.items():
            if isinstance(key, CQLConfig):
                run_key = (key.expert_proba, key.alpha)
            else:
                # Collect FSB and SCIP later
                continue

            test_nb_nodes[run_key].append(res["mean_nb_nodes"])

    mean_nb_nodes = {key: np.mean(vals) for key, vals in test_nb_nodes.items()}
    # 95 % CI is at mean +/- 1.98 * std
    ci_nb_nodes = {key: 1.98 * np.std(vals) for key, vals in test_nb_nodes.items()}

    mean_nb_nodes["FSB"] = test_results["FSB"]["easy"]["mean_nb_nodes"]
    ci_nb_nodes["FSB"] = 1.98 * test_results["FSB"]["easy"]["std_nb_nodes"]

    mean_nb_nodes["SCIP"] = test_results["SCIP"]["easy"]["mean_nb_nodes"]
    ci_nb_nodes["SCIP"] = 1.98 * test_results["SCIP"]["easy"]["std_nb_nodes"]

    # Test results keys:
    # 
    # Pour CQL, FSB et SCIP:
    # - mean_solve_time: Solving time moyen par instance (5 runs)
    # - mean_nb_nodes: Nb Nodes moyen par instance (5 runs)
    # - mean_lp_iters: LP iters moyen par instance (5 runs)
    #
    # Pour FSB et SCIP seulement:
    # - std_solve_time: Solving time moyen par instance (5 runs)
    # - std_nb_nodes: Nb Nodes moyen par instance (5 runs)
    # - std_lp_iters: LP iters moyen par instance (5 runs) 
    #
    # Pour CQL seulement:
    # - mean_sb_accuracy: Accuracy moyenne par rapport au SB

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--saving_path', type=str, default='.') # Index to the results folder
    
    args = parser.parse_args() 

    gather_results(args.saving_path)