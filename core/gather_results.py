import pickle
import os


class CQLConfig:
    def __init__(self, expert_proba, alpha, seed):
        self.expert_proba = expert_proba
        self.alpha = alpha
        self.seed = seed

    @classmethod
    def from_config_name(cls, config_name):
        _, expert_proba, _, alpha, _, seed = config_name.split("_")

        expert_proba = expert_proba.split("=")[-1]  # Can't cast to float because of 'mixed'
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
