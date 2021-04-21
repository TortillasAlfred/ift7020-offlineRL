from core import load_results, CQLConfig
from matplotlib import pyplot as plt
import numpy as np


def run_key_to_model_name(expert_proba, alpha):
    return f'expert{expert_proba}_alpha{alpha}'


def generate_figures(results, metrics):
    fig, axs = plt.subplots(3, 2)

    for model in results.keys():
        for i, metric in enumerate(metrics):
            axs[i%3, int(i/3)].plot(np.mean(results[model], axis=0)[i], label=model)

    for i in range(len(metrics)):
        axs[i%3, int(i/3)].set_title(metrics[i])
        axs[i%3, int(i/3)].legend(prop={'size': 6})

    plt.show()


if __name__ == '__main__':
    path = '/home/jeff/Documents/almost_all_train_results'
    train_results, test_results = load_results(path)

    metrics = ['val_nb_nodes',
               'val_solve_time',
               'val_lp_iters',
               'train_loss',
               'train_dqn_loss',
               'train_cql_loss']

    results = {}

    nb_epochs = 35
    nb_metrics = 6
    nb_seeds = 3

    for key, res in train_results.items():
        if isinstance(key, CQLConfig):
            model = run_key_to_model_name(key.expert_proba, key.alpha)
        else:
            model = key.name

        if model not in results.keys():
            results[model] = []

        for metric in metrics:
            results[model].append(np.asarray([_[2] for _ in res[metric]])[-nb_epochs:])

    for model, values in results.items():
        results[model] = np.asarray(values).reshape(nb_seeds, nb_metrics, nb_epochs)

    generate_figures(results, metrics)

