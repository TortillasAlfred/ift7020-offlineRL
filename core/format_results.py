from core import load_results, CQLConfig
from matplotlib import pyplot as plt
import numpy as np


def run_key_to_model_name(expert_proba, alpha):
    return f'expert{expert_proba}_alpha{alpha}'


def generate_train_epochs_figures(results, metrics):
    fig, axs = plt.subplots(3, 2)

    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            for i, metric in enumerate(metrics):
                axs[i%3, int(i/3)].plot(np.mean(results[model], axis=0)[i], label=model)

    for i in range(len(metrics)):
        axs[i%3, int(i/3)].set_title(metrics[i])
        axs[i%3, int(i/3)].legend(prop={'size': 6})

    fig.suptitle('Training epochs results')
    plt.show()


def compute_metrics(train_epoch_results, i, epoch):
    if int(i/3) == 0:
        return np.mean(train_epoch_results, axis=0)[i][epoch-1]
    else:
        return np.std(train_epoch_results, axis=0)[i%3][epoch-1]


def generate_train_figures(results, metrics, difficulty, epoch):
    fig, axs = plt.subplots(3, 2, constrained_layout=True)

    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            for i, metric in enumerate(metrics):
                axs[i%3, int(i/3)].bar(model.replace('expert', 'e').replace('alpha', 'a'), compute_metrics(results[model], i, epoch))
        else:
            for i, metric in enumerate(metrics):
                axs[i % 3, int(i / 3)].bar(model, results[model][difficulty][i])

    for i in range(len(metrics)):
        axs[i%3, int(i/3)].set_title(metrics[i])
        axs[i%3, int(i/3)].tick_params(axis='x', labelrotation=280, size=6)

    fig.suptitle(f'Training results with {epoch} epochs')
    plt.show()


def generate_test_figures(results, metrics, mean_std_metrics, difficulty):
    fig, axs = plt.subplots(2, 2, constrained_layout=True)

    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            for i, metric in enumerate(metrics):
                axs[i%2, int(i/2)].bar(model.replace('expert', 'e').replace('alpha', 'a'), np.mean(results[model][difficulty], axis=0)[i])
        else:
            for i, metric in enumerate(metrics[:3]):
                axs[i%2, int(i/2)].bar(model, results[model][difficulty][i])

    for i in range(len(metrics)):
        axs[i%2, int(i/2)].set_title(metrics[i])
        axs[i%2, int(i/2)].tick_params(axis='x', labelrotation=280, size=6)

    fig.suptitle(f'Testing results {difficulty.capitalize()}')
    plt.show()


if __name__ == '__main__':
    path = '/home/jeff/Documents/RL_datasets/all_train_results_base_tests'
    train_results, test_results = load_results(path)

    train_epoch_metrics = ['val_nb_nodes',
                           'val_solve_time',
                           'val_lp_iters',
                           'train_loss',
                           'train_dqn_loss',
                           'train_cql_loss']

    mean_std_metrics = ['mean_nb_nodes',
                        'mean_solve_time',
                        'mean_lp_iters',
                        'std_nb_nodes',
                        'std_solve_time',
                        'std_lp_iters']

    test_metrics = ['mean_nb_nodes',
                    'mean_solve_time',
                    'mean_lp_iters',
                    'mean_lp_iters_predicted']

    train_metric_results = {}
    test_metric_results = {}

    train_test_data = [('train_results', train_results, train_epoch_metrics, train_metric_results),
                       ('test_results', test_results, test_metrics, test_metric_results)]

    nb_epochs = 40
    nb_seeds = 3
    test_difficulties = ['easy']

    for data in train_test_data:
        from_cql = None
        for key, res in data[1].items():
            if isinstance(key, CQLConfig):
                model = run_key_to_model_name(key.expert_proba, key.alpha)
                from_cql = True
            else:
                model = key.name
                from_cql = None

            if data[0] == 'train_results':
                if from_cql:
                    if model not in data[3].keys():
                        data[3][model] = []

                    for metric in data[2]:
                        data[3][model].append(np.asarray([_[2] for _ in res[metric]])[-nb_epochs:])
                else:
                    if model not in data[3].keys():
                        data[3][model] = {}

                        for difficulty in test_difficulties:
                            data[3][model][difficulty] = []

                        for metric in mean_std_metrics:
                            for difficulty in test_difficulties:
                                data[3][model][difficulty].append(res[difficulty][metric.replace('val', 'mean').replace('train', 'std')])

            else:
                if from_cql:
                    if model not in data[3].keys():
                        data[3][model] = {}
                        for difficulty in test_difficulties:
                            data[3][model][difficulty] = []

                    for metric in data[2]:
                        for difficulty in test_difficulties:
                            data[3][model][difficulty].append(res[difficulty][metric])

                else:
                    if model not in data[3].keys():
                        data[3][model] = {}

                        for difficulty in test_difficulties:
                            data[3][model][difficulty] = []

                        for metric in mean_std_metrics:
                            for difficulty in test_difficulties:
                                data[3][model][difficulty].append(res[difficulty][metric.replace('val', 'mean').replace('train', 'std')])

        if data[0] == 'train_results':
            for model, values in data[3].items():
                if model not in ['FSB', 'SCIP']:
                    data[3][model] = np.asarray(values).reshape(nb_seeds, len(data[2]), nb_epochs)
                else:
                    for difficulty in test_difficulties:
                        data[3][model][difficulty] = np.asarray(values[difficulty])
        else:
            for model, values in data[3].items():
                for difficulty in test_difficulties:
                    if model not in ['FSB', 'SCIP']:
                        data[3][model][difficulty] = np.asarray(values[difficulty]).reshape(nb_seeds, len(data[2]))
                    else:
                        data[3][model][difficulty] = np.asarray(values[difficulty])

        if data[0] == 'train_results':
            generate_train_epochs_figures(data[3], train_epoch_metrics)
            for difficulty in test_difficulties:
                generate_train_figures(data[3], mean_std_metrics, 'easy', 40)
        else:
            for difficulty in test_difficulties:
                generate_test_figures(data[3], test_metrics, mean_std_metrics, difficulty)
