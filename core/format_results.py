import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from core import load_results, CQLConfig
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd


def run_key_to_model_name(expert_proba, alpha):
    return f'expert{expert_proba}_alpha{alpha}'


def generate_full_train_epochs_figures(results, metrics):
    color_list = list(colors.TABLEAU_COLORS.values()) * 2
    line_list = ['solid', 'dashed']

    for i, metric in enumerate(metrics):
        for c, model in enumerate(results.keys()):
            if model not in ['FSB', 'SCIP']:
                plt.plot(np.mean(results[model], axis=0)[i], label=model, linewidth=2.0, color=color_list[c],
                         linestyle=line_list[int(c / 10)])
                for run_result in results[model][:, i]:
                    plt.plot(run_result, alpha=0.4, color=color_list[c])

        plt.title(f'Training epochs results for {metric}')
        plt.legend(prop={'size': 6})
        plt.savefig(f'figures/training_epochs_results_{metric}.png')
        plt.clf()


def generate_full_train_epochs_figure(results, metrics):
    color_list = list(colors.TABLEAU_COLORS.values()) * 2
    line_list = ['solid', 'dashed']

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    for c, model in enumerate(results.keys()):
        if model not in ['FSB', 'SCIP']:
            for i, metric in enumerate(metrics):
                axs[i % 3, int(i / 3)].plot(np.mean(results[model], axis=0)[i], label=model, linewidth=2.0,
                                            color=color_list[c], linestyle=line_list[int(c / 10)])
                for run_result in results[model][:, i]:
                    axs[i % 3, int(i / 3)].plot(run_result, alpha=0.4, color=color_list[c])

    for i in range(len(metrics)):
        axs[i % 3, int(i / 3)].set_title(metrics[i])
        axs[i % 3, int(i / 3)].legend(prop={'size': 6})

    fig.suptitle('Training epochs results')
    plt.savefig(f'figures/training_epochs_results.png')
    plt.show()


def generate_expert_train_epochs_figures(results, metrics, alpha=10.0):
    color_list = list(colors.TABLEAU_COLORS.values())

    for i, metric in enumerate(metrics):
        c = 0
        for model in results.keys():
            if model not in ['FSB', 'SCIP'] and str(f'alpha{alpha}') in model:
                plt.plot(np.mean(results[model], axis=0)[i], label=model, linewidth=2.0, color=color_list[c])
                for run_result in results[model][:, i]:
                    plt.plot(run_result, alpha=0.4, color=color_list[c])
            c += 1

        plt.title(f'Training epochs results for {metric} (alpha={alpha})')
        plt.legend(prop={'size': 6})
        plt.savefig(f'figures/training_epochs_results_alpha{alpha}_{metric}.png')
        plt.clf()


def generate_expert_train_epochs_figure(results, metrics, alpha=10.0):
    color_list = list(colors.TABLEAU_COLORS.values())

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    c = 0
    for model in results.keys():
        if model not in ['FSB', 'SCIP'] and str(f'alpha{alpha}') in model:
            for i, metric in enumerate(metrics):
                axs[i % 3, int(i / 3)].plot(np.mean(results[model], axis=0)[i], label=model, linewidth=2.0,
                                            color=color_list[c])
                for run_result in results[model][:, i]:
                    axs[i % 3, int(i / 3)].plot(run_result, alpha=0.4, color=color_list[c])
        c += 1

    for i in range(len(metrics)):
        axs[i % 3, int(i / 3)].set_title(metrics[i])
        axs[i % 3, int(i / 3)].legend(prop={'size': 6})

    fig.suptitle(f'Training epochs results (alpha={alpha})')
    plt.savefig(f'figures/training_epochs_results_alpha{alpha}.png')
    plt.show()


def generate_alpha_train_epochs_figures(results, metrics, expert='mixed'):
    color_list = list(colors.TABLEAU_COLORS.values())

    for i, metric in enumerate(metrics):

        c = 0
        for model in results.keys():
            if model not in ['FSB', 'SCIP'] and str(f'expert{expert}') in model:
                plt.plot(np.mean(results[model], axis=0)[i], label=model, linewidth=2.0, color=color_list[c])
                for run_result in results[model][:, i]:
                    plt.plot(run_result, alpha=0.4, color=color_list[c])
                c += 1

        plt.title(f'Training epochs results for {metric} (expert={expert})')
        plt.legend(prop={'size': 6})
        plt.savefig(f'figures/training_epochs_results_expert{expert}_{metric}.png')
        plt.clf()


def generate_alpha_train_epochs_figure(results, metrics, expert='mixed'):
    color_list = list(colors.TABLEAU_COLORS.values())

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    c = 0
    for model in results.keys():
        if model not in ['FSB', 'SCIP'] and str(f'expert{expert}') in model:
            for i, metric in enumerate(metrics):
                axs[i % 3, int(i / 3)].plot(np.mean(results[model], axis=0)[i], label=model, linewidth=2.0,
                                            color=color_list[c])
                for run_result in results[model][:, i]:
                    axs[i % 3, int(i / 3)].plot(run_result, alpha=0.4, color=color_list[c])
            c += 1

    for i in range(len(metrics)):
        axs[i % 3, int(i / 3)].set_title(metrics[i])
        axs[i % 3, int(i / 3)].legend(prop={'size': 6})

    fig.suptitle(f'Training epochs results (expert={expert})')
    plt.savefig(f'figures/training_epochs_results_expert{expert}.png')
    plt.show()


def compute_metrics(results, i):
    if int(i / 3) == 0:
        return np.mean(results, axis=0)[i]
    if int(i / 3) == 1:
        return np.std(results, axis=0)[i % 3]


def generate_gpu_test_figures(results, metrics, difficulty):
    for i in range(len(metrics)):
        for model in results.keys():
            if model not in ['FSB', 'SCIP']:
                if difficulty in results[model].keys():
                    plt.bar(model.replace('expert', 'e').replace('alpha', 'a'),
                            compute_metrics(results[model][difficulty], i))
            else:
                if difficulty in results[model].keys():
                    plt.bar(model, results[model][difficulty][i])

        plt.title(f'GPU testing results for {metrics[i]} (difficulty={difficulty.capitalize()})')
        plt.legend(prop={'size': 6})
        plt.xticks(rotation=30, size=6, ha='right')
        plt.savefig(f'figures/gpu_testing_results_difficulty{difficulty}_{metrics[i]}.png')
        plt.clf()


def generate_gpu_test_figure(results, metrics, difficulty):
    fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(16, 12))

    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            for i in range(len(metrics)):
                if difficulty in results[model].keys():
                    axs[i % 3, int(i / 3)].bar(model.replace('expert', 'e').replace('alpha', 'a'),
                                               compute_metrics(results[model][difficulty], i))
        else:
            for i in range(len(metrics)):
                if difficulty in results[model].keys():
                    axs[i % 3, int(i / 3)].bar(model, results[model][difficulty][i])

    for i in range(len(metrics)):
        axs[i % 3, int(i / 3)].set_title(metrics[i])
        axs[i % 3, int(i / 3)].tick_params(axis='x', labelrotation=280, size=6)

    fig.suptitle(f'GPU testing results (difficulty={difficulty.capitalize()})')
    plt.savefig(f'figures/gpu_testing_results_difficulty{difficulty}.png')
    plt.show()


def generate_cpu_test_figures(results, metrics, difficulty):
    for i in range(len(metrics)):
        for model in results.keys():
            if model not in ['FSB', 'SCIP']:
                if difficulty in results[model].keys():
                    if results[model][difficulty].shape[1] > 3:
                        plt.bar(model.replace('expert', 'e').replace('alpha', 'a'),
                                compute_metrics(results[model][difficulty][:, 3:], i))
            else:
                if difficulty in results[model].keys():
                    plt.bar(model, results[model][difficulty][i])

        plt.title(f'CPU testing results for {metrics[i]} (difficulty={difficulty.capitalize()})')
        plt.legend(prop={'size': 6})
        plt.xticks(rotation=30, size=6, ha='right')
        plt.savefig(f'figures/cpu_testing_results_difficulty{difficulty}_{metrics[i]}.png')
        plt.clf()


def generate_cpu_test_figure(results, metrics, difficulty):
    fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(16, 12))

    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            for i in range(len(metrics)):
                if difficulty in results[model].keys():
                    if results[model][difficulty].shape[1] > 3:
                        axs[i % 3, int(i / 3)].bar(model.replace('expert', 'e').replace('alpha', 'a'),
                                                   compute_metrics(results[model][difficulty][:, 3:], i))
        else:
            for i in range(len(metrics)):
                if difficulty in results[model].keys():
                    axs[i % 3, int(i / 3)].bar(model, results[model][difficulty][i])

    for i in range(len(metrics)):
        axs[i % 3, int(i / 3)].set_title(metrics[i])
        axs[i % 3, int(i / 3)].tick_params(axis='x', labelrotation=280, size=6)

    fig.suptitle(f'CPU testing results (difficulty={difficulty.capitalize()})')
    plt.savefig(f'figures/cpu_testing_results_difficulty{difficulty}.png')
    plt.show()


def generate_accuracy_figures(results, difficulty):
    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            if difficulty in results[model].keys():
                if results[model][difficulty].shape[1] > 6:
                    plt.bar(model.replace('expert', 'e').replace('alpha', 'a'),
                            np.mean(results[model][difficulty][:, -1]))
        else:
            if difficulty in results[model].keys():
                if results[model][difficulty].shape[0] > 6:
                    plt.bar(model, results[model][difficulty][-1])

    plt.title(f'Mean accuracy (difficulty={difficulty.capitalize()})')
    plt.savefig(f'figures/mean_accuracies_difficulty{difficulty}.png')
    plt.clf()

    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            if difficulty in results[model].keys():
                if results[model][difficulty].shape[1] > 6:
                    plt.bar(model.replace('expert', 'e').replace('alpha', 'a'),
                            np.std(results[model][difficulty][:, -1]))
        else:
            if difficulty in results[model].keys():
                if results[model][difficulty].shape[0] > 6:
                    plt.bar(model, results[model][difficulty][-1])

    plt.title(f'Std accuracy (difficulty={difficulty.capitalize()})')
    plt.savefig(f'figures/std_accuracies_difficulty{difficulty}.png')
    plt.clf()


def generate_accuracy_figure(results, difficulty):
    fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(16, 12))

    for model in results.keys():
        if model not in ['FSB', 'SCIP']:
            if difficulty in results[model].keys():
                if results[model][difficulty].shape[1] > 6:
                    axs[0].bar(model.replace('expert', 'e').replace('alpha', 'a'),
                               np.mean(results[model][difficulty][:, -1]))
                    axs[1].bar(model.replace('expert', 'e').replace('alpha', 'a'),
                               np.std(results[model][difficulty][:, -1]))
        else:
            if difficulty in results[model].keys():
                if results[model][difficulty].shape[0] > 6:
                    axs[0].bar(model, results[model][difficulty][-1])

    axs[0].set_title('Mean accuracy')
    axs[0].tick_params(axis='x', labelrotation=280, size=6)
    axs[1].set_title('Std accuracy')
    axs[1].tick_params(axis='x', labelrotation=280, size=6)

    fig.suptitle(f'Accuracies (difficulty={difficulty.capitalize()})')
    plt.savefig(f'figures/accuracies_difficulty{difficulty}.png')
    plt.show()


def generate_gpu_csv(data, difficulty):
    gpu_table_header = ['gpu_time', 'gpu_time_ci', 'nb_nodes', 'nb_nodes_ci', 'lp_iters', 'lp_iters_ci']
    gpu_table_models = []
    table_height = len([x for x in data[3] if difficulty in list(data[3][x].keys())])
    gpu_table_data = np.zeros((table_height, 6))

    i = 0
    for model in data[3].keys():
        if difficulty in data[3][model].keys():
            gpu_table_models.append(model)

            if model not in ['FSB', 'SCIP']:
                mean_data = np.mean(data[3][model][difficulty], axis=0)
                ci_data = 1.98 * np.std(data[3][model][difficulty], axis=0)
            else:
                mean_data = data[3][model][difficulty][:3]
                ci_data = 1.98 * data[3][model][difficulty][3:]

            gpu_table_data[i, 0] = mean_data[1]
            gpu_table_data[i, 1] = ci_data[1]
            gpu_table_data[i, 2] = mean_data[0]
            gpu_table_data[i, 3] = ci_data[0]
            gpu_table_data[i, 4] = mean_data[2]
            gpu_table_data[i, 5] = ci_data[2]
            i += 1

    gpu_table_df = pd.DataFrame(gpu_table_data, columns=gpu_table_header, index=gpu_table_models)
    gpu_table_df.round(4).to_csv(f'tables/gpu_{difficulty}_table.csv')


def generate_gpu_cpu_time_csv(data):
    difficulty = 'easy'
    times_table_header = ['gpu_time', 'gpu_time_ci', 'cpu_time', 'cpu_time_ci']
    times_table_models = []
    table_height = len([x for x in data[3] if 'alpha10.0' in x]) + 2
    times_table_data = np.zeros((table_height, len(times_table_header)))

    i = 0
    for model in data[3].keys():
        if difficulty in data[3][model].keys() and ('alpha10.0' in model or model in ['FSB', 'SCIP']):
            times_table_models.append(model)

            if model not in ['FSB', 'SCIP']:
                mean_data = np.mean(data[3][model][difficulty][:, [1, 4]], axis=0)
                ci_data = 1.98 * np.std(data[3][model][difficulty][:, [1, 4]], axis=0)
            else:
                mean_data = [np.nan, data[3][model][difficulty][1]]
                ci_data = [np.nan, 1.98 * data[3][model][difficulty][4]]

            times_table_data[i, 0] = mean_data[0]
            times_table_data[i, 1] = ci_data[0]
            times_table_data[i, 2] = mean_data[1]
            times_table_data[i, 3] = ci_data[1]
            i += 1

    times_table_df = pd.DataFrame(times_table_data, columns=times_table_header, index=times_table_models)
    times_table_df.round(4).to_csv(f'tables/gpu_cpu_times_table.csv')


def generate_accuracies_csv(data):
    difficulty = 'easy'
    accuracies_table_header = ['accuracy', 'accuracy_ci']
    accuracies_table_models = []
    table_height = len([x for x in data[3] if 'alpha10.0' in x])
    accuracies_table_data = np.zeros((table_height, len(accuracies_table_header)))

    i = 0
    for model in data[3].keys():
        if difficulty in data[3][model].keys() and ('alpha10.0' in model or model in ['FSB', 'SCIP']):
            if model not in ['FSB', 'SCIP']:
                accuracies_table_models.append(model)
                accuracies_table_data[i, 0] = np.mean(data[3][model][difficulty][:, -1])
                accuracies_table_data[i, 1] = 1.98 * np.std(data[3][model][difficulty][:, -1], )
                i += 1

    accuracies_table_df = pd.DataFrame(accuracies_table_data, columns=accuracies_table_header,
                                       index=accuracies_table_models)
    accuracies_table_df.round(4).to_csv(f'tables/accuracies_table.csv')


if __name__ == '__main__':
    path = '/home/jeff/Documents/RL_datasets/all_results'
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
                    'mean_nb_nodes_cpu',
                    'mean_solve_time_cpu',
                    'mean_lp_iters_cpu',
                    'mean_sb_accuracy']

    train_metric_results = {}
    test_metric_results = {}

    train_test_data = [('train_results', train_results, train_epoch_metrics, train_metric_results),
                       ('test_results', test_results, test_metrics, test_metric_results)]

    nb_epochs = 40
    nb_seeds = 3
    test_difficulties = ['easy', 'medium']

    seperate_figures = True

    for data in train_test_data:
        from_cql = None
        for key, res in data[1].items():
            if isinstance(key, CQLConfig):
                model = run_key_to_model_name(key.expert_proba, key.alpha)
                from_cql = True
            else:
                model = key.name
                from_cql = False

            if data[0] == 'train_results':
                if from_cql:
                    if model not in data[3].keys():
                        data[3][model] = []

                    for metric in data[2]:
                        data[3][model].append(np.asarray([_[2] for _ in res[metric]])[-nb_epochs:])

            else:
                if from_cql:
                    if model not in data[3].keys():
                        data[3][model] = {}
                        for difficulty in test_difficulties:
                            if difficulty not in data[3][model].keys() and difficulty in res.keys():
                                data[3][model][difficulty] = []

                    for metric in data[2]:
                        for difficulty in test_difficulties:
                            if difficulty in res.keys():
                                if metric in res[difficulty]:
                                    data[3][model][difficulty].append(res[difficulty][metric])

                else:
                    if model not in data[3].keys():
                        data[3][model] = {}

                        for difficulty in test_difficulties:
                            if difficulty not in data[3][model].keys() and difficulty in res.keys():
                                data[3][model][difficulty] = []

                        for metric in mean_std_metrics:
                            for difficulty in test_difficulties:
                                if difficulty in res.keys():
                                    data[3][model][difficulty].append(
                                        res[difficulty][metric.replace('val', 'mean').replace('train', 'std')])

        if data[0] == 'train_results':
            for model, values in data[3].items():
                data[3][model] = np.asarray(values).reshape(nb_seeds, len(data[2]), nb_epochs)
        else:
            for model, values in data[3].items():
                for difficulty in test_difficulties:
                    if difficulty in values.keys():
                        if model not in ['FSB', 'SCIP']:
                            if 'alpha10.0' in model and difficulty == 'easy':
                                data[3][model][difficulty] = np.asarray(values[difficulty]).reshape(nb_seeds,
                                                                                                    len(data[2]))
                            else:
                                data[3][model][difficulty] = np.asarray(values[difficulty]).reshape(nb_seeds,
                                                                                                    len(data[2]) - 4)
                        else:
                            data[3][model][difficulty] = np.asarray(values[difficulty])

        if data[0] == 'train_results':
            if seperate_figures:
                generate_full_train_epochs_figures(data[3], train_epoch_metrics)
                generate_expert_train_epochs_figures(data[3], train_epoch_metrics, alpha=10.0)
                generate_alpha_train_epochs_figures(data[3], train_epoch_metrics)
            else:
                generate_full_train_epochs_figure(data[3], train_epoch_metrics)
                generate_expert_train_epochs_figure(data[3], train_epoch_metrics, alpha=10.0)
                generate_alpha_train_epochs_figure(data[3], train_epoch_metrics, expert='mixed')
        else:
            if seperate_figures:
                for difficulty in test_difficulties:
                    generate_gpu_test_figures(data[3], mean_std_metrics, difficulty)
                    generate_cpu_test_figures(data[3], mean_std_metrics, 'easy')
                    generate_accuracy_figures(data[3], 'easy')

            else:
                for difficulty in test_difficulties:
                    generate_gpu_test_figure(data[3], mean_std_metrics, difficulty)
                generate_cpu_test_figure(data[3], mean_std_metrics, 'easy')
                generate_accuracy_figure(data[3], 'easy')

            for difficulty in test_difficulties:
                generate_gpu_csv(data, difficulty)

            generate_gpu_cpu_time_csv(data)
            generate_accuracies_csv(data)
