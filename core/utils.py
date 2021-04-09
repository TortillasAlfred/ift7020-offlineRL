from matplotlib import pyplot as plt
import numpy as np


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
