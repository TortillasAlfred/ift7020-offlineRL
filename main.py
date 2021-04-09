from core import DataCollector, GraphDataset, data_collection_stats_figure
import time


if __name__ == '__main__':
    start = time.time()
    collection_name = '10_instances_collection'
    trajectories_name = '10_trajectories'

    # Instances are created or loaded at initialization
    data_collector = DataCollector(collection_name=collection_name,
                                   nb_train_instances=10,
                                   nb_val_instances=2,
                                   nb_test_instances=2)

    # Collect trajectories from the training instances collection
    cpu_pct, cpus_pct, ram_used, ram_active, ram_pct, collect_trajectory_times,\
    collect_trajectories_total_time = data_collector.collect_training_data(trajectories_name=trajectories_name,
                                                                                  nb_train_trajectories=10,
                                                                                  expert_probability=0.05,
                                                                                  verbose=True)

    print('collect_trajectories_total_time', collect_trajectories_total_time)
    data_collection_stats_figure(cpu_pct, cpus_pct, ram_pct,
                                 ram_used, ram_active, collect_trajectory_times,
                                 collection_name, trajectories_name)

    # Load encoded trajectories
    training_trajectories = GraphDataset(collection_name=collection_name,
                                         trajectories_name=trajectories_name)

    # Get a specific trajectory
    graphs = training_trajectories.get(0)
