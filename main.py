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
    data_collector.collect_training_data(trajectories_name=trajectories_name,
                                         nb_train_trajectories=10,
                                         expert_probability=0.05)

    # Load encoded trajectories
    training_trajectories = GraphDataset(collection_name=collection_name,
                                         trajectories_name=trajectories_name)

    # Get a specific trajectory
    graphs = training_trajectories.get(0)
