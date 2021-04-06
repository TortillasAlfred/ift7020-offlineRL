from core import DataCollector, GraphDataset
import time


if __name__ == '__main__':
    start = time.time()
    collection_name = 'small_collection'
    trajectories_name = 'expert_5_percent'

    # Instances are created or loaded at initialization
    data_collector = DataCollector(collection_name=collection_name,
                                   nb_train_instances=5,
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
