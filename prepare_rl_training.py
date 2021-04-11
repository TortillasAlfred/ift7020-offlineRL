from core import DataCollector, train_gnn


if __name__ == '__main__':

    # Configuration
    nb_instances = 100
    nb_trajectories = 10
    collect_trajectories = True
    collect_mixed_trajectories = True
    train_gnn_weights = False
    collection_root = '.'
    collection_name = f'{nb_instances}_instances_collection'
    base_trajectories_name = f'{nb_trajectories}_trajectories_expert'

    # Instances are created or loaded at initialization
    data_collector = DataCollector(collection_root=collection_root,
                                   collection_name=collection_name,
                                   nb_train_instances=nb_instances,
                                   nb_val_instances=nb_instances,
                                   nb_test_instances=nb_instances)

    # Data collection (trajectories and GNN weights)
    expert_probabilities = [0.0, 0.05, 0.2, 0.5, 1.0]

    for expert_probability in expert_probabilities:

        trajectories_name = f'{base_trajectories_name}_{expert_probability}'
        print(f'{trajectories_name.upper()}')

        if collect_trajectories:
            # Collect trajectories from the training instances collection
            data_collector.collect_training_data(trajectories_name=trajectories_name,
                                                 nb_train_trajectories=nb_trajectories,
                                                 expert_probability=expert_probability)
        if train_gnn_weights:
            train_gnn(collection_root, collection_name, trajectories_name)

    # Mixed expert probabilities
    trajectories_name = f'{base_trajectories_name}_mixed'

    if collect_mixed_trajectories:
        # Collect trajectories from the training instances collection
        data_collector.collect_mixed_training_data(base_trajectories_name=base_trajectories_name,
                                                   nb_instances=nb_instances,
                                                   nb_trajectories=nb_trajectories,
                                                   expert_probabilities=expert_probabilities)
    if train_gnn_weights:
        train_gnn(collection_root, collection_name, trajectories_name)
