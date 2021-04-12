from core import DataCollector, train_gnn

import argparse

def collect_all_trajectories(nb_instances,
                             nb_trajectories,
                             collection_root,
                             collection_name,
                             base_trajectories_name,
                             expert_probabilities):
    
    # Instances are created or loaded at initialization
    data_collector = DataCollector(collection_root=collection_root,
                                   collection_name=collection_name,
                                   nb_train_instances=nb_instances,
                                   nb_val_instances=nb_instances,
                                   nb_test_instances=nb_instances)

    # Data collection (trajectories)
    for expert_probability in expert_probabilities:
        print(f'Collecting trajectories with {expert_probability} expert probability.')

        trajectories_name = f'{base_trajectories_name}_{expert_probability}'

        # Collect trajectories from the training instances collection
        data_collector.collect_training_data(trajectories_name=trajectories_name,
                                             nb_train_trajectories=nb_trajectories,
                                             expert_probability=expert_probability)

    # Mixed expert probabilities
    trajectories_name = f'{base_trajectories_name}_mixed'
    print('Regrouping the mixed trajectories dataset')

    # Collect trajectories from the training instances collection
    data_collector.collect_mixed_training_data(base_trajectories_name=base_trajectories_name,
                                                nb_instances=nb_instances,
                                                nb_trajectories=nb_trajectories,
                                                expert_probabilities=expert_probabilities)

def run_all_behaviour_cloning(collection_root,
                              collection_name,
                              base_trajectories_name,
                              expert_probabilities):

    for expert_probability in expert_probabilities:
        print(f'Training BC for {expert_probability} expert probability dataset.')

        trajectories_name = f'{base_trajectories_name}_{expert_probability}'

        train_gnn(collection_root, collection_name, trajectories_name)

    # Mixed expert probabilities
    trajectories_name = f'{base_trajectories_name}_mixed'
    print('Training BC for mixed dataset.')
    
    train_gnn(collection_root, collection_name, trajectories_name)

def main(config):
    if config.collect_trajectories:
        collect_all_trajectories(config.nb_instances,
                                 config.nb_trajectories,
                                 config.collection_root,
                                 config.collection_name,
                                 config.base_trajectories_name,
                                 config.expert_probabilities)

    if config.train_gnn_weights:
        run_all_behaviour_cloning(config.collection_root,
                                  config.collection_name,
                                  config.base_trajectories_name,
                                  config.expert_probabilities)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_instances', type=int, default=500)
    parser.add_argument('--nb_trajectories', type=int, default=3)
    parser.add_argument('--collect_trajectories', type=int, default=1) # Fake boolean
    parser.add_argument('--train_bc', type=int, default=1) # Fake boolean
    parser.add_argument('--collection_root', type=str, default='.')

    args = parser.parse_args()

    # Recast fake booleans to true booleans
    args.collect_trajectories = bool(args.collect_trajectories)
    args.train_bc = bool(args.train_bc)
    
    # Default vals
    args.collection_name = f'{args.nb_instances}_instances_collection'
    args.base_trajectories_name = f'{args.nb_trajectories}_trajectories_expert'
    args.expert_probabilities =  [0.0, 0.25, 1.0]

    main(args)