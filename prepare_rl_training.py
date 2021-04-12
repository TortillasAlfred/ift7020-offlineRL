from core import DataCollector, train_gnn

import argparse

def collect_all_trajectories(nb_instances,
                             nb_trajectories,
                             collection_root,
                             collection_name,
                             base_trajectories_name,
                             expert_probability,
                             job_index):
    
    # Instances are created or loaded at initialization
    data_collector = DataCollector(collection_root=collection_root,
                                   collection_name=collection_name,
                                   nb_train_instances=nb_instances)

    print(f'Collecting trajectories with {expert_probability} expert probability.')

    trajectories_name = f'{base_trajectories_name}_{expert_probability}'

    # Collect trajectories from the training instances collection
    data_collector.collect_training_data(trajectories_name=trajectories_name,
                                         nb_train_trajectories=nb_trajectories,
                                         expert_probability=expert_probability,
                                         job_index=job_index)

def run_all_behaviour_cloning(collection_root,
                              collection_name,
                              base_trajectories_name,
                              expert_probability):

    print(f'Training BC for {expert_probability} expert probability dataset.')

    trajectories_name = f'{base_trajectories_name}_{expert_probability}'

    train_gnn(collection_root, collection_name, trajectories_name)

def mixed_run(base_trajectories_name,
              collection_root,
              collection_name,
              nb_instances,
              nb_trajectories,
              expert_probabilities=[0.0, 0.25, 1.0]):

    # Instances are created or loaded at initialization
    data_collector = DataCollector(collection_root=collection_root,
                                   collection_name=collection_name,
                                   nb_train_instances=nb_instances)
                                   
    # Mixed expert probabilities
    trajectories_name = f'{base_trajectories_name}_mixed'
    print('Regrouping the mixed trajectories dataset')

    # Collect trajectories from the training instances collection
    data_collector.collect_mixed_training_data(base_trajectories_name=base_trajectories_name,
                                                nb_instances=nb_instances,
                                                nb_trajectories=nb_trajectories,
                                                expert_probabilities=expert_probabilities)
    
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
                                 config.expert_probability,
                                 config.job_index)

    if config.train_bc:
        run_all_behaviour_cloning(config.collection_root,
                                  config.collection_name,
                                  config.base_trajectories_name,
                                  config.expert_probability)

    if config.mixed_run:
        mixed_run(config.base_trajectories_name,
                  config.collection_root,
                  config.collection_name,
                  config.nb_instances,
                  config.nb_trajectories,)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nb_instances', type=int, default=1000)
    parser.add_argument('--nb_trajectories', type=int, default=3)
    parser.add_argument('--collect_trajectories', type=int, default=0) # Fake boolean
    parser.add_argument('--train_bc', type=int, default=0) # Fake boolean
    parser.add_argument('--mixed_run', type=int, default=0) # Fake boolean
    parser.add_argument('--collection_root', type=str, default='.')
    parser.add_argument('--expert_probability', type=float, default=0.0)
    parser.add_argument('--job_index', type=int, default=-1)

    args = parser.parse_args()

    # Recast fake booleans to true booleans
    args.collect_trajectories = bool(args.collect_trajectories)
    args.train_bc = bool(args.train_bc)
    args.mixed_run = bool(args.mixed_run)
    
    # Default vals
    args.collection_name = f'{args.nb_instances}_instances_collection'
    args.base_trajectories_name = f'{args.nb_trajectories}_trajectories_expert'

    if args.job_index > -1:
        proba_index = int(args.job_index / 200)
        args.expert_probability = [0.0, 0.25, 1.0][proba_index]

        args.job_index = args.job_index % 200

    main(args)