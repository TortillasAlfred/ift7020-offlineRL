from core import DataCollector, GraphDataset, GNNPolicy, process_epoch
import torch
import torch.nn.functional as F
import torch_geometric
import glob


if __name__ == '__main__':
    collection_root = '.'
    collection_name = '100_instances_collection'

    # TODO mixing trajectoires mixte même nb de trajectoires que nb_train_instances=100

    expert_probabilities = [0.0, 0.05, 0.2, 0.5, 1.0]

    for expert_probability in expert_probabilities:

        trajectories_name = f'10_trajectories_expert_{expert_probability}'

        # Instances are created or loaded at initialization
        data_collector = DataCollector(collection_root=collection_root,
                                       collection_name=collection_name,
                                       nb_train_instances=100,
                                       nb_val_instances=100,
                                       nb_test_instances=100)

        # Collect trajectories from the training instances collection
        data_collector.collect_training_data(trajectories_name=trajectories_name,
                                             nb_train_trajectories=10,
                                             expert_probability=expert_probability)

        # Train GNN
        LEARNING_RATE = 0.001
        NB_EPOCHS = 50
        PATIENCE = 10
        EARLY_STOPPING = 20
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        path = f'{collection_root}/collections/{collection_name}/{trajectories_name}/'

        sample_files = [str(file_path) for file_path in glob.glob(f'{path}*')]
        sample_files.sort()
        train_files = sample_files[:int(0.8 * len(sample_files))]
        valid_files = sample_files[int(0.8 * len(sample_files)):]

        train_data = GraphDataset(train_files)
        train_loader = torch_geometric.data.DataLoader(train_data, batch_size=32, shuffle=True)
        valid_data = GraphDataset(valid_files)
        valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=128, shuffle=False)

        policy = GNNPolicy().to(DEVICE)

        observation = train_data[0].to(DEVICE)

        logits = policy(observation.constraint_features, observation.edge_index, observation.edge_attr,
                        observation.variable_features)
        action_distribution = F.softmax(logits[observation.candidates], dim=-1)

        print(action_distribution)

        optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)
        for epoch in range(NB_EPOCHS):
            print(f"Epoch {epoch + 1}")

            train_loss, train_acc = process_epoch(policy, train_loader, optimizer, device=DEVICE)
            print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}")

            valid_loss, valid_acc = process_epoch(policy, valid_loader, None, device=DEVICE)
            print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}")

        torch.save(policy.state_dict(), f'{path}GCNN_trained_params.pkl')
