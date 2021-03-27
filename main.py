from core import DatasetBuilder

if __name__ == '__main__':
    dataset_builder = DatasetBuilder(dataset_name='test_dataset', nb_episodes=10)
    dataset_builder.build_dataset()
    dataset = dataset_builder.load_dataset()
