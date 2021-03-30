from core import DataCollector
import time


if __name__ == '__main__':
    start = time.time()
    collection_name = 'test_collection'
    data_collector = DataCollector(collection_name=collection_name, nb_train_instances=5, nb_train_episodes=5,
                                   nb_val_instances=5, nb_test_instances=5)

    data_collector.collect_training_data()
    # training_trajectories = data_collector.load_training_trajectories()
    training_instances = data_collector.load_instances(name='train')
    validation_instances = data_collector.load_instances(name='validation')
    test_instances = data_collector.load_instances(name='test')
    print('Time:', time.time()-start)
