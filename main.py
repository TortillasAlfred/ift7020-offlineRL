from core import DataCollector

if __name__ == '__main__':
    data_collector = DataCollector(collection_name='test_collection', nb_episodes=10)
    data_collector.collect_data()
    data = data_collector.load_data()
    print()
