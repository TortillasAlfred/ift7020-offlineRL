import gzip
import pickle
import numpy as np
import ecole
from pathlib import Path
import os
import filecmp
import psutil
import time


class DataCollector:

    def __init__(self, collection_root='.', collection_name='collection', nb_train_instances=100,
                 nb_val_instances=20, nb_test_instances=20, set_cover_nb_rows=500, set_cover_nb_cols=1000,
                 set_cover_density=0.05):

        self.collection_name = collection_name
        self.collection_root = collection_root
        self.set_cover_nb_rows = set_cover_nb_rows
        self.set_cover_nb_cols = set_cover_nb_cols
        self.set_cover_density = set_cover_density
        self.instance_generator = ecole.instance.SetCoverGenerator(n_rows=self.set_cover_nb_rows,
                                                                   n_cols=self.set_cover_nb_cols,
                                                                   density=self.set_cover_density)
        self.nb_val_instances = nb_val_instances
        self.nb_test_instances = nb_test_instances
        self.nb_train_instances = nb_train_instances
        self.val_instances = []
        self.test_instances = []
        self.train_instances = []
        self.set_instances(name='validation')
        self.set_instances(name='test')
        self.set_instances(name='train')

    def collect_training_data(self, trajectories_name='trajectories_name', nb_train_trajectories=10,
                              expert_probability=0.05, verbose=False):
        if verbose:
            cpu_pct = []
            cpus_pct = []
            ram_active = []
            ram_used = []
            ram_pct = []
            collect_trajectory_times = []
            start_total_time = time.time()

        # We can pass custom SCIP parameters easily
        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

        # Note how we can tuple observation functions to return complex state information
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=expert_probability),
                                                                ecole.observation.NodeBipartite()),
                                          scip_params=scip_parameters,
                                          information_function={"nb_nodes": ecole.reward.NNodes(),
                                                                "lp_iterations": ecole.reward.LpIterations(),
                                                                "time": ecole.reward.SolvingTime()})

        # This will seed the environment for reproducibility
        env.seed(0)

        # We will solve problems (run episodes) until we reach the number of episodes

        strong_branching_list = []
        weak_branching_list = []
        discarded_trajectories = 0

        Path(f'{self.collection_root}/collections/{self.collection_name}/train_instances/').mkdir(parents=True, exist_ok=True)

        for i, instance in enumerate(self.train_instances):
            print(f"\nCollecting training trajectories for instance {i+1} ...")

            Path(f'{self.collection_root}/collections/{self.collection_name}/{trajectories_name}/instance_{i + 1}/').mkdir(
                parents=True, exist_ok=True)

            for j in range(nb_train_trajectories):
                observation, action_set, _, terminal, rewards = env.reset(instance)

                trajectory = []
                strong_branching = 0
                weak_branching = 0

                if verbose:
                    start_time = time.time()
                    cpu_pct.append(psutil.cpu_percent())
                    cpus_pct.append(psutil.cpu_percent(percpu=True))
                    ram_used.append(psutil.virtual_memory().used / 1e+9)
                    ram_active.append(psutil.virtual_memory().active / 1e+9)
                    ram_pct.append(psutil.virtual_memory().percent)

                while not terminal:
                    (scores, scores_are_expert), node_observation = observation
                    if scores_are_expert:
                        strong_branching += 1
                    else:
                        weak_branching += 1

                    node_observation = (node_observation.row_features,
                                        (node_observation.edge_features.indices,
                                         node_observation.edge_features.values),
                                        node_observation.column_features)

                    action = action_set[scores[action_set].argmax()]

                    trajectory.append([node_observation, action, action_set, rewards, terminal])
                    observation, action_set, _, terminal, rewards = env.step(action)
                    if terminal:
                        trajectory[-1][-1] = True
                if verbose:
                    collect_trajectory_times.append(time.time() - start_time)

                filename = f'{self.collection_root}/collections/{self.collection_name}/{trajectories_name}/instance_{i+1}/trajectory_{j+1}.pkl'
                if len(trajectory) > 0:
                    with gzip.open(filename, 'wb') as f:
                        pickle.dump(trajectory, f)

                    strong_branching_list.append(strong_branching)
                    weak_branching_list.append(weak_branching)
                    print(f"Trajectory for episode {j+1} contains {strong_branching} strong branching and "
                          f"{weak_branching} weak branching.")
                else:
                    discarded_trajectories += 1
                    print(f"Trajectory for episode {j+1} is empty and will be discarded.")

        print(f"\nCollected {(len(self.train_instances) * nb_train_trajectories) - discarded_trajectories} trajectories, containing"
              f" an average of {np.round(np.mean(strong_branching_list),2)} strong branching and an average"
              f" of {np.round(np.mean(weak_branching_list),2)} weak branching.")

        if verbose:
            collect_trajectories_total_time = time.time() - start_total_time
            return cpu_pct, cpus_pct, ram_used, ram_active, ram_pct, collect_trajectory_times, collect_trajectories_total_time

    def load_training_trajectories(self, trajectories_name='trajectories_name'):
        dataset_path = f'{self.collection_root}/collections/{self.collection_name}/{trajectories_name}/'
        trajectories = []

        for _, directories, _ in os.walk(dataset_path):
            for directory in directories:
                for _, _, files in os.walk(dataset_path+directory):
                    for file in files:
                        with gzip.open(f"{dataset_path}{directory}/{file}", 'rb') as f:
                            trajectories.append(pickle.load(f))
        return trajectories

    def save_instances(self, nb_instances, name='validation'):
        print(f"Saving {name} instances ...")

        path = f'{self.collection_root}/collections/{self.collection_name}/{name}_instances/'
        temp_path = f'{self.collection_root}/collections/{self.collection_name}/temp_instances/'
        Path(f'{path}').mkdir(parents=True, exist_ok=True)
        Path(f'{temp_path}').mkdir(parents=True, exist_ok=True)

        for i in range(nb_instances):
            file = f'{self.collection_root}/collections/{self.collection_name}/{name}_instances/instance_{i + 1}.lp'
            temp_file = f'{self.collection_root}/collections/{self.collection_name}/temp_instances/instance_{i + 1}.lp'

            instance = next(self.instance_generator)
            instance.write_problem(temp_file)

            if self.instance_unavailable(temp_file):
                print(f"{name.capitalize()} instance {i + 1} already exist and will be discarded.")
                os.remove(temp_file)
                continue

            os.rename(temp_file, file)

            if name == 'validation':
                self.val_instances.append(instance)
            if name == 'test':
                self.test_instances.append(instance)
            if name == 'train':
                self.train_instances.append(instance)

        os.rmdir(temp_path)

    def load_instances(self, name='validation'):
        path = f'{self.collection_root}/collections/{self.collection_name}/{name}_instances/'
        Path(f'{path}').mkdir(parents=True, exist_ok=True)
        loaded_instances = []
        for _, _, files in os.walk(path):
            for file in files:
                instance = ecole.scip.Model.from_file(path+file)
                loaded_instances.append(instance)
        if name == 'validation':
            self.val_instances = loaded_instances
        if name == 'test':
            self.test_instances = loaded_instances
        if name == 'train':
            self.train_instances = loaded_instances
        return loaded_instances

    def set_instances(self, name='validation'):
        self.load_instances(name=name)
        if name == 'validation':
            if len(self.val_instances) == 0:
                self.save_instances(self.nb_val_instances, name=name)
            else:
                print(f"Loading {name} instances ...")
        if name == 'test':
            if len(self.test_instances) == 0:
                self.save_instances(self.nb_test_instances, name=name)
            else:
                print(f"Loading {name} instances ...")
        if name == 'train':
            if len(self.train_instances) == 0:
                self.save_instances(self.nb_train_instances, name=name)
            else:
                print(f"Loading {name} instances ...")

    def instance_unavailable(self, instance_file):
        instance_folders = ['train', 'validation', 'test']
        for instance_folder in instance_folders:
            path = f'{self.collection_root}/collections/{self.collection_name}/{instance_folder}_instances/'
            for _, _, files in os.walk(path):
                for file in files:
                    if self.duplicated_instance(instance_file, path+file):
                        return True
        return False

    @staticmethod
    def duplicated_instance(instance1_file, instance2_file):
        return filecmp.cmp(instance1_file, instance2_file)


class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, terminal):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, terminal), True)
        else:
            return (self.pseudocosts_function.extract(model, terminal), False)
