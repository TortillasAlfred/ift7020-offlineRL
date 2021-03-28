import gzip
import pickle
import numpy as np
import ecole
from pathlib import Path
import os
import filecmp


class DataCollector:

    def __init__(self, collection_root='.', collection_name='collection', nb_train_instances=100, nb_train_episodes=10,
                 nb_val_instances=20, nb_test_instances=20, set_cover_nb_rows=500, set_cover_nb_cols=1000,
                 set_cover_density=0.05, expert_probability=0.05):

        self.collection_name = collection_name
        self.collection_root = collection_root
        self.expert_probability = expert_probability
        self.nb_train_episodes = nb_train_episodes
        self.nb_train_instances = nb_train_instances
        self.set_cover_nb_rows = set_cover_nb_rows
        self.set_cover_nb_cols = set_cover_nb_cols
        self.set_cover_density = set_cover_density
        self.instance_generator = ecole.instance.SetCoverGenerator(n_rows=self.set_cover_nb_rows, n_cols=self.set_cover_nb_cols,
                                                          density=self.set_cover_density)
        self.nb_val_instances = nb_val_instances
        self.nb_test_instances = nb_test_instances
        self.val_instances = []
        self.test_instances = []
        self.set_instances(name='validation')
        self.set_instances(name='test')

    def collect_training_data(self):
        # We can pass custom SCIP parameters easily
        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

        # Note how we can tuple observation functions to return complex state information
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=self.expert_probability),
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

        for i in range(self.nb_train_instances):
            print(f"\nCollecting training trajectories for instance {i+1} ...")

            Path(f'{self.collection_root}/collections/{self.collection_name}/train_instances/').mkdir(parents=True, exist_ok=True)
            Path(f'{self.collection_root}/collections/{self.collection_name}/train_trajectories/instance_{i+1}/').mkdir(parents=True, exist_ok=True)
            file = f'{self.collection_root}/collections/{self.collection_name}/train_instances/instance_{i+1}.lp'

            instance = next(self.instance_generator)
            instance.write_problem(file)

            if self.instance_unavailable(file):
                print(f"Training instance {i+1} is in the validation or test collections and will be discarded.")
                os.remove(file)
                continue

            for j in range(self.nb_train_episodes):
                observation, action_set, _, done, rewards = env.reset(instance)

                trajectory = []
                strong_branching = 0
                weak_branching = 0

                while not done:
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

                    trajectory.append([node_observation, action, action_set, rewards])
                    observation, action_set, _, done, rewards = env.step(action)

                filename = f'{self.collection_root}/collections/{self.collection_name}/train_trajectories/instance_{i+1}/trajectory_{j+1}.pkl'
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

        print(f"Collected {(self.nb_train_instances * self.nb_train_episodes) - discarded_trajectories} trajectories, containing"
              f" an average of {np.round(np.mean(strong_branching_list),2)} strong branching and an average"
              f" of {np.round(np.mean(weak_branching_list),2)} weak branching.")

    def load_training_trajectories(self):
        dataset_path = f'{self.collection_root}/collections/{self.collection_name}/train_trajectories/'
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
        Path(f'{path}').mkdir(parents=True, exist_ok=True)

        for i in range(nb_instances):
            instance = next(self.instance_generator)
            if instance not in (self.val_instances + self.test_instances):  # TODO
                instance.write_problem(f'{path}instance_{i+1}.lp')
                if name == 'validation':
                    self.val_instances.append(instance)
                if name == 'test':
                    self.test_instances.append(instance)
            else:
                continue

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
                self.save_instances(self.nb_val_instances, name=name)
            else:
                print(f"Loading {name} instances ...")

    def instance_unavailable(self, instance_file):
        path = f'{self.collection_root}/collections/{self.collection_name}/validation_instances/'
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

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)
