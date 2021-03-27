import gzip
import pickle
import numpy as np
import ecole
from pathlib import Path
import os


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


class DatasetBuilder:

    def __init__(self,  dataset_name='dataset', nb_episodes=100, set_cover_nb_rows=500, set_cover_nb_cols=1000,
                 set_cover_density=0.05, expert_probability=0.05):

        self.dataset_name = dataset_name
        self.expert_probability = expert_probability
        self.nb_episodes = nb_episodes
        self.set_cover_nb_rows = set_cover_nb_rows
        self.set_cover_nb_cols = set_cover_nb_cols
        self.set_cover_density = set_cover_density

    def build_dataset(self):
        instances = ecole.instance.SetCoverGenerator(n_rows=self.set_cover_nb_rows, n_cols=self.set_cover_nb_cols,
                                                     density=self.set_cover_density)

        # We can pass custom SCIP parameters easily
        scip_parameters = {'separating/maxrounds': 0, 'presolving/maxrestarts': 0, 'limits/time': 3600}

        # Note how we can tuple observation functions to return complex state information
        env = ecole.environment.Branching(observation_function=(ExploreThenStrongBranch(expert_probability=self.expert_probability),
                                                                ecole.observation.NodeBipartite()),
                                          scip_params=scip_parameters)

        # This will seed the environment for reproducibility
        env.seed(0)

        Path(f'./trajectories/{self.dataset_name}/').mkdir(parents=True, exist_ok=True)

        # We will solve problems (run episodes) until we reach the number of episodes

        strong_branching_list = []

        for i in range(self.nb_episodes):
            print(f"Collecting trajectory for episode {i+1} ...")

            trajectory = []
            strong_branching = 0

            observation, action_set, _, done, _ = env.reset(next(instances))
            while not done:
                (scores, scores_are_expert), node_observation = observation
                if scores_are_expert:
                    strong_branching += 1

                node_observation = (node_observation.row_features,
                                    (node_observation.edge_features.indices,
                                     node_observation.edge_features.values),
                                    node_observation.column_features)

                action = action_set[scores[action_set].argmax()]

                trajectory.append([node_observation, action, action_set, scores])
                observation, action_set, _, done, _ = env.step(action)

            filename = f'trajectories/{self.dataset_name}/trajectory_{i+1}.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump(trajectory, f)

            strong_branching_list.append(strong_branching)
            print(f"Trajectory for episode {i+1} contains {strong_branching} strong branching.\n")

        print(f"Collected {self.nb_episodes} trajectories, containing an average of {np.mean(strong_branching_list)}"
              f"strong branching samples.")

    def load_dataset(self):
        dataset_path = f'./trajectories/{self.dataset_name}/'
        trajectories = []

        for _, _, files in os.walk(dataset_path):
            for filename in files:
                with gzip.open(dataset_path+filename, 'rb') as f:
                    trajectories.append(pickle.load(f))
        return trajectories
