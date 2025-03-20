from stable_baselines3 import DQN  # to support reinforcement learning algorithms by offering an interface to train, evaluate agents and do hyperparameters tuning
import optuna  # hyperparameters optimization framework to deal with hyperparameters tuning
import pandas as pd  # to load data and analyse, manipulate and visualize the DataFrame
from pathlib import Path  # to work with file and directory paths (read and write files)

from ODU1_env import ODUEnvironment  # To import our working environment
from custom_callback import TrainLoggingCallback  # to deal with training loops for logs, agent evaluation and checkpoints


# to write the results
OUTPUT_FILE = './results.txt'
BEST_ARGS = None
BEST_SCORE = float('-inf') # to handle negative numbers if there is any

# Load the data
Dataset = pd.read_csv("ALL_data_30min_train.csv")

# Initialize the working environment
env = ODUEnvironment(Dataset=Dataset)
eval_env = ODUEnvironment(Dataset=Dataset)


# to define objective function to be optimized
def create_agent(trial):
    agent = DQN(
        policy='MlpPolicy',  # The policy model to use
        env=env,  # The environment to learn from registered in Gym class
        learning_rate=trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2, 1e-1]),  # learning rate for adam optimizer (controls how much the agent's policy is updated based on the observed rewards)
        buffer_size=trial.suggest_categorical('buffer_size', [10_000, 100_000, 1_000_000]),  # size of the replay memory
        tau=trial.suggest_categorical('tau', [0.0, 0.2, 0.5, 1.0]), #  controls the soft update of the target network
        gamma=trial.suggest_categorical('gamma', [0.99, 0.95, 0.8, 0.7]),  # controls how much importance is placed on future rewards
        train_freq=trial.suggest_categorical('train_freq', [4, 10]),  # controls how often the agent's policy is updated
        gradient_steps=trial.suggest_categorical('gradient_steps', [4, 1]), # determines how many gradient steps of the loss function is calculated and used to update the parameters of the policy
        target_update_interval=trial.suggest_categorical('target_update_interval', [1, 5, 10]), #  controls how often the target network is updated
        exploration_fraction=trial.suggest_categorical('exploration_fraction', [0.1, 0.2, 0.3, 0.4, 0.5]), # controls the amount of exploration that the agent performs (the probability that the agent will take a random action)
        seed=trial.suggest_categorical('seed', [1, 3, 13]), # to ensure that the results of the experiment are reproducible 
    )

    return agent


def objective(trial):

    agent = create_agent(trial)

    # define some training/callback parameters 
    train_timesteps = 1_000_000
    timesteps_save_freq = 10_000
    save_dir = Path('./saved_agents/')
    model_name = 'dqn'
    verbose = 1 # to display the training progress information
    eval_freq = 10

    callback = TrainLoggingCallback(
        save_freq=timesteps_save_freq,
        save_dir=save_dir,
        model_name=model_name,
        verbose=verbose,
        eval_env=eval_env,
        eval_freq = eval_freq,
        start_saving_timesteps = 50_000
    )
    agent.learn(total_timesteps=100_000, callback=callback) # train the agent
    score = callback.best_score_and_steps[0]  # assign the best score 

    del agent

    # return it
    return score


def main():
    """The main() driver function."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)


if __name__ == "__main__":
    print()
    main()

