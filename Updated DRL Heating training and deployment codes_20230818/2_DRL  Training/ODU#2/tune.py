from stable_baselines3 import DQN
import optuna
import pandas as pd
from pathlib import Path

from ODU2_env import ODUEnvironment
from custom_callback import TrainLoggingCallback


OUTPUT_FILE = './results.txt'
BEST_ARGS = None
BEST_SCORE = float('-inf')

Dataset = pd.read_csv("ALL_data_30min_train.csv")

env = ODUEnvironment(Dataset=Dataset)
eval_env = ODUEnvironment(Dataset=Dataset)


def create_agent(trial):
    agent = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2, 1e-1]),
        buffer_size=trial.suggest_categorical('buffer_size', [10_000, 100_000, 1_000_000]),
        tau=trial.suggest_categorical('tau', [0.0, 0.2, 0.5, 1.0]),
        gamma=trial.suggest_categorical('gamma', [0.99, 0.95, 0.8, 0.7]),
        train_freq=trial.suggest_categorical('train_freq', [4, 10]),
        gradient_steps=trial.suggest_categorical('gradient_steps', [4, 1]),
        target_update_interval=trial.suggest_categorical('target_update_interval', [1, 5, 10]),
        exploration_fraction=trial.suggest_categorical('exploration_fraction', [0.1, 0.2, 0.3, 0.4, 0.5]),
        seed=trial.suggest_categorical('seed', [1, 3, 13]),

    )

    return agent


def objective(trial):

    agent = create_agent(trial)

    # define some training/callback parameters here
    train_timesteps = 1_000_000
    timesteps_save_freq = 10_000
    save_dir = Path('./saved_agents/')
    model_name = 'dqn'
    verbose = 1
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
    agent.learn(total_timesteps=100_000, callback=callback)
    score = callback.best_score_and_steps[0]

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

