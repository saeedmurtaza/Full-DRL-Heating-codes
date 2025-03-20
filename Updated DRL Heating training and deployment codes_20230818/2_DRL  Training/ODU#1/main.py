import torch # to create deep neural networks
import random # to generate a random float values uniformly
import numpy as np # to deal with arrays and numerical values
import pandas as pd  # to load data into DataFrame

from pathlib import Path # to work with file and directory paths (read and write files)
from stable_baselines3 import DQN  # to support reinforcement learning algorithms by offering an interface to train, evaluate agents and do hyperparameters tuning
from stable_baselines3.common.env_checker import check_env # To check any possible errors in inherited environment

from ODU1_env import ODUEnvironment  # To initilaize our environment
from utils import force_empty_directory # To use contained tools and utilities for DQN and neural network training
from custom_callback import TrainLoggingCallback  # to deal with training loops for logs, agent evaluation and creating checkpoints



def main():
    """main driver function."""
    Dataset = pd.read_csv("ALL_data_30min_train.csv")
    env = ODUEnvironment(Dataset=Dataset)

    check_env(env) # to check for possible errors in our initialized environment

    # define the hyperparameters for the DQN agent
    policy = 'MlpPolicy'
    learning_rate = 0.01
    buffer_size = 10000
    tau = 0.2
    gamma = 0.99
    train_freq = 4
    gradient_steps = 1
    target_update_interval = 5
    exploration_fraction = 0.3
    tensorboard_log = './logs/'
    verbose = 0
    seed = 1
    device = 'auto'

    # empty the logs directory
    force_empty_directory(Path(tensorboard_log))

    # fix seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

    # Build the agent

    agent = DQN(
        policy=policy,
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        device=device
        )

    # define some training/callback parameters here
    train_timesteps = 80_000
    timesteps_save_freq = 10_000
    save_dir = Path('./saved_agents_1/')
    model_name = 'dqn'
    verbose = 1
    eval_env = ODUEnvironment(Dataset=Dataset)
    eval_freq = 10

    # Calling the callback here
    callback = TrainLoggingCallback(
        save_freq=timesteps_save_freq,
        save_dir=save_dir,
        model_name=model_name,
        verbose=verbose,
        eval_env=eval_env,
        eval_freq=eval_freq,
        start_saving_timesteps = 50_000
    )

    # train the model
    agent.learn(
        total_timesteps=train_timesteps,
        callback=callback
    )

    del agent

    # to execute the main driver function
if __name__ == "__main__":
    print()
    main()
