import torch
import random
import numpy as np
import pandas as pd

from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from ODU2_env import ODUEnvironment
from utils import force_empty_directory
from custom_callback import TrainLoggingCallback



def main():
    """main driver function."""
    Dataset = pd.read_csv("ALL_data_30min_train.csv")
    env = ODUEnvironment(Dataset=Dataset)

    check_env(env) # to check for possible errors in our initialized environment

    # define the hyperparameters for the DQN agent, refer here: used 41 trial
    policy = 'MlpPolicy'
    learning_rate = 0.01
    buffer_size = 100000
    tau = 0.2
    gamma = 0.95
    train_freq = 4
    gradient_steps = 1
    target_update_interval = 10
    exploration_fraction = 0.1
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
    train_timesteps = 1_000_000
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

    # train
    agent.learn(
        total_timesteps=train_timesteps,
        callback=callback
    )


    del agent

if __name__ == "__main__":
    print()
    main()
