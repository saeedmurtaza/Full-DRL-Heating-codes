import torch # used to create deep neural networks

from typing import Union  # to specify multiple datatypes
from pathlib import Path  # to work with file and directory paths (read and write files)
from stable_baselines3.common.callbacks import BaseCallback  # to access internal state of the RL model such as events (on_step()) or model variables (self._save_model)
from ODU1_env import ODUEnvironment  # To initilaize our environment
from utils import force_empty_directory  # # To use contained functions and classes 


class TrainLoggingCallback(BaseCallback):
    """Implements a basic callback that saves a model every `save_freq`
        steps."""

    def __init__(
            self,
            save_freq: int = None,
            save_dir: Path = None,
            model_name: str = None,
            verbose: int = 1,
            eval_env: ODUEnvironment = None,
            eval_freq: int = None,
            start_saving_timesteps: int = None
    ) -> None:
        """
        :param save_freq:
            Every how many steps to save the model.

        :param save_dir:
            Target directory where the model will be saved.

        :param model_name:
            The name of the model that will be used when saving it.

        :param verbose:
            Whether to print stuff on the console.

        :param eval_env:
            Environment used to evaluate the agent every `eval_freq` steps.

        :param eval_freq:
            How often to evaluate the agent.
        """
        super(TrainLoggingCallback, self).__init__(verbose=verbose)

        # set important variables
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.model_name = model_name
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_score_and_steps = (float('-inf'), -1)
        self.start_saving_timesteps = start_saving_timesteps

    def _init_callback(self) -> None:
        """Creates the empty save directory (recursively, if it
            doesn't exist)."""
        if self.save_dir is not None:
            force_empty_directory(self.save_dir)

    def _save_model(self, checkpoint: Union[int, str]) -> None:
        """Saves the current model."""
        model_filename = f'{self.model_name}_checkpoint_{checkpoint}'
        model_path = self.save_dir/model_filename
        if self.verbose > 0:
            print(f'Saving model at path: {model_path}')
        self.model.save(model_path)

    def _evaluate_model(self) -> None:
        """Evaluates the agent and saves the corresponding variables. Saves
            the agent if it has improved, and if a save directory has been
            specified."""
        obs = self.eval_env.reset()
        with torch.no_grad():
            action = self.model.predict(observation=obs)[0]
        _, reward, _, _ = self.eval_env.step(action)
        if reward > self.best_score_and_steps[0]:
            self.best_score_and_steps = (reward, self.n_calls)
            if self.save_dir is not None:
                self._save_model('best_model')

    def _on_step(self) -> bool:
        """What to do after every step, when the callback is called. Save the
            agent if a specific number of steps has elapsed. Returns whether
            training should continue or not."""
        # save the model if it's time to
        if self.n_calls % self.save_freq == 0 and self.save_dir is not None:
            self._save_model(self.n_calls)

        # check if it's time to evaluate the agent
        if self.eval_env is not None and self.n_calls > self.start_saving_timesteps:
            self._evaluate_model()

        # continue training
        return True

    def _on_training_end(self) -> None:
        """When training ends, save the final model."""
        if self.save_dir is not None:
            self._save_model(self.n_calls)
