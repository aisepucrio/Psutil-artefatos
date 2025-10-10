import psutil
import csv
import os
import time
import gym
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from torch import normal
from codecarbon import EmissionsTracker

import settings

# Start CodeCarbon tracker
tracker = EmissionsTracker(project_name=r"C:\Users\guicu\OneDrive\Documentos\prog\aise\artifact\artifacts\benchmarking_rlmut\RLMutation\RLMT\mutated_env.py")
tracker.start()


class Buffer:
    """
    Buffer for storing the transitions of the agent.
    """

    def __init__(self, max_size: int) -> None:
        """
        Buffer for storing the transitions of the agent.

        Args:
            max_size (int): maximum buffer size
        """
        self.max_size = max_size
        self.buffer = []

    def add(self, transition: Dict) -> None:
        """
        Add a transition to the buffer.

        Args:
            transition (Dict): Transition to add. Each transition is a dictionary containing the following keys:
                - state: Current state.
                - action: Action taken.
                - reward: Reward received.
                - next_state: Next state.
        """
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)

        self.buffer.append(transition)

    def sample(self, batch_size=None) -> Dict:
        """
        Sample a batch of transitions.

        Args:
            batch_size (int): Size of the batch to sample. If None, a single transition will be returned.

        Returns:
            Dict: A single transitions.
        """
        if batch_size is None:
            return random.sample(self.buffer, 1)
        else:
            return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """
        Return the number of transitions in the buffer.

        Returns:
            int: Number of transitions in the buffer.
        """
        return len(self.buffer)

    def __getitem__(self, index: int) -> Dict:
        """
        Returns the transition at the given index.

        Args:
            index (int): index of the transition to return

        Returns:
            Dict: Transition at the given index.
        """
        return self.buffer[index]


class MutatedMonitor(Monitor):
    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        mutation: dict = None,
    ):
        super().__init__(env=env, filename=filename)
        self.past_obs = None
        self.buffer = Buffer(max_size=1000)
        self.mutation = mutation
        assert (
            "repeat" in self.mutation
            or "random" in self.mutation
            or "mangled" in self.mutation
            or "reward_noise" in self.mutation
        ), "Invalid environment mutation in mutation dictionary {}".format(self.mutation)

    def _apply_mutation(self, observation, reward):
        mut_env = list(set(self.mutation).intersection(settings.MUTATION_ENVIRONMENT_LIST))[0]
        mutated = False
        for mut_ in mut_env:
            if random.random() < float(self.mutation[mut_]):
                if "repeat" in self.mutation and self.total_steps > 0:
                    step_info = self.buffer[-2]
                    observation = step_info["next_state"]
                    reward = step_info["reward"]
                if "random" in self.mutation:
                    step_info = self.buffer.sample()[0]
                    observation = step_info["next_state"]
                    reward = step_info["reward"]
                if "mangled" in self.mutation and self.total_steps > 0:
                    batch_size = min(10, self.total_steps)
                    transition_pool = self.buffer.sample(batch_size=batch_size)
                    observation = random.choice(transition_pool)["next_state"]
                    reward = random.choice(transition_pool)["reward"]
                if "reward_noise" in self.mutation:
                    reward += random.normalvariate(0, 0.1 * reward)
                mutated = True
        return observation, reward, mutated

    def _handle_episode_end(self, info):
        ep_rew = sum(self.rewards)
        ep_len = len(self.rewards)
        ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
        for key in self.info_keywords:
            ep_info[key] = info[key]
        self.episode_returns.append(ep_rew)
        self.episode_lengths.append(ep_len)
        self.episode_times.append(time.time() - self.t_start)
        ep_info.update(self.current_reset_info)
        if self.results_writer:
            self.results_writer.write_row(ep_info)
        info["episode"] = ep_info
        return info

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        step_info = {"state": self.past_obs, "action": None, "reward": None, "next_state": None}

        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        observation, reward, done, info = self.env.step(action)

        step_info["action"] = action
        step_info["reward"] = reward
        step_info["next_state"] = observation
        self.past_obs = observation
        self.buffer.add(step_info)

        observation, reward, mutated = self._apply_mutation(observation, reward)

        self.rewards.append(reward)

        if done:
            self.needs_reset = True
            info = self._handle_episode_end(info)
        self.total_steps += 1
        return observation, reward, done, info


tracker.stop()
