from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

import gym
import sys
import warnings
import time
import numpy as np
import torch as th

from gym import spaces
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.buffers import (
    RolloutBuffer,
    ReplayBuffer,
    DictRolloutBuffer,
)
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.noise import ActionNoise

from stable_baselines3.common.utils import (
    get_linear_fn,
    get_parameters_by_name,
    is_vectorized_observation,
    polyak_update,
)
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
    safe_mean,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.dqn.policies import (
    CnnPolicy,
    DQNPolicy,
    MlpPolicy,
    MultiInputPolicy,
)

import settings

try:
    import psutil
except ImportError:
    psutil = None


class MutatedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        mutation: dict = None,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )
        self.mutation = mutation

    def _handle_discrete_action(self, action: np.ndarray) -> np.ndarray:
        if isinstance(self.action_space, spaces.Discrete):
            return action.reshape((self.n_envs, self.action_dim))
        return action

    def _handle_discrete_obs(self, obs: np.ndarray) -> np.ndarray:
        if isinstance(self.observation_space, spaces.Discrete):
            return obs.reshape((self.n_envs,) + self.obs_shape)
        return obs

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        obs = self._handle_discrete_obs(obs)
        next_obs = self._handle_discrete_obs(next_obs)
        action = self._handle_discrete_action(action)

        if "missing_state_update" in self.mutation:
            next_obs = np.array(obs).copy()

        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()

        if "missing_terminal_state" in self.mutation:
            self.dones[self.pos] = 0
        else:
            self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(
                [info.get("TimeLimit.truncated", False) for info in infos]
            )

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0


class MutatedRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        mutation: dict = None,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.mutation = mutation

    def _calculate_delta(
        self, step: int, next_non_terminal: float, next_values: th.Tensor
    ) -> float:
        return (
            self.rewards[step]
            + self.gamma * next_values * next_non_terminal
            - self.values[step]
        )

    def _calculate_gae(self, delta: float, next_non_terminal: float, last_gae_lam: float) -> float:
        return delta + self.gae_lambda * next_non_terminal * last_gae_lam

    def compute_returns_and_advantage(
        self, last_values: th.Tensor, dones: np.ndarray
    ) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0

        if "no_reverse" in self.mutation:
            for step in range(self.buffer_size):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_values = self.values[step + 1]
                delta = self._calculate_delta(step, next_non_terminal, next_values)
                last_gae_lam = self._calculate_gae(delta, next_non_terminal, last_gae_lam)
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.values

        elif "no_discount_factor" in self.mutation:
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_values = self.values[step + 1]
                delta = self._calculate_delta(step, next_non_terminal, next_values)
                last_gae_lam = self._calculate_gae(delta, next_non_terminal, last_gae_lam)
                self.advantages[step] = last_gae_lam
            self.returns = self.advantages + self.values
        else:
            raise Exception("Wrong mutation mode for RolloutBuffer")


class MutatedOnPolicyAlgorithm(OnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        mutation: dict = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=supported_action_spaces,
        )
        self.mutation = mutation
        self._configure_policy_mutation()

    def _configure_policy_mutation(self):
        if "policy_optimizer_change" in self.mutation:
            optimizer_name = self.mutation["policy_optimizer_change"]
            optimizer_map = {
                "SGD": th.optim.SGD,
                "Adam": th.optim.Adam,
                "RMSProp": th.optim.RMSprop,
            }
            if optimizer_name in optimizer_map:
                self.policy_kwargs["optimizer_class"] = optimizer_map[optimizer_name]
            else:
                raise ValueError(
                    f"Mutation magnitude {optimizer_name} is not supported for mutation policy_optimizer_change"
                )

        if "policy_activation_change" in self.mutation:
            activation_name = self.mutation["policy_activation_change"]
            activation_map = {"Sigmoid": th.nn.Sigmoid, "ReLU": th.nn.ReLU}
            if activation_name in activation_map:
                self.policy_kwargs["activation_fn"] = activation_map[activation_name]
            else:
                raise ValueError(
                    f"Mutation magnitude {activation_name} is not supported for mutation policy_activation_change"
                )

    def _setup_rollout_buffer(self):
        if (
            "no_reverse" not in self.mutation
            and "no_discount_factor" not in self.mutation
        ):
            buffer_cls = (
                DictRolloutBuffer
                if isinstance(self.observation_space, gym.spaces.Dict)
                else RolloutBuffer
            )
            self.rollout_buffer = buffer_cls(
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )
        else:
            assert (
                "no_reverse" in self.mutation or "no_discount_factor" in self.mutation
            ), "Invalid mutation mode for rollout buffer"
            self.rollout_buffer = MutatedRolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                mutation=self.mutation,
            )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self._setup_rollout_buffer()
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def _handle_timeout_reward(self, rewards: np.ndarray, dones: np.ndarray, infos: List[Dict[str, Any]]) -> np.ndarray:
        if "missing_terminal_state" not in self.mutation:
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
        return rewards

    def _collect_rollout_step(self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer):
        if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
            self.policy.reset_noise(env.num_envs)

        with th.no_grad():
            obs_tensor = obs_as_tensor(self._last_obs, self.device)
            actions, values, log_probs = self.policy(obs_tensor)
        actions = actions.cpu().numpy()

        clipped_actions = actions
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )

        new_obs, rewards, dones, infos = env.step(clipped_actions)
        self.num_timesteps += env.num_envs

        callback.update_locals(locals())
        if callback.on_step() is False:
            return False

        self._update_info_buffer(infos)
        n_steps += 1

        if isinstance(self.action_space, gym.spaces.Discrete):
            actions = actions.reshape(-1, 1)

        rewards = self._handle_timeout_reward(rewards, dones, infos)

        rollout_buffer.add(
            self._last_obs,
            actions,
            rewards,
            self._last_episode_starts,
            values,
            log_probs,
        )
        if "missing_state_update" in self.mutation:
            self._last_episode_starts = dones
        else:
            self._last_obs = new_obs
            self._last_episode_starts = dones

        return new_obs, rewards, dones, infos, actions, values, log_probs

    def mutated_collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            result = self._collect_rollout_step(env, callback, rollout_buffer)
            if result is False:
                return False

            new_obs, rewards, dones, infos, actions, values, log_probs = result

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MutatedOnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "MutatedOnPolicyAlgorithm":
        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.mutated_collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            ) if ("missing_terminal_state" in self.mutation or "missing_state_update" in self.mutation) else self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if log_interval is not None and iteration % log_interval == 0:
                fps = int(
                    (self.num_timesteps - self._num_timesteps_at_start)
                    / (time.time() - self.start_time)
                )
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "rollout/ep_rew_mean",
                        safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                    )
                    self.logger.record(
                        "rollout/ep_len_mean",
                        safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                    )
                self.logger.record("time/fps", fps)
                self.logger.record(
                    "time/time_elapsed",
                    int(time.time() - self.start_time),
                    exclude="tensorboard",
                )
                self.logger.record(
                    "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                )
                self.logger.dump(step=self.num_timesteps)

            self.train()
        callback.on_training_end()
        return self


class MutatedOffPolicyAlgorithm(OffPolicyAlgorithm):
    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        mutation: dict = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_support=sde_support,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.mutation = mutation
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer_class = replay_buffer_class
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self._episode_storage = None
        self.train_freq = train_freq
        self.actor = None
        self.replay_buffer = None
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        self.use_sde_at_warmup = use_sde_at_warmup
        self._configure_policy_mutation()

    def _configure_policy_mutation(self):
        if "policy_optimizer_change" in self.mutation:
            optimizer_name = self.mutation["policy_optimizer_change"]
            optimizer_map = {
                "SGD": th.optim.SGD,
                "Adam": th.optim.Adam,
                "RMSProp": th.optim.RMSprop,
            }
            if optimizer_name in optimizer_map:
                self.policy_kwargs["optimizer_class"] = optimizer_map[optimizer_name]
            else:
                raise ValueError(
                    f"Mutation magnitude {optimizer_name} is not supported for mutation policy_optimizer_change"
                )

        if "policy_activation_change" in self.mutation:
            activation_name = self.mutation["policy_activation_change"]
            activation_map = {"Sigmoid": th.nn.Sigmoid, "ReLU": th.nn.ReLU}
            if activation_name in activation_map:
                self.policy_kwargs["activation_fn"] = activation_map[activation_name]
            else:
                raise ValueError(
                    f"Mutation magnitude {activation_name} is not supported for mutation policy_activation_change"
                )

    def _setup_replay_buffer(self):
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                raise (
                    "We cant use DictReplayBuffer yet. You can't use vectorized environments with this"
                )
            else:
                if (
                    "missing_terminal_state" in self.mutation
                    or "missing_state_update" in self.mutation
                ):
                    if 'mutation' not in self.replay_buffer_kwargs:
                        self.replay_buffer_kwargs['mutation'] = self.mutation
                    self.replay_buffer_class = MutatedReplayBuffer
                else:
                    self.replay_buffer_class = ReplayBuffer
        elif self.replay_buffer_class == HerReplayBuffer:
            raise (
                "We cant use HerReplayBuffer yet. You can't use vectorized environments with this"
            )

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self._setup_replay_buffer()
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)
        self._convert_train_freq()


class MutatedA2C(MutatedOnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        mutation: dict = None,
    ):
        super(MutatedA2C, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            mutation=mutation,
        )
        self.normalize_advantage = normalize_advantage
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(
                alpha=0.99, eps=rms_prop_eps, weight_decay=0
            )
        if _init_setup_model:
            self._setup_model()

    def _calculate_policy_loss(self, advantages, log_prob):
        if "incorrect_loss_function" in self.mutation:
            return (advantages * log_prob).mean()
        else:
            return -(advantages * log_prob).mean()

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for rollout_data in self.rollout_buffer.get(batch_size=None):
            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.long().flatten()

            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions
            )
            values = values.flatten()
            advantages = rollout_data.advantages

            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            policy_loss = self._calculate_policy_loss(advantages, log_prob)
            value_loss = F.mse_loss(rollout_data.returns, values)

            if entropy is None:
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "MA2C",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )


class MutatedPPO(MutatedOnPolicyAlgorithm):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        mutation: dict = None,
    ):
        super(MutatedPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
