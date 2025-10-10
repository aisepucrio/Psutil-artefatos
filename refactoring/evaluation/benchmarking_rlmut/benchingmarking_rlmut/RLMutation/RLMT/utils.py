import psutil
import csv
import os
import gym, sys
import numpy as np
import stable_baselines3
import torch as th
import pandas as pd
from gym import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from codecarbon import EmissionsTracker
import settings
from patsy import dmatrices
import statsmodels.api as sm
import statsmodels.stats.power as pw

# Start CodeCarbon tracker with project name
tracker = EmissionsTracker()
tracker.start()

# Initial system metrics (removed unnecessary calls)
mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)

########################################################################################################################
# The goal is to simulate the faults that can occur in user's code.
########################################################################################################################

########################################################################################################################
# Implemented Algorithms
########################################################################################################################
SUPPORTED_ALGORITHMS = ["A2C", "PPO", "DQN"]


def calculate_p_value_glm(orig_accuracy_list, accuracy_list):
    """Calculate p-value using GLM method."""
    list_length = len(orig_accuracy_list)
    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list
    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)
    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    pv = str(glm_results.summary().tables[1][2][4])
    p_value_g = float(pv)
    return p_value_g


def calculate_cohen_d(orig_accuracy_list, accuracy_list):
    """Calculates Cohen's kappa value."""
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2 + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return abs(result)


def calculate_power(orig_accuracy_list, mutation_accuracy_list):
    """Calculate test method power."""
    eff_size = calculate_cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow_ = pw.FTestAnovaPower().solve_power(effect_size=eff_size,
                                            nobs=len(orig_accuracy_list) + len(mutation_accuracy_list), alpha=0.05)
    return pow_


def calculate_hellinger_distance(p, q):
    """Calculate the Hellinger distance between discrete distributions."""
    from scipy.spatial.distance import euclidean
    return euclidean(np.sqrt(p), np.sqrt(q)) / np.sqrt(2.0)


def _create_a2c_model(environment, hyper_parameters, tensorboard_directory, mutation):
    """Creates an A2C model."""
    from stable_baselines3.a2c.a2c import A2C
    from stable_baselines3.a2c.policies import MlpPolicy

    if mutation is None or not any(mut in mutation for mut in settings.MUTATION_AGENT_LIST):
        model = A2C(
            policy=MlpPolicy,
            env=environment,
            learning_rate=hyper_parameters.get("learning_rate", 0.001),
            n_steps=hyper_parameters.get("n_steps", 5),
            gamma=hyper_parameters.get("gamma", 0.99),
            ent_coef=hyper_parameters.get("ent_coef", 0.0),
            vf_coef=hyper_parameters.get("vf_coef", 0.5),
            use_rms_prop=hyper_parameters.get("use_rms_prop", False),
            normalize_advantage=hyper_parameters.get("normalize_advantage", False),
            tensorboard_log=tensorboard_directory,
            seed=hyper_parameters.get("seed"),
        )
    else:
        from mutated_algorithms import MutatedA2C
        model = MutatedA2C(
            policy=MlpPolicy,
            env=environment,
            learning_rate=hyper_parameters.get("learning_rate", 0.001),
            n_steps=hyper_parameters.get("n_steps", 5),
            gamma=hyper_parameters.get("gamma", 0.99),
            ent_coef=hyper_parameters.get("ent_coef", 0.0),
            vf_coef=hyper_parameters.get("vf_coef", 0.5),
            use_rms_prop=hyper_parameters.get("use_rms_prop", False),
            normalize_advantage=hyper_parameters.get("normalize_advantage", False),
            tensorboard_log=tensorboard_directory,
            seed=hyper_parameters.get("seed"),
            mutation=mutation
        )
    return model


def _create_ppo_model(environment, hyper_parameters, tensorboard_directory, mutation):
    """Creates a PPO model."""
    from stable_baselines3.ppo.ppo import PPO
    from stable_baselines3.ppo.policies import MlpPolicy

    if mutation is None or not any(mut in mutation for mut in settings.MUTATION_AGENT_LIST):
        model = PPO(
            policy=MlpPolicy,
            env=environment,
            seed=hyper_parameters.get("seed", 0),
            n_steps=hyper_parameters.get("n_steps", 2048),
            batch_size=hyper_parameters.get("batch_size", 64),
            gae_lambda=hyper_parameters.get("gae_lambda", 0.95),
            gamma=hyper_parameters.get("gamma", 0.99),
            n_epochs=hyper_parameters.get("n_epochs", 4),
            ent_coef=hyper_parameters.get("ent_coef", 0.0),
            tensorboard_log=tensorboard_directory,
        )
    else:
        from mutated_algorithms import MutatedPPO
        model = MutatedPPO(
            policy=MlpPolicy,
            env=environment,
            seed=hyper_parameters.get("seed", 0),
            n_steps=hyper_parameters.get("n_steps", 2048),
            batch_size=hyper_parameters.get("batch_size", 64),
            gae_lambda=hyper_parameters.get("gae_lambda", 0.95),
            gamma=hyper_parameters.get("gamma", 0.99),
            n_epochs=hyper_parameters.get("n_epochs", 4),
            ent_coef=hyper_parameters.get("ent_coef", 0.0),
            tensorboard_log=tensorboard_directory,
            mutation=mutation
        )
    return model


def _create_dqn_model(environment, hyper_parameters, tensorboard_directory, mutation):
    """Creates a DQN model."""
    from stable_baselines3.dqn.dqn import DQN
    from stable_baselines3.dqn.policies import MlpPolicy

    if mutation is None or not any(mut in mutation for mut in settings.MUTATION_AGENT_LIST):
        model = DQN(
            policy=MlpPolicy,
            env=environment,
            learning_rate=hyper_parameters.get("learning_rate", 1e-4),
            gamma=hyper_parameters.get("gamma", 0.99),
            batch_size=hyper_parameters.get("batch_size", 32),
            buffer_size=hyper_parameters.get("buffer_size", 1e6),
            learning_starts=hyper_parameters.get("learning_starts", 50000),
            target_update_interval=hyper_parameters.get("target_update_interval", 10000),
            train_freq=hyper_parameters.get("train_freq", 4),
            gradient_steps=hyper_parameters.get("gradient_steps", 1),
            exploration_fraction=hyper_parameters.get("exploration_fraction", 0.1),
            exploration_final_eps=hyper_parameters.get("exploration_final_eps", 0.05),
            policy_kwargs=hyper_parameters.get("policy_kwargs"),
            tensorboard_log=tensorboard_directory,
            seed=hyper_parameters.get("seed"),
        )
    else:
        from mutated_algorithms import MutatedDQN
        model = MutatedDQN(
            policy=MlpPolicy,
            env=environment,
            learning_rate=hyper_parameters.get("learning_rate", 1e-4),
            gamma=hyper_parameters.get("gamma", 0.99),
            batch_size=hyper_parameters.get("batch_size", 32),
            buffer_size=hyper_parameters.get("buffer_size", 1e6),
            learning_starts=hyper_parameters.get("learning_starts", 50000),
            target_update_interval=hyper_parameters.get("target_update_interval", 10000),
            train_freq=hyper_parameters.get("train_freq", 4),
            gradient_steps=hyper_parameters.get("gradient_steps", 1),
            exploration_fraction=hyper_parameters.get("exploration_fraction", 0.1),
            exploration_final_eps=hyper_parameters.get("exploration_final_eps", 0.05),
            policy_kwargs=hyper_parameters.get("policy_kwargs"),
            tensorboard_log=tensorboard_directory,
            seed=hyper_parameters.get("seed"),
            mutation=mutation
        )
    return model


def create_model(
        algorithm: str,
        environment: gym.Env,
        hyper_parameters: dict,
        tensorboard_directory: str,
        mutation: dict = None,
):
    """Create an agent's model."""
    algorithm = algorithm.upper()
    assert algorithm in SUPPORTED_ALGORITHMS, f"Algorithm {algorithm} is not supported yet"
    assert "seed" in hyper_parameters, "Seed must be specified"

    if algorithm == "A2C":
        return _create_a2c_model(environment, hyper_parameters, tensorboard_directory, mutation)
    elif algorithm == "PPO":
        return _create_ppo_model(environment, hyper_parameters, tensorboard_directory, mutation)
    elif algorithm == "DQN":
        return _create_dqn_model(environment, hyper_parameters, tensorboard_directory, mutation)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def _load_a2c_model(model_path, environment):
    """Loads an A2C model."""
    from stable_baselines3.a2c.a2c import A2C
    model = A2C.load(
        model_path,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
        },
    )
    model.set_env(environment)
    return model


def _load_ppo_model(model_path, environment):
    """Loads a PPO model."""
    from stable_baselines3.ppo.ppo import PPO
    model = PPO.load(
        model_path,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )
    model.set_env(environment)
    return model


def _load_dqn_model(model_path, environment):
    """Loads a DQN model."""
    from stable_baselines3.dqn.dqn import DQN
    model = DQN.load(
        model_path,
        custom_objects={
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "exploration_schedule": None
        },
    )
    model.set_env(environment)
    return model


def load_model(algorithm: str, environment: gym.Env, model_path: str):
    """Load agent's model from the specified path."""
    algorithm = algorithm.upper()
    if algorithm == "A2C":
        return _load_a2c_model(model_path, environment)
    elif algorithm == "PPO":
        return _load_ppo_model(model_path, environment)
    elif algorithm == "DQN":
        return _load_dqn_model(model_path, environment)
    else:
        raise ValueError("Model not implemented")


if __name__ == "__main__":
    test_model = create_model(
        "ma2c",
        gym.make("CartPole-v1"),
        {"seed": 0},
        "runs/healthy",
        mutation={"incorrect_loss_function": None},
    )
    test_model.learn(total_timesteps=100000)
    print("hi")

tracker.stop()
