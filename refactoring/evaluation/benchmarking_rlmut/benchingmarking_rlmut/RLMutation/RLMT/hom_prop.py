import psutil
import csv
import os
import re
import ast
import multiprocessing as mp
import argparse
from typing import List, Tuple

import torch as th
import numpy as np
import statsmodels as sm

from codecarbon import EmissionsTracker
from gym.envs.registration import register
import gym

import utils as utils
from agent import Agent
import settings
from dataclasses import dataclass

MAX_CPU = None


@dataclass
class EnvConfig:
    init_val: List[float]
    limits: List[Tuple[float]]


def natural_keys(text):
    """
    Sorts strings with embedded numbers naturally.
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


def correct_reward_list(rew_dict):
    """
    Orders agent rewards correctly.
    """
    my_list = list(rew_dict.keys())
    my_list.sort(key=natural_keys)
    return [rew_dict[k][0] for k in my_list if k in rew_dict]


def calculate_hellinger_distances(s, s2, num_iterations=2000):
    """
    Calculates Hellinger distances for mt_dtr.
    """
    bin_edges = np.histogram_bin_edges(np.concatenate((s, s2)), bins='auto')
    dist_orig, dist = [], []

    for _ in range(num_iterations):
        ind = np.arange(len(s))
        ind_unknown = np.random.choice(ind, size=len(s) // 2, replace=False)
        ind_choice = list(set(ind) - set(ind_unknown))
        acc_choice2 = np.array(s)[ind_unknown]
        acc_choice = np.array(s)[ind_choice]
        hist, _ = np.histogram(acc_choice, density=True, bins=bin_edges)
        hist_mut, _ = np.histogram(acc_choice2, density=True, bins=bin_edges)
        dist_orig.append(utils.hellinger_distance(hist, hist_mut))

    for _ in range(num_iterations):
        acc_choice2 = np.random.choice(s2, size=len(s) // 2, replace=False)
        acc_choice = np.random.choice(s, size=len(s) // 2, replace=False)
        hist, _ = np.histogram(acc_choice, density=True, bins=bin_edges)
        hist_mut, _ = np.histogram(acc_choice2, density=True, bins=bin_edges)
        dist.append(utils.hellinger_distance(hist, hist_mut))

    return dist_orig, dist


def perform_statistical_tests(dist_orig, dist):
    """
    Performs statistical tests for mt_dtr.
    """
    if dist != dist_orig:
        try:
            p_value = utils.p_value_glm(dist_orig, dist)
            effect_size = utils.cohen_d(dist_orig, dist)
            power = utils.power(dist_orig, dist)
        except sm.tools.sm_exceptions.PerfectSeparationError:
            return True

        if power < 0.8:
            return False
        else:
            return p_value < 0.05 and effect_size >= 0.5
    return False


def mt_dtr(s, s2):
    """
    'dtr' test method, i.e. statistical test of the distribution of the intra/inter hellinger distance
    """
    dist_orig, dist = calculate_hellinger_distances(s, s2)
    return perform_statistical_tests(dist_orig, dist)


def mt_rew(s, s2):
    """
    'r' test method, i.e. statistical test of the distribution of rewards
    """
    if s != s2:
        p_value = utils.p_value_glm(s, s2)
        effect_size = utils.cohen_d(s, s2)
        power = utils.power(s, s2)

        if power < 0.8:
            return False
        else:
            return p_value < 0.05 and effect_size >= 0.5
    return False


def hom_check_property(dict_mut: dict, list_hom: list, current_mut_name: str):
    """
    Checking the type of HOM we have.
    """
    foms = current_mut_name.split('-')
    fom_1 = dict_mut[foms[0]]
    fom_2 = dict_mut[foms[1]]

    hom_kills_count = np.sum(list_hom)
    fom_union_count = np.sum(fom_1 + fom_2)
    fom_intersection = fom_1 * fom_2

    if (hom_kills_count < fom_union_count or (hom_kills_count == fom_union_count and np.array_equal(fom_1 + fom_2, fom_intersection))) and hom_kills_count > 0:
        if np.sum((fom_1 + fom_2) * list_hom) == 0:
            return 'Weakly Subsuming and Decoupled'
        elif sum([x == True and y == False for (x, y) in zip(list_hom, fom_intersection)]) > 0:
            return 'Weakly Subsuming and Coupled'
        else:
            return 'Strongly Subsuming and Coupled'
    else:
        return 'Non-subsuming'


def load_agents(algorithm, environment_name, agent_type, mutation_name=None):
    """Loads and initializes agents."""
    base_path = os.path.join('..', 'experiments', 'Mutated_Agents' if agent_type == 'Mutated' else 'Healthy_Agents',
                             'HighOrderMutation' if agent_type == 'Mutated' else '', environment_name, algorithm)
    if mutation_name:
        base_path = os.path.join(base_path, mutation_name, "logs")
    else:
        base_path = os.path.join(base_path, "logs")

    agent_count = 20
    agents = [
        Agent(
            algorithm, {"n_episodes": 10}, environment_name, i, False,
            os.path.join(base_path, f"{mutation_name if mutation_name else agent_type}_{algorithm}_{environment_name}_{i}", "best_model")
        )
        for i in range(agent_count)
    ]
    return agents


def run_agent_tests(agents, environment_name, max_cpu):
    """Runs tests for agents and returns the results."""
    manager = mp.Manager()
    return_dict = manager.dict()

    with mp.Pool(processes=max_cpu):
        processes = [mp.Process(target=agent.test, args=(return_dict,)) for agent in agents]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    return correct_reward_list(return_dict)


def register_custom_environment(environment_name, params):
    """Registers a custom environment with specific parameters."""
    register(
        id=f"Custom{environment_name}",
        entry_point=f"custom_test_env:Custom{environment_name.split('-')[0]}",
        max_episode_steps=settings.CUSTOM_ENV_SETTINGS[environment_name]['max_episode_steps'],
        reward_threshold=settings.CUSTOM_ENV_SETTINGS[environment_name]['reward_threshold'],
        kwargs={'params': params}
    )


def process_test_environment(h_agents, m_agents, test_env_params, args, max_cpu):
    """Processes a single test environment."""
    environment_name = args.environment_name
    register_custom_environment(environment_name, test_env_params)

    for agent in h_agents + m_agents:
        agent.environment = f"Custom{environment_name}"
        agent.init_agent(test=True)

    healthy_rewards = run_agent_tests(h_agents, environment_name, max_cpu)
    mutated_rewards = run_agent_tests(m_agents, environment_name, max_cpu)

    if args.test_mode == 'r':
        return mt_rew(healthy_rewards, mutated_rewards)
    else:
        return mt_dtr(healthy_rewards, mutated_rewards)


def main():
    """Main function to orchestrate the HOM analysis."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment_name", help="Environment's name", required=False, default="CartPole-v1")
    parser.add_argument("-a", "--algorithm", help="DRL algorithm to use", required=False, default="ppo")
    parser.add_argument("-t", "--test_mode", help="Test mode. 'r' for reward based or 'dtr' for distance to reward",
                        required=False, default="r")
    parser.add_argument("-n", "--ncpus",
                        help="Number of cpus to run agents on. GPU is not handle for now. Default is number of cpus available - 1",
                        required=False, default=mp.cpu_count() - 1)
    args = parser.parse_args()

    assert args.environment_name in settings.CUSTOM_ENV_SETTINGS, f"Unknown environment name {args.environment_name}"

    global MAX_CPU
    MAX_CPU = int(args.ncpus)

    file_name_prefix = f"mut_{args.algorithm}_{args.environment_name}"
    file_name = f"{file_name_prefix}{'_dtr' if args.test_mode == 'dtr' else ''}"

    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker()
    tracker.start()

    # Load FOM killed data
    fom_killed_file = os.path.join('fom_test_env_killed', f"{file_name_prefix}.csv")
    with open(fom_killed_file, mode='r') as infile:
        reader = csv.reader(infile, delimiter=";")
        csv_dict = {}
        for ind, rows in enumerate(reader):
            if ind != 0:
                csv_dict.update({rows[0]: np.array([True if x == 'True' else False for x in rows[1:-1]])})
            else:
                csv_dict.update({"Env": rows[1:-1]})

    test_env = [ast.literal_eval(x) for x in csv_dict['Env']]

    # Load Healthy Agents
    h_agents = load_agents(args.algorithm, args.environment_name, 'Healthy')

    # Get HOM list
    hom_dir = os.path.join('..', 'experiments', 'Mutated_Agents', 'HighOrderMutation', args.environment_name, args.algorithm)
    considered_mutation = [f for f in os.listdir(hom_dir) if os.path.isdir(os.path.join(hom_dir, f))]

    # Prepare output file
    output_file_name = f"hom_{file_name}.csv"
    with open(output_file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        fields = ['Mutation'] + ['{}'.format(p) for p in test_env] + ['Total'] + ['Properties']
        csv_writer.writerow(fields)

    for k, mut in enumerate(considered_mutation):
        print(f"Evaluating mutations... {k + 1}/{len(considered_mutation)}")

        # Load Mutated Agents
        m_agents = load_agents(args.algorithm, args.environment_name, 'Mutated', mut)

        list_mut = []
        for ind, p in enumerate(test_env):
            print(f"\nTesting {ind + 1}/{len(test_env)} environment, parameters {p}")
            result = process_test_environment(h_agents, m_agents, p, args, MAX_CPU)
            list_mut.append(result)

        # Write results
        with open(output_file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=';')
            row = [mut] + [l for l in list_mut] + [sum(list_mut)] + [hom_check_property(csv_dict, list_mut, mut)]
            csv_writer.writerow(row)

    tracker.stop()


if __name__ == '__main__':
    th.set_num_threads(1)
    gym.logger.set_level(40)
    main()
