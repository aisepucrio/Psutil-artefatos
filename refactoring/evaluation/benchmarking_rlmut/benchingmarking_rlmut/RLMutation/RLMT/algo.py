import psutil
import csv
import os
import re
import ast
import multiprocessing as mp
import numpy as np
import statsmodels as sm
from dataclasses import dataclass
import argparse
import torch as th
from gym.envs.registration import register
import gym
from codecarbon import EmissionsTracker
from typing import List, Tuple
import utils
from agent import Agent
import settings

th.set_num_threads(1)
gym.logger.set_level(40)
MAX_CPU = None


@dataclass
class EnvConfig:
    init_val: List[float]
    limits: List[Tuple[float]]


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)$', text)]


def correct_reward_list(rew_dict):
    my_list = list(rew_dict.keys())
    my_list.sort(key=natural_keys)
    return [rew_dict[k][0] for k in my_list if k in rew_dict]


def _calculate_hellinger_distance(s, s2, bin_edges):
    hist, _ = np.histogram(s, density=True, bins=bin_edges)
    hist_mut, _ = np.histogram(s2, density=True, bins=bin_edges)
    return utils.hellinger_distance(hist, hist_mut)


def _perform_dtr_test(s, s2):
    dist_orig, dist = [], []
    bin_edges = np.histogram_bin_edges(np.concatenate((s, s2)), bins='auto')

    for _ in range(2000):
        ind = np.arange(len(s))
        ind_unknown = np.random.choice(ind, size=len(s) // 2, replace=False)
        ind_choice = list(set(ind) - set(ind_unknown))
        acc_choice2 = np.array(s)[ind_unknown]
        acc_choice = np.array(s)[ind_choice]
        dist_orig.append(_calculate_hellinger_distance(acc_choice, acc_choice2, bin_edges))

    for _ in range(2000):
        acc_choice2 = np.random.choice(s2, size=len(s) // 2, replace=False)
        acc_choice = np.random.choice(s, size=len(s) // 2, replace=False)
        dist.append(_calculate_hellinger_distance(acc_choice, acc_choice2, bin_edges))

    if dist == dist_orig:
        return False

    try:
        p_value = utils.p_value_glm(dist_orig, dist)
        effect_size = utils.cohen_d(dist_orig, dist)
        power = utils.power(dist_orig, dist)
    except sm.tools.sm_exceptions.PerfectSeparationError:
        return True

    return power >= 0.8 and p_value < 0.05 and effect_size >= 0.5


def mt_dtr(s, s2):
    return _perform_dtr_test(s, s2)


def mt_rew(s, s2):
    if s == s2:
        return False
    p_value = utils.p_value_glm(s, s2)
    effect_size = utils.cohen_d(s, s2)
    power = utils.power(s, s2)
    return power >= 0.8 and p_value < 0.05 and effect_size >= 0.5


def _register_custom_env(environment_name, current_val):
    register(
        id=f"Custom{environment_name}",
        entry_point=f"custom_test_env:Custom{environment_name.split('-')[0]}",
        max_episode_steps=settings.CUSTOM_ENV_SETTINGS[environment_name]['max_episode_steps'],
        reward_threshold=settings.CUSTOM_ENV_SETTINGS[environment_name]['reward_threshold'],
        kwargs={'params': current_val}
    )


def _test_agent(agent, return_dict):
    agent.test(return_dict)


def test_val(agents: List[Agent], comp_list: List[float], current_val: List[float], environment_name: str,
             test_mode: str):
    _register_custom_env(environment_name, current_val)
    for i in range(len(agents)):
        agents[i].environment = f"Custom{environment_name}"
        agents[i].init_env(test=True)

    manager = mp.Manager()
    return_dict = manager.dict()

    with mp.Pool(processes=MAX_CPU) as pool:
        processes = [pool.apply_async(_test_agent, args=(agents[i], return_dict,)) for i in range(len(agents))]
        [p.get() for p in processes]

    return mt_dtr(comp_list, correct_reward_list(return_dict)) if test_mode == 'dtr' else mt_rew(comp_list,
                                                                                                 correct_reward_list(
                                                                                                     return_dict))


def _search_boundary(init_env, agents, comp_list, precision, environment_name, test_mode, index, direction):
    print(f"Checking parameter n°{index}")
    print(f"Initial value is {init_env.init_val[index]}\n")
    boundary_value = init_env.init_val[:]
    limit_index = 1 if direction == "upper" else 0
    limit_value = init_env.limits[index][limit_index]
    print(f"{direction.capitalize()} bound...")

    if init_env.init_val[index] != limit_value:
        curr_low = np.array(init_env.init_val)
        curr_up = np.array([init_env.init_val[k] if k != index else limit_value for k in range(len(init_env.init_val))])

        if not test_val(agents, comp_list, curr_up, environment_name, test_mode):
            boundary_value[index] = limit_value
        else:
            while abs((curr_up - curr_low)[index]) > precision[index]:
                m = (curr_up + curr_low) / 2
                print(f"\nUpdating current value to {m[index]}")
                print("Test configuration: ", m)
                if not test_val(agents, comp_list, m, environment_name, test_mode):
                    curr_low = m
                else:
                    curr_up = m
            boundary_value[index] = curr_low[index]
    print(f"Found: {boundary_value[index]}")
    return boundary_value


def _generate_mid_points(boundaries, init_env, agents, comp_list, environment_name, test_mode):
    mid_bounds = []
    for i in range(len(boundaries) - 1):
        for j in range(i + 1, len(boundaries)):
            for p in range(2):
                for l in range(2):
                    if boundaries[i][p] != init_env.init_val and boundaries[j][l] != init_env.init_val:
                        curr_low = np.array(init_env.init_val)
                        curr_up = np.array([boundaries[i][p][k] if k != j else boundaries[j][l][k] for k in
                                            range(len(init_env.init_val))])
                        print("\nSearching middle points between {} and {}".format(curr_low, curr_up))
                        if not test_val(agents, comp_list, curr_up, environment_name, test_mode):
                            mid_bounds.append(list(curr_up))
                        else:
                            while np.linalg.norm(curr_up - curr_low) > np.linalg.norm(precision):
                                m = (curr_up + curr_low) / 2
                                print("\nUpdating current points to {}".format(m))
                                print("Test configuration: ", m)
                                if not test_val(agents, comp_list, m, environment_name, test_mode):
                                    curr_low = m
                                else:
                                    curr_up = m
                            mid_bounds.append(list(curr_low))
    return mid_bounds


def determine_boundaries(init_env: EnvConfig, agents: List[Agent], comp_list: List[float], precision: List[float],
                         environment_name: str, test_mode: str):
    for ind in range(len(init_env.init_val)):
        assert init_env.limits[ind][1] >= init_env.init_val[ind] >= init_env.limits[ind][
            0], f'Initial value of parameter n°{ind} is outside of search limits'

    boundaries = []
    print("Searching boundaries on axis...")
    for ind in range(len(init_env.init_val)):
        b_up = _search_boundary(init_env, agents, comp_list, precision, environment_name, test_mode, ind, "upper")
        b_low = _search_boundary(init_env, agents, comp_list, precision, environment_name, test_mode, ind, "lower")
        print("Bound axis values are [{}, {}]\n".format(b_low[ind], b_up[ind]))
        boundaries.append([b_low, b_up])

    print("Searching mid points boundaries...")
    mid_bounds = _generate_mid_points(boundaries, init_env, agents, comp_list, environment_name, test_mode)
    mid_bounds = [mid_bounds]
    bounds = [list(x) for x in set(tuple(x) for x in [item for sublist in boundaries + mid_bounds for item in sublist])]
    if init_env.init_val not in bounds:
        bounds.append(init_env.init_val)
    return bounds


def _create_agents(algorithm, environment_name, agent_count, is_healthy, agent_type, mutation_mode=None, path_prefix=None):
    agent_list = []
    for i in range(agent_count):
        if agent_type == "healthy":
            path = os.path.join('..', "experiments", 'Healthy_Agents', environment_name, algorithm.upper(), "logs")
            agent = Agent(
                algorithm, {"n_episodes": 10}, environment_name, i, is_healthy, os.path.join(
                    path, f"Healthy_{algorithm}_{environment_name}_{i}", "best_model"
                )
            )
        elif agent_type == "mutated":
            path = os.path.join(path_prefix, mutation_mode, environment_name, algorithm.upper())
            agent = Agent(
                algorithm, {"n_episodes": 10}, environment_name, i, is_healthy, os.path.join(
                    path, f"{mutation_mode}_{algorithm}_{environment_name}_{i}", "best_model"
                )
            )
        agent.init_agent(test=True)
        agent_list.append(agent)
    return agent_list


def _get_comp_list(environment_name, algorithm, mut_mode):
    csv_path = os.path.join('..', 'csv_res', environment_name, algorithm, f'results_{mut_mode}.csv')
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        next(csv_reader)
        row = next(csv_reader)
        comp_list = [float(r) for r in row[1:21]]
    return comp_list


def _test_mutations(h_agents, bounds, args, mut, mut_dict, MAX_CPU):
    res = re.match(r"(results_)(.*)(\.csv)", mut)
    mut_mode = res.groups()[1]
    sp = mut_mode.split('_')
    path_prefix = os.path.join('..', 'csv_res')
    path = os.path.join('..', "experiments", 'Mutated_Agents', 'SingleOrderMutation',
                        list(mut_dict.keys())[list(mut_dict.values()).index(sp[0])], args.environment_name,
                        args.algorithm.upper())
    if len(sp) == 2:
        path = os.path.join(path, sp[1])
    path = os.path.join(path, "logs")

    m_agents = _create_agents(args.algorithm, args.environment_name, 20, False, "mutated", mut_mode, path)

    file_name = f'mut_{args.algorithm}_{args.environment_name}_{"_dtr" if args.test_mode == "dtr" else ""}.csv'
    with open(file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        fields = ['Mutation'] + ['{}'.format(p) for p in bounds] + ['Total']
        csv_writer.writerow(fields)
        dict_mut = {}

        for ind, p in enumerate(bounds):
            print(f"\nTesting {ind + 1}/{len(bounds)} environment, parameters {p}")

            _register_custom_env(args.environment_name, p)

            for agent in h_agents:
                agent.environment = f"Custom{args.environment_name}"
                agent.init_agent(test=True)
            for m_ag in m_agents:
                m_ag.environment = f"Custom{args.environment_name}"
                m_ag.init_agent(test=True)

            manager = mp.Manager()
            return_dict = manager.dict()

            with mp.Pool(processes=MAX_CPU) as pool:
                processes = [pool.apply_async(_test_agent, args=(h_agents[i], return_dict,)) for i in
                             range(len(h_agents))]
                [p.get() for p in processes]

            acc_choice = correct_reward_list(return_dict)

            manager = mp.Manager()
            m_dict_temp = manager.dict()

            with mp.Pool(processes=MAX_CPU) as pool:
                processes = [pool.apply_async(_test_agent, args=(m_agents[i], m_dict_temp,)) for i in
                             range(len(m_agents))]
                [p.get() for p in processes]

            acc_choice2 = correct_reward_list(m_dict_temp)

            if args.test_mode == 'dtr':
                if mut_mode in mut_dict:
                    dict_mut[mut_mode].append(mt_dtr(acc_choice, acc_choice2))
                else:
                    dict_mut[mut_mode] = [mt_dtr(acc_choice, acc_choice2)]
            else:
                if mut_mode in mut_dict:
                    dict_mut[mut_mode].append(mt_rew(acc_choice, acc_choice2))
                else:
                    dict_mut[mut_mode] = [mt_rew(acc_choice, acc_choice2)]

        for key in dict_mut:
            col = [key]
            col = col + dict_mut[key] + [sum(dict_mut[key])]
            csv_writer.writerow(col)


if __name__ == '__main__':
    tracker = EmissionsTracker()
    tracker.start()
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("-e", "--environment_name", help="Environment's name", required=False,
                           default="CartPole-v1")
    my_parser.add_argument("-a", "--algorithm", help="DRL algorithm to use", required=False, default="ppo")
    my_parser.add_argument('-i', '--init_val', type=float, nargs='+',
                           help='Initial environment parameters values to consider. Ex: "-i 1.0 0.1', required=True)
    my_parser.add_argument('-l', '--limits', type=float, nargs='+',
                           help='Upper/Lower bound for each parameters for the search. Synthax is "-l param1_lower param1_upper param2_lower .... Ex for two parameters: "-l 1.0 20.0 0.1 20.0',
                           required=True)
    my_parser.add_argument("-b", "--bounds", type=str,
                           help="Optional. If bounds are provided as a string of a list of list, then search is bypassed.",
                           required=False, default=None)
    my_parser.add_argument("-t", "--test_mode", help="Test mode. 'r' for reward based or 'dtr' for distance to reward",
                           required=False, default="r")
    my_parser.add_argument("-n", "--ncpus",
                           help="Number of cpus to run agents on. GPU is not handle for now. Default is number of cpus available - 1",
                           required=False, default=mp.cpu_count() - 1)
    args = my_parser.parse_args()

    assert 2 * len(args.init_val) == len(
        args.limits), "'--limits' argument should have 2 * number of parameters in '--init_val' argument"
    assert args.environment_name in settings.CUSTOM_ENV_SETTINGS, "Unknown environment name {}".format(
        args.environment_name)

    lim = [args.limits[i:i + 2] for i in range(0, len(args.limits), 2)]
    init_env = EnvConfig(init_val=args.init_val, limits=lim)
    MAX_CPU = args.ncpus

    if args.bounds is None:
        mut_mode = 'Healthy'
        h_agents = _create_agents(args.algorithm, args.environment_name, 20, False, "healthy")
        comp_list = _get_comp_list(args.environment_name, args.algorithm, mut_mode)
        bounds = determine_boundaries(init_env=init_env, agents=h_agents, comp_list=comp_list,
                                      precision=[0.5, 0.05],
                                      environment_name=args.environment_name, test_mode=args.test_mode)
    else:
        bounds = ast.literal_eval(args.bounds)

    print("Environment to consider: ", bounds)
    h_agents = _create_agents(args.algorithm, args.environment_name, 20, False, "healthy")

    considered_mutation = [f for f in os.listdir(os.path.join('..', 'csv_res', args.environment_name, args.algorithm))
                           if 'Healthy' not in f]
    mut_dict = dict(settings.MUTATION_AGENT_LIST)
    mut_dict.update(settings.MUTATION_ENVIRONMENT_LIST)

    for mut in considered_mutation:
        _test_mutations(h_agents, bounds, args, mut, mut_dict, MAX_CPU)

    tracker.stop()
